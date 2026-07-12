"""Orchestratore per l'analisi di testi lunghi (V10.18.0). Passo 2.

Prende un libro, lo segmenta (ft_segmenter, passo 1) e analizza ogni
segmento con la pipeline esistente, in modo RESUMABILE e robusto agli
errori. Non sostituisce la pipeline: la invoca, una volta per segmento.

GARANZIE
--------
- RESUME: un job interrotto (crash, Ctrl-C, llama.cpp caduto, .bat
  sbagliato) riparte dall'ultimo segmento non completato, non da capo. Lo
  stato vive nel manifest, scritto in modo atomico a ogni checkpoint.
- IDEMPOTENZA: rilanciare il runner sulla stessa cartella non rifa' il
  lavoro gia' fatto -- i segmenti 'done' vengono saltati.
- ERRORI per-segmento: un segmento che fallisce non uccide il libro
  (skip-and-continue + dead-letter, salvo --halt-on-failure). Gli errori
  transitori si ritentano con backoff esponenziale.

CHECKPOINT
----------
Dopo ogni transizione di stato di un segmento (running -> done/failed) il
manifest viene riscritto interamente, in modo atomico (write_json_atomic).
La granularita' e' il segmento: su crash si perde al piu' il lavoro di UN
segmento. Risoluzione piu' fine non vale la complessita' per la v1.

PROFONDITA' PER SEGMENTO
------------------------
Default 'base': ogni segmento passa per L0->L4 (classificazione + analisi
causale), niente expand/bridge/magistrale. Costruire ponti cross-scale su
un frammento di poche pagine, senza vedere il resto del libro, darebbe
ponti spuri moltiplicati per N segmenti. La frattalita' d'insieme spetta
alla riduzione gerarchica (passo 3). 'full' resta disponibile per chi la
vuole comunque.
"""
from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from .ft_model import BookManifest, SegmentRecord
from .ft_segmenter import segment_text
from .ft_session import ExplorerSession
from .io_utils import write_json_atomic
from .json_utils import read_json
from .llm import LLMClient


MANIFEST_FILENAME = "book_manifest.json"

# Errori considerati TRANSITORI: vale la pena ritentarli con backoff.
# Tutto cio' che sa di timeout o di rete che non risponde.
_TRANSIENT_MARKERS = ("timeout", "connection", "contattare", "refused",
                      "temporarily", "unavailable", "reset")

# Backoff esponenziale: attesa = base * 2**(tentativo-1), in secondi.
_BACKOFF_BASE_SECONDS = 4.0
_BACKOFF_MAX_SECONDS = 120.0


def _is_transient(err: str) -> bool:
    """True se il messaggio d'errore sembra un guasto transitorio."""
    low = err.lower()
    return any(m in low for m in _TRANSIENT_MARKERS)


def _backoff_seconds(attempt: int) -> float:
    """Attesa prima del tentativo `attempt` (1-based), con tetto."""
    return min(_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), _BACKOFF_MAX_SECONDS)


class BookRunner:
    """Orchestratore resumabile per l'analisi di un libro.

    Uso tipico:
        runner = BookRunner(client, out_dir, num_ctx=8192)
        manifest = runner.run(text, book_id="mio_libro")

    Rilanciando run() sulla stessa out_dir, il manifest esistente viene
    ricaricato e i segmenti 'done' saltati: e' il resume.
    """

    def __init__(
        self,
        client: LLMClient,
        out_dir: str | Path,
        *,
        num_ctx: int = 8192,
        overlap_ratio: float = 0.15,
        per_segment_depth: str = "base",
        max_retries: int = 3,
        halt_on_failure: bool = False,
        max_cross_scale: int = 8,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        sleep_fn: iniettabile, cosi' i test non aspettano davvero il backoff.
        """
        self.client = client
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.segments_root = self.out_dir / "segments"
        self.segments_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / MANIFEST_FILENAME
        self.num_ctx = num_ctx
        self.overlap_ratio = overlap_ratio
        self.per_segment_depth = per_segment_depth
        self.max_retries = max_retries
        self.halt_on_failure = halt_on_failure
        self.max_cross_scale = max_cross_scale
        self.sleep_fn = sleep_fn

    # ----- manifest: load / build / checkpoint ------------------------------

    def _load_manifest(self) -> BookManifest | None:
        """Ricarica il manifest da disco, se esiste. None al primo run."""
        if not self.manifest_path.exists():
            return None
        raw = read_json(self.manifest_path)
        segs = [SegmentRecord(**s) for s in raw.get("segments", [])]
        data = {k: v for k, v in raw.items() if k != "segments"}
        return BookManifest(segments=segs, **data)

    def _build_manifest(self, text: str, book_id: str, source_input_id: str) -> BookManifest:
        """Segmenta il testo e costruisce un manifest fresco (tutti 'pending')."""
        seg_result = segment_text(
            text, num_ctx=self.num_ctx, overlap_ratio=self.overlap_ratio
        )
        now = datetime.now().isoformat(timespec="seconds")
        records = [
            SegmentRecord(
                id=s.id,
                index=s.index,
                chapter_index=s.chapter_index,
                chapter_title=s.chapter_title,
                status="pending",
                out_dir=str((self.segments_root / s.id).relative_to(self.out_dir)),
                est_tokens=s.est_tokens,
            )
            for s in seg_result.segments
        ]
        return BookManifest(
            book_id=book_id,
            source_input_id=source_input_id,
            created_at=now,
            updated_at=now,
            num_ctx=self.num_ctx,
            overlap_ratio=self.overlap_ratio,
            token_budget=seg_result.token_budget,
            used_chapters=seg_result.used_chapters,
            per_segment_depth=self.per_segment_depth,
            halt_on_failure=self.halt_on_failure,
            max_retries=self.max_retries,
            segments=records,
            segmenter_notes=seg_result.notes,
        )

    def _checkpoint(self, manifest: BookManifest) -> None:
        """Scrive il manifest su disco, in modo atomico. Chiamato a ogni
        transizione di stato di un segmento."""
        manifest.updated_at = datetime.now().isoformat(timespec="seconds")
        write_json_atomic(manifest, self.manifest_path)

    # ----- esecuzione di un singolo segmento --------------------------------

    def _segment_text_for(self, full_text: str, index: int) -> str:
        """Ri-deriva il testo del segmento `index`.

        Il manifest non salva il testo dei segmenti (sarebbe enorme e
        ridondante). Lo si ri-ottiene segmentando di nuovo: l'operazione e'
        deterministica, quindi al resume produce identici segmenti.
        """
        seg_result = segment_text(
            full_text, num_ctx=self.num_ctx, overlap_ratio=self.overlap_ratio
        )
        return seg_result.segments[index].text

    def _run_one_segment(
        self, rec: SegmentRecord, seg_text: str, progress: Callable[[str], None]
    ) -> None:
        """Esegue la pipeline su un segmento. Aggiorna `rec` in place.

        Due categorie di fallimento, gestite diversamente:

        1. ECCEZIONE da analyze() -- guasto vero (rete, I/O). Se transitorio,
           retry con backoff fino a max_retries; se definitivo, stop subito.
        2. DEGRADO SILENZIOSO -- analyze() non solleva ma la pipeline ha
           degradato (RoleAgent cattura gli errori dei singoli agent). Il
           sintomo e' un esito vuoto: ft senza item. Va trattato come
           fallimento, altrimenti un segmento "vuoto" passerebbe per 'done'.

        Non solleva: il chiamante decide se proseguire o fermarsi.
        """
        seg_out = self.out_dir / rec.out_dir
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            rec.attempts = attempt
            started = time.perf_counter()
            try:
                sess = ExplorerSession.analyze(
                    self.client,
                    seg_out,
                    seg_text,
                    source_input_id=rec.id,
                    max_cross_scale=self.max_cross_scale,
                )
                # categoria 2: la pipeline non ha sollevato, ma ha prodotto
                # un esito vuoto -> degrado silenzioso, e' un fallimento.
                ok, why = self._outcome_is_valid(sess, seg_out)
                if not ok:
                    last_error = why
                    transient = _is_transient(last_error)
                    progress(f"   {rec.id}: esito non valido (tentativo "
                             f"{attempt}/{self.max_retries}) "
                             f"{'[transitorio]' if transient else '[definitivo]'}: {why}")
                    if not transient:
                        break
                    if attempt < self.max_retries:
                        wait = _backoff_seconds(attempt)
                        progress(f"   {rec.id}: backoff {wait:.0f}s prima di ritentare")
                        self.sleep_fn(wait)
                    continue

                if self.per_segment_depth == "full":
                    sess.auto_explore(progress=lambda m: progress(f"      {m}"))
                rec.elapsed_seconds = round(time.perf_counter() - started, 3)
                rec.status = "done"
                rec.error = ""
                progress(f"   {rec.id}: done in {rec.elapsed_seconds}s "
                         f"(tentativo {attempt})")
                return
            except Exception as exc:  # noqa: BLE001 -- categoria 1: vogliamo tutto
                last_error = f"{type(exc).__name__}: {exc}"
                transient = _is_transient(last_error)
                progress(f"   {rec.id}: errore (tentativo {attempt}/"
                         f"{self.max_retries}) {'[transitorio]' if transient else '[definitivo]'}: "
                         f"{last_error}")
                if not transient:
                    break  # errore definitivo: inutile ritentare
                if attempt < self.max_retries:
                    wait = _backoff_seconds(attempt)
                    progress(f"   {rec.id}: backoff {wait:.0f}s prima di ritentare")
                    self.sleep_fn(wait)

        # qui solo se tutti i tentativi sono falliti o errore definitivo
        rec.status = "failed"
        rec.dead_letter = True
        rec.error = last_error
        rec.elapsed_seconds = 0.0
        progress(f"   {rec.id}: FAILED definitivo -> dead-letter. {last_error}")

    @staticmethod
    def _outcome_is_valid(sess: "ExplorerSession", seg_out: Path) -> tuple[bool, str]:
        """Valida l'esito di analyze() quando non ha sollevato eccezioni.

        La pipeline degrada silenziosamente: se TUTTI gli agent falliscono,
        analyze() ritorna comunque, ma con un ft senza item. Quel segmento
        non e' 'done', e' fallito. Diagnosi: ft vuoto.

        Ritorna (valido, motivo). Il motivo, se non valido, e' classificato
        transitorio/definitivo da _is_transient sul testo della telemetria.
        """
        if sess.ft.items:
            return True, ""
        # ft vuoto: leggiamo la telemetria per capire perche', cosi'
        # _is_transient puo' decidere se vale la pena ritentare.
        why = "pipeline degradata: nessun item prodotto (ft vuoto)"
        tel = seg_out / "telemetry.jsonl"
        if tel.exists():
            txt = tel.read_text(encoding="utf-8", errors="replace").lower()
            if "timeout" in txt or "contattare" in txt or "connection" in txt:
                why = f"timeout/connessione durante la pipeline ({why})"
        return False, why

    # ----- ciclo principale -------------------------------------------------

    def run(
        self,
        text: str,
        *,
        book_id: str = "book",
        source_input_id: str = "input_001",
        retry_dead_letter: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> BookManifest:
        """Analizza il libro `text`. Resumabile: se esiste gia' un manifest
        nella out_dir, riprende da li'.

        retry_dead_letter: se True, i segmenti finiti in dead-letter in un run
        precedente vengono ri-tentati (utile dopo aver risolto la causa
        esterna -- es. llama.cpp riavviato). Se False, restano saltati.

        Ritorna il BookManifest finale (con i conteggi per stato).
        """
        say = progress if progress is not None else (lambda _m: None)

        manifest = self._load_manifest()
        if manifest is None:
            say(f"[book] nuovo job '{book_id}': segmento il testo "
                f"({len(text)} caratteri, num_ctx={self.num_ctx})")
            manifest = self._build_manifest(text, book_id, source_input_id)
            self._checkpoint(manifest)
            say(f"[book] {len(manifest.segments)} segmenti "
                f"(capitoli={'si' if manifest.used_chapters else 'no, fallback paragrafo'}, "
                f"budget={manifest.token_budget} token)")
        else:
            c = manifest.counts()
            say(f"[book] RESUME job '{manifest.book_id}': "
                f"done={c['done']} pending={c['pending']} failed={c['failed']} "
                f"-- riprendo i non completati")

        # un eventuale 'running' rimasto da un crash precedente va ritrattato
        # come 'pending': il segmento era a meta', va rifatto da capo.
        for rec in manifest.segments:
            if rec.status == "running":
                say(f"[book] {rec.id}: era 'running' (crash precedente) -> "
                    f"ripristino a 'pending'")
                rec.status = "pending"
            # resume con retry esplicito dei dead-letter
            if retry_dead_letter and rec.status == "failed" and rec.dead_letter:
                say(f"[book] {rec.id}: dead-letter ri-abilitato per nuovo tentativo")
                rec.status = "pending"
                rec.dead_letter = False
                rec.attempts = 0

        for rec in manifest.segments:
            if rec.status == "done":
                continue  # idempotenza: gia' fatto
            if rec.status == "failed" and rec.dead_letter:
                say(f"[book] {rec.id}: in dead-letter, salto "
                    f"(usa --retry-failed per ritentarlo)")
                continue

            # marca running + checkpoint (cosi' un crash lascia traccia)
            rec.status = "running"
            self._checkpoint(manifest)

            seg_text = self._segment_text_for(text, rec.index)
            say(f"[book] {rec.id} ({rec.index + 1}/{len(manifest.segments)}): "
                f"~{rec.est_tokens} token, avvio pipeline")
            self._run_one_segment(rec, seg_text, say)

            # checkpoint dopo l'esito (done o failed)
            self._checkpoint(manifest)

            if rec.status == "failed" and self.halt_on_failure:
                say(f"[book] --halt-on-failure: fermo il job su {rec.id} fallito")
                break

        c = manifest.counts()
        say(f"[book] job concluso: done={c['done']} failed={c['failed']} "
            f"pending={c['pending']}")
        if c["failed"]:
            failed_ids = [r.id for r in manifest.segments if r.status == "failed"]
            say(f"[book] segmenti in dead-letter: {', '.join(failed_ids)} "
                f"-- rilancia il comando per ritentarli dopo aver risolto la causa")
        return manifest
