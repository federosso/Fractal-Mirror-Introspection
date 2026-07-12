"""Lettura tematica di un libro intero (V10.19.2). Passo 3.

Estende la lettura tematica (ft_thematic, quattro lenti) a un testo lungo:
lo segmenta, applica le lenti a ogni segmento in modo RESUMABILE, e riduce
le osservazioni in una lettura tematica dell'opera.

E' l'equivalente tematico di book-analyze. Riusa gli stessi mattoni:
- ft_segmenter per spezzare il libro su confini naturali;
- il pattern manifest + checkpoint atomico + retry/backoff + dead-letter
  del book runner causale (qui replicato in forma focalizzata sul tematico,
  per non toccare il BookRunner causale che e' gia' collaudato);
- ThematicReader per le quattro lenti su ogni segmento.

La riduzione e' diversa da quella causale: invece di fondere item e cercare
catene, raccoglie le OSSERVAZIONI di ogni lente da tutti i segmenti e le
sintetizza -- prima per lente ("cosa vede la lente simbolica nell'intero
libro"), poi in una sintesi plurale dell'opera.

ONESTA' EPISTEMICA: vale qui come in ft_thematic. Le lenti osservano COME il
testo costruisce il suo discorso, non SE cio' che afferma sia vero. La
sintesi e' plurale, non un verdetto.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .ft_book_runner import _backoff_seconds, _is_transient
from .ft_model import (
    THEMATIC_LENSES,
    BookManifest,
    Observation,
    SegmentRecord,
    ThematicBookReduction,
    ThematicReading,
)
from .ft_segmenter import segment_text
from .ft_thematic import ThematicReader, _LENS_PROMPTS, _HONESTY_CLAUSE
from .io_utils import write_json_atomic, write_text
from .json_utils import read_json
from .llm import LLMClient, RoleAgent
from .ft_budget import budget


MANIFEST_FILENAME = "thematic_book_manifest.json"


class ThematicBookRunner:
    """Lettura tematica resumabile di un libro intero.

    Uso:
        runner = ThematicBookRunner(client, out_dir, num_ctx=8192)
        manifest = runner.run(text, book_id="mio_libro")
        # poi: ThematicBookReducer(client, out_dir).reduce(manifest)
    """

    def __init__(
        self,
        client: LLMClient,
        out_dir: str | Path,
        *,
        num_ctx: int = 8192,
        overlap_ratio: float = 0.15,
        max_retries: int = 3,
        halt_on_failure: bool = False,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.segments_root = self.out_dir / "segments"
        self.segments_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / MANIFEST_FILENAME
        self.num_ctx = num_ctx
        self.overlap_ratio = overlap_ratio
        self.max_retries = max_retries
        self.halt_on_failure = halt_on_failure
        self.sleep_fn = sleep_fn

    # ----- manifest ---------------------------------------------------------

    def _load_manifest(self) -> BookManifest | None:
        if not self.manifest_path.exists():
            return None
        raw = read_json(self.manifest_path)
        segs = [SegmentRecord(**s) for s in raw.get("segments", [])]
        return BookManifest(segments=segs,
                            **{k: v for k, v in raw.items() if k != "segments"})

    def _build_manifest(self, text: str, book_id: str, source_input_id: str) -> BookManifest:
        seg_result = segment_text(
            text, num_ctx=self.num_ctx, overlap_ratio=self.overlap_ratio
        )
        now = datetime.now().isoformat(timespec="seconds")
        records = [
            SegmentRecord(
                id=s.id, index=s.index,
                chapter_index=s.chapter_index, chapter_title=s.chapter_title,
                status="pending",
                out_dir=str((self.segments_root / s.id).relative_to(self.out_dir)),
                est_tokens=s.est_tokens,
            )
            for s in seg_result.segments
        ]
        return BookManifest(
            book_id=book_id, source_input_id=source_input_id,
            created_at=now, updated_at=now,
            num_ctx=self.num_ctx, overlap_ratio=self.overlap_ratio,
            token_budget=seg_result.token_budget,
            used_chapters=seg_result.used_chapters,
            per_segment_depth="thematic",       # marca: e' un run tematico
            halt_on_failure=self.halt_on_failure,
            max_retries=self.max_retries,
            segments=records,
            segmenter_notes=seg_result.notes,
        )

    def _checkpoint(self, manifest: BookManifest) -> None:
        manifest.updated_at = datetime.now().isoformat(timespec="seconds")
        write_json_atomic(manifest, self.manifest_path)

    # ----- esecuzione di un segmento ----------------------------------------

    def _segment_text_for(self, full_text: str, index: int) -> str:
        seg_result = segment_text(
            full_text, num_ctx=self.num_ctx, overlap_ratio=self.overlap_ratio
        )
        return seg_result.segments[index].text

    def _run_one_segment(
        self, rec: SegmentRecord, seg_text: str, progress: Callable[[str], None]
    ) -> None:
        """Applica le quattro lenti a un segmento, con retry/backoff.

        L'esito vuoto (zero osservazioni da tutte le lenti) e' trattato come
        fallimento -- stessa logica del book runner causale: un segmento
        "vuoto" non deve passare per 'done'.
        """
        seg_out = self.out_dir / "segments" / rec.id
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            rec.attempts = attempt
            started = time.perf_counter()
            try:
                reader = ThematicReader(self.client, seg_out)
                reading = reader.read(seg_text)
                if not reading.observations:
                    last_error = "lettura tematica vuota: nessuna osservazione"
                    transient = _is_transient(last_error)
                    progress(f"   {rec.id}: esito vuoto (tentativo "
                             f"{attempt}/{self.max_retries})")
                    if not transient and attempt >= 1:
                        # esito vuoto: di norma e' il modello, non un guasto
                        # transitorio. Un solo tentativo, poi dead-letter.
                        break
                    if attempt < self.max_retries:
                        self.sleep_fn(_backoff_seconds(attempt))
                    continue
                rec.elapsed_seconds = round(time.perf_counter() - started, 3)
                rec.status = "done"
                rec.error = ""
                progress(f"   {rec.id}: done in {rec.elapsed_seconds}s "
                         f"({len(reading.observations)} osservazioni)")
                return
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                transient = _is_transient(last_error)
                progress(f"   {rec.id}: errore (tentativo {attempt}/"
                         f"{self.max_retries}) "
                         f"{'[transitorio]' if transient else '[definitivo]'}: {last_error}")
                if not transient:
                    break
                if attempt < self.max_retries:
                    self.sleep_fn(_backoff_seconds(attempt))

        rec.status = "failed"
        rec.dead_letter = True
        rec.error = last_error
        rec.elapsed_seconds = 0.0
        progress(f"   {rec.id}: FAILED -> dead-letter. {last_error}")

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
        """Legge il libro `text` con le quattro lenti, segmento per segmento.
        Resumabile: se esiste gia' un manifest, riprende da li'."""
        say = progress if progress is not None else (lambda _m: None)

        manifest = self._load_manifest()
        if manifest is None:
            say(f"[tema-libro] nuovo job '{book_id}': segmento il testo "
                f"({len(text)} caratteri, num_ctx={self.num_ctx})")
            manifest = self._build_manifest(text, book_id, source_input_id)
            self._checkpoint(manifest)
            say(f"[tema-libro] {len(manifest.segments)} segmenti")
        else:
            c = manifest.counts()
            say(f"[tema-libro] RESUME '{manifest.book_id}': "
                f"done={c['done']} pending={c['pending']} failed={c['failed']}")

        for rec in manifest.segments:
            if rec.status == "running":
                rec.status = "pending"
            if retry_dead_letter and rec.status == "failed" and rec.dead_letter:
                say(f"[tema-libro] {rec.id}: dead-letter ri-abilitato")
                rec.status = "pending"
                rec.dead_letter = False
                rec.attempts = 0

        for rec in manifest.segments:
            if rec.status == "done":
                continue
            if rec.status == "failed" and rec.dead_letter:
                say(f"[tema-libro] {rec.id}: in dead-letter, salto")
                continue

            rec.status = "running"
            self._checkpoint(manifest)

            seg_text = self._segment_text_for(text, rec.index)
            say(f"[tema-libro] {rec.id} ({rec.index + 1}/{len(manifest.segments)}): "
                f"~{rec.est_tokens} token, quattro lenti")
            self._run_one_segment(rec, seg_text, say)
            self._checkpoint(manifest)

            if rec.status == "failed" and self.halt_on_failure:
                say(f"[tema-libro] --halt-on-failure: stop su {rec.id}")
                break

        c = manifest.counts()
        say(f"[tema-libro] job concluso: done={c['done']} failed={c['failed']} "
            f"pending={c['pending']}")
        return manifest


# -----------------------------------------------------------------------------
# Riduzione tematica del libro.
# -----------------------------------------------------------------------------


# prompt e contratto per la sintesi di lente sull'intero libro
_LENS_BOOK_PROMPT = """Sei la LENTE {lens_upper} di un sistema di lettura tematica.
Ricevi le osservazioni che la lente {lens} ha raccolto da TUTTO un libro,
segmento per segmento. Il tuo compito: una sintesi di cosa la lente {lens}
vede nell'intera opera -- i temi, le immagini, le forme ricorrenti.
NON un verdetto: una sintesi di osservazioni. 5-10 frasi, in italiano.
""" + _HONESTY_CLAUSE

_LENS_BOOK_CONTRACT = {"synthesis": "<5-10 frasi: cosa la lente vede nell'opera>"}

_OPERA_PROMPT = """Sei il SINTETIZZATORE di un sistema di lettura tematica.
Ricevi quattro sintesi -- una per lente (simbolica, strutturale, relazionale,
esperienziale) -- riferite a un intero libro.
Il tuo compito: una SINTESI PLURALE dell'opera. NON un verdetto su cosa il
libro "sia" o se cio' che afferma sia vero: un riepilogo onesto delle quattro
angolazioni, di dove convergono e di dove offrono lo stesso testo da
prospettive diverse. 8-14 frasi, in italiano, prosa piana.
""" + _HONESTY_CLAUSE

_OPERA_CONTRACT = {"synthesis": "<8-14 frasi: sintesi plurale dell'opera>"}


class ThematicBookReducer:
    """Riduce le letture tematiche dei segmenti in una lettura dell'opera.

    Due stadi: prima una sintesi per lente (tutte le osservazioni di una
    lente, da tutti i segmenti), poi una sintesi plurale dell'opera dalle
    quattro sintesi di lente.
    """

    def __init__(self, client: LLMClient, out_dir: str | Path) -> None:
        self.client = client
        self.out_dir = Path(out_dir)
        self.reduction_dir = self.out_dir / "reduction"
        self.reduction_dir.mkdir(parents=True, exist_ok=True)
        self.llm_calls_dir = self.reduction_dir / "llm_calls"
        self.llm_calls_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.reduction_dir / "telemetry.jsonl"

    def _load_segment_reading(self, segment_id: str) -> ThematicReading | None:
        """Ricarica le osservazioni di un segmento dai suoi record llm_calls.

        Le osservazioni stanno nei record delle chiamate LLM delle lenti
        (NNNN_L_Thematic_<lente>.json). Si rileggono da li': non si dipende
        da un report markdown per-segmento, che il runner non scrive.
        """
        seg_dir = self.out_dir / "segments" / segment_id
        reading = ThematicReading()
        calls_dir = seg_dir / "llm_calls"
        if not calls_dir.exists():
            return None
        import json
        for call_file in sorted(calls_dir.glob("*.json")):
            # la lente si ricava dal NOME del file: i record sono salvati
            # come NNNN_L_Thematic_<lente>.json (il campo 'role' nel record
            # non e' garantito). La sintesi (..._Synthesis) va saltata.
            stem = call_file.stem            # es. 0001_L_Thematic_simbolica
            if "L_Thematic_" not in stem or stem.endswith("Synthesis"):
                continue
            lens = stem.split("L_Thematic_", 1)[1]
            if lens not in THEMATIC_LENSES:
                continue
            try:
                rec = json.loads(call_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            parsed = rec.get("parsed_json") or {}
            for o in parsed.get("observations", []) or []:
                if not isinstance(o, dict):
                    continue
                reading.observations.append(
                    Observation(
                        lens=lens,
                        focus=str(o.get("focus", "")).strip(),
                        note=str(o.get("note", "")).strip(),
                        evidence=str(o.get("evidence", "")).strip(),
                    )
                )
        return reading

    def _synthesize_lens(self, lens: str, observations: list[Observation],
                         progress: Callable[[str], None]) -> str:
        """Sintesi di una lente su tutto il libro."""
        if not observations:
            return f"(nessuna osservazione della lente {lens})"
        agent = RoleAgent(
            self.client,
            role_name=f"L_ThematicBook_{lens}",
            role_prompt=_LENS_BOOK_PROMPT.format(
                lens=lens, lens_upper=lens.upper()
            ),
            out_dir=self.llm_calls_dir,
            max_output_tokens=budget("thematic_book"),
        )
        # le osservazioni potrebbero essere tante: passiamo focus+note,
        # compatti. Se fossero troppe, si limita per non sfondare il contesto.
        MAX_OBS = 120
        obs_payload = [
            {"focus": o.focus, "note": o.note}
            for o in observations[:MAX_OBS]
        ]
        trace: list[str] = []
        raw, _meta = agent.run_json(
            {"lens": lens, "observations": obs_payload},
            _LENS_BOOK_CONTRACT, trace, telemetry_path=self.telemetry_path,
        )
        text = ""
        if isinstance(raw, dict):
            text = str(raw.get("synthesis", "")).strip()
        n = len(observations)
        extra = f" (su {n} osservazioni" + (
            f", prime {MAX_OBS} usate)" if n > MAX_OBS else ")")
        progress(f"   lente {lens}: sintesi prodotta{extra}")
        return text or f"(sintesi della lente {lens} non riuscita)"

    def _synthesize_opera(self, per_lens: dict[str, str],
                          progress: Callable[[str], None]) -> str:
        """Sintesi plurale dell'opera dalle quattro sintesi di lente."""
        agent = RoleAgent(
            self.client,
            role_name="L_ThematicBook_Opera",
            role_prompt=_OPERA_PROMPT,
            out_dir=self.llm_calls_dir,
            max_output_tokens=budget("thematic_book_opera"),
        )
        trace: list[str] = []
        raw, _meta = agent.run_json(
            {"lens_syntheses": per_lens},
            _OPERA_CONTRACT, trace, telemetry_path=self.telemetry_path,
        )
        text = ""
        if isinstance(raw, dict):
            text = str(raw.get("synthesis", "")).strip()
        progress("   sintesi plurale dell'opera prodotta")
        return text or "(sintesi dell'opera non riuscita)"

    def reduce(
        self,
        manifest: BookManifest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> ThematicBookReduction:
        """Riduce le letture tematiche dei segmenti 'done' in una lettura
        dell'opera. Scrive thematic_book_reduction.md."""
        say = progress if progress is not None else (lambda _m: None)
        reduction = ThematicBookReduction(book_id=manifest.book_id)

        done = [s for s in manifest.segments if s.status == "done"]
        skipped = len(manifest.segments) - len(done)
        if skipped:
            reduction.notes.append(
                f"{skipped} segmenti non completati esclusi dalla riduzione."
            )
        if not done:
            reduction.notes.append("Nessun segmento completato: niente da ridurre.")
            say("[tema-reduce] nessun segmento 'done'")
            self._write_report(reduction)
            return reduction
        reduction.total_segments_used = len(done)

        # raccoglie tutte le osservazioni, raggruppate per lente
        by_lens: dict[str, list[Observation]] = {l: [] for l in THEMATIC_LENSES}
        for rec in done:
            reading = self._load_segment_reading(rec.id)
            if reading is None:
                continue
            for o in reading.observations:
                if o.lens in by_lens:
                    by_lens[o.lens].append(o)
        reduction.total_observations = sum(len(v) for v in by_lens.values())
        say(f"[tema-reduce] {reduction.total_observations} osservazioni "
            f"da {len(done)} segmenti")

        # stadio 1: una sintesi per lente
        say("[tema-reduce] stadio 1: sintesi per lente")
        for lens in THEMATIC_LENSES:
            reduction.per_lens_synthesis[lens] = self._synthesize_lens(
                lens, by_lens[lens], say
            )

        # stadio 2: sintesi plurale dell'opera
        say("[tema-reduce] stadio 2: sintesi dell'opera")
        reduction.opera_synthesis = self._synthesize_opera(
            reduction.per_lens_synthesis, say
        )

        self._write_report(reduction)
        say("[tema-reduce] riduzione completata -> thematic_book_reduction.md")
        return reduction

    def _write_report(self, reduction: ThematicBookReduction) -> None:
        lines = [f"# Lettura tematica dell'opera -- {reduction.book_id}", ""]
        lines += [
            "_Non un verdetto: una lettura plurale, da quattro lenti, "
            "dell'intero libro._", "",
            "## Sintesi dell'opera", "",
            reduction.opera_synthesis or "_(non disponibile)_", "",
            "## Sintesi per lente", "",
        ]
        for lens in THEMATIC_LENSES:
            lines.append(f"### Lente {lens}")
            lines.append("")
            lines.append(reduction.per_lens_synthesis.get(lens, "_(non disponibile)_"))
            lines.append("")
        if reduction.notes:
            lines += ["## Note", ""]
            for n in reduction.notes:
                lines.append(f"- {n}")
            lines.append("")
        lines += ["---",
                  f"Segmenti usati: {reduction.total_segments_used} | "
                  f"osservazioni totali: {reduction.total_observations}"]
        write_text("\n".join(lines) + "\n",
                   self.out_dir / "thematic_book_reduction.md")
