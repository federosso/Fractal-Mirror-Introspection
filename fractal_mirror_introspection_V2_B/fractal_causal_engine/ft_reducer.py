"""Riduzione gerarchica per l'analisi di un libro (V10.18.2). Passo 3.

Dopo che il book runner ha analizzato ogni segmento (un ft per segmento),
i risultati vanno RICOMPOSTI. Concatenarli darebbe N micro-analisi
scollegate; il reducer li fonde in una lettura unitaria, in due stadi:

    segmenti  --stadio 1-->  capitoli  --stadio 2-->  opera

STADIO 1 -- sintesi per capitolo
    I segmenti di uno stesso capitolo hanno ciascuno un ft. Si fondono in
    un ft di capitolo (deduplica degli item equivalenti, unione delle
    ipotesi cross-scale), poi una magistrale RACCONTA il capitolo.

STADIO 2 -- sintesi globale
    Le sintesi di capitolo diventano l'input di una magistrale di secondo
    livello: la lettura causale dell'intera opera.

E' il motore che ricorre su se stesso: la stessa magistrale, applicata a
due livelli di scala nuovi (capitolo, opera). Coerente con la natura
frattale del progetto.

PERCHE' GERARCHICA E NON PIATTA
    Un libro ha una scala naturale -- frase, paragrafo, capitolo, opera. Il
    merge piatto (un unico ft con tutti gli item) la perde. La riduzione a
    due stadi la rispetta: ogni stadio sintetizza il livello sotto.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from .ft_magistrale import MagistraleReportBuilder
from .ft_model import (
    BookManifest,
    BookReduction,
    ChapterSynthesis,
    ClassifiedItem,
    CrossScaleHypothesis,
    FractalTriadResult,
    SCALE_DEPTH,
)
from .ft_session import ExplorerSession
from .llm import LLMClient


REDUCTION_FILENAME = "book_reduction.md"


# -----------------------------------------------------------------------------
# Fusione di ft: deduplica deterministica, niente LLM.
# -----------------------------------------------------------------------------


def _norm_quote(quote: str) -> str:
    """Normalizza una quote per il confronto: minuscole, spazi compattati,
    punteggiatura di bordo rimossa. Due item con la stessa quote normalizzata
    e la stessa scala sono considerati lo stesso item."""
    q = (quote or "").lower().strip()
    q = re.sub(r"\s+", " ", q)
    return q.strip(" .,;:!?\"'")


def _dedup_key(item: ClassifiedItem) -> tuple[str, str]:
    """Chiave di deduplica: (quote normalizzata, scala)."""
    return (_norm_quote(item.quote), item.scale)


def merge_fts(fts: list[FractalTriadResult]) -> FractalTriadResult:
    """Fonde piu' ft in uno solo.

    - items: deduplicati per (quote normalizzata, scala). Il primo vince;
      i duplicati successivi sono scartati. La provenienza (quale ft) e'
      annotata nei metadata dell'item conservato.
    - cross_scale: unite tutte; i duplicati per (cause, effect, scale) sono
      scartati.
    - locked_reports / vision / double_cone: non si fondono (sono viste di
      sintesi di un singolo run); il ft fuso ne resta privo e la magistrale
      lavora comunque sui soli items + cross_scale, che e' quanto le serve.

    Deterministico: nessun LLM.
    """
    merged = FractalTriadResult()
    seen_items: dict[tuple[str, str], ClassifiedItem] = {}
    for fi, ft in enumerate(fts):
        for it in ft.items:
            key = _dedup_key(it)
            if key in seen_items:
                # annota che l'item ricorre anche in un altro segmento
                kept = seen_items[key]
                seen = kept.metadata.setdefault("merge_seen_in", [])
                if fi not in seen:
                    seen.append(fi)
                continue
            md = dict(it.metadata or {})
            md["merge_seen_in"] = [fi]
            seen_items[key] = ClassifiedItem(
                id=it.id, quote=it.quote, predicate=it.predicate,
                nature=it.nature, scale=it.scale, rationale=it.rationale,
                source_input_id=it.source_input_id,
                epistemic_status=it.epistemic_status, metadata=md,
            )
    merged.items = list(seen_items.values())

    seen_cs: set[tuple[str, str, str, str]] = set()
    for ft in fts:
        for h in ft.cross_scale:
            key = (h.cause_item_id, h.effect_item_id, h.cause_scale, h.effect_scale)
            if key in seen_cs:
                continue
            seen_cs.add(key)
            merged.cross_scale.append(h)
    return merged


# -----------------------------------------------------------------------------
# Il reducer.
# -----------------------------------------------------------------------------


class BookReducer:
    """Riduzione gerarchica a due stadi dei risultati di un book runner.

    Uso:
        reducer = BookReducer(client, out_dir)
        reduction = reducer.reduce(manifest)
    """

    def __init__(self, client: LLMClient, out_dir: str | Path) -> None:
        self.client = client
        self.out_dir = Path(out_dir)
        self.reduction_dir = self.out_dir / "reduction"
        self.reduction_dir.mkdir(parents=True, exist_ok=True)
        self.llm_calls_dir = self.reduction_dir / "llm_calls"
        self.llm_calls_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.reduction_dir / "telemetry.jsonl"

    # ----- caricamento degli ft di segmento ---------------------------------

    def _load_segment_ft(self, segment_id: str) -> FractalTriadResult | None:
        """Carica l'ft di un segmento dal suo session.json. None se assente
        o illeggibile (segmento mai completato).

        V10.19.1: il path si RICOSTRUISCE da segments/<id>, non si prende dal
        campo out_dir del manifest. Quel campo, scritto su Windows, contiene
        separatori '\\' che su un altro OS non sono separatori di path -- un
        manifest creato su Windows e ripreso su Linux (o viceversa) si
        romperebbe. L'id del segmento e' sempre noto e portabile.
        """
        seg_dir = self.out_dir / "segments" / segment_id
        if not (seg_dir / "session.json").exists():
            return None
        try:
            sess = ExplorerSession.load(seg_dir, self.client)
            return sess.ft
        except Exception:
            return None

    # ----- stadio 1: sintesi per capitolo -----------------------------------

    # numero massimo di item che si passano alla magistrale in una sola
    # chiamata. Oltre, si sintetizza a sotto-blocchi (vedi _synthesize_merged).
    # 60 item ~ stanno comodi in un contesto da 8k; valore conservativo.
    MAX_ITEMS_PER_MAGISTRALE = 60

    def _run_magistrale(self, ft: FractalTriadResult) -> tuple[str, bool]:
        """Esegue una magistrale su un ft. Ritorna (testo, degraded)."""
        builder = MagistraleReportBuilder(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
        )
        trace: list[str] = []
        report = builder.build(ft, trace=trace)
        if report.degraded:
            return "", True
        return (report.sintesi_magistrale or ""), False

    def _synthesize_merged(
        self, merged: FractalTriadResult, label: str,
        progress: Callable[[str], None],
    ) -> tuple[str, bool]:
        """Sintetizza un ft fuso, gestendo il caso di troppi item.

        Se il ft sta entro MAX_ITEMS_PER_MAGISTRALE, una sola magistrale.
        Altrimenti la riduzione diventa gerarchica ANCHE qui dentro: gli item
        si dividono in sotto-blocchi, ognuno produce una sintesi parziale, e
        una magistrale finale sintetizza le sintesi parziali. E' la
        correzione del difetto V10.18: prima si passavano tutti gli item in
        un'unica chiamata e un libro senza capitoli sfondava il contesto.
        """
        n = len(merged.items)
        if n <= self.MAX_ITEMS_PER_MAGISTRALE:
            return self._run_magistrale(merged)

        # troppi item: spezza in sotto-blocchi
        size = self.MAX_ITEMS_PER_MAGISTRALE
        blocks = [merged.items[i:i + size] for i in range(0, n, size)]
        progress(f"   {label}: {n} item oltre la soglia "
                 f"({self.MAX_ITEMS_PER_MAGISTRALE}) -> sintesi a "
                 f"{len(blocks)} sotto-blocchi")

        partial_texts: list[str] = []
        for bi, block in enumerate(blocks, 1):
            sub_ft = FractalTriadResult()
            sub_ft.items = block
            # le ipotesi cross-scale che coinvolgono solo item del blocco
            block_ids = {it.id for it in block}
            sub_ft.cross_scale = [
                h for h in merged.cross_scale
                if h.cause_item_id in block_ids and h.effect_item_id in block_ids
            ]
            text, degraded = self._run_magistrale(sub_ft)
            if degraded or not text:
                progress(f"   {label}: sotto-blocco {bi}/{len(blocks)} non riuscito, "
                         f"escluso")
                continue
            partial_texts.append(text)
            progress(f"   {label}: sotto-blocco {bi}/{len(blocks)} sintetizzato")

        if not partial_texts:
            return "", True

        # sintesi delle sintesi parziali: ogni parziale diventa un item
        from .ft_model import EpistemicStatus, Nature, PredicateType
        final_ft = FractalTriadResult()
        for pi, ptext in enumerate(partial_texts):
            final_ft.items.append(
                ClassifiedItem(
                    id=f"part_{pi:03d}",
                    quote=ptext.strip()[:240],
                    predicate=PredicateType.CLAIMED_PROPERTY,
                    nature=Nature.CONTEXT,
                    scale="sociale",
                    rationale=f"Sintesi parziale {pi + 1} di {len(partial_texts)}",
                    epistemic_status=EpistemicStatus.TEXT_OBSERVED,
                    metadata={"full_synthesis": ptext.strip()},
                )
            )
        progress(f"   {label}: sintesi finale da {len(partial_texts)} parziali")
        return self._run_magistrale(final_ft)

    def _synthesize_chapter(
        self,
        chapter_index: int,
        chapter_title: str,
        segment_ids: list[str],                # solo gli id; il path si ricostruisce
        progress: Callable[[str], None],
    ) -> ChapterSynthesis:
        """Fonde gli ft dei segmenti di un capitolo e ne genera la magistrale."""
        syn = ChapterSynthesis(
            chapter_index=chapter_index, chapter_title=chapter_title
        )
        fts: list[FractalTriadResult] = []
        for seg_id in segment_ids:
            ft = self._load_segment_ft(seg_id)
            if ft is not None and ft.items:
                fts.append(ft)
                syn.segment_ids.append(seg_id)
            else:
                progress(f"   capitolo {chapter_index}: segmento {seg_id} "
                         f"senza ft utilizzabile, escluso dalla sintesi")

        if not fts:
            syn.degraded = True
            syn.magistrale_text = "(nessun segmento utilizzabile in questo capitolo)"
            return syn

        merged = merge_fts(fts)
        syn.merged_items = len(merged.items)
        syn.merged_cross_scale = len(merged.cross_scale)
        progress(f"   capitolo {chapter_index}: {len(fts)} segmenti fusi -> "
                 f"{syn.merged_items} item, {syn.merged_cross_scale} cross-scale")

        text, degraded = self._synthesize_merged(
            merged, f"capitolo {chapter_index}", progress
        )
        if degraded:
            syn.degraded = True
            syn.magistrale_text = "(sintesi di capitolo non riuscita)"
        else:
            syn.magistrale_text = text
        return syn

    # ----- stadio 2: sintesi globale ----------------------------------------

    def _synthesize_global(
        self, chapters: list[ChapterSynthesis], book_id: str,
        progress: Callable[[str], None],
    ) -> str:
        """Sintesi dell'opera: una magistrale di secondo livello che legge le
        sintesi di capitolo come item.

        Le sintesi di capitolo vengono trasformate in ClassifiedItem (uno per
        capitolo) e date in pasto alla stessa MagistraleReportBuilder. Il
        motore ricorre su se stesso a un livello di scala piu' alto.
        """
        usable = [c for c in chapters if not c.degraded and c.magistrale_text]
        if not usable:
            return "(nessuna sintesi di capitolo utilizzabile per la sintesi globale)"

        # ogni capitolo diventa un item osservato a scala 'sociale' (la scala
        # di un discorso strutturato); la magistrale li legge come materiale.
        from .ft_model import EpistemicStatus, Nature, PredicateType

        global_ft = FractalTriadResult()
        for c in usable:
            quote = c.magistrale_text.strip()
            # la quote di un ClassifiedItem e' pensata breve; qui passiamo la
            # sintesi intera tramite metadata e teniamo la quote come incipit.
            incipit = quote[:240]
            global_ft.items.append(
                ClassifiedItem(
                    id=f"cap_{c.chapter_index:03d}",
                    quote=incipit,
                    predicate=PredicateType.CLAIMED_PROPERTY,
                    nature=Nature.CONTEXT,
                    scale="sociale",
                    rationale=f"Sintesi del capitolo {c.chapter_index}",
                    epistemic_status=EpistemicStatus.TEXT_OBSERVED,
                    metadata={"full_synthesis": quote,
                              "chapter_title": c.chapter_title},
                )
            )
        progress(f"   sintesi globale: {len(usable)} capitoli come materiale")

        builder = MagistraleReportBuilder(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
        )
        trace: list[str] = []
        report = builder.build(global_ft, trace=trace)
        if report.degraded:
            return "(sintesi globale non riuscita)"
        return report.sintesi_magistrale or ""

    # ----- API pubblica -----------------------------------------------------

    def reduce(
        self, manifest: BookManifest,
        *, progress: Callable[[str], None] | None = None,
    ) -> BookReduction:
        """Esegue la riduzione gerarchica completa a partire dal manifest.

        Usa solo i segmenti 'done'. Scrive book_reduction.md nella out_dir.
        """
        say = progress if progress is not None else (lambda _m: None)
        reduction = BookReduction(book_id=manifest.book_id)

        done = [s for s in manifest.segments if s.status == "done"]
        if not done:
            reduction.notes.append("Nessun segmento completato: niente da ridurre.")
            say("[reduce] nessun segmento 'done', riduzione vuota")
            self._write_report(reduction)
            return reduction

        skipped = len(manifest.segments) - len(done)
        if skipped:
            reduction.notes.append(
                f"{skipped} segmenti non completati esclusi dalla riduzione."
            )
        reduction.total_segments_used = len(done)

        # raggruppa i segmenti 'done' per capitolo, preservando l'ordine.
        # chapter_index == -1 (nessun capitolo: fallback paragrafo) -> tutti
        # i segmenti finiscono in un unico capitolo sintetico.
        chapters_map: dict[int, list] = {}
        order: list[int] = []
        for rec in done:
            ci = rec.chapter_index
            if ci not in chapters_map:
                chapters_map[ci] = []
                order.append(ci)
            chapters_map[ci].append(rec)

        say(f"[reduce] stadio 1: sintesi di {len(order)} capitolo/i "
            f"da {len(done)} segmenti")
        for ci in order:
            recs = chapters_map[ci]
            title = recs[0].chapter_title
            seg_ids = [r.id for r in recs]
            syn = self._synthesize_chapter(ci, title, seg_ids, say)
            reduction.chapters.append(syn)
            reduction.total_items_merged += syn.merged_items

        say("[reduce] stadio 2: sintesi globale dell'opera")
        reduction.global_magistrale_text = self._synthesize_global(
            reduction.chapters, manifest.book_id, say
        )

        self._write_report(reduction)
        say(f"[reduce] riduzione completata -> {REDUCTION_FILENAME}")
        return reduction

    # ----- render -----------------------------------------------------------

    def _write_report(self, reduction: BookReduction) -> None:
        (self.out_dir / REDUCTION_FILENAME).write_text(
            render_reduction_md(reduction), encoding="utf-8"
        )


def render_reduction_md(reduction: BookReduction) -> str:
    """Rende la BookReduction in markdown leggibile."""
    out: list[str] = [f"# Lettura causale dell'opera -- {reduction.book_id}\n"]

    out.append("\n## Sintesi globale\n")
    out.append("\n" + (reduction.global_magistrale_text.strip()
                       or "(sintesi globale non disponibile)") + "\n")

    out.append("\n## Sintesi per capitolo\n")
    if not reduction.chapters:
        out.append("\n_(nessun capitolo)_\n")
    for c in reduction.chapters:
        head = c.chapter_title or f"Capitolo {c.chapter_index}"
        if c.chapter_index == -1:
            head = "Testo integrale (nessun capitolo riconosciuto)"
        flag = "  [degradato]" if c.degraded else ""
        out.append(
            f"\n### {head}{flag}\n"
            f"- segmenti fusi: {len(c.segment_ids)} "
            f"({', '.join(c.segment_ids) if c.segment_ids else 'nessuno'})\n"
            f"- item dopo deduplica: {c.merged_items} | "
            f"cross-scale: {c.merged_cross_scale}\n\n"
            f"{c.magistrale_text.strip()}\n"
        )

    if reduction.notes:
        out.append("\n## Note\n")
        for n in reduction.notes:
            out.append(f"- {n}")

    out.append(
        f"\n\n---\nSegmenti usati: {reduction.total_segments_used} | "
        f"item totali fusi: {reduction.total_items_merged}\n"
    )
    return "\n".join(out) + "\n"
