"""Test della riduzione gerarchica (V10.18, passo 3).

Verifica la fusione degli ft, i due stadi di sintesi, il fallback senza
capitoli e l'esclusione dei segmenti non completati. Backend mock.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fractal_causal_engine.ft_book_runner import BookRunner
from fractal_causal_engine.ft_reducer import (
    BookReducer, merge_fts, render_reduction_md,
)
from fractal_causal_engine.ft_model import (
    ClassifiedItem, CrossScaleHypothesis, FractalTriadResult, Nature,
    PredicateType,
)
from fractal_causal_engine.llm import LLMClient, LLMConfig


def _mock_client() -> LLMClient:
    return LLMClient(LLMConfig(mock=True))


def _item(iid: str, quote: str, scale: str) -> ClassifiedItem:
    return ClassifiedItem(
        id=iid, quote=quote, predicate=PredicateType.EVENT,
        nature=Nature.CAUSE, scale=scale,
    )


# --- merge_fts ---------------------------------------------------------------


def test_merge_fts_unions_items():
    a = FractalTriadResult()
    a.items = [_item("a1", "fatto uno", "atomico")]
    b = FractalTriadResult()
    b.items = [_item("b1", "fatto due", "molecolare")]
    merged = merge_fts([a, b])
    assert len(merged.items) == 2


def test_merge_fts_dedups_equivalent_items():
    # stesso (quote, scala) in due ft: l'overlap fra segmenti li duplica
    a = FractalTriadResult()
    a.items = [_item("a1", "fatto condiviso", "atomico")]
    b = FractalTriadResult()
    b.items = [_item("b1", "fatto condiviso", "atomico")]   # duplicato
    merged = merge_fts([a, b])
    assert len(merged.items) == 1


def test_merge_fts_dedups_cross_scale():
    h = CrossScaleHypothesis(
        id="h1", cause_item_id="x", effect_item_id="y",
        cause_scale="atomico", effect_scale="cosmologico",
        verdict="genuine", reasoning="r",
    )
    a = FractalTriadResult()
    a.cross_scale = [h]
    b = FractalTriadResult()
    b.cross_scale = [CrossScaleHypothesis(
        id="h2", cause_item_id="x", effect_item_id="y",   # stessa coppia
        cause_scale="atomico", effect_scale="cosmologico",
        verdict="genuine", reasoning="r2",
    )]
    merged = merge_fts([a, b])
    assert len(merged.cross_scale) == 1


def test_merge_fts_empty():
    merged = merge_fts([])
    assert merged.items == []
    assert merged.cross_scale == []


# --- riduzione end-to-end ----------------------------------------------------


def _run_book(td: str, text: str) -> "BookManifest":
    runner = BookRunner(_mock_client(), td, num_ctx=2048,
                        sleep_fn=lambda _s: None)
    return runner.run(text, book_id="libro_test")


def _book_with_chapters() -> str:
    return "\n\n".join(
        f"## Capitolo {c}\nLa causa {c} genera un effetto osservabile. " * 8
        for c in range(3)
    )


def test_reduce_produces_chapter_and_global_synthesis():
    with tempfile.TemporaryDirectory() as td:
        manifest = _run_book(td, _book_with_chapters())
        reducer = BookReducer(_mock_client(), td)
        reduction = reducer.reduce(manifest)
        # una sintesi per capitolo
        assert len(reduction.chapters) == 3
        # tutti i segmenti done usati
        assert reduction.total_segments_used == manifest.counts()["done"]
        # la sintesi globale c'e'
        assert reduction.global_magistrale_text
        # il report e' su disco
        assert (Path(td) / "book_reduction.md").exists()


def test_reduce_fallback_single_chapter_when_no_markers():
    # testo senza marcatori: il segmenter usa il fallback paragrafo,
    # tutti i segmenti hanno chapter_index == -1 -> un solo capitolo
    with tempfile.TemporaryDirectory() as td:
        plain = "\n\n".join(
            f"Paragrafo {i}: la causa produce un effetto. " * 6
            for i in range(30)
        )
        manifest = _run_book(td, plain)
        reducer = BookReducer(_mock_client(), td)
        reduction = reducer.reduce(manifest)
        # un unico capitolo sintetico
        assert len(reduction.chapters) == 1
        assert reduction.chapters[0].chapter_index == -1


def test_reduce_with_no_done_segments_is_graceful():
    with tempfile.TemporaryDirectory() as td:
        manifest = _run_book(td, _book_with_chapters())
        # forza tutti i segmenti a 'failed'
        for r in manifest.segments:
            r.status = "failed"
        reducer = BookReducer(_mock_client(), td)
        reduction = reducer.reduce(manifest)
        assert reduction.chapters == []
        assert any("Nessun segmento" in n for n in reduction.notes)


def test_reduce_excludes_failed_segments():
    with tempfile.TemporaryDirectory() as td:
        manifest = _run_book(td, _book_with_chapters())
        n_done = manifest.counts()["done"]
        assert n_done >= 2
        # marca uno dei segmenti come fallito
        manifest.segments[0].status = "failed"
        reducer = BookReducer(_mock_client(), td)
        reduction = reducer.reduce(manifest)
        assert reduction.total_segments_used == n_done - 1
        assert any("non completati esclusi" in n for n in reduction.notes)


# --- render ------------------------------------------------------------------


def test_render_reduction_md_has_sections():
    with tempfile.TemporaryDirectory() as td:
        manifest = _run_book(td, _book_with_chapters())
        reducer = BookReducer(_mock_client(), td)
        reduction = reducer.reduce(manifest)
        md = render_reduction_md(reduction)
        assert "Lettura causale dell'opera" in md
        assert "Sintesi globale" in md
        assert "Sintesi per capitolo" in md


# --- CLI book-analyze end-to-end ---------------------------------------------


def test_cli_book_analyze_end_to_end():
    """Il subcommand book-analyze gira tutto: segmenta, analizza, riduce."""
    from fractal_causal_engine.cli import build_parser, run_book_analyze
    with tempfile.TemporaryDirectory() as td:
        book = Path(td) / "libro.txt"
        book.write_text(
            "## Capitolo 1\n\nLa causa genera un effetto. Il fenomeno cresce.\n\n"
            "## Capitolo 2\n\nLa risposta riduce la tensione accumulata.\n",
            encoding="utf-8",
        )
        out = Path(td) / "out"
        parser = build_parser()
        args = parser.parse_args([
            "book-analyze", "-i", str(book), "-o", str(out),
            "--mock", "--num-ctx", "2048", "--book-id", "libro_test",
        ])
        run_book_analyze(args)
        # gli output attesi sono su disco
        assert (out / "book_manifest.json").exists()
        assert (out / "book_reduction.md").exists()
        assert (out / "segments").is_dir()


def test_cli_book_analyze_no_reduce_skips_reduction():
    from fractal_causal_engine.cli import build_parser, run_book_analyze
    with tempfile.TemporaryDirectory() as td:
        book = Path(td) / "libro.txt"
        book.write_text("Un paragrafo. Un altro paragrafo con una causa.\n",
                        encoding="utf-8")
        out = Path(td) / "out"
        args = build_parser().parse_args([
            "book-analyze", "-i", str(book), "-o", str(out),
            "--mock", "--num-ctx", "2048", "--no-reduce",
        ])
        run_book_analyze(args)
        assert (out / "book_manifest.json").exists()
        # con --no-reduce non si produce il book_reduction.md
        assert not (out / "book_reduction.md").exists()


# --- V10.19.1: sintesi a sotto-blocchi quando gli item sono troppi -----------


def test_synthesize_merged_single_call_when_few_items():
    """Pochi item: una sola magistrale, nessuna sotto-divisione."""
    import tempfile
    from fractal_causal_engine.ft_model import FractalTriadResult
    with tempfile.TemporaryDirectory() as td:
        reducer = BookReducer(_mock_client(), td)
        ft = FractalTriadResult()
        ft.items = [_item(f"i{n}", f"fatto {n}", "atomico") for n in range(10)]
        msgs: list[str] = []
        text, degraded = reducer._synthesize_merged(ft, "test", msgs.append)
        # nessun messaggio di sotto-blocchi
        assert not any("sotto-blocchi" in m for m in msgs)
        assert not degraded


def test_synthesize_merged_splits_when_too_many_items():
    """Troppi item (oltre la soglia): la sintesi si spezza in sotto-blocchi.
    E' la correzione del difetto V10.18 -- prima sfondava il contesto."""
    import tempfile
    from fractal_causal_engine.ft_model import FractalTriadResult
    with tempfile.TemporaryDirectory() as td:
        reducer = BookReducer(_mock_client(), td)
        n = reducer.MAX_ITEMS_PER_MAGISTRALE * 3 + 5   # ben oltre la soglia
        ft = FractalTriadResult()
        ft.items = [_item(f"i{k}", f"fatto {k}", "atomico") for k in range(n)]
        msgs: list[str] = []
        text, degraded = reducer._synthesize_merged(ft, "capitolo X", msgs.append)
        # ha spezzato in sotto-blocchi
        assert any("sotto-blocchi" in m for m in msgs)
        # 3*60+5 = 185 item -> 4 blocchi (60,60,60,5)
        assert any("4 sotto-blocchi" in m for m in msgs)
        # e ha fatto la sintesi finale dei parziali
        assert any("sintesi finale" in m for m in msgs)


def test_oversized_chapter_does_not_crash_reduce():
    """Un libro lungo senza capitoli (tutti i segmenti in un blocco unico
    con molti item) viene ridotto senza errori -- il caso che falliva."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # libro lungo senza marcatori di capitolo
        plain = "\n\n".join(
            f"Paragrafo {i}: una causa porta a un effetto distinto. " * 5
            for i in range(50)
        )
        manifest = _run_book(td, plain)
        reducer = BookReducer(_mock_client(), td)
        reduction = reducer.reduce(manifest)
        # la riduzione completa senza eccezioni
        assert reduction.total_segments_used == manifest.counts()["done"]
