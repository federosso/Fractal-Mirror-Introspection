"""Test della lettura tematica di un libro intero (V10.19.2).

Verifica il runner resumabile, la riduzione tematica a due stadi, il resume
dopo crash. Backend mock.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fractal_causal_engine.ft_thematic_book import (
    ThematicBookRunner, ThematicBookReducer, MANIFEST_FILENAME,
)
from fractal_causal_engine.ft_model import THEMATIC_LENSES, BookManifest
from fractal_causal_engine.llm import LLMClient, LLMConfig


def _mock_client() -> LLMClient:
    return LLMClient(LLMConfig(mock=True))


def _no_sleep(_s: float) -> None:
    return None


def _long_book(n: int = 30) -> str:
    return "\n\n".join(
        f"Paragrafo {i}: la luce guida, le voci parlano, la coscienza si espande. " * 4
        for i in range(n)
    )


# --- runner ------------------------------------------------------------------


def test_thematic_book_run_completes():
    with tempfile.TemporaryDirectory() as td:
        runner = ThematicBookRunner(_mock_client(), td, num_ctx=2048,
                                    sleep_fn=_no_sleep)
        manifest = runner.run(_long_book(), book_id="libro_tema")
        assert len(manifest.segments) > 1
        c = manifest.counts()
        assert c["done"] == len(manifest.segments)
        assert (Path(td) / MANIFEST_FILENAME).exists()


def test_thematic_book_resume_after_crash():
    with tempfile.TemporaryDirectory() as td:
        runner = ThematicBookRunner(_mock_client(), td, num_ctx=2048,
                                    sleep_fn=_no_sleep)
        manifest = runner.run(_long_book(), book_id="libro_tema")
        n = len(manifest.segments)
        # simula crash: ultimo segmento torna pending
        manifest.segments[-1].status = "pending"
        runner._checkpoint(manifest)
        msgs: list[str] = []
        final = runner.run(_long_book(), book_id="libro_tema", progress=msgs.append)
        assert any("RESUME" in m for m in msgs)
        assert final.counts()["done"] == n


def test_thematic_book_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        runner = ThematicBookRunner(_mock_client(), td, num_ctx=2048,
                                    sleep_fn=_no_sleep)
        runner.run(_long_book(), book_id="libro_tema")
        msgs: list[str] = []
        m2 = runner.run(_long_book(), book_id="libro_tema", progress=msgs.append)
        assert m2.counts()["done"] == len(m2.segments)
        assert any("RESUME" in m for m in msgs)


# --- reducer -----------------------------------------------------------------


def test_thematic_book_reduction_two_stages():
    with tempfile.TemporaryDirectory() as td:
        runner = ThematicBookRunner(_mock_client(), td, num_ctx=2048,
                                    sleep_fn=_no_sleep)
        manifest = runner.run(_long_book(), book_id="libro_tema")
        reducer = ThematicBookReducer(_mock_client(), td)
        reduction = reducer.reduce(manifest)
        # una sintesi per ciascuna delle quattro lenti
        assert set(reduction.per_lens_synthesis.keys()) == set(THEMATIC_LENSES)
        # sintesi dell'opera prodotta
        assert reduction.opera_synthesis
        # osservazioni raccolte da tutti i segmenti
        assert reduction.total_observations > 0
        assert (Path(td) / "thematic_book_reduction.md").exists()


def test_thematic_reduction_no_done_segments_graceful():
    with tempfile.TemporaryDirectory() as td:
        runner = ThematicBookRunner(_mock_client(), td, num_ctx=2048,
                                    sleep_fn=_no_sleep)
        manifest = runner.run(_long_book(), book_id="libro_tema")
        for r in manifest.segments:
            r.status = "failed"
        reducer = ThematicBookReducer(_mock_client(), td)
        reduction = reducer.reduce(manifest)
        assert reduction.total_segments_used == 0
        assert any("Nessun segmento" in n for n in reduction.notes)


# --- CLI ---------------------------------------------------------------------


def test_cli_theme_book_analyze_end_to_end():
    from fractal_causal_engine.cli import build_parser, run_theme_book_analyze
    with tempfile.TemporaryDirectory() as td:
        book = Path(td) / "libro.txt"
        book.write_text(_long_book(20), encoding="utf-8")
        out = Path(td) / "out"
        args = build_parser().parse_args([
            "theme-book-analyze", "-i", str(book), "-o", str(out),
            "--mock", "--num-ctx", "2048", "--book-id", "test",
        ])
        run_theme_book_analyze(args)
        assert (out / MANIFEST_FILENAME).exists()
        assert (out / "thematic_book_reduction.md").exists()
