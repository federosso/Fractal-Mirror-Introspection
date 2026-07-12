"""Smoke test della pipeline V10.14.0.

Verifica:
- L1 produce ClassifiedItem con scale valide e quote presenti nel testo.
- L2 produce un report per scala con orphans coerenti.
- L3A produce un UnlockedReport con almeno una sezione popolata o degraded.
- L3B produce CrossScaleHypothesis con verdict in {genuine,spurious,uncertain}.
- L4 produce DoubleCone e GlobalVision e final_report.md non vuoto.

Tutto in mock: il backend non viene contattato.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# espone la radice del progetto al path (il package è sorgente semplice)
RADICE = Path(__file__).resolve().parents[2]
if str(RADICE) not in sys.path:
    sys.path.insert(0, str(RADICE))

from fractal_causal_engine.ft_model import (
    SCALES_CANONICAL,
    Nature,
)
from fractal_causal_engine.ft_pipeline import FractalTriadPipeline
from fractal_causal_engine.llm import LLMClient, LLMConfig


SAMPLE_TEXT = (
    "L'evento ha innescato una risposta corporea. Dopo qualche tempo si e' manifestato un effetto duraturo."
)


def _make_client_mock() -> LLMClient:
    return LLMClient(LLMConfig(mock=True))


def test_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    pipeline = FractalTriadPipeline(_make_client_mock(), out_dir=tmp_path)
    ft = pipeline.run(SAMPLE_TEXT, source_input_id="t1")
    pipeline.write_outputs(ft)
    # file di output prodotti
    assert (tmp_path / "ft_analysis.json").exists()
    assert (tmp_path / "final_report.md").exists()
    assert (tmp_path / "trace.md").exists()
    assert (tmp_path / "llm_calls").is_dir()


def test_l1_items_have_valid_scales_and_quotes_in_text(tmp_path: Path) -> None:
    pipeline = FractalTriadPipeline(_make_client_mock(), out_dir=tmp_path)
    ft = pipeline.run(SAMPLE_TEXT, source_input_id="t2")
    assert ft.items, "L1 deve produrre almeno un item"
    for it in ft.items:
        assert it.scale in SCALES_CANONICAL, f"Scala fuori dalle 9 canoniche: {it.scale}"
        norm_text = SAMPLE_TEXT.lower()
        assert it.quote.lower() in norm_text, (
            f"Quote non presente nel testo: {it.quote!r}"
        )
        assert len(it.quote.split()) <= 25, "Quote oltre 25 parole"


def test_l2_orphans_are_consistent(tmp_path: Path) -> None:
    pipeline = FractalTriadPipeline(_make_client_mock(), out_dir=tmp_path)
    ft = pipeline.run(SAMPLE_TEXT, source_input_id="t3")
    all_item_ids = {it.id for it in ft.items}
    for rep in ft.locked_reports:
        for orph in rep.orphans:
            assert orph.item_id in all_item_ids, "Orphan punta a item inesistente"
            assert orph.scale == rep.scale


def test_l3a_unlocked_structure(tmp_path: Path) -> None:
    pipeline = FractalTriadPipeline(_make_client_mock(), out_dir=tmp_path)
    ft = pipeline.run(SAMPLE_TEXT, source_input_id="t4")
    u = ft.unlocked
    assert u is not None
    # con il mock generico, almeno knowledge e principles devono essere non vuoti
    assert u.domain_knowledge or u.degraded
    assert u.causal_principles or u.degraded


def test_l3b_verdicts_in_allowed_set(tmp_path: Path) -> None:
    pipeline = FractalTriadPipeline(_make_client_mock(), out_dir=tmp_path)
    ft = pipeline.run(SAMPLE_TEXT, source_input_id="t5")
    for h in ft.cross_scale:
        assert h.verdict in {"genuine", "spurious", "uncertain"}
        assert 0.0 <= h.confidence <= 1.0
        assert h.cause_scale in SCALES_CANONICAL
        assert h.effect_scale in SCALES_CANONICAL
        assert h.cause_scale != h.effect_scale, "L3B deve trattare solo coppie cross-scale"


def test_double_cone_uses_only_canonical_scales(tmp_path: Path) -> None:
    pipeline = FractalTriadPipeline(_make_client_mock(), out_dir=tmp_path)
    ft = pipeline.run(SAMPLE_TEXT, source_input_id="t6")
    for s in ft.double_cone.cone_of_causes:
        assert s in SCALES_CANONICAL
    for s in ft.double_cone.cone_of_effects:
        assert s in SCALES_CANONICAL


def test_no_same_item_in_both_cones(tmp_path: Path) -> None:
    """Un item con nature=cause non puo' anche essere nel cono effetti, e viceversa."""
    pipeline = FractalTriadPipeline(_make_client_mock(), out_dir=tmp_path)
    ft = pipeline.run(SAMPLE_TEXT, source_input_id="t7")
    cause_ids = {iid for ids in ft.double_cone.cone_of_causes.values() for iid in ids}
    effect_ids = {iid for ids in ft.double_cone.cone_of_effects.values() for iid in ids}
    assert not (cause_ids & effect_ids), "Item presente in entrambi i coni: violato"


def test_no_legacy_quality_fallback_in_v14(tmp_path: Path) -> None:
    """Il vecchio quality_fallback per prossimita' non deve mai essere chiamato in V14."""
    pipeline = FractalTriadPipeline(_make_client_mock(), out_dir=tmp_path)
    ft = pipeline.run(SAMPLE_TEXT, source_input_id="t8")
    # ft.trace deve essere fatto solo dai livelli L0..L4 nuovi
    trace_joined = "\n".join(ft.trace).lower()
    assert "quality_fallback" not in trace_joined
    assert "domain_quality_fallback" not in trace_joined


def test_ft_analysis_json_is_valid(tmp_path: Path) -> None:
    pipeline = FractalTriadPipeline(_make_client_mock(), out_dir=tmp_path)
    ft = pipeline.run(SAMPLE_TEXT, source_input_id="t9")
    pipeline.write_outputs(ft)
    data = json.loads((tmp_path / "ft_analysis.json").read_text(encoding="utf-8"))
    assert "items" in data
    assert "locked_reports" in data
    assert "unlocked" in data
    assert "cross_scale" in data
    assert "double_cone" in data
    assert "vision" in data
