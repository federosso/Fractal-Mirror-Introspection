"""Test delle modifiche A e B ai report (V10.18.3).

A: il final_report include il testo analizzato.
B: le ipotesi cross-scale duplicate non vengono renderizzate piu' volte.
"""
from __future__ import annotations

from fractal_causal_engine.ft_orchestrator import render_final_report_md
from fractal_causal_engine.ft_model import (
    ClassifiedItem, CrossScaleHypothesis, FractalTriadResult, Nature,
    PredicateType,
)


def _ft_with_pair() -> FractalTriadResult:
    ft = FractalTriadResult()
    ft.items = [
        ClassifiedItem("a", "la causa", PredicateType.EVENT, Nature.CAUSE, "atomico"),
        ClassifiedItem("b", "l'effetto", PredicateType.EVENT, Nature.EFFECT, "cosmologico"),
    ]
    return ft


# --- modifica A --------------------------------------------------------------


def test_report_includes_original_text():
    ft = _ft_with_pair()
    md = render_final_report_md(ft, original_text="La frase di input da analizzare.")
    assert "## 0. Testo analizzato" in md
    assert "La frase di input da analizzare." in md


def test_report_without_original_text_has_no_section():
    ft = _ft_with_pair()
    md = render_final_report_md(ft)            # nessun testo passato
    assert "## 0. Testo analizzato" not in md


def test_report_original_text_multiline_is_blockquoted():
    ft = _ft_with_pair()
    md = render_final_report_md(ft, original_text="Riga uno.\nRiga due.")
    # entrambe le righe presenti come blockquote
    assert "> Riga uno." in md
    assert "> Riga due." in md


# --- modifica B --------------------------------------------------------------


def test_duplicate_cross_scale_hypotheses_deduped():
    ft = _ft_with_pair()
    # tre ipotesi, tutte sulla stessa coppia a->b
    ft.cross_scale = [
        CrossScaleHypothesis(f"h{i}", "a", "b", "atomico", "cosmologico",
                             "genuine", f"reasoning {i}")
        for i in range(3)
    ]
    md = render_final_report_md(ft)
    # una sola riga di ipotesi renderizzata, non tre
    assert md.count("[validata]") == 1
    # e il report segnala le omesse
    assert "2 ipotesi duplicate omesse" in md


def test_distinct_cross_scale_hypotheses_all_kept():
    ft = FractalTriadResult()
    ft.items = [
        ClassifiedItem("a", "causa A", PredicateType.EVENT, Nature.CAUSE, "atomico"),
        ClassifiedItem("b", "effetto B", PredicateType.EVENT, Nature.EFFECT, "cosmologico"),
        ClassifiedItem("c", "causa C", PredicateType.EVENT, Nature.CAUSE, "molecolare"),
    ]
    # due ipotesi su coppie DIVERSE: entrambe vanno tenute
    ft.cross_scale = [
        CrossScaleHypothesis("h1", "a", "b", "atomico", "cosmologico", "genuine", "r1"),
        CrossScaleHypothesis("h2", "c", "b", "molecolare", "cosmologico", "genuine", "r2"),
    ]
    md = render_final_report_md(ft)
    assert md.count("[validata]") == 2
    assert "duplicate omesse" not in md
