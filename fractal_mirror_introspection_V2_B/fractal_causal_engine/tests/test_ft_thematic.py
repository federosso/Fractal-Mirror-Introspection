"""Test della lettura tematica (V10.19.0).

Verifica le quattro lenti, la robustezza ai fallimenti di lente, il render
e la clausola di onesta' nei prompt.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fractal_causal_engine.ft_thematic import (
    THEMATIC_LENSES, ThematicReader, render_thematic_md,
    _LENS_PROMPTS, _HONESTY_CLAUSE,
)
from fractal_causal_engine.ft_model import (
    Observation, ThematicMotif, ThematicReading,
)
from fractal_causal_engine.llm import LLMClient, LLMConfig


def _mock_reader(td: str) -> ThematicReader:
    return ThematicReader(LLMClient(LLMConfig(mock=True)), td)


# --- struttura di base -------------------------------------------------------


def test_four_canonical_lenses():
    assert THEMATIC_LENSES == ["simbolica", "strutturale", "relazionale", "esperienziale"]


def test_every_lens_has_a_prompt():
    for lens in THEMATIC_LENSES:
        assert lens in _LENS_PROMPTS
        assert _LENS_PROMPTS[lens].strip()


def test_honesty_clause_in_every_lens_prompt():
    """Ogni lente deve includere la clausola di onesta' epistemica: osserva
    come il testo costruisce il discorso, non se sia vero."""
    for lens in THEMATIC_LENSES:
        assert _HONESTY_CLAUSE.strip()[:30] in _LENS_PROMPTS[lens]


# --- esecuzione --------------------------------------------------------------


def test_read_runs_all_lenses():
    with tempfile.TemporaryDirectory() as td:
        reader = _mock_reader(td)
        msgs: list[str] = []
        reading = reader.read("Un testo di prova con immagini e voci.",
                              progress=msgs.append)
        # le quattro lenti sono state tutte invocate
        for lens in THEMATIC_LENSES:
            assert any(f"'{lens}'" in m for m in msgs)
        assert isinstance(reading, ThematicReading)


def test_empty_text_yields_note():
    with tempfile.TemporaryDirectory() as td:
        reading = _mock_reader(td).read("   ")
        assert reading.observations == []
        assert any("vuoto" in n for n in reading.notes)


def test_thematic_trace_written():
    with tempfile.TemporaryDirectory() as td:
        _mock_reader(td).read("Testo di prova.")
        assert (Path(td) / "thematic_trace.txt").exists()


# --- by_lens -----------------------------------------------------------------


def test_reading_by_lens_filters():
    reading = ThematicReading()
    reading.observations = [
        Observation(lens="simbolica", focus="luce", note="n1"),
        Observation(lens="relazionale", focus="voci", note="n2"),
        Observation(lens="simbolica", focus="pianta", note="n3"),
    ]
    assert len(reading.by_lens("simbolica")) == 2
    assert len(reading.by_lens("relazionale")) == 1
    assert reading.by_lens("strutturale") == []


# --- render ------------------------------------------------------------------


def test_render_has_all_sections():
    reading = ThematicReading()
    reading.observations = [
        Observation(lens="simbolica", focus="la luce", note="immagine centrale",
                    evidence="esseri di luce", salience=0.9),
    ]
    reading.motifs = [
        ThematicMotif(name="la luce", lens="simbolica",
                      occurrences=["premessa", "finale"],
                      transformation="da esterna a interiore"),
    ]
    reading.synthesis = "Le quattro lenti mostrano un testo costruito su immagini di luce."
    md = render_thematic_md(reading, original_text="Testo originale di prova.")
    assert "Testo analizzato" in md
    assert "Sintesi plurale" in md
    assert "Osservazioni per lente" in md
    assert "Motivi ricorrenti" in md
    assert "la luce" in md
    # la sintesi e' dichiarata non-verdetto
    assert "Non un verdetto" in md


def test_render_handles_empty_reading():
    md = render_thematic_md(ThematicReading())
    # non esplode e segnala l'assenza di contenuto
    assert "Lettura tematica" in md
    assert "Nessuna osservazione" in md


# --- robustezza: una lente che fallisce non blocca le altre ------------------


def test_lens_failure_is_isolated(monkeypatch):
    """Le quattro lenti vengono eseguite tutte e il sistema produce una
    ThematicReading coerente senza esplodere."""
    with tempfile.TemporaryDirectory() as td:
        reader = _mock_reader(td)
        reading = reader.read("Testo di prova con piu' frasi. Seconda frase.")
        assert isinstance(reading, ThematicReading)
        # col mock ogni lente produce osservazioni: tutte e quattro presenti
        lenti_viste = {o.lens for o in reading.observations}
        assert lenti_viste == set(THEMATIC_LENSES)
