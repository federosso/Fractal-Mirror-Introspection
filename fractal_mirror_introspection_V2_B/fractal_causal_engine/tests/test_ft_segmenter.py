"""Test della segmentazione di testi lunghi (V10.18.0, passo 1).

Tutto deterministico: nessun LLM, nessuna rete.
"""
from __future__ import annotations

from fractal_causal_engine.ft_segmenter import (
    CHARS_PER_TOKEN,
    compute_token_budget,
    estimate_tokens,
    segment_text,
    _is_chapter_heading,
    _split_into_chapters,
    _tail_on_sentence_boundary,
)
from fractal_causal_engine.ft_model import Segment, SegmentationResult


# --- stima token e budget ----------------------------------------------------


def test_estimate_tokens_monotonic():
    assert estimate_tokens("") >= 1
    short = estimate_tokens("una frase breve")
    long = estimate_tokens("una frase breve " * 50)
    assert long > short


def test_compute_token_budget_subtracts_overheads():
    b = compute_token_budget(8192, prompt_overhead_tokens=1200, output_margin_tokens=1000)
    assert b == 8192 - 1200 - 1000


def test_compute_token_budget_floor_when_ctx_tiny():
    # num_ctx troppo piccolo per gli overhead: budget non va sotto il minimo
    b = compute_token_budget(500, prompt_overhead_tokens=1200, output_margin_tokens=1000)
    assert b == 256


# --- riconoscimento capitoli -------------------------------------------------


def test_chapter_heading_recognizes_common_formats():
    assert _is_chapter_heading("## Il conflitto")
    assert _is_chapter_heading("Capitolo 3")
    assert _is_chapter_heading("Capitolo IV")
    assert _is_chapter_heading("Chapter 12")
    assert _is_chapter_heading("Cap. 5")
    assert _is_chapter_heading("Parte II")
    assert _is_chapter_heading("12. Le origini")


def test_chapter_heading_rejects_prose():
    # una riga lunga di prosa non e' un titolo, anche se inizia con un numero
    long_line = "12. " + "parola " * 40
    assert not _is_chapter_heading(long_line)
    assert not _is_chapter_heading("Questa e' una frase normale di prosa.")


def test_split_into_chapters_with_markers():
    text = (
        "Prefazione senza titolo.\n\n"
        "## Capitolo uno\nCorpo del primo capitolo.\n\n"
        "## Capitolo due\nCorpo del secondo capitolo.\n"
    )
    chapters = _split_into_chapters(text)
    # prefazione + 2 capitoli
    assert len(chapters) == 3
    assert chapters[0][0] == ""              # prefazione senza titolo
    assert chapters[1][0] == "## Capitolo uno"
    assert chapters[2][0] == "## Capitolo due"


def test_split_into_chapters_no_markers_returns_empty():
    text = "Solo prosa.\n\nAltro paragrafo.\n\nAncora prosa, nessun capitolo."
    assert _split_into_chapters(text) == []


# --- segmentazione: testo corto ----------------------------------------------


def test_short_text_single_segment():
    res = segment_text("Una frase breve e innocua.", num_ctx=8192)
    assert isinstance(res, SegmentationResult)
    assert len(res.segments) == 1
    assert res.segments[0].index == 0
    assert res.segments[0].overlap_chars == 0   # primo segmento, niente overlap


def test_empty_text_no_segments():
    res = segment_text("   \n  \n ", num_ctx=8192)
    assert res.segments == []


# --- segmentazione: testo lungo, fallback a paragrafo ------------------------


def _long_text_no_chapters(n_paragraphs: int = 60) -> str:
    # paragrafi distinti e non vuoti, nessun marcatore di capitolo
    return "\n\n".join(
        f"Paragrafo numero {i}: " + "contenuto di prova ripetuto. " * 8
        for i in range(n_paragraphs)
    )


def test_long_text_splits_into_multiple_segments():
    # num_ctx piccolo per forzare piu' segmenti
    res = segment_text(_long_text_no_chapters(), num_ctx=2048, overlap_ratio=0.0)
    assert len(res.segments) > 1
    assert res.used_chapters is False          # nessun capitolo: fallback
    # ogni segmento e' sotto (o vicino a) il budget
    for seg in res.segments:
        assert seg.est_tokens <= res.token_budget * 1.5


def test_segment_ids_are_sequential():
    res = segment_text(_long_text_no_chapters(), num_ctx=2048, overlap_ratio=0.0)
    ids = [s.id for s in res.segments]
    assert ids == [f"seg_{i+1:04d}" for i in range(len(ids))]
    indices = [s.index for s in res.segments]
    assert indices == list(range(len(indices)))


# --- overlap -----------------------------------------------------------------


def test_overlap_applied_to_non_first_segments():
    res = segment_text(_long_text_no_chapters(), num_ctx=2048, overlap_ratio=0.15)
    assert len(res.segments) > 1
    assert res.segments[0].overlap_chars == 0           # il primo mai
    # almeno un segmento successivo ha overlap > 0
    assert any(s.overlap_chars > 0 for s in res.segments[1:])


def test_overlap_zero_when_ratio_zero():
    res = segment_text(_long_text_no_chapters(), num_ctx=2048, overlap_ratio=0.0)
    assert all(s.overlap_chars == 0 for s in res.segments)


def test_tail_on_sentence_boundary_starts_clean():
    text = "Prima frase. Seconda frase. Terza frase finale."
    tail = _tail_on_sentence_boundary(text, max_chars=30)
    # la coda non inizia a meta' di una parola: o e' vuota o inizia maiuscola
    assert tail == "" or tail[0].isupper()


# --- segmentazione: testo lungo con capitoli ---------------------------------


def _long_text_with_chapters() -> str:
    chapters = []
    for c in range(4):
        body = "\n\n".join(
            f"Paragrafo {p} del capitolo {c}: " + "testo di prova. " * 10
            for p in range(15)
        )
        chapters.append(f"## Capitolo {c}\n{body}")
    return "\n\n".join(chapters)


def test_chapters_recognized_and_tracked():
    res = segment_text(_long_text_with_chapters(), num_ctx=2048, overlap_ratio=0.0)
    assert res.used_chapters is True
    # i segmenti riportano un chapter_index valido (>= 0)
    assert all(s.chapter_index >= 0 for s in res.segments)
    # piu' di un capitolo rappresentato
    assert len({s.chapter_index for s in res.segments}) > 1


def test_chapter_segments_carry_title():
    res = segment_text(_long_text_with_chapters(), num_ctx=2048, overlap_ratio=0.0)
    titled = [s for s in res.segments if s.chapter_title]
    assert titled
    assert all(s.chapter_title.startswith("## Capitolo") for s in titled)


# --- paragrafo gigante spezzato per frase ------------------------------------


def test_offsets_are_monotonic_and_contiguous():
    """char_start/char_end formano una catena contigua: char_start del
    segmento k+1 == char_end del k. E' la proprieta' garantita per
    costruzione (non un taglio byte-esatto, vedi docstring di Segment)."""
    res = segment_text(_long_text_with_chapters(), num_ctx=2048, overlap_ratio=0.15)
    assert len(res.segments) > 1
    prev_end = 0
    for s in res.segments:
        assert s.char_start == prev_end
        assert s.char_end >= s.char_start
        prev_end = s.char_end


def test_huge_paragraph_split_into_sentences():
    # un solo paragrafo, molte frasi, ben oltre il budget
    huge = " ".join(f"Frase numero {i} di prova." for i in range(400))
    res = segment_text(huge, num_ctx=2048, overlap_ratio=0.0)
    assert len(res.segments) > 1
    # la spezzatura fine e' stata annotata
    assert any("spezzato in" in n for n in res.notes)
