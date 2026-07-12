"""Segmentazione di testi lunghi (V10.18.0). Passo 1 del sottosistema libro.

Spezza un testo che eccede la finestra di contesto del modello in `Segment`:
unita' di lavoro dentro il budget di token, tagliate su confini NATURALI e
con overlap fra l'uno e il successivo.

PERCHE'
-------
La pipeline mette l'intero testo nel prompt di L1. Per un libro questo
sfora `num_ctx` e llama.cpp tronca il prompt IN INGRESSO, silenziosamente:
si analizzerebbe solo l'inizio del libro credendo sia tutto. La
segmentazione e' la difesa da questo.

STRATEGIA -- chunking ricorsivo per confini
-------------------------------------------
Si prova a tagliare sul confine piu' grosso possibile, scendendo solo se
necessario (e' la baseline robusta della letteratura, piu' economica e
spesso piu' accurata del chunking semantico LLM-based):

  1. CAPITOLO   -- se il testo ha marcatori riconoscibili.
  2. PARAGRAFO  -- confine \\n\\n. Fallback automatico se non ci sono
                   capitoli, e taglio interno quando un capitolo sfora.
  3. FRASE      -- ultimo ripiego, se un singolo paragrafo sfora il budget.

Non si scende mai sotto la frase: meglio un segmento un filo sopra taglia
che una frase spezzata a meta'.

OVERLAP
-------
Ogni segmento (tranne il primo) ripete in testa una coda del precedente
(default ~15% del budget). Cosi' una catena causale a cavallo del taglio
non va persa. L'overlap e' contato in `Segment.overlap_chars`; `char_start`
/`char_end` sono offset monotoni e contigui che localizzano il segmento nel
testo (non un taglio byte-esatto -- vedi docstring di Segment).

TUTTO DETERMINISTICO: nessun LLM, nessuna rete. Interamente testabile.
"""
from __future__ import annotations

import re

from .ft_model import Segment, SegmentationResult


# -----------------------------------------------------------------------------
# Stima dei token. Euristica: per l'italiano ~3.3-4 caratteri per token con i
# tokenizer BPE comuni. Usiamo 3.3 -- CONSERVATIVO di proposito: sovrastimare
# i token significa fare segmenti piu' piccoli, mai piu' grandi del lecito.
# -----------------------------------------------------------------------------
CHARS_PER_TOKEN: float = 3.3

# Overhead fisso del prompt di L1 (role + contratto + struttura), in token.
# Misurato sull'ordine di grandezza dei prompt reali (~7000 char osservati,
# di cui ~3500 fissi). Conservativo.
DEFAULT_PROMPT_OVERHEAD_TOKENS: int = 1200

# Margine riservato all'output del modello, in token. La risposta di L1 per
# un segmento non deve competere col testo per lo spazio di contesto.
DEFAULT_OUTPUT_MARGIN_TOKENS: int = 1000

# Frazione del budget ripetuta come overlap fra segmenti consecutivi.
DEFAULT_OVERLAP_RATIO: float = 0.15

# Pattern di inizio capitolo. L'ordine non conta: si prova riga per riga.
# Volutamente largo, per reggere formati .txt diversi; se nessuno matcha si
# degrada al fallback paragrafo senza errori.
_CHAPTER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*#{1,6}\s+\S"),                       # markdown ## Titolo
    re.compile(r"^\s*capitolo\s+[\divxlcIVXLC]+", re.I),  # Capitolo 3 / Capitolo IV
    re.compile(r"^\s*cap\.?\s+[\divxlcIVXLC]+", re.I),    # Cap. 3
    re.compile(r"^\s*chapter\s+[\divxlcIVXLC]+", re.I),   # Chapter 3
    re.compile(r"^\s*parte\s+[\divxlcIVXLC]+", re.I),     # Parte II
    re.compile(r"^\s*[IVXLC]{1,6}\.\s+\S"),               # IV. Titolo
    re.compile(r"^\s*\d{1,3}\.\s+\S"),                    # 12. Titolo
]


def estimate_tokens(text: str) -> int:
    """Stima conservativa dei token di `text`."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def compute_token_budget(
    num_ctx: int,
    *,
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    output_margin_tokens: int = DEFAULT_OUTPUT_MARGIN_TOKENS,
) -> int:
    """Token di CONTENUTO ammessi per segmento, derivati da num_ctx.

        budget = num_ctx - overhead_prompt - margine_output

    Se il risultato non e' positivo (num_ctx troppo piccolo per gli
    overhead) si ritorna un minimo di sicurezza: meglio segmenti minuscoli
    che un budget <= 0.
    """
    budget = num_ctx - prompt_overhead_tokens - output_margin_tokens
    return max(budget, 256)


# -----------------------------------------------------------------------------
# Riconoscimento dei capitoli.
# -----------------------------------------------------------------------------


def _is_chapter_heading(line: str) -> bool:
    """True se la riga sembra l'inizio di un capitolo.

    Vincolo anti-falso-positivo: una riga molto lunga e' prosa, non un
    titolo. I titoli sono righe brevi.
    """
    if len(line.strip()) > 120:
        return False
    return any(p.match(line) for p in _CHAPTER_PATTERNS)


def _split_into_chapters(text: str) -> list[tuple[str, str]]:
    """Spezza il testo in (titolo, corpo) per capitolo.

    Se non riconosce alcun confine ritorna una lista vuota: il chiamante
    interpreta il vuoto come 'usa il fallback a paragrafo'. Il testo che
    precede il primo heading (prefazione, frontespizio) diventa un capitolo
    senza titolo.
    """
    lines = text.splitlines(keepends=True)
    heading_idxs = [i for i, ln in enumerate(lines) if _is_chapter_heading(ln)]
    if not heading_idxs:
        return []

    chapters: list[tuple[str, str]] = []
    # eventuale testo prima del primo heading
    if heading_idxs[0] > 0:
        pre = "".join(lines[: heading_idxs[0]])
        if pre.strip():
            chapters.append(("", pre))

    for k, start in enumerate(heading_idxs):
        end = heading_idxs[k + 1] if k + 1 < len(heading_idxs) else len(lines)
        title = lines[start].strip()
        body = "".join(lines[start:end])
        chapters.append((title, body))
    return chapters


# -----------------------------------------------------------------------------
# Spezzettamento entro un blocco di testo: paragrafi, e frasi come ripiego.
# -----------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[str]:
    """Paragrafi non vuoti, separati da righe vuote."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Frasi. Ripiego usato solo quando un paragrafo da solo sfora il budget."""
    parts = _SENTENCE_END.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _pack_units(units: list[str], token_budget: int) -> list[str]:
    """Raggruppa `units` (paragrafi, o frasi) in blocchi sotto il budget.

    Greedy: accumula finche' aggiungere l'unita' successiva sforerebbe; allora
    chiude il blocco e ne apre uno nuovo. Una singola unita' piu' grande del
    budget viene emessa da sola (la spezzatura fine e' gia' avvenuta a monte).
    """
    blocks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        ut = estimate_tokens(unit)
        if current and current_tokens + ut > token_budget:
            blocks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += ut
    if current:
        blocks.append("\n\n".join(current))
    return blocks


def _blocks_for_chapter(
    body: str, token_budget: int, notes: list[str], *, heading: str = ""
) -> list[str]:
    """Blocchi di testo per un capitolo, tutti sotto il budget.

    Prima per paragrafo; se un singolo paragrafo sfora, quel paragrafo viene
    spezzato per frase. Ogni spezzatura fine viene annotata in `notes`.

    Se `heading` e' fornito (la riga di titolo del capitolo) viene anteposto
    al primo blocco: il titolo e' contesto utile per la classificazione, e
    cosi' nessun carattere del testo originale resta fuori dai segmenti --
    i confini char_start/char_end restano una partizione esatta.
    """
    paragraphs = _split_paragraphs(body)
    if not paragraphs:
        # capitolo con solo l'heading e nessun corpo: l'heading e' il blocco
        return [heading.strip()] if heading.strip() else []

    # se il primo paragrafo coincide con l'heading (lo split lo ha gia'
    # assorbito) non va anteposto due volte
    head = heading.strip()
    if head and paragraphs[0].startswith(head):
        head = ""

    # espandi i paragrafi troppo grandi in frasi PRIMA del packing
    units: list[str] = []
    for para in paragraphs:
        if estimate_tokens(para) <= token_budget:
            units.append(para)
            continue
        sentences = _split_sentences(para)
        if len(sentences) > 1:
            notes.append(
                f"Paragrafo di ~{estimate_tokens(para)} token oltre budget "
                f"({token_budget}): spezzato in {len(sentences)} frasi."
            )
            units.extend(sentences)
        else:
            # una sola frase enorme: non si scende oltre, si emette com'e'
            notes.append(
                f"Unita' indivisibile di ~{estimate_tokens(para)} token oltre "
                f"budget: emessa intera (potrebbe troncare in ingresso)."
            )
            units.append(para)
    blocks = _pack_units(units, token_budget)
    if head and blocks:
        blocks[0] = head + "\n" + blocks[0]
    return blocks


# -----------------------------------------------------------------------------
# API pubblica.
# -----------------------------------------------------------------------------


def segment_text(
    text: str,
    *,
    num_ctx: int = 8192,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    output_margin_tokens: int = DEFAULT_OUTPUT_MARGIN_TOKENS,
) -> SegmentationResult:
    """Segmenta `text` in unita' di lavoro dentro il budget di token.

    num_ctx: finestra di contesto del modello. Da qui si deriva il budget.
    overlap_ratio: frazione del budget ripetuta in testa a ogni segmento dal
        precedente (0 disattiva l'overlap).

    Ritorna un SegmentationResult. Se il testo e' corto (sta in un solo
    segmento) il risultato ha un Segment unico: il chiamante non deve gestire
    il caso a parte.
    """
    result = SegmentationResult(
        total_chars=len(text),
        num_ctx=num_ctx,
    )
    token_budget = compute_token_budget(
        num_ctx,
        prompt_overhead_tokens=prompt_overhead_tokens,
        output_margin_tokens=output_margin_tokens,
    )
    result.token_budget = token_budget

    if not text.strip():
        return result

    # --- fase 1: capitoli (con fallback) ------------------------------------
    chapters = _split_into_chapters(text)
    if chapters:
        result.used_chapters = True
    else:
        # fallback: l'intero testo come un unico 'capitolo' senza titolo
        chapters = [("", text)]
        result.used_chapters = False

    # --- fase 2: blocchi sotto budget, per capitolo -------------------------
    # raw_blocks: lista di (chapter_index, chapter_title, blocco_di_testo)
    raw_blocks: list[tuple[int, str, str]] = []
    for ch_idx, (title, body) in enumerate(chapters):
        blocks = _blocks_for_chapter(body, token_budget, result.notes, heading=title)
        for blk in blocks:
            # chapter_index resta -1 quando non c'erano veri capitoli
            stored_idx = ch_idx if result.used_chapters else -1
            raw_blocks.append((stored_idx, title, blk))

    # --- fase 3: overlap + costruzione dei Segment --------------------------
    # Offset: char_start/char_end sono MONOToni e contigui per costruzione
    # (char_start del segmento k+1 == char_end del segmento k). Localizzano il
    # segmento nel testo in modo ordinato; non sono pero' un taglio byte-esatto
    # del testo originale, perche' i blocchi sono ricostruiti dai paragrafi
    # (strip + rejoin). La fonte autoritativa del contenuto e' `Segment.text`,
    # non gli offset. Vedi docstring di Segment.
    overlap_budget_chars = int(token_budget * overlap_ratio * CHARS_PER_TOKEN)
    cursor = 0
    for i, (ch_idx, title, blk) in enumerate(raw_blocks):
        char_start = cursor
        char_end = char_start + len(blk)
        cursor = char_end

        # overlap: coda del segmento precedente in testa a questo
        overlap_text = ""
        if i > 0 and overlap_budget_chars > 0:
            prev_blk = raw_blocks[i - 1][2]
            overlap_text = _tail_on_sentence_boundary(prev_blk, overlap_budget_chars)

        seg_text = (overlap_text + "\n\n" + blk) if overlap_text else blk
        result.segments.append(
            Segment(
                id=f"seg_{i + 1:04d}",
                index=i,
                text=seg_text,
                char_start=char_start,
                char_end=char_end,
                chapter_index=ch_idx,
                chapter_title=title,
                overlap_chars=len(overlap_text),
                est_tokens=estimate_tokens(seg_text),
            )
        )
    return result


def _tail_on_sentence_boundary(text: str, max_chars: int) -> str:
    """Coda di `text` lunga al piu' max_chars, allineata a inizio frase.

    Si prende la coda e la si fa partire dalla prima frase intera che vi
    rientra, per non aprire un segmento a meta' periodo.
    """
    if max_chars <= 0 or not text:
        return ""
    tail = text[-max_chars:]
    m = _SENTENCE_END.search(tail)
    if m:
        return tail[m.end():].strip()
    return tail.strip()
