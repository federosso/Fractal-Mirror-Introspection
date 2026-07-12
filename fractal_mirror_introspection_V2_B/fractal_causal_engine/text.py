"""Utility testo per V10.16.0.

Versione minimale: solo cio' che V15 usa davvero. Le funzioni V13.1
(split_text_into_spans, label_from_claim, slugify) sono state rimosse,
cosi' come le dataclass RawInput/TextSpan che vivevano in schemas.py.

RawInput resta come dataclass locale a questo modulo, perche' load_text_file
lo ritorna e il CLI lo passa alla pipeline.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawInput:
    """Input testuale grezzo: un solo file o un solo blocco di testo."""
    id: str
    text: str
    source: str = ""


def stable_hash(text: str, size: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:size]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def short(text: str, limit: int = 220) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "..."


def load_text_file(path: str | Path) -> list[RawInput]:
    """Legge un file di testo. Ritorna una lista con un solo RawInput.

    L'API ritorna una lista per coerenza con eventuali estensioni future
    (lettura batch); per ora e' sempre lunghezza 1.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return [RawInput(
        id="input_" + stable_hash(str(p.resolve()) + text),
        text=text,
        source=str(p),
    )]
