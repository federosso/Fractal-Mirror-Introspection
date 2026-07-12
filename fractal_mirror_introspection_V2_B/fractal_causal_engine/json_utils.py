from __future__ import annotations

import ast
import json
import re
from typing import Any


class JSONParseError(ValueError):
    pass


def read_json(path) -> Any:
    """Carica un file JSON. Solleva JSONParseError se il file e' assente o
    malformato, con un messaggio che dice quale file."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise JSONParseError(f"File JSON non trovato: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JSONParseError(f"JSON malformato in {p}: {exc}") from exc


def _strip_code_fence(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    return fence.group(1).strip() if fence else text


def _strip_think(text: str) -> str:
    """Rimuove i blocchi di ragionamento dei modelli 'thinking' (Qwen3.5, QwQ,
    DeepSeek-R1, ecc.) che antepongono il loro pensiero al JSON.

    Questi modelli emettono, prima della risposta vera, un blocco delimitato
    (di solito <think>...</think>, talvolta non chiuso se la generazione viene
    troncata). Quel blocco puo' contenere graffe e virgolette che spiazzano
    l'estrazione del JSON. Lo togliamo a monte, in modo conservativo:
      - <think>...</think> chiuso  -> via tutto il blocco;
      - <think> senza chiusura     -> via tutto cio' che precede la prima '{'
        o '[' reale dopo l'apertura (il JSON, se c'e', viene dopo il pensiero).
    Se non c'e' alcun blocco think, il testo torna invariato.
    """
    t = text or ""
    # blocchi chiusi, anche multipli, case-insensitive, multilinea
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    # apertura senza chiusura (troncamento): tieni dalla prima graffa/quadra
    open_tag = re.search(r"<think>", t, flags=re.IGNORECASE)
    if open_tag:
        rest = t[open_tag.end():]
        first = min((i for i in (rest.find("{"), rest.find("[")) if i >= 0), default=-1)
        t = rest[first:] if first >= 0 else ""
    return t.strip()


def _candidate_slices(text: str) -> list[str]:
    out: list[str] = []
    t = _strip_code_fence(text.strip())
    out.append(t)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        end = t.rfind(closer)
        if start >= 0 and end > start:
            out.append(t[start:end + 1])
    return list(dict.fromkeys(out))


def _basic_repair(candidate: str) -> str:
    c = candidate.strip()
    # Normalizzazioni conservative: non tentano di "capire" il contenuto,
    # rimuovono solo errori formali frequenti dei modelli locali.
    c = c.replace("\ufeff", "")
    c = c.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    c = re.sub(r",\s*([}\]])", r"\1", c)
    # Se il modello ha aggiunto testo prima/dopo, il caller prova già le slice.
    return c


def _balance_object(candidate: str) -> str:
    # Recupero minimo per risposte tronche con parentesi graffe quadre mancanti.
    c = candidate.strip()
    diff_curly = c.count("{") - c.count("}")
    diff_square = c.count("[") - c.count("]")
    if diff_square > 0:
        c += "]" * diff_square
    if diff_curly > 0:
        c += "}" * diff_curly
    return c


def extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise JSONParseError("Risposta vuota")
    text = _strip_think(text)
    if not text:
        raise JSONParseError("Risposta vuota dopo rimozione del blocco <think> "
                             "(il modello ha consumato i token nel ragionamento "
                             "senza emettere JSON: alza num_predict o disattiva il thinking)")

    attempts: list[str] = []
    for cand in _candidate_slices(text):
        repaired = _basic_repair(cand)
        attempts.append(repaired)
        attempts.append(_balance_object(repaired))

    last_error = ""
    for cand in list(dict.fromkeys(attempts)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            # Fallback prudente per modelli che usano True/False/None o apici singoli.
            try:
                obj = ast.literal_eval(cand)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

    raise JSONParseError("Impossibile estrarre JSON valido dalla risposta: " + text[:500] + (f" | last_error={last_error}" if last_error else ""))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, x))


def clean_str(value: Any, limit: int | None = None) -> str:
    s = " ".join(str(value or "").split())
    if limit is not None:
        return s[:limit]
    return s
