from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(value: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def write_json_atomic(value: Any, path: str | Path) -> Path:
    """Scrittura JSON ATOMICA: scrive su un file temporaneo nella stessa
    cartella, poi os.replace() sul file finale.

    V10.18.0. Serve al manifest del book runner: un checkpoint si scrive
    interamente o per niente. Un crash a meta' scrittura lascia intatto il
    manifest precedente invece di corromperlo -- e' lo stesso principio per
    cui un record 'started' mai chiuso (v10.17.3) e' un difetto: una
    scrittura non atomica osservata da fuori.
    """
    import os

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, p)  # atomico sulla stessa partizione
    return p


def write_jsonl_line(value: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(value), ensure_ascii=False) + "\n")


def write_text(text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
