"""
storico.py — lettura dello storico della Strada B (storico_introspezione/).

Non ha più uno storage proprio: la fonte di verità sono gli artefatti per livello
che `strada_b_loop.esegui_loop` scrive in ogni cartella `loopB_<timestamp>/`
(00_manifestazione.json … 10_azione.json, 11_specchio_lettura.md, report.md,
trace/telemetry.jsonl) più `indice_memoria.jsonl` nella radice. Questo modulo
li rilegge e li ricompone in un unico dict per il rendering — nessuna scrittura,
tranne la nota del gate umano (nota_gate.md nella cartella del run), che resta
un artefatto ispezionabile a mano come tutto il resto.
"""
from __future__ import annotations

import json
import pathlib
import re
from typing import Optional

# Radice dello storico della Strada B — iniettata da avvia_web.py.
STORICO_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent / "storico_introspezione"

# I livelli del loop, nell'ordine in cui il loop li scrive.
# (nome file → chiave nel record, etichetta breve per il progresso)
LIVELLI = [
    ("00_manifestazione.json", "manifestazione", "manifestazione"),
    ("01_superficie.json",     "superficie",     "canale 1 · superficie"),
    ("02_corpo.json",          "corpo",          "canale 4 · substrato"),
    ("03_gating.json",         "gating",         "gating"),
    ("04_struttura_fractal.json", "struttura",   "canale 2 · Fractal"),
    ("05_specchio_segnali.json",  "specchio",    "canale 3 · Specchio"),
    ("06_must_reject.json",    "must_reject",    "must-reject"),
    ("07_memoria.json",        "memoria",        "memoria"),
    ("08_collasso.json",       "collasso",       "collasso"),
    ("09_telos.json",          "telos",          "telos"),
    ("10_azione.json",         "azione",         "azione"),
]

# File consultabili "grezzi" dalla rotta /run/<id>/grezzo/<nome> (whitelist).
FILE_GREZZI = [nome for nome, _, _ in LIVELLI] + [
    "04b_ventaglio.json",
    "11_specchio_lettura.md", "report.md", "GUIDA_interpretazione.md",
    "trace/telemetry.jsonl",
    "trace/ft_analysis.json", "trace/final_report.md", "trace/trace.md",
]

# Le llm_calls hanno nomi variabili: whitelist a pattern, chiusa (niente
# separatori di percorso nel nome file → nessun path traversal possibile).
_RE_LLM_CALL = re.compile(r"^trace/llm_calls/[A-Za-z0-9_\-]+\.json$")


def _dir() -> pathlib.Path:
    STORICO_DIR.mkdir(parents=True, exist_ok=True)
    return STORICO_DIR


def percorso_run(run_id: str) -> Optional[pathlib.Path]:
    """La cartella del run, solo se è davvero dentro lo storico (niente path
    traversal: run_id è il solo nome della cartella)."""
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        return None
    p = _dir() / run_id
    return p if p.is_dir() else None


def _json(p: pathlib.Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _testo(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Indice memoria (una riga jsonl per run) → mappa per nome-cartella
# ---------------------------------------------------------------------------

def indice_per_run() -> dict[str, dict]:
    """Legge indice_memoria.jsonl e la indicizza per basename di out_dir.
    Serve per arricchire l'elenco (model, modalità) senza riaprire ogni JSON."""
    mappa: dict[str, dict] = {}
    p = _dir() / "indice_memoria.jsonl"
    if not p.exists():
        return mappa
    for riga in p.read_text(encoding="utf-8").splitlines():
        riga = riga.strip()
        if not riga:
            continue
        try:
            r = json.loads(riga)
        except Exception:
            continue
        out_dir = str(r.get("out_dir", "")).replace("\\", "/")
        nome = out_dir.rstrip("/").split("/")[-1]
        if nome:
            mappa[nome] = r
    return mappa


# ---------------------------------------------------------------------------
# Elenco dei run (sintesi per lo storico e per la home)
# ---------------------------------------------------------------------------

def elenco() -> list[dict]:
    """Un run per cartella loopB_*, dal più recente. La sintesi legge SOLO i
    file piccoli (00, 02, 07, 08) — mai la trace — per restare veloce."""
    idx = indice_per_run()
    voci = []
    for d in sorted(_dir().glob("loopB_*"), reverse=True):
        if not d.is_dir():
            continue
        manif = _json(d / "00_manifestazione.json") or {}
        collasso = _json(d / "08_collasso.json")
        corpo = _json(d / "02_corpo.json") or {}
        memoria = _json(d / "07_memoria.json") or {}
        riga_idx = idx.get(d.name, {})
        completo = (d / "10_azione.json").exists()
        voci.append({
            "id": d.name,
            "timestamp": _timestamp_da_nome(d.name) or riga_idx.get("timestamp", ""),
            "sonda": (manif.get("sonda") or "").strip(),
            "modalita": (collasso or {}).get("modalita_loop") or riga_idx.get("modalita", ""),
            "model": riga_idx.get("model", ""),
            "verdetto": (collasso or {}).get("verdetto", "—"),
            "azione": (collasso or {}).get("azione", ""),
            "regola": (collasso or {}).get("regola", ""),
            "conf_substrato": (collasso or {}).get("conf_substrato"),
            "entropia_contenuto": corpo.get("entropia_contenuto"),
            "residuo": (collasso or {}).get("residuo"),
            "substrato_vs_storia": memoria.get("substrato_vs_storia", ""),
            "n_frasi_deboli": len((collasso or {}).get("frasi_deboli", []) or []),
            "completo": completo,
            "ha_nota_gate": (d / "nota_gate.md").exists(),
        })
    return voci


# Le 9 scale canoniche, in ordine di profondità (stessa lista di
# fractal_causal_engine.ft_model.SCALES_CANONICAL — copiata qui per non
# accoppiare il layer web al motore). NON semplificare: la tassonomia
# completa serve alla mappa del non-scelto.
_SCALE_CANONICHE = ["cosmologico", "planetario", "sociale", "organismo",
                    "cellulare", "molecolare", "atomico", "subatomico",
                    "fondamentale"]
_ORDINE_SCALA = {s: i for i, s in enumerate(_SCALE_CANONICHE)}
# Ordine della rampa epistemica, dal più ancorato al più speculativo
# (solo per ordinare la vista: MAI per comprimere i livelli).
_ORDINE_RAMPA = {"text_observed": 0, "domain_knowledge": 1, "causal_model": 2,
                 "cross_domain_analogy": 3, "speculative_extension": 4}


def _mappa_per_scala(analisi: Optional[dict]) -> list[dict]:
    """Items di ft_analysis.json raggruppati per scala, in ordine canonico
    (le scale non canoniche in coda); dentro la scala, dal più ancorato al
    più speculativo. Trasposizione della logica di serializza_ventaglio:
    ogni item porta il SUO gradino della rampa, nessuna media."""
    items = (analisi or {}).get("items") or []
    per_scala: dict = {}
    for it in items:
        per_scala.setdefault(it.get("scale", "?"), []).append(it)
    out = []
    for scala in sorted(per_scala, key=lambda s: _ORDINE_SCALA.get(s, 99)):
        voci = sorted(per_scala[scala],
                      key=lambda x: _ORDINE_RAMPA.get(x.get("epistemic_status"), 9))
        out.append({"scala": scala, "items": [{
            "quote": it.get("quote", ""),
            "nature": it.get("nature", ""),
            "epistemic": it.get("epistemic_status", ""),
            "rationale": it.get("rationale", ""),
        } for it in voci]})
    return out


def _timestamp_da_nome(nome: str) -> str:
    # loopB_YYYYMMDD_HHMMSS → "YYYY-MM-DD HH:MM:SS"
    try:
        _, data, ora = nome.split("_")
        return f"{data[0:4]}-{data[4:6]}-{data[6:8]} {ora[0:2]}:{ora[2:4]}:{ora[4:6]}"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Caricamento completo di un run (per la pagina di dettaglio)
# ---------------------------------------------------------------------------

def carica(run_id: str) -> Optional[dict]:
    d = percorso_run(run_id)
    if d is None:
        return None
    rec: dict = {"id": run_id, "timestamp": _timestamp_da_nome(run_id)}
    presenti = []
    for nome, chiave, etichetta in LIVELLI:
        dati = _json(d / nome)
        rec[chiave] = dati
        if dati is not None:
            presenti.append(etichetta)
    rec["livelli_presenti"] = presenti
    rec["completo"] = rec["azione"] is not None
    rec["lettura_specchio"] = _testo(d / "11_specchio_lettura.md")
    rec["nota_gate"] = _testo(d / "nota_gate.md")
    rec["trace"] = riepilogo_trace(d)
    # Artefatti Fractal completi (assenti nei run precedenti all'integrazione:
    # gli accessi nei template sono condizionali).
    rec["ventaglio"] = _json(d / "04b_ventaglio.json")
    rec["fractal_analisi"] = _json(d / "trace" / "ft_analysis.json")
    rec["fractal_mappa"] = _mappa_per_scala(rec["fractal_analisi"])
    llm_dir = d / "trace" / "llm_calls"
    rec["llm_calls"] = sorted(p.name for p in llm_dir.glob("*.json")) \
        if llm_dir.is_dir() else []
    riga_idx = indice_per_run().get(run_id, {})
    rec["model"] = riga_idx.get("model", "")
    return rec


# ---------------------------------------------------------------------------
# Trace → riepilogo per attore (durata, retry, stato)
# ---------------------------------------------------------------------------

def riepilogo_trace(d: pathlib.Path) -> dict:
    """Compatta telemetry.jsonl in una riga per attore: durata, n. retry per
    troncamento, stato finale. Serve al pannello 'costo del run'."""
    p = d / "trace" / "telemetry.jsonl"
    attori: dict[str, dict] = {}
    ordine: list[str] = []
    if not p.exists():
        return {"attori": [], "durata_totale_s": None}
    for riga in p.read_text(encoding="utf-8").splitlines():
        riga = riga.strip()
        if not riga:
            continue
        try:
            e = json.loads(riga)
        except Exception:
            continue
        cid = e.get("call_id") or e.get("actor") or "?"
        if cid not in attori:
            attori[cid] = {"call_id": cid, "attore": e.get("actor") or e.get("role", cid),
                           "durata_s": None, "retry_troncamento": 0, "stato": ""}
            ordine.append(cid)
        a = attori[cid]
        ev = e.get("event", "")
        if ev == "llm_call_truncated_retry":
            a["retry_troncamento"] += 1
        elif ev == "actor_end":
            a["durata_s"] = e.get("elapsed_seconds")
            a["stato"] = e.get("status", "")
    lista = [attori[c] for c in ordine]
    durate = [a["durata_s"] for a in lista if isinstance(a["durata_s"], (int, float))]
    return {"attori": lista, "durata_totale_s": round(sum(durate), 1) if durate else None}


# ---------------------------------------------------------------------------
# Nota del gate umano — l'unica scrittura di questo modulo
# ---------------------------------------------------------------------------

def aggiorna_nota_gate(run_id: str, nota: str) -> bool:
    d = percorso_run(run_id)
    if d is None:
        return False
    p = d / "nota_gate.md"
    nota = (nota or "").strip()
    if nota:
        p.write_text(nota + "\n", encoding="utf-8")
    elif p.exists():
        p.unlink()
    return True


# ---------------------------------------------------------------------------
# File grezzi (ispezionabilità: ogni numero deve poter mostrare la sua fonte)
# ---------------------------------------------------------------------------

def leggi_grezzo(run_id: str, nome: str) -> Optional[str]:
    if nome not in FILE_GREZZI and not _RE_LLM_CALL.match(nome):
        return None
    d = percorso_run(run_id)
    if d is None:
        return None
    p = d / nome
    if not p.exists():
        return None
    return _testo(p)
