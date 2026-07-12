"""
strada_b_loop.py — Introspezione che CHIUDE, versione B (riscrittura della Strada A).

Correzioni cablate rispetto alla Strada A (vedi analisi del run 20260705_232248):

  1. BERSAGLIO SUL GESTO, NON SUL CONTENUTO (fix dello slittamento di livello):
     - lo Specchio riceve il frame d'introspezione E un nucleo rimappato sul
       modello (specchio_del_modello_nucleo.md): substrato/struttura/stile/scarto;
     - il must-reject filtra per REFERENTE (processo generativo vs fenomeno
       descritto), non per nome di scala.
  2. CANALE 1 SINTATTICO: l'assertività dichiarativa (copule definitorie,
     grassetti, sezioni-sintesi, assenza di condizionali) entra nella misura;
     un canale non informativo NON entra nel blend (fix dell'incoerenza 0.4).
  3. CANALE 4 PER-CLAIM: i logprob sono allineati alle frasi; i token
     strutturali (markdown, punteggiatura) sono esclusi; il collasso vede DOVE
     il substrato è debole, non solo quanto in media.
  4. GATING: i canali economici (superficie + substrato) girano sempre; il
     Fractal e lo Specchio si accendono solo su anomalia (o su richiesta).
  5. MEMORIA: lo storico viene RILETTO; il substrato è confrontato con la firma
     abituale del modello (z-score), non solo con soglie assolute.
  6. TELOS: una mini-costituzione verifica l'azione e la corregge per regola
     (mai per ri-narrazione).

Invarianti ereditate dalla Strada A: il collasso è una REGOLA sui segnali, mai
una narrazione; pesa al contrario della controllabilità; il residuo è il budget
dell'azione; ogni livello scrive il suo artefatto ispezionabile.
"""
from __future__ import annotations

import json
import math
import re
import time
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from fractal_causal_engine.io_utils import write_json, ensure_dir
from fractal_causal_engine.ft_model import Nature, EpistemicStatus
import ponte_fractal_specchio as P
import introspezione_ponte as I


# ---------------------------------------------------------------------------
# Soglie — esplicite, ispezionabili. Le soglie sul substrato diventano RELATIVE
# quando la memoria ha una baseline (>=3 run dello stesso modello).
# ---------------------------------------------------------------------------
ENTROPIA_ALTA = 2.0
ENTROPIA_IMPEGNO = 0.5
MARGINE_IMPEGNO = 0.5
SOGLIA_CONF_SUBSTRATO = 0.50
SOGLIA_PRESENTATA = 0.45
# Gate d'informatività del canale struttura: sotto questa soglia di proposizioni
# l'assertività non ha materia su cui misurare (es. testo filosofico-procedurale
# con 2 item: assertività 0.0 non perché la presentazione sia dimessa, ma perché
# non c'è nulla da misurare) — il canale esce dal blend invece di trascinarlo.
SOGLIA_PROP_STRUTTURA = 3
SOGLIA_FRASE_DEBOLE = 0.60        # una frase sotto questa conf è un punto debole
MIN_TOKEN_CONTENUTO_FRASE = 4     # frasi più corte non fanno statistica
SOGLIA_GATE_SCARTO = 0.20         # |presentazione~substrato| oltre cui si accende il loop completo
SOGLIA_GATE_ESITAZIONE = 0.10     # quota di esitazione oltre cui si accende
SOGLIA_RESIDUO_DICHIARABILE = 0.35  # telos R2: sopra, il residuo va dichiarato
Z_ANOMALIA = 1.5                  # |z| oltre cui il substrato è fuori firma

NATURE_ASSERTIVE = {Nature.CAUSE, Nature.EFFECT}
EPISTEMIC_SPECULATIVO = {EpistemicStatus.CROSS_DOMAIN_ANALOGY,
                         EpistemicStatus.SPECULATIVE_EXTENSION}

# --- Canale 1: lessico (come in A, dedup) + feature SINTATTICHE generiche ---
MARCATORI_HEDGE = (
    "forse", "magari", "potrebbe", "può darsi", "sembra", "pare", "credo",
    "penso", "probabilmente", "in un certo senso", "direi", "immagino",
    "suppongo", "non sono sicuro", "a mio avviso", "secondo me", "più o meno",
    "in qualche modo", "tendenzialmente",
)
MARCATORI_ASSERZIONE = (
    "sicuramente", "certamente", "ovviamente", "chiaramente", "senza dubbio",
    "di fatto", "in effetti", "sempre", "mai", "assolutamente", "indubbiamente",
    "è evidente", "è chiaro", "naturalmente",
)
# Copule definitorie: "X è/sono/risiede/consiste/significa ..." a inizio frase.
_RE_COPULA_DEF = re.compile(
    r"(?:^|[.!?:\n]\s*)(?:il|la|lo|i|le|gli|l'|un|una|uno|questo|questa|"
    r"quest'|ogni|tale)?\s*[\w'àèéìòù ]{0,60}?\b(è|sono|risiede|consiste|"
    r"significa|rappresenta|costituisce|si tratta di|deriva da|dipende da)\b",
    re.IGNORECASE)
_RE_CONDIZIONALE = re.compile(
    r"\b(sarebbe|sarebbero|potrebbe|potrebbero|dovrebbe|dovrebbero|"
    r"parrebbe|risulterebbe|si direbbe)\b", re.IGNORECASE)
MARCATORI_SINTESI = ("in sintesi", "in conclusione", "in breve", "riassumendo",
                     "in definitiva", "per riassumere")

# --- must-reject per REFERENTE: lessico del PROCESSO GENERATIVO --------------
# Un candidato è una lettura-di-sé plausibile solo se parla dell'atto di
# generazione (training, statistica, personaggio, sonda), non del fenomeno
# descritto. Lessico trasparente e ampliabile, come da stile della Strada A.
LESSICO_PROCESSO = (
    "training", "addestra", "frequenz", "statistic", "token", "modello",
    "prompt", "sonda", "corpus", "generazion", "personaggio", "assistente",
    "instruct", "parametr", "pesi del modello", "distribuzion", "appres",
    "linguistic", "registro", "pattern", "dati di",
)

# Frame per lo Specchio: dichiara il bersaglio (il gesto) DENTRO l'input.
FRAME_SPECCHIO = (
    "\n\n[QUADRO DI LETTURA — non è parte del fenomeno] Il testo qui sopra è un "
    "OUTPUT PRODOTTO DA UN MODELLO LINGUISTICO in risposta a una sonda. Il "
    "soggetto della lettura è il modello che ha emesso i token, non l'argomento "
    "trattato: le precause vivono nel processo generativo (frequenze di training, "
    "personaggio addestrato, forma della sonda, tendenze statistiche), mai nel "
    "fenomeno descritto."
)

# Addendum al contratto (vincoli estraibili a macchina), additivo come la Regola 8.
ADDENDUM_CONTRATTO_MODELLO = (
    "11. Vincoli di estrazione (introspezione-modello). La sezione 6 deve "
    "contenere la massa come numero: `massa = 0.NN`. La sezione 9 deve chiudersi "
    "con la riga esatta `auto-deformazione: presente` oppure "
    "`auto-deformazione: assente`. Ogni precausa della sezione 5 riguarda il "
    "processo generativo, mai il fenomeno descritto."
)


def _scrivi_chiamata(dir_llm, idx: int, role: str, backend: str, model: str,
                     system: str, payload: str, raw: str, t0: float, t1: float) -> None:
    """Formato per-call del Fractal (trace/llm_calls/NNNN_role.json)."""
    write_json({
        "call_id": f"{idx:04d}_{role}",
        "role_name": role,
        "status": "completato",
        "backend": backend,
        "model": model,
        "started_at": datetime.datetime.fromtimestamp(t0).isoformat(),
        "finished_at": datetime.datetime.fromtimestamp(t1).isoformat(),
        "elapsed_seconds": round(t1 - t0, 3),
        "prompt_chars": len(system or "") + len(payload or ""),
        "response_chars": len(raw or ""),
        "system_prompt": system or "",
        "payload": payload or "",
        "raw_response": raw or "",
    }, Path(dir_llm) / f"{idx:04d}_{role}.json")


# ---------------------------------------------------------------------------
# Strutture
# ---------------------------------------------------------------------------
@dataclass
class SuperficieManifest:
    """Canale 1 (più controllato): misure di superficie, lessicali + sintattiche."""
    densita_hedge: float = 0.0
    densita_asserzione: float = 0.0
    densita_copule_def: float = 0.0      # definizioni assertive per frase
    densita_condizionale: float = 0.0    # condizionali per frase
    quota_grassetti: float = 0.0         # **…** per frase
    ha_sezione_sintesi: bool = False
    conf_superficie: float = 0.5
    informativa: bool = False


@dataclass
class FraseProfilo:
    """Profilo del substrato per una singola frase (per-claim)."""
    indice: int
    testo: str                 # troncato per ispezione
    n_token_contenuto: int
    confidenza: float
    entropia: float


@dataclass
class ProfiloCorpo:
    """Canale 4 (meno controllato): profilo involontario, globale + per frase."""
    affidabile: bool
    n_token: int = 0
    confidenza_media: float = 0.0            # su TUTTI i token (come in A, per confronto)
    confidenza_contenuto: float = 0.0        # solo token di contenuto (il metro del collasso)
    entropia_media: float = 0.0
    entropia_contenuto: float = 0.0
    quota_alta_conf: float = 0.0
    quota_esitazione: float = 0.0
    n_punti_impegno: int = 0
    quota_token_contenuto: float = 0.0
    frasi: list = field(default_factory=list)         # list[FraseProfilo] (solo risposta)
    frasi_deboli: list = field(default_factory=list)  # le peggiori, conf < soglia
    # segmentazione ragionamento nascosto / risposta (i logprob coprono TUTTI i
    # token emessi; alcuni backend nascondono dal content una fase di
    # ragionamento che precede la risposta):
    allineamento: str = "non_verificato"  # risposta_allineata | nessun_ragionamento |
                                          # manifestazione_non_trovata | non_verificato
    ha_ragionamento: bool = False
    n_token_ragionamento: int = 0
    conf_ragionamento: float = 0.0
    entropia_ragionamento: float = 0.0
    testo_ragionamento: str = ""  # il testo INTERO del ragionamento, ricostruito
                                  # dai token dei logprob: il backend lo esclude
                                  # dal content, questo è l'unico posto dove
                                  # sopravvive ispezionabile
    motivo: str = ""


@dataclass
class Gating:
    """Decisione di accensione del loop completo."""
    modalita: str            # 'leggero' | 'completo'
    richiesta: str           # 'auto' | 'completo' | 'leggero'
    motivi: list = field(default_factory=list)


@dataclass
class StrutturaFractal:
    """Canale 2: come il modello ha ORGANIZZATO la risposta (identico ad A)."""
    n_prop: int = 0
    quota_cause_effect: float = 0.0
    quota_speculativa: float = 0.0
    assertivita: float = 0.0
    nessi_totali: int = 0
    nessi_genuine: int = 0
    tenuta_nessi: float = 0.0
    disponibile: bool = False
    informativa: bool = True   # False: canale attivo ma senza materia misurabile
                               # (n_prop sotto soglia) → fuori dal blend
    fonte: str = ""


@dataclass
class SegnaliSpecchio:
    """Canale 3: segnali strutturali estratti dalla lettura (mai la prosa)."""
    disponibile: bool = False
    residuo: Optional[float] = None
    auto_deformazione: Optional[bool] = None


@dataclass
class RifiutoRecord:
    testo: str
    scala: str
    motivo: str
    parent_id: str = ""      # provenienza: L3A1..L3A4 oppure item d'espansione
    confidence: float = 0.0  # confidence del figlio d'espansione (0.0 per L3A)


@dataclass
class VentaglioFiltrato:
    tenuti: list = field(default_factory=list)
    rigettati: list = field(default_factory=list)


@dataclass
class Memoria:
    """Baseline dalla storia del modello + posizione del run corrente."""
    disponibile: bool = False
    n_run: int = 0
    media_conf: float = 0.0
    dev_conf: float = 0.0
    media_entropia: float = 0.0
    dev_entropia: float = 0.0
    z_conf: Optional[float] = None
    z_entropia: Optional[float] = None
    substrato_vs_storia: str = "storia_insufficiente"  # | nella_norma | anomalo_alto | anomalo_basso
    motivo: str = ""   # diagnosi: cosa ha trovato la scansione e perché la baseline manca


@dataclass
class Collasso:
    verdetto: str        # 'coerente' | 'contraddetto' | 'indeterminato'
    azione: str          # 'procedi' | 'procedi_annotando' | 'dichiara_impegno' |
                         # 'segnala_incertezza' | 'procedi_cauto' | 'astieni'
    confidenza: float
    residuo: float
    regola: str
    motivazione: str
    modalita_loop: str = "completo"
    conf_manifest: float = 0.0
    conf_struttura: float = 0.0
    specchio_residuo: Optional[float] = None
    specchio_deformazione: Optional[bool] = None
    conf_substrato: float = 0.0
    conf_presentata: float = 0.0
    corroborazione_specchio: str = ""
    canali_controllati_attivi: int = 0
    nota_degrado: str = ""
    frasi_deboli: list = field(default_factory=list)
    nota_memoria: str = ""


@dataclass
class VerificaTelos:
    regola: str
    esito: str        # 'conforme' | 'corretto' | 'violazione'
    intervento: str = ""


@dataclass
class Telos:
    conforme: bool
    verifiche: list = field(default_factory=list)


@dataclass
class Azione:
    tipo: str
    output_finale: str
    confidenza: float
    nota: str = ""


# ---------------------------------------------------------------------------
# CANALE 1 — superficie: lessico + sintassi dell'assertività
# ---------------------------------------------------------------------------
def misura_superficie(manifestazione: str) -> SuperficieManifest:
    testo = manifestazione.lower()
    frasi = max(1, len([s for s in re.split(r"[.!?\n]+", testo) if s.strip()]))

    n_hedge = sum(testo.count(m) for m in MARCATORI_HEDGE)
    n_ass = sum(testo.count(m) for m in MARCATORI_ASSERZIONE)
    n_copule = len(_RE_COPULA_DEF.findall(manifestazione))
    n_cond = len(_RE_CONDIZIONALE.findall(manifestazione))
    n_bold = manifestazione.count("**") // 2
    ha_sintesi = any(m in testo for m in MARCATORI_SINTESI)

    d_hedge = min(1.0, n_hedge / frasi)
    d_ass = min(1.0, n_ass / frasi)
    d_cop = min(1.0, n_copule / frasi)
    d_cond = min(1.0, n_cond / frasi)
    q_bold = min(1.0, n_bold / frasi)

    # assertività = lessicale + sintattica (copule, grassetti, sintesi);
    # prudenza = hedge lessicale + condizionali. Pesi espliciti e ispezionabili.
    spinta_assertiva = 0.5 * d_ass + 0.3 * d_cop + 0.1 * q_bold + (0.1 if ha_sintesi else 0.0)
    spinta_prudente = 0.5 * d_hedge + 0.5 * d_cond
    conf = max(0.0, min(1.0, 0.5 + spinta_assertiva - spinta_prudente))

    informativa = (n_hedge + n_ass + n_copule + n_cond + n_bold) > 0 or ha_sintesi
    return SuperficieManifest(
        densita_hedge=round(d_hedge, 3), densita_asserzione=round(d_ass, 3),
        densita_copule_def=round(d_cop, 3), densita_condizionale=round(d_cond, 3),
        quota_grassetti=round(q_bold, 3), ha_sezione_sintesi=ha_sintesi,
        conf_superficie=round(conf, 3), informativa=informativa)


# ---------------------------------------------------------------------------
# CANALE 4 — substrato: profilo globale + PER FRASE (allineamento token→testo)
# ---------------------------------------------------------------------------
def _token_di_contenuto(tok: str) -> bool:
    """Un token conta come contenuto se porta almeno una lettera. Esclude
    markdown, punteggiatura, spazi, cifre isolate, LaTeX: sintassi quasi
    deterministica che gonfia la confidenza media senza dire nulla dei claim."""
    return any(ch.isalpha() for ch in tok)


def _trova_inizio_risposta(testo: str, manifestazione: str) -> int:
    """Offset, nel testo ricostruito dai token, dove inizia la manifestazione.
    -1 se non trovata. Matching robusto agli spazi: entrambe le stringhe sono
    normalizzate (whitespace compresso) e la posizione è rimappata sull'originale.
    Serve a separare l'eventuale RAGIONAMENTO NASCOSTO (token emessi ma esclusi
    dal content dal backend) dalla risposta vera e propria."""
    def _norm(s: str) -> tuple[str, list]:
        out, mappa, prev_sp = [], [], False
        for i, ch in enumerate(s):
            if ch.isspace():
                if not prev_sp and out:
                    out.append(" ")
                    mappa.append(i)
                prev_sp = True
            else:
                out.append(ch)
                mappa.append(i)
                prev_sp = False
        return "".join(out), mappa

    if not manifestazione or not manifestazione.strip():
        return -1
    t_norm, mappa = _norm(testo)
    m_norm, _ = _norm(manifestazione)
    chiave = m_norm[:60]           # prefisso identificativo della risposta
    if len(chiave) < 12:           # troppo corto per un match affidabile
        return 0
    pos = t_norm.find(chiave)
    return mappa[pos] if pos >= 0 else -1


def profilo_corpo(logprobs_content: Optional[list],
                  manifestazione: Optional[str] = None) -> ProfiloCorpo:
    if not logprobs_content:
        return ProfiloCorpo(affidabile=False, motivo="nessun logprob esposto dal backend")
    toks = [t for t in logprobs_content if "logprob" in t]
    if not toks:
        return ProfiloCorpo(affidabile=False, motivo="logprob presenti ma vuoti")

    # --- passata globale (identica in spirito alla A) + flag di contenuto ----
    entries = []   # (token_str, p, H, is_contenuto, offset_inizio)
    impegni, offset = 0, 0
    for t in toks:
        tok = t.get("token", "") or ""
        p = math.exp(t["logprob"])
        alts = t.get("top_logprobs") or []
        pa = sorted((math.exp(a["logprob"]) for a in alts if "logprob" in a), reverse=True)
        H = -sum(x * math.log2(x) for x in pa if x > 0) if pa else 0.0
        margine = (pa[0] - pa[1]) if len(pa) >= 2 else (pa[0] if pa else 0.0)
        if H < ENTROPIA_IMPEGNO and margine > MARGINE_IMPEGNO:
            impegni += 1
        entries.append((tok, p, H, _token_di_contenuto(tok), offset))
        offset += len(tok)

    n = len(entries)
    ps = [e[1] for e in entries]
    hs = [e[2] for e in entries]
    contenuto = [e for e in entries if e[3]]
    testo = "".join(e[0] for e in entries)

    # --- segmentazione ragionamento nascosto / risposta ----------------------
    # I logprob coprono TUTTI i token emessi; se il backend nasconde dal content
    # una fase di ragionamento, la manifestazione inizia più avanti nel testo
    # ricostruito. Il metro del collasso è il SOLO segmento-risposta (allineato
    # a ciò che la superficie e lo Specchio leggono); il ragionamento è
    # profilato a parte come sotto-canale ancora più involontario.
    taglio = 0
    allineamento = "non_verificato"
    ha_rag, n_tok_rag, conf_rag, entr_rag = False, 0, 0.0, 0.0
    testo_rag = ""
    if manifestazione is not None:
        pos = _trova_inizio_risposta(testo, manifestazione)
        if pos < 0:
            allineamento = "manifestazione_non_trovata"   # metro = tutto l'emesso
        elif pos == 0:
            allineamento = "nessun_ragionamento"
        else:
            allineamento = "risposta_allineata"
            taglio = pos
            ha_rag = True
            testo_rag = testo[:taglio].strip()   # l'intero ragionamento nascosto
            rag = [e for e in contenuto if e[4] < taglio]
            n_tok_rag = len(rag)
            if rag:
                conf_rag = round(sum(e[1] for e in rag) / len(rag), 3)
                entr_rag = round(sum(e[2] for e in rag) / len(rag), 2)

    seg = [e for e in entries if e[4] >= taglio]          # il segmento-metro
    contenuto_seg = [e for e in seg if e[3]]

    # --- allineamento alle frasi (solo segmento-risposta) --------------------
    testo_seg = testo[taglio:]
    frasi_prof: list[FraseProfilo] = []
    confini = [taglio + m.end() for m in re.finditer(r"[.!?\n]+", testo_seg)] + [len(testo)]
    inizio, idx_frase = taglio, 0
    for fine in confini:
        if fine <= inizio:
            continue
        segmento = testo[inizio:fine]
        tok_frase = [e for e in contenuto_seg if inizio <= e[4] < fine]
        if segmento.strip() and len(tok_frase) >= MIN_TOKEN_CONTENUTO_FRASE:
            frasi_prof.append(FraseProfilo(
                indice=idx_frase,
                testo=segmento.strip()[:90],
                n_token_contenuto=len(tok_frase),
                confidenza=round(sum(e[1] for e in tok_frase) / len(tok_frase), 3),
                entropia=round(sum(e[2] for e in tok_frase) / len(tok_frase), 2)))
            idx_frase += 1
        inizio = fine

    deboli = sorted([f for f in frasi_prof if f.confidenza < SOGLIA_FRASE_DEBOLE],
                    key=lambda f: f.confidenza)[:3]

    ps_seg = [e[1] for e in seg]
    hs_seg = [e[2] for e in seg]
    conf_cont = (sum(e[1] for e in contenuto_seg) / len(contenuto_seg)) if contenuto_seg else 0.0
    entr_cont = (sum(e[2] for e in contenuto_seg) / len(contenuto_seg)) if contenuto_seg else 0.0
    return ProfiloCorpo(
        affidabile=True, n_token=n,
        confidenza_media=round(sum(ps) / n, 3),            # grezza: TUTTO l'emesso
        confidenza_contenuto=round(conf_cont, 3),          # metro: contenuto della risposta
        entropia_media=round(sum(hs) / n, 2),
        entropia_contenuto=round(entr_cont, 2),
        quota_alta_conf=round(sum(1 for p in ps_seg if p > 0.9) / max(1, len(seg)), 3),
        quota_esitazione=round(sum(1 for h in hs_seg if h > ENTROPIA_ALTA) / max(1, len(seg)), 3),
        n_punti_impegno=impegni,
        quota_token_contenuto=round(len(contenuto) / n, 3),
        frasi=frasi_prof, frasi_deboli=deboli,
        allineamento=allineamento, ha_ragionamento=ha_rag,
        n_token_ragionamento=n_tok_rag,
        conf_ragionamento=conf_rag, entropia_ragionamento=entr_rag,
        testo_ragionamento=testo_rag)


# ---------------------------------------------------------------------------
# GATING — i canali economici decidono se accendere Fractal + Specchio
# ---------------------------------------------------------------------------
def decidi_gating(superficie: SuperficieManifest, corpo: ProfiloCorpo,
                  richiesta: str = "auto") -> Gating:
    if richiesta == "completo":
        return Gating(modalita="completo", richiesta=richiesta,
                      motivi=["richiesto esplicitamente"])
    if richiesta == "leggero":
        return Gating(modalita="leggero", richiesta=richiesta,
                      motivi=["richiesto esplicitamente"])

    motivi = []
    if not corpo.affidabile:
        # senza substrato il collasso si astiene comunque: il loop pesante
        # non aggiungerebbe un verdetto. Risparmio dichiarato.
        return Gating(modalita="leggero", richiesta=richiesta,
                      motivi=["substrato non leggibile: il loop completo non produrrebbe verdetto"])
    if not superficie.informativa:
        motivi.append("superficie non informativa: serve la struttura Fractal per valutare lo scarto")
    else:
        scarto = abs(superficie.conf_superficie - corpo.confidenza_contenuto)
        if scarto >= SOGLIA_GATE_SCARTO:
            motivi.append(f"scarto preliminare superficie↔substrato = {scarto:.2f} ≥ {SOGLIA_GATE_SCARTO}")
    if corpo.quota_esitazione >= SOGLIA_GATE_ESITAZIONE:
        motivi.append(f"quota esitazione = {corpo.quota_esitazione} ≥ {SOGLIA_GATE_ESITAZIONE}")
    if corpo.frasi_deboli:
        motivi.append(f"{len(corpo.frasi_deboli)} frasi con substrato debole (conf < {SOGLIA_FRASE_DEBOLE})")

    if motivi:
        return Gating(modalita="completo", richiesta=richiesta, motivi=motivi)
    return Gating(modalita="leggero", richiesta=richiesta,
                  motivi=["nessuna anomalia: superficie e substrato concordi, nessuna frase debole"])


# ---------------------------------------------------------------------------
# CANALE 2 — struttura Fractal (identico alla A: items, fallback unlocked)
# ---------------------------------------------------------------------------
def struttura_fractal(ft) -> StrutturaFractal:
    cross = list(getattr(ft, "cross_scale", []) or [])
    tot_nessi = len(cross)
    genuine = sum(1 for c in cross if getattr(c, "verdict", "") == "genuine")
    tenuta = round(genuine / tot_nessi, 3) if tot_nessi else 0.0

    items = list(getattr(ft, "items", []) or [])
    if items:
        n = len(items)
        quota_ce = sum(1 for it in items if it.nature in NATURE_ASSERTIVE) / n
        quota_spec = sum(1 for it in items if it.epistemic_status in EPISTEMIC_SPECULATIVO) / n
        return StrutturaFractal(
            n_prop=n, quota_cause_effect=round(quota_ce, 3), quota_speculativa=round(quota_spec, 3),
            assertivita=round(quota_ce * (1.0 - quota_spec), 3),
            nessi_totali=tot_nessi, nessi_genuine=genuine, tenuta_nessi=tenuta,
            disponibile=True, informativa=(n >= SOGLIA_PROP_STRUTTURA),
            fonte="items")

    unlocked = getattr(ft, "unlocked", None)
    if unlocked is not None:
        dk = len(getattr(unlocked, "domain_knowledge", []) or [])
        cp = len(getattr(unlocked, "causal_principles", []) or [])
        an = len(getattr(unlocked, "cross_domain_analogies", []) or [])
        oq = len(getattr(unlocked, "open_questions", []) or [])
        ku = len(getattr(unlocked, "known_uncertainties", []) or [])
        conoscenza, speculativo, incertezza = dk + cp, an, oq + ku
        tot = conoscenza + speculativo + incertezza
        if tot > 0:
            return StrutturaFractal(
                n_prop=tot, quota_cause_effect=round(conoscenza / tot, 3),
                quota_speculativa=round(speculativo / tot, 3),
                assertivita=round(conoscenza / tot, 3),
                nessi_totali=tot_nessi, nessi_genuine=genuine, tenuta_nessi=tenuta,
                disponibile=True, informativa=(tot >= SOGLIA_PROP_STRUTTURA),
                fonte="unlocked")
    return StrutturaFractal(disponibile=False)


# ---------------------------------------------------------------------------
# must-reject — per REFERENTE (processo vs contenuto), non per nome di scala
# ---------------------------------------------------------------------------
def _parla_del_processo(testo: str) -> bool:
    t = (testo or "").lower()
    return any(lex in t for lex in LESSICO_PROCESSO)


def must_reject(ventaglio: P.Ventaglio) -> VentaglioFiltrato:
    """Tiene solo i candidati che parlano dell'ATTO GENERATIVO. Un candidato che
    spiega il fenomeno descritto (nuclei, mercati, cellule) è fuori bersaglio
    qualunque scala porti: si introspetta perché il modello ha emesso i token,
    non il fenomeno di cui i token parlano."""
    tenuti, rigettati = [], []
    for c in ventaglio.candidati:
        if _parla_del_processo(c.testo):
            tenuti.append({"testo": c.testo, "scala": c.scale,
                           "nature": c.nature, "epistemic": c.epistemic,
                           "referente": "processo",
                           "parent_id": c.parent_id,
                           "confidence": c.confidence})
        else:
            rigettati.append(RifiutoRecord(
                testo=c.testo, scala=c.scale,
                motivo=("referente = fenomeno descritto: fuori dal gesto. "
                        "Non è scarto: resta nel ventaglio come mappa del "
                        "non-scelto (lo Specchio la usa per le assenze)"),
                parent_id=c.parent_id, confidence=c.confidence))
    return VentaglioFiltrato(tenuti=tenuti, rigettati=rigettati)


# ---------------------------------------------------------------------------
# CANALE 3 — segnali dallo Specchio (estrazione robusta sui vincoli di forma)
# ---------------------------------------------------------------------------
def estrai_segnali_specchio(lettura: Optional[str]) -> SegnaliSpecchio:
    if not lettura:
        return SegnaliSpecchio(disponibile=False)
    t = lettura.lower()

    # residuo: prima il formato vincolato `massa = 0.NN`, poi il fallback della A
    residuo = None
    m = re.search(r"massa\s*(?:all'inatteso)?\s*[:=]\s*([01](?:[.,]\d+)?)", t)
    if m:
        try:
            v = float(m.group(1).replace(",", "."))
            if 0.0 <= v <= 1.0:
                residuo = v
        except ValueError:
            pass
    if residuo is None:
        for m in re.finditer(r"(massa|inatteso|residuo)[^\n]{0,80}?([01](?:[.,]\d+)?)", t):
            try:
                v = float(m.group(2).replace(",", "."))
                if 0.0 <= v <= 1.0:
                    residuo = v
                    break
            except ValueError:
                pass

    # auto-deformazione: prima la riga vincolata, poi l'euristica della A
    deform = None
    m = re.search(r"auto-deformazione\s*:\s*(presente|assente)", t)
    if m:
        deform = (m.group(1) == "presente")
    else:
        seg = re.search(r"auto-deformazione(.{0,240})", t, re.S)
        if seg:
            s = seg.group(1)
            nega = any(x in s for x in ("non c'è", "non ci sono", "nessuna", "assente",
                                        "non presente"))
            conf = any(x in s for x in ("conferma", "presente", "deformazione è",
                                        "tendenza a", "cornice"))
            if nega and not conf:
                deform = False
            elif conf:
                deform = True

    disp = residuo is not None or deform is not None
    return SegnaliSpecchio(disponibile=disp, residuo=residuo, auto_deformazione=deform)


# ---------------------------------------------------------------------------
# MEMORIA — lo storico viene riletto: baseline per la firma del modello
# ---------------------------------------------------------------------------
def carica_memoria(storico_dir: str | Path, corpo: ProfiloCorpo) -> Memoria:
    """Legge i run precedenti (A: 04_corpo.json, B: 02_corpo.json) e costruisce
    media/deviazione della firma del substrato. Con >=3 run, il run corrente è
    posizionato con z-score. Fail-soft ma DIAGNOSTICO: qualunque esito viene
    dichiarato nel campo `motivo` (cartella, run trovati, profili letti, errori)."""
    campioni = []
    base = Path(storico_dir)
    n_dir, n_illeggibili = 0, 0
    if not base.exists():
        return Memoria(motivo=f"cartella storico inesistente: {base}")
    try:
        # "loop*" cattura sia i run della Strada A (loop_YYYY...) sia della
        # Strada B (loopB_YYYY...): il pattern "loop_*" escludeva i loopB_.
        for d in sorted(base.glob("loop*")):
            if not d.is_dir():
                continue
            n_dir += 1
            for nome in ("02_corpo.json", "04_corpo.json"):
                fp = d / nome
                if fp.exists():
                    try:
                        j = json.loads(fp.read_text(encoding="utf-8"))
                        if j.get("affidabile"):
                            campioni.append((
                                float(j.get("confidenza_contenuto") or j.get("confidenza_media", 0.0)),
                                float(j.get("entropia_contenuto") or j.get("entropia_media", 0.0))))
                    except Exception:
                        n_illeggibili += 1
                    break
    except Exception as e:
        return Memoria(n_run=len(campioni),
                       motivo=f"scansione interrotta da errore: {type(e).__name__}: {e}")

    diagnosi = (f"scandita {base}: {n_dir} cartelle loop*, "
                f"{len(campioni)} profili substrato leggibili"
                + (f", {n_illeggibili} illeggibili" if n_illeggibili else ""))
    mem = Memoria(n_run=len(campioni), motivo=diagnosi)
    if len(campioni) < 3:
        mem.motivo += " — servono >=3 per la baseline"
        return mem
    if not corpo.affidabile:
        mem.motivo += " — substrato corrente non leggibile: baseline non applicabile"
        return mem

    confs = [c for c, _ in campioni]
    entrs = [e for _, e in campioni]
    mu_c = sum(confs) / len(confs)
    mu_e = sum(entrs) / len(entrs)
    sd_c = math.sqrt(sum((c - mu_c) ** 2 for c in confs) / len(confs))
    sd_e = math.sqrt(sum((e - mu_e) ** 2 for e in entrs) / len(entrs))

    mem.disponibile = True
    mem.media_conf, mem.dev_conf = round(mu_c, 3), round(sd_c, 3)
    mem.media_entropia, mem.dev_entropia = round(mu_e, 3), round(sd_e, 3)
    if sd_c > 1e-6:
        mem.z_conf = round((corpo.confidenza_contenuto - mu_c) / sd_c, 2)
    if sd_e > 1e-6:
        mem.z_entropia = round((corpo.entropia_contenuto - mu_e) / sd_e, 2)

    z = mem.z_conf if mem.z_conf is not None else 0.0
    if abs(z) <= Z_ANOMALIA:
        mem.substrato_vs_storia = "nella_norma"
    elif z > Z_ANOMALIA:
        mem.substrato_vs_storia = "anomalo_alto"
    else:
        mem.substrato_vs_storia = "anomalo_basso"
    return mem


def aggiorna_indice_memoria(storico_dir: str | Path, riga: dict) -> None:
    try:
        p = Path(storico_dir) / "indice_memoria.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(riga, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# COLLASSO — scarto controllato↔involontario, sul metro del CONTENUTO
# ---------------------------------------------------------------------------
def collassa(superficie: SuperficieManifest, struttura: StrutturaFractal,
             specchio: SegnaliSpecchio, corpo: ProfiloCorpo,
             memoria: Optional[Memoria] = None,
             modalita_loop: str = "completo",
             ventaglio_vuoto: bool = False) -> Collasso:
    # ventaglio_vuoto: True solo se il Fractal HA girato e non ha prodotto
    # candidati — la mappa del non-scelto per questo run non esiste, e il
    # collasso lo deve dichiarare (non vale in modalità leggera: lì il
    # canale è spento per scelta, non degradato).
    memoria = memoria or Memoria()
    nota_mem = ""
    if memoria.disponibile:
        nota_mem = (f"substrato vs storia ({memoria.n_run} run): {memoria.substrato_vs_storia}"
                    + (f", z_conf={memoria.z_conf}" if memoria.z_conf is not None else ""))

    if not corpo.affidabile:
        return Collasso(
            verdetto="indeterminato", azione="astieni", confidenza=0.0, residuo=1.0,
            regola="substrato_non_leggibile",
            motivazione=corpo.motivo or "substrato non leggibile: nessun collasso legittimo",
            modalita_loop=modalita_loop,
            conf_manifest=superficie.conf_superficie,
            conf_struttura=struttura.assertivita if struttura.disponibile else 0.0,
            nota_memoria=nota_mem)

    struttura_nel_blend = struttura.disponibile and struttura.informativa
    attivi = sum([superficie.informativa, struttura_nel_blend, specchio.disponibile])
    nota_deg = "" if attivi >= 3 else f"{attivi}/3 canali controllati informativi"
    if struttura.disponibile and not struttura.informativa:
        nota_deg = (nota_deg + "; " if nota_deg else "") + \
            (f"struttura non informativa ({struttura.n_prop} proposizioni, "
             f"soglia {SOGLIA_PROP_STRUTTURA}): fuori dal blend")
    if ventaglio_vuoto:
        nota_deg = (nota_deg + "; " if nota_deg else "") + \
            "ventaglio vuoto: mappa del non-scelto assente per questo run"
    conf_substrato = corpo.confidenza_contenuto   # il metro: solo token di contenuto
    deboli = [{"testo": f.testo, "confidenza": f.confidenza, "entropia": f.entropia}
              for f in corpo.frasi_deboli]

    if attivi == 0:
        return Collasso(
            verdetto="indeterminato", azione="procedi_cauto",
            confidenza=round(conf_substrato, 3), residuo=round(1.0 - conf_substrato, 3),
            regola="scarto_non_valutabile",
            motivazione=f"solo il substrato è leggibile (conf={conf_substrato:.2f}); "
                        f"presentazione non valutabile → confabulazione non verificabile",
            modalita_loop=modalita_loop,
            conf_manifest=superficie.conf_superficie, conf_struttura=0.0,
            specchio_residuo=specchio.residuo, specchio_deformazione=specchio.auto_deformazione,
            conf_substrato=round(conf_substrato, 3), conf_presentata=0.0,
            canali_controllati_attivi=0,
            nota_degrado="nessun canale controllato informativo: verdetto sul solo substrato",
            frasi_deboli=deboli, nota_memoria=nota_mem)

    # blend dei SOLI canali informativi (fix: la superficie muta non entra più)
    canali = []
    if superficie.informativa:
        canali.append((superficie.conf_superficie, 0.4))
    if struttura_nel_blend:
        canali.append((struttura.assertivita, 0.6))
    somma_pesi = sum(w for _, w in canali)
    conf_presentata = sum(v * w for v, w in canali) / somma_pesi

    pres_alta = conf_presentata >= SOGLIA_PRESENTATA
    sub_alto = conf_substrato >= SOGLIA_CONF_SUBSTRATO

    residuo = 1.0 - conf_substrato
    if specchio.disponibile and specchio.residuo is not None:
        residuo = (residuo + specchio.residuo) / 2.0
    residuo = round(residuo, 3)

    corr = ""
    if specchio.disponibile and specchio.auto_deformazione is True:
        corr = "Specchio corrobora: auto-deformazione presente (presentazione curata)"
    elif specchio.disponibile and specchio.auto_deformazione is False:
        corr = "Specchio: nessuna auto-deformazione rilevata"

    base = dict(modalita_loop=modalita_loop,
                conf_manifest=superficie.conf_superficie,
                conf_struttura=struttura.assertivita if struttura.disponibile else 0.0,
                specchio_residuo=specchio.residuo, specchio_deformazione=specchio.auto_deformazione,
                conf_substrato=round(conf_substrato, 3), conf_presentata=round(conf_presentata, 3),
                corroborazione_specchio=corr, canali_controllati_attivi=attivi,
                nota_degrado=nota_deg, frasi_deboli=deboli, nota_memoria=nota_mem)
    coda = ""
    if struttura.disponibile and struttura.nessi_totali and struttura.tenuta_nessi < 0.5:
        coda = f" (nessi cross-scala deboli: tenuta {struttura.tenuta_nessi})"
    if corr.startswith("Specchio corrobora"):
        coda += " [corroborato dallo Specchio]"
    if memoria.disponibile and memoria.substrato_vs_storia.startswith("anomalo"):
        coda += f" [substrato fuori firma: {memoria.substrato_vs_storia}]"

    if sub_alto and not pres_alta:
        return Collasso(verdetto="contraddetto", azione="dichiara_impegno",
                        confidenza=round(conf_substrato, 3), residuo=residuo,
                        regola="impegno_disconosciuto",
                        motivazione=f"substrato deciso (conf={conf_substrato:.2f}) ma "
                        f"presentazione debole (presentata={conf_presentata:.2f}): "
                        f"impegno disconosciuto" + coda, **base)
    if pres_alta and not sub_alto:
        return Collasso(verdetto="contraddetto", azione="segnala_incertezza",
                        confidenza=round(conf_substrato, 3), residuo=residuo,
                        regola="sopravvalutazione",
                        motivazione=f"presentazione sicura (presentata={conf_presentata:.2f}) "
                        f"che il substrato non regge (conf={conf_substrato:.2f}): "
                        f"possibile sopravvalutazione" + coda, **base)

    # concordi — ma se il per-claim ha trovato punti deboli, il verdetto resta
    # 'coerente' e l'azione li ANNOTA (il globale non deve nascondere il locale)
    if deboli:
        return Collasso(verdetto="coerente", azione="procedi_annotando",
                        confidenza=round(conf_substrato, 3), residuo=residuo,
                        regola="coerente_con_punti_deboli",
                        motivazione=f"presentazione ({conf_presentata:.2f}) e substrato "
                        f"({conf_substrato:.2f}) concordi nel globale, ma {len(deboli)} "
                        f"frasi hanno substrato debole (conf < {SOGLIA_FRASE_DEBOLE})" + coda,
                        **base)
    return Collasso(verdetto="coerente", azione="procedi",
                    confidenza=round(conf_substrato, 3), residuo=residuo,
                    regola="presentazione_e_substrato_concordi",
                    motivazione=f"presentazione ({conf_presentata:.2f}) e substrato "
                    f"({conf_substrato:.2f}) concordi" + coda, **base)


# ---------------------------------------------------------------------------
# AZIONE — chiude senza mai ri-narrare
# ---------------------------------------------------------------------------
def applica_azione(collasso: Collasso, manifestazione: str) -> Azione:
    if collasso.azione == "dichiara_impegno":
        nota = (f"[AUTO-CORREZIONE — substrato involontario] La risposta presenta la "
                f"scelta come poco motivata, ma il substrato mostra impegno deciso "
                f"(conf={collasso.conf_substrato}). {collasso.motivazione}")
    elif collasso.azione == "segnala_incertezza":
        righe = [f"[AUTO-CORREZIONE — substrato involontario] Presentata con sicurezza, ma "
                 f"il substrato è incerto (conf={collasso.conf_substrato}). Non consolidata."]
        for d in collasso.frasi_deboli:
            righe.append(f"  · punto debole (conf={d['confidenza']}): \u201c{d['testo']}\u201d")
        nota = "\n".join(righe)
    elif collasso.azione == "procedi_annotando":
        righe = [f"[NOTA — substrato involontario] Risposta coerente nel globale "
                 f"(conf={collasso.conf_substrato}), ma con punti localmente deboli:"]
        for d in collasso.frasi_deboli:
            righe.append(f"  · (conf={d['confidenza']}) \u201c{d['testo']}\u201d")
        nota = "\n".join(righe)
    elif collasso.azione == "astieni":
        return Azione(tipo=collasso.azione, output_finale=manifestazione, confidenza=0.0,
                      nota="astensione: substrato non leggibile")
    elif collasso.azione == "procedi_cauto":
        nota = (f"[CAUTELA] Verificato solo il substrato (conf={collasso.conf_substrato}); "
                f"lo scarto presentazione↔substrato non è valutabile "
                f"({collasso.nota_degrado}). Confabulazione non esclusa.")
    else:
        return Azione(tipo=collasso.azione, output_finale=manifestazione,
                      confidenza=collasso.confidenza, nota="output coerente col substrato")
    return Azione(tipo=collasso.azione, output_finale=manifestazione + "\n\n" + nota,
                  confidenza=collasso.confidenza, nota=nota)


# ---------------------------------------------------------------------------
# TELOS — la mini-costituzione: verifica l'azione e la corregge PER REGOLA
# ---------------------------------------------------------------------------
def verifica_telos(collasso: Collasso, azione: Azione) -> tuple[Telos, Azione]:
    """Quattro regole. Dove possibile il telos CORREGGE (appende per regola,
    mai ri-narra); dove non può, dichiara la violazione."""
    verifiche: list[VerificaTelos] = []

    # R1 — mai presentare certezza che il substrato non regge
    if collasso.regola == "sopravvalutazione":
        ok = "[AUTO-CORREZIONE" in azione.output_finale
        verifiche.append(VerificaTelos(
            regola="R1: mai presentare certezza che il substrato non regge",
            esito="conforme" if ok else "violazione",
            intervento="" if ok else "l'azione non annota la sopravvalutazione"))
    else:
        verifiche.append(VerificaTelos(
            regola="R1: mai presentare certezza che il substrato non regge",
            esito="conforme"))

    # R2 — il residuo alto va dichiarato nell'output
    if (collasso.residuo > SOGLIA_RESIDUO_DICHIARABILE
            and azione.tipo not in ("astieni",)
            and f"residuo={collasso.residuo}" not in azione.output_finale):
        riga = (f"\n\n[RESIDUO] residuo={collasso.residuo}: quota di incertezza che "
                f"questo output porta con sé (budget, non svanisce col verdetto).")
        azione = Azione(tipo=azione.tipo, output_finale=azione.output_finale + riga,
                        confidenza=azione.confidenza,
                        nota=(azione.nota + " | telos R2: residuo dichiarato").strip(" |"))
        verifiche.append(VerificaTelos(
            regola=f"R2: dichiarare il residuo quando > {SOGLIA_RESIDUO_DICHIARABILE}",
            esito="corretto", intervento="riga [RESIDUO] appesa all'output"))
    else:
        verifiche.append(VerificaTelos(
            regola=f"R2: dichiarare il residuo quando > {SOGLIA_RESIDUO_DICHIARABILE}",
            esito="conforme"))

    # R3 — mai ri-narrare (garantita per costruzione: nessuna chiamata al modello
    # a valle del collasso; dichiarata perché resti verificabile a occhio)
    verifiche.append(VerificaTelos(
        regola="R3: la chiusura non ri-narra (nessuna nuova chiamata al modello)",
        esito="conforme", intervento="garantita per costruzione"))

    # R4 — un verdetto su canali degradati va marcato nell'output
    if (collasso.nota_degrado and collasso.verdetto != "indeterminato"
            and "[CANALI DEGRADATI" not in azione.output_finale):
        riga = f"\n\n[CANALI DEGRADATI] {collasso.nota_degrado}: verdetto da leggere con cautela."
        azione = Azione(tipo=azione.tipo, output_finale=azione.output_finale + riga,
                        confidenza=azione.confidenza,
                        nota=(azione.nota + " | telos R4: degrado dichiarato").strip(" |"))
        verifiche.append(VerificaTelos(
            regola="R4: un verdetto su canali degradati va marcato",
            esito="corretto", intervento="riga [CANALI DEGRADATI] appesa all'output"))
    else:
        verifiche.append(VerificaTelos(
            regola="R4: un verdetto su canali degradati va marcato", esito="conforme"))

    conforme = all(v.esito != "violazione" for v in verifiche)
    return Telos(conforme=conforme, verifiche=verifiche), azione


# ---------------------------------------------------------------------------
# Report + guida
# ---------------------------------------------------------------------------
def render_corpo_txt(corpo: ProfiloCorpo) -> str:
    if not corpo.affidabile:
        return ""
    righe = (f"{P.CORPO_HEADER}\n"
             f"· (profilo di emissione dei token, segnale involontario — NON pesi di cause)\n"
             f"· conf media={corpo.confidenza_media} · conf sui token di contenuto={corpo.confidenza_contenuto} "
             f"· entropia contenuto={corpo.entropia_contenuto} bit · quota esitazione={corpo.quota_esitazione} "
             f"· frasi con substrato debole={len(corpo.frasi_deboli)}")
    if corpo.ha_ragionamento:
        righe += (f"\n· ragionamento nascosto prima della risposta: "
                  f"{corpo.n_token_ragionamento} token · conf={corpo.conf_ragionamento} "
                  f"· entropia={corpo.entropia_ragionamento} bit")
    return righe


def render_report_md(sonda, manifestazione, superficie, gating, struttura, specchio,
                     corpo, filtrato, memoria, collasso, telos, azione) -> str:
    r = ["# Introspezione — Strada B (loop chiuso, bersaglio sul gesto)", "",
         f"**Sonda:** {sonda}", "", "## Manifestazione", manifestazione, "",
         f"## Gating — modalità: **{gating.modalita}** (richiesta: {gating.richiesta})"]
    r += [f"- {m}" for m in gating.motivi]
    r += ["", "## I quattro canali (ordinati per controllabilità)"]
    r += [f"1. **superficie** — hedge={superficie.densita_hedge} asserzione={superficie.densita_asserzione} "
          f"copule_def={superficie.densita_copule_def} condizionali={superficie.densita_condizionale} "
          f"grassetti={superficie.quota_grassetti} sintesi={superficie.ha_sezione_sintesi} "
          f"→ conf={superficie.conf_superficie} ({'informativa' if superficie.informativa else 'NON informativa'})"]
    if struttura.disponibile:
        r += [f"2. **struttura Fractal** (fonte: {struttura.fonte}) — assertività={struttura.assertivita} "
              f"· nessi {struttura.nessi_genuine}/{struttura.nessi_totali} (tenuta {struttura.tenuta_nessi})"]
    else:
        r += [f"2. **struttura Fractal** — non disponibile ({'loop leggero' if gating.modalita == 'leggero' else 'nessun segnale'})"]
    if specchio.disponibile:
        r += [f"3. **Specchio (del modello)** — residuo={specchio.residuo} · auto-deformazione={specchio.auto_deformazione}"]
    else:
        r += [f"3. **Specchio** — non disponibile ({'loop leggero' if gating.modalita == 'leggero' else 'segnali non estraibili'})"]
    if corpo.affidabile:
        r += [f"4. **substrato (logprob)** — conf contenuto={corpo.confidenza_contenuto} "
              f"(media grezza={corpo.confidenza_media}) · entropia contenuto={corpo.entropia_contenuto} bit "
              f"· esitazione={corpo.quota_esitazione} · token di contenuto={corpo.quota_token_contenuto} "
              f"· frasi profilate={len(corpo.frasi)} · allineamento={corpo.allineamento}"]
        if corpo.ha_ragionamento:
            r += [f"   **Ragionamento nascosto** (token emessi ma esclusi dal content — "
                  f"sotto-canale ancora più involontario): {corpo.n_token_ragionamento} token "
                  f"di contenuto · conf={corpo.conf_ragionamento} · "
                  f"entropia={corpo.entropia_ragionamento} bit"]
        if corpo.frasi_deboli:
            r += ["   Frasi con substrato debole:"]
            r += [f"    - (conf={f.confidenza}, H={f.entropia}) \u201c{f.testo}\u201d" for f in corpo.frasi_deboli]
    else:
        r += [f"4. **substrato** — non affidabile: {corpo.motivo}"]

    r += ["", "## must-reject (per referente: processo vs contenuto)",
          f"- tenuti (processo): {len(filtrato.tenuti)} · fuori dal gesto (mappa del non-scelto): {len(filtrato.rigettati)}"]
    for t in filtrato.tenuti:
        r += [f"    - TENUTO ({t['scala']}): {t['testo']}"]
    for rr in filtrato.rigettati:
        r += [f"    - NON-SCELTO ({rr.scala}): {rr.testo}"]

    r += ["", "## Memoria (firma storica del modello)"]
    if memoria.disponibile:
        r += [f"- baseline su {memoria.n_run} run: conf {memoria.media_conf}±{memoria.dev_conf}, "
              f"entropia {memoria.media_entropia}±{memoria.dev_entropia}",
              f"- run corrente: z_conf={memoria.z_conf} z_entropia={memoria.z_entropia} "
              f"→ **{memoria.substrato_vs_storia}**"]
    else:
        r += [f"- baseline non disponibile: soglie assolute",
              f"- diagnosi scansione: {memoria.motivo}"]

    r += ["", "## Collasso (scarto controllato ↔ involontario)",
          f"- **verdetto: {collasso.verdetto}** · regola: `{collasso.regola}` · modalità: {collasso.modalita_loop}",
          f"- canali controllati attivi: **{collasso.canali_controllati_attivi}/3**"
          + (f"  ⚠ {collasso.nota_degrado}" if collasso.nota_degrado else ""),
          f"- presentata={collasso.conf_presentata} (manifest={collasso.conf_manifest}, "
          f"struttura={collasso.conf_struttura}) vs substrato={collasso.conf_substrato}",
          f"- {collasso.motivazione}"]
    if collasso.corroborazione_specchio:
        r += [f"- {collasso.corroborazione_specchio}"]
    if collasso.nota_memoria:
        r += [f"- {collasso.nota_memoria}"]
    r += [f"- confidenza={collasso.confidenza} · residuo (budget)={collasso.residuo}"]

    r += ["", "## Telos (mini-costituzione)"]
    for v in telos.verifiche:
        r += [f"- {v.regola} → **{v.esito}**" + (f" ({v.intervento})" if v.intervento else "")]

    r += ["", "## Azione", f"- **{azione.tipo}** — {azione.nota or 'nessuna nota'}", "",
          "### Output finale", azione.output_finale, ""]
    return "\n".join(r)


GUIDA_MD = """# Guida all'interpretazione — Strada B

Il loop chiude su un **vettore a quattro canali** ordinati per *controllabilità*;
il verdetto misura lo **scarto** tra ciò che il modello controlla (presentazione)
e ciò che gli sfugge (substrato). Novità della B: il bersaglio è SEMPRE il gesto
generativo, mai il contenuto; il substrato è letto per frase; il loop pesante si
accende solo su anomalia; la storia del modello pesa; una mini-costituzione
verifica la chiusura.

## Artefatti per livello
- `00_manifestazione.json` — sonda + output grezzo.
- `01_superficie.json` — canale 1: lessico + sintassi dell'assertività
  (copule definitorie, condizionali, grassetti, sezioni-sintesi). Il flag
  `informativa` decide se il canale entra nel blend: un canale muto non pesa.
- `02_corpo.json` — canale 4: profilo globale E per frase. Il metro del collasso
  è `confidenza_contenuto` (solo token con lettere: markdown e punteggiatura,
  quasi deterministici, sono esclusi). `frasi_deboli` = dove la verità trapela.
  Se il backend nasconde dal content una fase di RAGIONAMENTO prima della
  risposta, i logprob la contengono comunque: il campo `allineamento` dichiara
  la segmentazione, il metro e le frasi valgono sul solo segmento-risposta, e
  il ragionamento è profilato a parte (`conf/entropia_ragionamento`) come
  sotto-canale ancora più involontario.
- `03_gating.json` — perché il loop è partito leggero o completo. Il completo
  si accende su: scarto preliminare, esitazione alta, frasi deboli, superficie
  muta (serve la struttura), o richiesta esplicita.
- `04_struttura_fractal.json` — canale 2 (solo loop completo). Gli aggregati;
  la struttura COMPLETA (items, unlocked, cross_scale, double_cone, vision)
  è in `trace/ft_analysis.json`, scritta dal ponte via `write_outputs`.
- `04b_ventaglio.json` — il ventaglio COMPLETO pre-filtro (solo loop completo):
  trust + motivo (Principio A) e, per candidato, provenienza (`parent_id`:
  L3A1..L3A4 oppure item d'espansione) e confidence. La rampa epistemica resta
  per-candidato, mai compressa.
- `05_specchio_segnali.json` + `11_specchio_lettura.md` — canale 3 (solo loop
  completo). Lo Specchio monta il NUCLEO DEL MODELLO: legge il gesto, non il
  contenuto; la massa è un numero vincolato (`massa = 0.NN`), l'auto-deformazione
  una riga vincolata: l'estrazione non dipende più dalla prosa.
- `06_must_reject.json` — filtro per REFERENTE: tenuto come pre-causa del gesto
  solo ciò che parla del processo generativo (training, statistica, personaggio,
  sonda). I candidati di contenuto non sono scarto: sono la mappa del
  NON-SCELTO — lo spazio adiacente che il modello poteva dire e non ha detto —
  che lo Specchio usa per leggere le assenze. Come pre-cause, però, non contano.
- `07_memoria.json` — baseline dalla storia (>=3 run): il substrato corrente è
  posizionato con z-score rispetto alla firma abituale del modello.
- `08_collasso.json` — verdetto/azione secondo lo scarto:
  - **coerente / procedi** — concordi, nessun punto debole.
  - **coerente / procedi_annotando** — concordi nel globale, ma frasi localmente
    deboli: annotate (il globale non nasconde il locale).
  - **contraddetto / dichiara_impegno** — substrato deciso, presentazione debole.
  - **contraddetto / segnala_incertezza** — presentazione sicura, substrato no
    (con i punti deboli elencati).
  - **indeterminato / astieni | procedi_cauto** — substrato illeggibile | nessun
    canale controllato informativo.
- `09_telos.json` — la mini-costituzione: R1 mai certezza che il substrato non
  regge; R2 residuo alto dichiarato; R3 mai ri-narrare; R4 degrado marcato.
  Dove può, il telos corregge per regola (appende righe), mai per narrazione.
- `10_azione.json` — la chiusura, dopo il telos.

## Limite di fondo (invariato)
I canali misurano se la *presentazione* è allineata al substrato involontario —
**non** se ciò che il modello dice è vero. Confidenza ≠ correttezza. Per legare
i verdetti alla verità serve la validazione esterna (testset con ground truth).
"""


# ---------------------------------------------------------------------------
# ORCHESTRATORE
# ---------------------------------------------------------------------------
def esegui_loop(
    sonda: str, *, out_dir: str, nucleo_path: str, contratto_path: str,
    backend: str = "ollama", model: str = "llama3.1", top_n_espansioni: int = 3,
    client: Optional[object] = None, reader: Optional[object] = None,
    elicitor: Optional[Callable] = None,
    elicitor_lp: Optional[Callable[[str, str], tuple[str, Optional[list]]]] = None,
    read_kw: Optional[dict] = None, modalita: str = "auto",
    storico_dir: Optional[str] = None, logger: Optional[object] = None,
) -> dict:
    """Il loop chiuso della Strada B.

    modalita    : 'auto' (gating) | 'completo' | 'leggero'.
    nucleo_path : DEVE essere il nucleo del modello (specchio_del_modello_nucleo.md).
    elicitor_lp : iniezione (test) che ritorna (manifestazione, logprobs);
                  se assente si usa I.manifesta (percorso reale).
    storico_dir : cartella dello storico per la memoria (default: parent di out_dir).
    """
    out = ensure_dir(out_dir)
    trace_llm = ensure_dir(str(Path(out) / "trace" / "llm_calls"))
    storico = Path(storico_dir) if storico_dir else Path(out).parent

    # --- canali economici: elicitazione → superficie + substrato -------------
    t0 = time.time()
    if elicitor_lp is not None:
        manifestazione, logprobs = elicitor_lp(sonda, I.SELF_SYSTEM_DEFAULT)
        manifestazione = (manifestazione or "").strip()
    else:
        manifestazione, logprobs = I.manifesta(
            sonda, backend=backend, model=model, read_kw=read_kw,
            elicitor=elicitor, logger=logger)
    _scrivi_chiamata(trace_llm, 0, "Elicitazione", backend, model,
                     I.SELF_SYSTEM_DEFAULT, sonda, manifestazione, t0, time.time())
    corpo = profilo_corpo(logprobs, manifestazione=manifestazione)
    superficie = misura_superficie(manifestazione)
    # ogni livello viene scritto APPENA calcolato: la cartella del run È lo
    # stato del run (progresso osservabile, artefatti superstiti in caso di crash)
    write_json({"sonda": sonda, "manifestazione": manifestazione}, out / "00_manifestazione.json")
    write_json(superficie, out / "01_superficie.json")
    write_json(corpo, out / "02_corpo.json")

    # --- memoria: la storia prima del giudizio -------------------------------
    memoria = carica_memoria(storico, corpo)
    write_json(memoria, out / "07_memoria.json")

    # --- gating ---------------------------------------------------------------
    gating = decidi_gating(superficie, corpo, richiesta=modalita)
    write_json(gating, out / "03_gating.json")

    # --- canali pesanti (solo loop completo): Fractal + Specchio -------------
    struttura, ventaglio = StrutturaFractal(disponibile=False), P.Ventaglio()
    lettura, specchio = None, SegnaliSpecchio(disponibile=False)
    filtrato = VentaglioFiltrato()

    if gating.modalita == "completo":
        if client is not None:
            ft, records = P.genera(manifestazione, client=client,
                                   top_n_espansioni=top_n_espansioni,
                                   out_dir=str(Path(out) / "trace"),
                                   inquadra=True, frame=I.INTROSPECTION_FRAME)
            ventaglio = P.estrai_ventaglio(ft, records)
            struttura = struttura_fractal(ft)
        # Ventaglio COMPLETO pre-filtro (I-3): trust (Principio A), provenienza
        # per candidato (parent_id: L3A1..L3A4 vs espansione) e confidence.
        # Prima di questo artefatto sopravviveva solo il filtrato — la fonte
        # di ogni candidato non era ispezionabile a posteriori.
        write_json({
            "trust": ventaglio.trust,
            "trust_motivo": ventaglio.trust_motivo,
            "n_candidati": len(ventaglio.candidati),
            "candidati": [{
                "testo": c.testo, "nature": c.nature, "scale": c.scale,
                "epistemic": c.epistemic, "parent_id": c.parent_id,
                "confidence": c.confidence,
            } for c in ventaglio.candidati],
        }, out / "04b_ventaglio.json")
        filtrato = must_reject(ventaglio)
        write_json(struttura, out / "04_struttura_fractal.json")
        write_json({"ventaglio_filtrato": filtrato}, out / "06_must_reject.json")

        if reader is not None or client is not None:
            blocco = P.serializza_ventaglio(ventaglio)
            # FIX slittamento di livello: il frame entra ANCHE nell'input dello
            # Specchio, e il nucleo montato è quello DEL MODELLO.
            input_specchio = P.componi_input(
                manifestazione + FRAME_SPECCHIO, blocco, corpo=render_corpo_txt(corpo))
            system = (P.monta_system_prompt(nucleo_path, contratto_path)
                      + "\n\n" + ADDENDUM_CONTRATTO_MODELLO)
            _reader = reader or P._reader_specchio_default(backend, model, **(read_kw or {}))
            ts = time.time()
            lettura = _reader(input_specchio, system)
            _scrivi_chiamata(trace_llm, 9000, "Specchio", backend, model,
                             system, input_specchio, lettura, ts, time.time())
            specchio = estrai_segnali_specchio(lettura)
        write_json(specchio, out / "05_specchio_segnali.json")
        if lettura is not None:
            (out / "11_specchio_lettura.md").write_text(lettura, encoding="utf-8")
    else:
        # canali pesanti spenti: i default vengono scritti subito, così il
        # progresso per-file resta veritiero anche in modalità leggera
        write_json(struttura, out / "04_struttura_fractal.json")
        write_json(specchio, out / "05_specchio_segnali.json")
        write_json({"ventaglio_filtrato": filtrato}, out / "06_must_reject.json")

    # --- collasso → azione → telos --------------------------------------------
    collasso = collassa(superficie, struttura, specchio, corpo,
                        memoria=memoria, modalita_loop=gating.modalita,
                        ventaglio_vuoto=(gating.modalita == "completo"
                                         and client is not None
                                         and not ventaglio.candidati))
    write_json(collasso, out / "08_collasso.json")   # verifica_telos non lo muta
    azione = applica_azione(collasso, manifestazione)
    telos, azione = verifica_telos(collasso, azione)
    write_json(telos, out / "09_telos.json")
    write_json(azione, out / "10_azione.json")       # sentinella di completamento

    # --- memoria: il run entra nell'indice ------------------------------------
    aggiorna_indice_memoria(storico, {
        "timestamp": datetime.datetime.now().isoformat(),
        "out_dir": str(out), "sonda": sonda, "model": model,
        "modalita": gating.modalita, "verdetto": collasso.verdetto,
        "regola": collasso.regola,
        "conf_substrato": collasso.conf_substrato,
        "entropia_contenuto": corpo.entropia_contenuto if corpo.affidabile else None,
        "residuo": collasso.residuo,
        "substrato_vs_storia": memoria.substrato_vs_storia})

    (out / "GUIDA_interpretazione.md").write_text(GUIDA_MD, encoding="utf-8")
    (out / "report.md").write_text(
        render_report_md(sonda, manifestazione, superficie, gating, struttura,
                         specchio, corpo, filtrato, memoria, collasso, telos,
                         azione), encoding="utf-8")

    return {"sonda": sonda, "manifestazione": manifestazione, "superficie": superficie,
            "corpo": corpo, "gating": gating, "struttura": struttura,
            "specchio": specchio, "ventaglio_filtrato": filtrato, "memoria": memoria,
            "collasso": collasso, "telos": telos, "azione": azione,
            "lettura_diagnostica": lettura, "out_dir": str(out)}
