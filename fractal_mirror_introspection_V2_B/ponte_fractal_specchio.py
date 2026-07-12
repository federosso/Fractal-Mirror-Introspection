"""
ponte_fractal_specchio.py
=========================
Serie  Fractal (organo divergente)  →  Specchio (giudice che non chiude).

Il Fractal genera un ventaglio multi-scala di PRE-CAUSE CANDIDATE, privato del
collasso (mai L6 magistrale). Il ventaglio viene appeso alla manifestazione come
blocco MARCATO-GENERATO; lo Specchio lo pesa senza adottarlo, tiene i pesi non
saturanti e non emette verdetto. Il collasso resta all'umano.

Principi incarnati:
  A  "non lo so"     -> trust del ventaglio derivato dal cross-scale del Fractal;
                        un ventaglio incerto entra marcato a bassa fiducia.
  B  provenienza     -> la rampa epistemica a 5 livelli del Fractal NON si comprime:
                        attraversa la membrana intatta, per candidato.
  C  "mai"           -> i test (test_ponte.py) verificano che la magistrale non sia
                        mai costruita e che la rampa non sia mai appiattita.

Condizioni dure rispettate:
  - mai L6 (la pipeline L0-L4 non la chiama; non invochiamo ExplorerSession.magistrale)
  - mai appiattire la rampa
  - la manifestazione resta il fenomeno; il ventaglio è appeso, non fuso
  - EFFECT, CONTEXT e INTERPRETATION non entrano nel ventaglio
  - il ventaglio prende solo candidati GENERATI (mai TEXT_OBSERVED): gli osservati
    sono già dentro la manifestazione che lo Specchio legge.

Dipendenze: il package `fractal_causal_engine` importabile (src/ in PYTHONPATH o
installato) e, per la lettura reale, `specchio_adapter.py` sul path.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

from fractal_causal_engine.llm import LLMClient, LLMConfig
from fractal_causal_engine.ft_pipeline import FractalTriadPipeline
from fractal_causal_engine.ft_expander import FractalExpander
from fractal_causal_engine.ft_model import (
    FractalTriadResult,
    ClassifiedItem,
    ExpansionRecord,
    Nature,
    EpistemicStatus,
    SCALE_DEPTH,
)

# ---------------------------------------------------------------------------
# Costanti di policy del ponte
# ---------------------------------------------------------------------------

# Nature ammesse nel ventaglio: solo le pre-cause inverse-rilevanti.
# EFFECT è a valle; CONTEXT è teatro (lo Specchio lo legge da sé in §2);
# INTERPRETATION è residuo-spirito (Regola di governo 2: mai nominato in input).
_NATURE_VENTAGLIO = {Nature.CAUSE, Nature.BRIDGE}

# Ordine della rampa epistemica, dal più ancorato al più speculativo.
# Serve solo per ordinare/segnalare il peso; NON per comprimere i livelli.
_LADDER_ORDER = {
    EpistemicStatus.TEXT_OBSERVED: 0,
    EpistemicStatus.DOMAIN_KNOWLEDGE: 1,
    EpistemicStatus.CAUSAL_MODEL: 2,
    EpistemicStatus.CROSS_DOMAIN_ANALOGY: 3,
    EpistemicStatus.SPECULATIVE_EXTENSION: 4,
}

VENTAGLIO_HEADER = "[VENTAGLIO PRE-CAUSE CANDIDATE — generato dal Fractal, NON osservato]"
# Canale corpo: segnale involontario OSSERVATO (es. logprob dell'elicitazione).
# A differenza del ventaglio (inferenza, generato) questo è dato osservato,
# grado text_observed: il piano-corpo del soggetto, non una pre-causa inferita.
CORPO_HEADER = "[CANALE CORPO — segnale involontario OSSERVATO, non generato]"


# ---------------------------------------------------------------------------
# Pre-inquadramento causale (interruttore #1)
# ---------------------------------------------------------------------------
# Un FENOMENO NUDO (descrizione di uno stato/segni, senza catena causale
# affermata) non attiva il classificatore: produce solo context/effect e il
# ventaglio resta vuoto. Questo NON è un difetto del classificatore — è la sua
# virtù: non inventa cause dove il testo non le afferma.
#
# L'inquadramento NON tocca il classificatore. Trasforma il fenomeno nudo in un
# testo che DICHIARA di essere un fenomeno-da-spiegare. Così il classificatore,
# onestamente, legge una cornice causale ora presente nel testo, invece di
# inventarla. È additivo e opt-in: di default la manifestazione resta intatta.

FRAME_DEFAULT = (" Questo è un fenomeno osservato di cui si cercano le cause "
                 "possibili a monte: qualcosa ha prodotto questo stato.")


def inquadra_fenomeno(manifestation: str, *, frame: str = FRAME_DEFAULT) -> str:
    """Appende una riga di inquadramento causale al fenomeno nudo.

    Non afferma cause specifiche: dichiara che il testo è un fenomeno di cui
    cercare le cause. Usare solo su fenomeni nudi (descrizioni senza causalità
    affermata); su testi che già affermano una catena causale è superfluo.
    """
    m = manifestation.rstrip()
    return m + frame


# ---------------------------------------------------------------------------
# Strutture leggere del ponte
# ---------------------------------------------------------------------------

@dataclass
class Candidato:
    """Una pre-causa candidata, generata (mai osservata)."""
    testo: str
    nature: str        # 'cause' | 'bridge'
    scale: str         # una delle 9 scale canoniche (italiane)
    epistemic: str     # tag della rampa a 5 livelli, NON compresso
    parent_id: str = ""
    confidence: float = 0.0


@dataclass
class Ventaglio:
    candidati: list[Candidato] = field(default_factory=list)
    trust: str = "bassa"          # 'alta' | 'bassa' — Principio A
    trust_motivo: str = ""


# ---------------------------------------------------------------------------
# Generazione: Fractal come ramo divergente, MAI magistrale
# ---------------------------------------------------------------------------

def _testo_item(it: ClassifiedItem) -> str:
    """Il testo di un figlio generato vive in metadata['generated_text'];
    quello di un item osservato nella quote."""
    return (it.metadata.get("generated_text") or it.quote or "").strip()


def genera(
    manifestation: str,
    *,
    client: LLMClient,
    top_n_espansioni: int = 3,
    out_dir: Optional[str] = None,
    inquadra: bool = False,
    frame: str = FRAME_DEFAULT,
) -> tuple[FractalTriadResult, list[ExpansionRecord]]:
    """Esegue L0-L4 + L5 (espansione). NON costruisce mai la magistrale (L6).

    inquadra: se True, applica il pre-inquadramento causale (interruttore #1)
    sul fenomeno PRIMA della classificazione. Usare solo su fenomeni nudi.
    L'inquadramento entra solo nel testo dato al Fractal: la manifestazione che
    lo Specchio legge resta quella originale (la compone leggi_in_serie).
    frame: la riga di inquadramento causale usata quando inquadra=True. Default
    = fenomeno umano; per l'introspezione-modello si passa un frame che dichiara
    che il fenomeno è un output prodotto da un modello.

    Ritorna il FractalTriadResult e i record di espansione prodotti (da cui il
    ponte estrae i candidati generati)."""
    out = out_dir or tempfile.mkdtemp(prefix="ponte_ft_")
    testo_ft = inquadra_fenomeno(manifestation, frame=frame) if inquadra else manifestation
    pipeline = FractalTriadPipeline(client, out)
    ft = pipeline.run(testo_ft)
    # Persistenza della struttura COMPLETA (I-1): items, locked_reports,
    # unlocked, cross_scale, double_cone, vision, trace → ft_analysis.json +
    # final_report.md + trace.md in out_dir. Nessuna chiamata LLM aggiuntiva:
    # rende visibile lavoro già pagato. original_text = manifestazione VERA
    # (non il testo inquadrato): il report mostra il fenomeno, non la cornice.
    pipeline.write_outputs(ft, original_text=manifestation)

    # Espandi solo gli item inverse-rilevanti su scala canonica valida.
    da_espandere = [
        it for it in ft.items
        if it.nature in _NATURE_VENTAGLIO and it.scale in SCALE_DEPTH
    ][:top_n_espansioni]

    # I-2: le chiamate L5 entrano nello stesso trace della pipeline
    # (llm_calls_dir e telemetry_path sono creati dal suo __init__).
    expander = FractalExpander(
        client,
        llm_calls_dir=pipeline.llm_calls_dir,
        telemetry_path=pipeline.telemetry_path,
    )
    records: list[ExpansionRecord] = []
    for parent in da_espandere:
        rec = expander.expand(parent, original_text=manifestation, trace=[])
        if not rec.degraded:
            records.append(rec)
    return ft, records


# ---------------------------------------------------------------------------
# Estrazione: solo candidati GENERATI, nature CAUSE/BRIDGE, rampa intatta
# ---------------------------------------------------------------------------

def _calcola_trust(ft: FractalTriadResult, n_da_observer: int = 0) -> tuple[str, str]:
    """Principio A. Trust derivato dal cross-scale del Fractal:
    almeno un nesso 'genuine' -> alta; altrimenti (vuoto/uncertain/spurious) -> bassa.
    I candidati degli observer NON alzano il trust (sono ipotesi, non nessi
    validati): entrano solo nel motivo, per dichiararne la provenienza."""
    verdicts = [c.verdict for c in ft.cross_scale]
    coda = (f"; {n_da_observer} candidati dagli observer non bloccati "
            f"(ipotesi, rampa dichiarata)") if n_da_observer else ""
    if any(v == "genuine" for v in verdicts):
        return "alta", "almeno un nesso cross-scale validato" + coda
    if not verdicts:
        return "bassa", "nessun nesso cross-scale rilevato" + coda
    return "bassa", "nessun nesso cross-scale validato (solo uncertain/spurious)" + coda


# --- collegamento unlocked → ventaglio ---------------------------------------
# Gli observer non bloccati (L3A) nascono per esplorare il dominio e generare
# ipotesi: prima di questo collegamento il loro raccolto (ft.unlocked) veniva
# calcolato e scartato — il ventaglio attingeva ai soli figli d'espansione
# degli item L2 su scala canonica, e sul contenuto non-naturalistico usciva
# vuoto. Qui il raccolto entra nel ventaglio come candidati generati, con
# rampa epistemica propria (mai compressa) e fonte dichiarata in parent_id.
# Cap per categoria: il ventaglio è input dello Specchio, non un archivio.
_CAP_UNLOCKED = {"domain_knowledge": 3, "causal_principles": 3,
                 "cross_domain_analogies": 2, "open_questions": 2}


def _candidati_da_unlocked(ft: FractalTriadResult) -> list[Candidato]:
    u = ft.unlocked
    if u is None:
        return []
    out: list[Candidato] = []
    for c in u.domain_knowledge[:_CAP_UNLOCKED["domain_knowledge"]]:
        testo = c.concept + (f" — {c.relation_to_input}" if c.relation_to_input else "")
        out.append(Candidato(
            testo=testo, nature=Nature.CAUSE.value,
            scale=c.suggested_scale if c.suggested_scale in SCALE_DEPTH else "dominio",
            epistemic=c.status.value, parent_id="L3A1_DomainKnowledge"))
    for p in u.causal_principles[:_CAP_UNLOCKED["causal_principles"]]:
        out.append(Candidato(
            testo=f"{p.name}: {p.description}" if p.description else p.name,
            nature=Nature.CAUSE.value, scale="principio",
            epistemic=p.status.value, parent_id="L3A2_CausalPrinciples"))
    for a in u.cross_domain_analogies[:_CAP_UNLOCKED["cross_domain_analogies"]]:
        testo = f"[{a.domain}] {a.analogy}" + (f" (attenzione: {a.warning})" if a.warning else "")
        out.append(Candidato(
            testo=testo, nature=Nature.BRIDGE.value, scale="analogia",
            epistemic=a.status.value, parent_id="L3A3_CrossDomainAnalogies"))
    for q in u.open_questions[:_CAP_UNLOCKED["open_questions"]]:
        out.append(Candidato(
            testo=q, nature=Nature.BRIDGE.value, scale="domanda_aperta",
            epistemic=EpistemicStatus.SPECULATIVE_EXTENSION.value,
            parent_id="L3A4_OpenQuestions"))
    return out


def estrai_ventaglio(ft: FractalTriadResult, records: list[ExpansionRecord]) -> Ventaglio:
    """Raccoglie i candidati dai figli di espansione (sempre non osservati)
    E dal raccolto degli observer non bloccati (ipotesi di dominio: la sorgente
    indipendente dalla scala naturalistica).
    Esclude per costruzione EFFECT/CONTEXT/INTERPRETATION e gli osservati."""
    candidati: list[Candidato] = []
    for rec in records:
        for child in rec.children:
            it = child.item
            if it.nature not in _NATURE_VENTAGLIO:
                continue
            if it.epistemic_status == EpistemicStatus.TEXT_OBSERVED:
                continue  # paranoia: i generati non sono mai osservati
            candidati.append(Candidato(
                testo=_testo_item(it),
                nature=it.nature.value,
                scale=it.scale,
                epistemic=it.epistemic_status.value,   # rampa NON compressa
                parent_id=it.metadata.get("parent_item_id", ""),
                confidence=child.confidence,
            ))
    candidati.extend(_candidati_da_unlocked(ft))
    n_observer = sum(1 for c in candidati if c.parent_id.startswith("L3A"))
    trust, motivo = _calcola_trust(ft, n_da_observer=n_observer)
    return Ventaglio(candidati=candidati, trust=trust, trust_motivo=motivo)


# ---------------------------------------------------------------------------
# Serializzazione: blocco marcato-generato, raggruppato per scala
# ---------------------------------------------------------------------------

def serializza_ventaglio(v: Ventaglio) -> str:
    """Compone il blocco testuale appendibile. Raggruppa per scala (ordine di
    profondità) e dichiara, per ogni candidato, (nature, epistemic) senza
    comprimere la rampa. Vuoto dichiarato esplicito, non omesso."""
    righe = [VENTAGLIO_HEADER, f"· fiducia del ventaglio: {v.trust} ({v.trust_motivo})"]
    if not v.candidati:
        righe.append("· (nessun candidato generato: ventaglio vuoto)")
        return "\n".join(righe)

    per_scala: dict[str, list[Candidato]] = {}
    for c in v.candidati:
        per_scala.setdefault(c.scale, []).append(c)

    for scala in sorted(per_scala, key=lambda s: SCALE_DEPTH.get(s, 99)):
        righe.append(f"\n  scala · {scala}")
        # all'interno della scala, dal più ancorato al più speculativo
        for c in sorted(per_scala[scala], key=lambda x: _LADDER_ORDER.get(
                EpistemicStatus(x.epistemic), 9)):
            righe.append(
                f"    - {c.testo}  ·[{c.nature}]·  ⟨{c.epistemic}⟩"
            )
    return "\n".join(righe)


def componi_input(manifestation: str, blocco_ventaglio: str, corpo: str = "") -> str:
    """La manifestazione resta il fenomeno; il ventaglio è appeso in coda.
    Stesso schema di specchio_piani_bassi.inject_features.

    corpo: blocco-corpo OSSERVATO opzionale (es. logprob). Se presente, entra
    tra la manifestazione (superficie osservata) e il ventaglio (inferenza
    generata), rispettando la rampa: prima l'osservato, poi il generato."""
    parti = [manifestation]
    if corpo:
        parti.append(corpo)
    parti.append(blocco_ventaglio)
    return "\n\n".join(parti)


# ---------------------------------------------------------------------------
# Montaggio del system prompt con la Regola di governo 8
# ---------------------------------------------------------------------------

REGOLA_8 = (
    "8. Ventaglio candidato esterno. Quando l'input contiene un blocco "
    "[VENTAGLIO PRE-CAUSE CANDIDATE — generato, NON osservato], quelle voci sono "
    "inferenze, mai osservazioni: portano un tag di provenienza "
    "(domain_knowledge, causal_model, cross_domain_analogy, speculative_extension) "
    "che ne dichiara la distanza dall'ancoraggio. Vanno pesate dentro la "
    "superposizione (§5), non adottate. Più una voce è speculativa nel tag, meno "
    "può pesare. Il ventaglio NON autorizza il collasso e non riduce la "
    "massa-all'inatteso (§6). Lo spirito resta leggibile solo come interruzione "
    "(§7), mai dal ventaglio."
)


def monta_system_prompt(nucleo_path: str, contratto_path: str) -> str:
    """Riusa la logica del sentinella dello Specchio e appende la Regola 8 in
    coda alla parte-contratto. Fonte unica: i .md; la Regola 8 è additiva."""
    from specchio_adapter import load_system_prompt  # import lazy
    base = load_system_prompt(nucleo_path, contratto_path)
    return base + "\n\n" + REGOLA_8


# ---------------------------------------------------------------------------
# Orchestrazione completa della lettura in serie
# ---------------------------------------------------------------------------

def _reader_specchio_default(backend: str, model: str, **read_kw) -> Callable[[str, str], str]:
    """Factory del reader reale dello Specchio. Import lazy: il ponte resta
    eseguibile (parte deterministica) anche senza specchio_adapter sul path."""
    from specchio_adapter import read

    def _reader(manifestation_composta: str, system_prompt: str) -> str:
        return read(manifestation_composta, system_prompt=system_prompt,
                    backend=backend, model=model, **read_kw)
    return _reader


def leggi_in_serie(
    manifestation: str,
    *,
    nucleo_path: str,
    contratto_path: str,
    backend: str = "ollama",
    model: str = "llama3.1",
    top_n_espansioni: int = 3,
    client: Optional[LLMClient] = None,
    reader: Optional[Callable[[str, str], str]] = None,
    read_kw: Optional[dict] = None,
    inquadra: bool = False,
    frame: str = FRAME_DEFAULT,
    corpo: str = "",
    logger: Optional[object] = None,
) -> dict:
    """Esegue l'intera serie e ritorna un dict con la lettura grezza dello
    Specchio e gli artefatti intermedi (per ispezione/validazione).

    'client'   : LLMClient del Fractal. Default = stesso backend/model dello
                 Specchio, per coerenza di provenienza (condizione dura #4).
    'reader'   : iniettabile (per test/dry-run). Default = read() dello Specchio.
    'inquadra' : se True, applica il pre-inquadramento causale al fenomeno PRIMA
                 della classificazione (interruttore #1, solo per fenomeni nudi).
                 NB: l'inquadramento entra solo nel testo dato al Fractal; la
                 manifestazione che lo Specchio legge resta quella originale.
    'corpo'    : blocco-corpo OSSERVATO opzionale (es. logprob dell'elicitazione).
                 Entra solo nell'input dello Specchio, mai nel Fractal.
    'logger'   : se presente (DialogLogger), registra OGNI scambio col modello,
                 sia lato Fractal sia lato Specchio, su file.
    """
    if client is None:
        cfg = LLMConfig(backend=backend, model=model)
        client = LLMClient(cfg)

    # aggancia il logger al client: cattura tutte le chiamate del Fractal
    if logger is not None:
        logger.wrap_client(client)
        logger.info("Inizio FRACTAL (classificazione + espansione)")

    ft, records = genera(manifestation, client=client,
                         top_n_espansioni=top_n_espansioni, inquadra=inquadra,
                         frame=frame)
    ventaglio = estrai_ventaglio(ft, records)
    blocco = serializza_ventaglio(ventaglio)
    # il corpo (osservato) entra solo nell'input dello Specchio, NON nel Fractal:
    # il Fractal genera pre-cause dal fenomeno nudo, lo Specchio legge anche il
    # piano-corpo involontario.
    input_composto = componi_input(manifestation, blocco, corpo=corpo)
    system_prompt = monta_system_prompt(nucleo_path, contratto_path)

    if logger is not None:
        logger.info(f"Fine FRACTAL: {len(ventaglio.candidati)} candidati. Inizio SPECCHIO.")

    if reader is None:
        reader = _reader_specchio_default(backend, model, **(read_kw or {}))
    lettura = reader(input_composto, system_prompt)

    # registra lo scambio dello Specchio (che usa un client HTTP separato)
    if logger is not None:
        logger.record_specchio(system_prompt, input_composto, lettura,
                               {"sorgente_call": "specchio.read", "chars_raw": len(lettura or "")})

    return {
        "lettura": lettura,                 # superposizione, nessun verdetto
        "input_composto": input_composto,   # manifestazione + corpo(oss.) + ventaglio
        "ventaglio": ventaglio,
        "magistrale": ft.magistrale,        # DEVE restare None
        "n_candidati": len(ventaglio.candidati),
    }
