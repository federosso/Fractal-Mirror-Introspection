"""Fractal Triad — modello dati canonico V10.14.0.

Riallineamento al paper federosso/fractal-triad:
- un solo asse di scala (9 scale ontologiche, da cosmologico a fondamentale);
- due coordinate per ogni elemento: (nature, scale);
- ZoomCoherencePrinciple: i legami causa->effetto sono validi SOLO same-scale;
  le transizioni cross-scale sono ammesse ma vanno ragionate, mai promosse per
  prossimità testuale.

Questo modulo NON sostituisce schemas.py: ci convive durante la transizione.
schemas.py contiene i tipi legacy (Node, Relation, AnalysisResult); ft_model.py
contiene i tipi nuovi della pipeline FT (ClassifiedItem, SameScaleLink, ...).

Alla fine della transizione, AnalysisResult ospitera' entrambi via campi nuovi
opzionali; i tipi legacy possono essere ignorati senza romperla.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# -----------------------------------------------------------------------------
# Le 9 scale canoniche del paper, dall'alto verso il basso.
# Indici: 0 = cosmologico (shallow / superficie degli effetti),
#         8 = fondamentale (deep / radice delle cause).
# -----------------------------------------------------------------------------
SCALES_CANONICAL: list[str] = [
    "cosmologico",
    "planetario",
    "sociale",
    "organismo",
    "cellulare",
    "molecolare",
    "atomico",
    "subatomico",
    "fondamentale",
]

SCALE_DEPTH: dict[str, int] = {name: i for i, name in enumerate(SCALES_CANONICAL)}


def is_valid_scale(scale: str) -> bool:
    return scale in SCALE_DEPTH


def scale_distance(a: str, b: str) -> int:
    """Distanza assoluta tra due scale sull'asse canonico.

    0 = stessa scala (same-scale).
    >=1 = cross-scale, e va validato esplicitamente dall'Unlocked.
    """
    if a not in SCALE_DEPTH or b not in SCALE_DEPTH:
        return -1
    return abs(SCALE_DEPTH[a] - SCALE_DEPTH[b])


# -----------------------------------------------------------------------------
# Nature: la natura causale di un elemento.
# -----------------------------------------------------------------------------
class Nature(str, Enum):
    CAUSE = "cause"
    EFFECT = "effect"
    CONTEXT = "context"            # fa da sfondo, non e' ne' causa ne' effetto
    BRIDGE = "bridge"              # passaggio intermedio possibile, non valido come causa
    INTERPRETATION = "interpretation"  # lettura simbolica/spirituale, non causale


# -----------------------------------------------------------------------------
# Predicato dei claim probatori: distingue cosa fa una frase nel testo.
# Serve per impedire che una definizione venga promossa come trigger causale.
# -----------------------------------------------------------------------------
class PredicateType(str, Enum):
    DEFINITION = "definition"               # "X e' un termine/etichetta che indica Y"
    PROCESS_DESCRIPTION = "process_description"  # "X stimola/agisce su Y"
    CLAIMED_PROPERTY = "claimed_property"   # "X ha la proprieta' Y" (incluse assenze)
    EVENT = "event"                          # "e' successo X"
    STATE = "state"                          # "X e' in stato Y"
    COMPARISON = "comparison"                # "X a differenza di Y..."
    QUESTION = "question"                    # "Che cos'e' X?" -- quesito, non asserzione
    UNKNOWN = "unknown"


# -----------------------------------------------------------------------------
# Stato epistemico: a quale categoria di certezza appartiene un'affermazione.
# -----------------------------------------------------------------------------
class EpistemicStatus(str, Enum):
    TEXT_OBSERVED = "text_observed"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CAUSAL_MODEL = "causal_model"
    CROSS_DOMAIN_ANALOGY = "cross_domain_analogy"
    SPECULATIVE_EXTENSION = "speculative_extension"


# -----------------------------------------------------------------------------
# L1 -- Item classificato: una proposizione minima estratta dal testo,
# con le due coordinate del paper (nature + scale) e il tipo di predicato.
# -----------------------------------------------------------------------------
@dataclass
class ClassifiedItem:
    id: str
    quote: str                                  # testuale, <= 25 parole
    predicate: PredicateType
    nature: Nature
    scale: str                                  # una delle 9 SCALES_CANONICAL
    rationale: str = ""                         # breve, perche' questa classificazione
    source_input_id: str = ""
    epistemic_status: EpistemicStatus = EpistemicStatus.TEXT_OBSERVED
    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# L2 -- Locked-per-scala: un legame same-scale e una segnalazione di orfani.
# -----------------------------------------------------------------------------
@dataclass
class SameScaleLink:
    id: str
    scale: str
    cause_item_id: str
    effect_item_id: str
    rationale: str
    confidence: float = 0.0


@dataclass
class Orphan:
    """Item che non ha trovato un partner same-scale; candidato per cross-scale."""
    item_id: str
    nature: Nature
    scale: str
    reason: str = ""


@dataclass
class LockedScaleReport:
    scale: str
    same_scale_links: list[SameScaleLink] = field(default_factory=list)
    orphans: list[Orphan] = field(default_factory=list)
    items_seen: list[str] = field(default_factory=list)
    summary: str = ""


# -----------------------------------------------------------------------------
# L3.A -- Unlocked Domain Explorer: conoscenza di dominio, evocata.
# Quattro micro-output indipendenti, ciascuno marcato epistemic.
# -----------------------------------------------------------------------------
@dataclass
class DomainConcept:
    concept: str
    relation_to_input: str
    status: EpistemicStatus = EpistemicStatus.DOMAIN_KNOWLEDGE
    suggested_scale: str = ""                   # opzionale, quale scala evoca
    not_in_input: bool = True


@dataclass
class CausalPrinciple:
    name: str
    description: str
    status: EpistemicStatus = EpistemicStatus.CAUSAL_MODEL


@dataclass
class CrossDomainAnalogy:
    domain: str
    analogy: str
    warning: str = ""
    status: EpistemicStatus = EpistemicStatus.CROSS_DOMAIN_ANALOGY


@dataclass
class UnlockedReport:
    domain: str
    domain_knowledge: list[DomainConcept] = field(default_factory=list)
    causal_principles: list[CausalPrinciple] = field(default_factory=list)
    cross_domain_analogies: list[CrossDomainAnalogy] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    known_uncertainties: list[str] = field(default_factory=list)
    degraded: bool = False              # True se una o piu' micro-chiamate sono fallite
    degraded_parts: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# L3.B -- Cross-Scale Validator: ipotesi cross-scale ragionate dall'LLM.
# -----------------------------------------------------------------------------
@dataclass
class CrossScaleHypothesis:
    id: str
    cause_item_id: str
    effect_item_id: str
    cause_scale: str
    effect_scale: str
    verdict: str                                # genuine | spurious | uncertain
    reasoning: str
    confidence: float = 0.0


# -----------------------------------------------------------------------------
# L4 -- Visione globale dell'Orchestratore.
# -----------------------------------------------------------------------------
@dataclass
class DoubleCone:
    cone_of_causes: dict[str, list[str]] = field(default_factory=dict)   # scale -> [item_id]
    cone_of_effects: dict[str, list[str]] = field(default_factory=dict)  # scale -> [item_id]


@dataclass
class GlobalVision:
    core_image: str = ""
    human_summary: str = ""
    epistemic_warning: str = ""
    dominant_domain: str = ""
    primary_lenses: list[str] = field(default_factory=list)
    blocked_lenses: list[str] = field(default_factory=list)


@dataclass
class FractalTriadResult:
    """Risultato della pipeline V10.14.0. Vive accanto ad AnalysisResult."""
    items: list[ClassifiedItem] = field(default_factory=list)
    locked_reports: list[LockedScaleReport] = field(default_factory=list)
    unlocked: UnlockedReport | None = None
    cross_scale: list[CrossScaleHypothesis] = field(default_factory=list)
    double_cone: DoubleCone = field(default_factory=DoubleCone)
    vision: GlobalVision = field(default_factory=GlobalVision)
    trace: list[str] = field(default_factory=list)
    # V10.15 -- arricchimenti opzionali. Non rompono nulla: i moduli V14 li ignorano.
    expansions: list["ExpansionRecord"] = field(default_factory=list)
    bridges: list["BridgeRecord"] = field(default_factory=list)
    magistrale: "MagistraleReport | None" = None


# =============================================================================
# V10.15.0 -- arricchimenti additivi: espansione frattale, bridge esplicito,
# relazione magistrale. Tutti i nuovi dataclass riusano gli stessi vincoli
# di V14: nature, scale, predicate, epistemic_status. Niente bypass.
# =============================================================================


class ExpansionDirection(str, Enum):
    """Le 4 direzioni canoniche del 'frazionamento' di un item.

    Ispirate al Fractal Triad: stessa scala (causa orizzontale), scale-up
    (propagazione fenomenologica emergente), scale-down (meccanismo che
    sostiene), bridge (passaggio di coerenza cross-scale ragionato).
    """
    SAME_SCALE_CAUSE = "same_scale_cause"
    SCALE_UP_PROPAGATION = "scale_up_propagation"
    SCALE_DOWN_MECHANISM = "scale_down_mechanism"
    COHERENCE_BRIDGE = "coherence_bridge"


@dataclass
class ExpansionChild:
    """Un singolo item generato dall'espansione di un item esistente.

    L'item generato vive con i suoi predicati e scale, e mantiene un puntatore
    al padre. Non e' un 'fatto osservato dal testo' (TEXT_OBSERVED): e' al
    massimo DOMAIN_KNOWLEDGE o CAUSAL_MODEL, e va trattato come tale.
    """
    item: ClassifiedItem
    direction: ExpansionDirection
    relation_to_parent: str = ""           # breve, perche' questo figlio si lega al padre
    confidence: float = 0.0


@dataclass
class ExpansionRecord:
    """Tracciamento di una singola espansione: padre + figli + cross-scale generate.

    Le cross-scale generate sono coppie (figlio, padre) per le 3 direzioni
    cross-scale (scale_up, scale_down, bridge). La same_scale_cause produce
    invece un SameScaleLink puro.
    """
    parent_item_id: str
    direction_set: list[ExpansionDirection] = field(default_factory=list)
    children: list[ExpansionChild] = field(default_factory=list)
    same_scale_links_added: list[SameScaleLink] = field(default_factory=list)
    cross_scale_added: list[CrossScaleHypothesis] = field(default_factory=list)
    degraded: bool = False
    notes: str = ""


@dataclass
class BridgeRecord:
    """Un bridge esplicito costruito tra due item su scale diverse.

    Il bridge_item e' un ClassifiedItem con nature=BRIDGE e scale=gap_scale.
    Il suo epistemic_status e' CAUSAL_MODEL: e' un meccanismo proposto, non un
    fatto osservato.
    """
    source_item_id: str
    target_item_id: str
    gap_scale: str                         # la scala intermedia mancante (canonica)
    bridge_item: ClassifiedItem
    mechanism_reasoning: str = ""
    cross_scale_added: list[CrossScaleHypothesis] = field(default_factory=list)
    degraded: bool = False


@dataclass
class MagistraleCones:
    """Strutturazione dei due coni per la relazione magistrale.

    Le 4 liste del cono cause sono ruoli causali (predisposizione = condizione
    di sfondo, trigger = innesco prossimo, proximate = causa diretta osservata,
    bridge = meccanismo intermedio). Gli effetti hanno una scomposizione
    analoga ma orientata alla propagazione.
    """
    predispositions: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    proximate_causes: list[str] = field(default_factory=list)
    bridge_mechanisms: list[str] = field(default_factory=list)


@dataclass
class MagistraleEffects:
    direct_effects: list[str] = field(default_factory=list)
    downstream_effects: list[str] = field(default_factory=list)
    interpretations: list[str] = field(default_factory=list)
    social_propagations: list[str] = field(default_factory=list)


@dataclass
class MagistraleReport:
    """Sintesi finale dialogica della FractalTriadResult.

    Una sola chiamata LLM, nutrita dai dati gia' validati (items, links,
    cross-scale verdetti, vision). NESSUNA nuova validazione causale qui:
    questo modulo formatta e racconta, non aggiunge prove.
    """
    sintesi_magistrale: str = ""
    cono_cause: MagistraleCones = field(default_factory=MagistraleCones)
    cono_effetti: MagistraleEffects = field(default_factory=MagistraleEffects)
    propagazione_multi_scala: str = ""
    stato_epistemico: str = ""
    verdetto_finale: str = ""
    degraded: bool = False


# =============================================================================
# V10.17.0 -- L7 Director (il Regista).
#
# Un meta-osservatore che guarda l'Attore (la pipeline auto_explore) mentre
# lavora. Non e' un nuovo asse causale: riusa scale canoniche, verdict
# cross-scale ed epistemic_status. E' il motore che ricorre su se stesso.
#
# Il Regista, dopo ogni fase dell'Attore, valuta TRE condizioni. Rompe il
# silenzio (interviene) SOLO quando scattano tutte e tre insieme -- e' cio'
# che distingue una presa di coscienza da un trigger a orologeria:
#   1. divergenza di scala  -> l'Attore accumula ipotesi cross-scale spurious;
#   2. costo del silenzio   -> integrale di deriva del baricentro di scala
#                              oltre una banda di tolleranza;
#   3. irreversibilita'     -> la fase osservata e' l'ultima prima della
#                              chiusura (la magistrale).
# =============================================================================


@dataclass
class DirectorReading:
    """Una singola lettura del Regista, dopo una fase dell'Attore.

    Registra le tre misure e se la fase ha fatto scattare l'intervento.
    E' il 'voltarsi indietro a guardare se stesso mentre lavora'.
    """
    phase: str                              # quale fase dell'Attore e' stata osservata
    scale_divergence: float = 0.0           # frazione di cross-scale 'spurious' [0..1]
    silence_cost: float = 0.0               # integrale di deriva del baricentro di scala
    is_irreversible: bool = False           # la fase osservata precede la chiusura
    intervened: bool = False                # True se le 3 condizioni sono scattate insieme
    note: str = ""                          # spiegazione testuale della lettura


@dataclass
class DirectorIntervention:
    """L'intromissione del Regista.

    V10.17.0: modulava solo i parametri della fase successiva.
    V10.17.1: PIENO CONTROLLO. Il Regista puo' anche governare il FLUSSO
    dell'Attore, decidendo quale fase eseguire dopo.

    Il campo `control` e' il verbo di regia:
      - PROCEED: prosegui con la fase successiva nell'ordine naturale
                 (eventualmente modulata da param_overrides). E' il default,
                 equivale al comportamento V10.17.0.
      - SKIP:    salta la fase successiva e passa a quella dopo ancora.
                 L'Attore non eseguira' quella fase.
      - REPEAT:  ri-esegui la fase appena conclusa, prima di proseguire.
                 Serve quando il Regista giudica il risultato incompleto.
      - GOTO:    salta direttamente alla fase indicata da `goto_phase`,
                 anche all'indietro (es. tornare a 'expand' dopo 'bridge').
      - HALT:    ferma l'Attore. Nessuna fase ulteriore viene eseguita.

    `param_overrides` resta valido e si applica alla fase che verra' eseguita
    (quale che sia, secondo `control`).
    """
    after_phase: str                        # dopo quale fase il Regista e' intervenuto
    target_phase: str                       # quale fase viene influenzata
    control: str = "proceed"                # proceed | skip | repeat | goto | halt
    goto_phase: str = ""                    # destinazione, solo se control=goto
    param_overrides: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""                     # perche' il Regista ha scelto di intromettersi


@dataclass
class DirectorReport:
    """Esito completo dell'osservazione del Regista su una sessione auto_explore.

    readings: una lettura per ogni fase osservata.
    interventions: solo le fasi dove le 3 condizioni hanno fatto scattare
                   una correzione.
    executed_phases: la sequenza REALE di fasi eseguite dall'Attore. Con il
                   Regista a pieno controllo puo' divergere dall'ordine
                   canonico (salti, ripetizioni, halt anticipato).
    halted: True se il Regista ha fermato l'Attore prima della chiusura.
    """
    readings: list[DirectorReading] = field(default_factory=list)
    interventions: list[DirectorIntervention] = field(default_factory=list)
    executed_phases: list[str] = field(default_factory=list)
    halted: bool = False
    silence_band: float = 0.0               # soglia di costo del silenzio usata
    divergence_threshold: float = 0.0       # soglia di divergenza di scala usata
    summary: str = ""


# =============================================================================
# V10.18.0 -- Segmentazione di testi lunghi (libri).
#
# Per un testo che eccede la finestra di contesto del modello, ft_segmenter
# lo spezza in Segment: unita' di lavoro dentro il budget di token, tagliate
# su confini naturali (capitolo -> paragrafo -> frase) con overlap tra l'uno
# e il successivo. Ogni Segment viene poi analizzato dalla pipeline normale.
# Questi dataclass sono puro contenitore: nessuna logica, nessun LLM.
# =============================================================================


@dataclass
class Segment:
    """Un segmento di testo, unita' di lavoro per l'analisi di un libro.

    id            -- identificatore stabile ('seg_0001'); chiave del manifest.
    index         -- posizione 0-based nella sequenza dei segmenti.
    text          -- il testo del segmento, overlap incluso. E' la fonte
                     AUTORITATIVA del contenuto: e' questo che va analizzato.
    char_start    -- offset (incluso) che localizza il segmento nel testo
                     originale. Monotono e contiguo (char_start del segmento
                     k+1 == char_end del k). NB: non e' un taglio byte-esatto
                     del testo -- i blocchi sono ricostruiti dai paragrafi --
                     ma serve a ordinare e orientare, non a ri-tagliare.
    char_end      -- offset (escluso) di fine, stessa semantica di char_start.
    chapter_index -- indice del capitolo di appartenenza; -1 se il testo non
                     aveva confini di capitolo riconoscibili (fallback paragrafo).
    chapter_title -- titolo/etichetta del capitolo, '' se non disponibile.
    overlap_chars -- quanti caratteri, in testa a `text`, sono ripetuti dalla
                     coda del segmento precedente (0 per il primo segmento).
    est_tokens    -- stima dei token di `text` (euristica caratteri/token).
    """
    id: str
    index: int
    text: str
    char_start: int
    char_end: int
    chapter_index: int = -1
    chapter_title: str = ""
    overlap_chars: int = 0
    est_tokens: int = 0


@dataclass
class SegmentationResult:
    """Esito completo della segmentazione di un testo.

    segments       -- la lista ordinata dei Segment.
    total_chars    -- lunghezza del testo originale.
    num_ctx        -- finestra di contesto usata per calcolare il budget.
    token_budget   -- token massimi di contenuto per segmento (derivato).
    used_chapters  -- True se la segmentazione ha riconosciuto confini di
                      capitolo; False se ha usato il fallback a paragrafo.
    notes          -- avvisi non bloccanti (es. un paragrafo troppo lungo
                      spezzato a livello di frase).
    """
    segments: list[Segment] = field(default_factory=list)
    total_chars: int = 0
    num_ctx: int = 0
    token_budget: int = 0
    used_chapters: bool = False
    notes: list[str] = field(default_factory=list)


# =============================================================================
# V10.18.0 -- Manifest del book runner.
#
# Il manifest e' il LEDGER di stato dell'analisi di un libro: traccia, per
# ogni segmento, se e' da fare / in corso / fatto / fallito. E' cio' che
# rende il job resumabile: su crash, si riparte leggendo il manifest invece
# di rifare tutto. Si scrive in modo ATOMICO a ogni checkpoint.
# =============================================================================


@dataclass
class SegmentRecord:
    """Stato di un segmento nel manifest.

    status -- vocabolario del ledger:
      'pending' : non ancora elaborato;
      'running' : in corso (scritto all'inizio, come marcatore di crash);
      'done'    : pipeline completata, output su disco;
      'failed'  : fallito dopo tutti i retry.
    dead_letter -- True se 'failed' in via definitiva: messo da parte, non
                   piu' ritentato nei resume successivi.
    attempts    -- quante volte e' stato tentato (per il backoff e la
                   diagnostica).
    """
    id: str
    index: int
    chapter_index: int = -1
    chapter_title: str = ""
    status: str = "pending"
    attempts: int = 0
    dead_letter: bool = False
    out_dir: str = ""                       # sotto-cartella di output del segmento
    error: str = ""                         # ultimo errore, se status='failed'
    elapsed_seconds: float = 0.0
    est_tokens: int = 0


@dataclass
class BookManifest:
    """Il registro completo di un job di analisi-libro.

    E' serializzato in book_manifest.json accanto agli output. Un resume
    ricarica questo file, salta i segmenti 'done' e riprende dagli altri.
    """
    book_id: str
    source_input_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    num_ctx: int = 0
    overlap_ratio: float = 0.0
    token_budget: int = 0
    used_chapters: bool = False
    per_segment_depth: str = "base"         # 'base' (L0->L4) | 'full'
    halt_on_failure: bool = False
    max_retries: int = 3
    segments: list[SegmentRecord] = field(default_factory=list)
    segmenter_notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Conteggio dei segmenti per stato. Comodo per log e riepiloghi."""
        out = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for s in self.segments:
            out[s.status] = out.get(s.status, 0) + 1
        return out


# =============================================================================
# V10.18.2 -- Riduzione gerarchica (book reducer).
#
# Dopo che il book runner ha analizzato ogni segmento, il reducer fonde i
# risultati in due stadi: segmenti -> capitoli -> opera. Questi dataclass
# descrivono l'esito della riduzione.
# =============================================================================


@dataclass
class ChapterSynthesis:
    """Sintesi di un capitolo: gli ft dei suoi segmenti fusi in uno solo,
    piu' la magistrale di capitolo."""
    chapter_index: int
    chapter_title: str = ""
    segment_ids: list[str] = field(default_factory=list)
    merged_items: int = 0                   # quanti item dopo la deduplica
    merged_cross_scale: int = 0
    magistrale_text: str = ""               # sintesi magistrale del capitolo
    degraded: bool = False


@dataclass
class BookReduction:
    """Esito completo della riduzione gerarchica di un libro."""
    book_id: str = ""
    chapters: list[ChapterSynthesis] = field(default_factory=list)
    global_magistrale_text: str = ""        # sintesi dell'intera opera
    total_segments_used: int = 0
    total_items_merged: int = 0
    notes: list[str] = field(default_factory=list)


# =============================================================================
# V10.19.0 -- Lettura tematica (ft_thematic).
#
# Un fork concettuale del motore causale: invece di cercare catene di causa
# ed effetto, genera OSSERVAZIONI da quattro lenti (angolazioni) diverse.
# Adatto a testi non argomentativi -- diari, dialoghi, testi simbolici --
# dove la griglia causale forzerebbe una struttura assente.
#
# IMPORTANTE -- onesta' epistemica: una lente RILEVA e MAPPA come un testo
# costruisce il suo discorso (anche spirituale, religioso, metafisico). NON
# si pronuncia sulla VERITA' di cio' che il testo afferma. Offre angolazioni
# di lettura; cio' che il testo descrive resta cio' che e', indipendentemente
# dalla lente. La lente legge il testo, non il referente del testo.
# =============================================================================


# Le quattro lenti canoniche della lettura tematica.
THEMATIC_LENSES: list[str] = [
    "simbolica",        # immagini, metafore, archetipi (incl. simboli del sacro)
    "strutturale",      # organizzazione del discorso, come costruisce autorita'
    "relazionale",      # voci, interlocutori, posizioni reciproche
    "esperienziale",    # vissuto interiore, stati, esperienza riferita (incl. metafisica/energetica)
]


@dataclass
class Observation:
    """Una singola osservazione prodotta da una lente su un segmento di testo.

    NON e' un'affermazione di verita': e' cio' che la lente NOTA nel testo.
    """
    lens: str                               # quale lente (THEMATIC_LENSES)
    focus: str                              # il punto del testo osservato (breve)
    note: str                               # l'osservazione vera e propria
    evidence: str = ""                      # citazione/riferimento dal testo
    salience: float = 0.5                   # quanto e' centrale [0..1]


@dataclass
class ThematicMotif:
    """Un motivo ricorrente: un tema/immagine che ritorna nel testo.

    E' l'equivalente tematico dell'ipotesi cross-scale -- ma non collega
    cause, collega RICORRENZE: la stessa immagine che riappare, magari
    trasformata, in punti diversi del testo.
    """
    name: str                               # nome breve del motivo
    lens: str                               # lente che lo ha rilevato
    occurrences: list[str] = field(default_factory=list)  # focus/punti dove ricorre
    transformation: str = ""                # come il motivo cambia attraverso il testo


@dataclass
class ThematicReading:
    """Esito completo della lettura tematica di un testo (o segmento).

    observations: tutte le osservazioni, da tutte le lenti.
    motifs: i motivi ricorrenti individuati.
    synthesis: sintesi PLURALE -- non un verdetto, ma un riepilogo delle
               angolazioni. Dichiaratamente non conclusivo.
    """
    observations: list[Observation] = field(default_factory=list)
    motifs: list[ThematicMotif] = field(default_factory=list)
    synthesis: str = ""
    notes: list[str] = field(default_factory=list)

    def by_lens(self, lens: str) -> list[Observation]:
        """Le osservazioni di una data lente."""
        return [o for o in self.observations if o.lens == lens]


# =============================================================================
# V10.19.2 -- Riduzione tematica di un libro.
#
# L'equivalente di BookReduction, ma per la lettura tematica: invece di
# fondere item causali, raccoglie le osservazioni delle quattro lenti da
# tutti i segmenti e le sintetizza in una lettura tematica dell'opera.
# =============================================================================


@dataclass
class ThematicBookReduction:
    """Esito della riduzione tematica di un intero libro.

    per_lens_synthesis: una sintesi per ciascuna delle quattro lenti,
        costruita su tutte le osservazioni di quella lente in tutti i
        segmenti -- "cosa vede la lente simbolica nell'intero libro".
    opera_synthesis: la sintesi plurale finale dell'opera.
    """
    book_id: str = ""
    total_segments_used: int = 0
    total_observations: int = 0
    per_lens_synthesis: dict = field(default_factory=dict)   # lens -> testo
    opera_synthesis: str = ""
    notes: list[str] = field(default_factory=list)
