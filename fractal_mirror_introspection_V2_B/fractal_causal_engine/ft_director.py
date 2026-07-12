"""L7 -- Director (V10.17.0). Il Regista.

Un meta-osservatore che guarda l'Attore (la pipeline auto_explore) mentre
lavora, e decide se "rompere il silenzio" per correggere la traiettoria.

ARCHITETTURA ATTORE / REGISTA
-----------------------------
- L'ATTORE e' la pipeline esistente: auto_explore esegue, fase dopo fase,
  expand -> bridge -> revalidate -> magistrale. Segue la corrente. E' fuso
  con la dinamica stimolo->risposta: dato un input fa sempre le stesse cose.
- Il REGISTA e' questo modulo. Dopo OGNI fase dell'Attore osserva lo stato
  (il FractalTriadResult) e produce una DirectorReading.

LA SOGLIA: cosa distingue una presa di coscienza da un trigger a orologeria
---------------------------------------------------------------------------
Un timer dice "sono passate N fasi, intervieni": esogeno, cieco. Il Regista
interviene solo quando rileva una DISCREPANZA tra come l'Attore sta operando
e come dovrebbe operare. Tre condizioni, e servono TUTTE E TRE insieme:

  1. DIVERGENZA DI SCALA
     L'Attore sta accumulando ipotesi cross-scale con verdict 'spurious':
     sta tessendo ponti tra scale che non reggono. Misura = frazione di
     cross_scale spurious sul totale. Riusa il verdict di L3.B: nessuna
     euristica nuova.

  2. COSTO DEL SILENZIO
     Non e' una soglia istantanea ma un INTEGRALE: di quanto il baricentro
     di scala degli item generati si e' allontanato, fase dopo fase, dal
     baricentro degli item text_observed (la radice nel testo). Piccole
     derive sono tollerate; l'integrale rompe il silenzio solo oltre banda.
     Questo impedisce l'intervento a singhiozzo.

  3. IRREVERSIBILITA'
     Il Regista interviene solo se la fase osservata precede la CHIUSURA
     (la magistrale: il momento in cui l'Attore "spegne il processo").
     Non si intromette nel mezzo di un pensiero in formazione: aspetta lo
     spazio fra lo stimolo e l'azione.

QUANDO LE TRE SCATTANO INSIEME, il Regista emette una DirectorIntervention:
una correzione dei PARAMETRI della fase successiva. NON riscrive il ft, NON
bypassa i guard di V14 (predicate, nature, scale canonica, zoom coherence).
Modula solo lo "zoom di coinvolgimento": e' lo zoom-in / zoom-out.

Il modulo e' STATELESS rispetto al ft come FractalExpander: si costruisce,
osserva, e ritorna un DirectorReport. La parte deterministica (le 3 misure)
e' puro Python -- e' il circuito ricorsivo. La sola chiamata LLM e'
opzionale: serve solo a RACCONTARE l'auto-osservazione, non a deciderla.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .ft_model import (
    SCALE_DEPTH,
    ClassifiedItem,
    DirectorIntervention,
    DirectorReading,
    DirectorReport,
    FractalTriadResult,
)
from .llm import LLMClient, RoleAgent
from .ft_budget import budget


# Le fasi dell'Attore, in ordine. 'magistrale' e' la chiusura.
ACTOR_PHASES: list[str] = ["expand", "bridge", "revalidate", "magistrale"]
CLOSING_PHASE: str = "magistrale"

# I verbi di regia (V10.17.1: pieno controllo del flusso).
CONTROL_PROCEED: str = "proceed"
CONTROL_SKIP: str = "skip"
CONTROL_REPEAT: str = "repeat"
CONTROL_GOTO: str = "goto"
CONTROL_HALT: str = "halt"


# -----------------------------------------------------------------------------
# Parte deterministica: le tre misure. Niente LLM qui.
# -----------------------------------------------------------------------------


def _scale_centroid(items: list[ClassifiedItem]) -> float:
    """Baricentro di scala (profondita' media) di una lista di item.

    Ritorna -1.0 se non ci sono item con scala canonica valida.
    """
    depths = [SCALE_DEPTH[it.scale] for it in items if it.scale in SCALE_DEPTH]
    if not depths:
        return -1.0
    return sum(depths) / len(depths)


def _observed_items(ft: FractalTriadResult) -> list[ClassifiedItem]:
    """Item nati dal testo (text_observed). Sono la radice: la posizione di
    scala da cui l'Attore non dovrebbe allontanarsi senza ragionarci."""
    return [
        it for it in ft.items
        if (it.metadata or {}).get("origin", "text_observed") == "text_observed"
        and it.scale in SCALE_DEPTH
    ]


def _generated_items(ft: FractalTriadResult) -> list[ClassifiedItem]:
    """Item nati da espansione o bridge. Sono il prodotto dell'Attore."""
    return [
        it for it in ft.items
        if (it.metadata or {}).get("origin") in ("expansion", "bridge")
        and it.scale in SCALE_DEPTH
    ]


def measure_scale_divergence(ft: FractalTriadResult) -> float:
    """Condizione 1: frazione di ipotesi cross-scale con verdict 'spurious'.

    [0.0 .. 1.0]. 0.0 = nessun ponte fasullo; alto = l'Attore sta legando
    scale che non reggono. Riusa direttamente il verdict di L3.B.
    """
    if not ft.cross_scale:
        return 0.0
    spurious = sum(1 for h in ft.cross_scale if h.verdict == "spurious")
    return spurious / len(ft.cross_scale)


def measure_silence_cost(ft: FractalTriadResult) -> float:
    """Condizione 2: deriva del baricentro di scala.

    Distanza assoluta tra il baricentro degli item generati e quello degli
    item osservati. 0.0 se non c'e' ancora nulla di generato (l'Attore non
    si e' ancora mosso) o se manca la radice osservata.

    NB: il chiamante ACCUMULA questo valore fase dopo fase per ottenere
    l'integrale di deriva (vedi Director.observe).
    """
    observed = _observed_items(ft)
    generated = _generated_items(ft)
    if not observed or not generated:
        return 0.0
    c_obs = _scale_centroid(observed)
    c_gen = _scale_centroid(generated)
    if c_obs < 0 or c_gen < 0:
        return 0.0
    return abs(c_gen - c_obs)


def is_phase_irreversible(phase: str) -> bool:
    """Condizione 3: osservare dopo 'phase' e' l'ultimo momento utile per
    correggere prima della chiusura.

    L'Attore chiude con 'magistrale'. Il Regista osserva DOPO una fase e
    modula la fase SUCCESSIVA. Il momento irreversibile e' dunque quello in
    cui la fase successiva e' l'ultima modulabile prima della magistrale,
    cioe' 'revalidate'. L'Attore osserva dopo 'bridge' -> la fase successiva
    e' 'revalidate' -> e' l'ultimo spazio fra lo stimolo e l'azione, prima
    dello 'spengo il processo'. Quindi: osservare dopo 'bridge' e'
    irreversibile.

    In generale: phase e' irreversibile se la sua fase successiva e' l'ultima
    fase non-chiusura (quella subito prima di CLOSING_PHASE).
    """
    if phase not in ACTOR_PHASES:
        return False
    idx = ACTOR_PHASES.index(phase)
    closing_idx = ACTOR_PHASES.index(CLOSING_PHASE)
    next_idx = idx + 1
    # la fase successiva esiste, non e' la chiusura, ed e' l'ultima prima di essa
    return next_idx == closing_idx - 1


# -----------------------------------------------------------------------------
# La correzione: come il Regista modula la fase successiva.
# -----------------------------------------------------------------------------


def _next_phase(phase: str) -> str:
    """Fase successiva nell'ordine dell'Attore; '' se phase e' la chiusura."""
    if phase not in ACTOR_PHASES:
        return ""
    idx = ACTOR_PHASES.index(phase)
    return ACTOR_PHASES[idx + 1] if idx + 1 < len(ACTOR_PHASES) else ""


def _build_intervention(
    after_phase: str,
    divergence: float,
    silence_cost: float,
    divergence_threshold: float,
) -> DirectorIntervention | None:
    """Traduce le misure in un atto di regia a PIENO CONTROLLO.

    Il Regista non modula piu' solo i parametri: governa il FLUSSO. La stessa
    misura (divergenza di scala) viene graduata in intensita' di risposta --
    e' lo zoom di coinvolgimento, dal piu' lieve al piu' drastico:

    - divergenza appena sopra soglia  -> PROCEED + param_overrides
        zoom-in lieve: prosegui, ma correggi i parametri della fase dopo.
    - divergenza molto sopra soglia   -> REPEAT
        l'Attore ha prodotto un risultato troppo compromesso: ri-esegui la
        fase appena conclusa con i parametri corretti, prima di andare avanti.
    - divergenza estrema              -> GOTO 'expand'
        i ponti sono cosi' fasulli che non basta rifarli: il Regista riporta
        l'Attore all'espansione, per ri-radicare l'esplorazione nel testo
        prima di ritentare i bridge.
    - costo del silenzio estremo e nessun ancoraggio osservato recuperabile
        -> HALT: l'Attore ha derivato cosi' lontano dalla radice testuale
        che proseguire produrrebbe solo rumore. Il Regista spegne il processo.

    Le soglie relative (1.5x, 2.5x) sono dichiarate qui, niente magia.
    Ritorna None se non c'e' nulla da fare (es. fase senza successore).
    """
    target = _next_phase(after_phase)

    # Rapporto di superamento della soglia di divergenza.
    over = divergence / divergence_threshold if divergence_threshold > 0 else 0.0

    # --- HALT: deriva estrema, l'Attore ha perso la radice testuale ---------
    # silence_cost e' un integrale: oltre ~3 scale accumulate l'esplorazione
    # non e' piu' ancorata. Combinato con divergenza estrema -> spegni.
    if over >= 2.5 and silence_cost >= 3.0:
        return DirectorIntervention(
            after_phase=after_phase,
            target_phase=CLOSING_PHASE,
            control=CONTROL_HALT,
            reasoning=(
                f"Divergenza {divergence:.2f} ({over:.1f}x soglia) e costo del "
                f"silenzio {silence_cost:.2f}: l'Attore ha derivato oltre ogni "
                f"ancoraggio testuale. Proseguire produrrebbe solo rumore. Il "
                f"Regista spegne il processo."
            ),
        )

    # --- GOTO expand: ponti irrecuperabili, ri-radica l'esplorazione --------
    if over >= 2.5 and after_phase in ("bridge", "revalidate"):
        return DirectorIntervention(
            after_phase=after_phase,
            target_phase="expand",
            control=CONTROL_GOTO,
            goto_phase="expand",
            param_overrides={"expand_top_n": 2},
            reasoning=(
                f"Divergenza {divergence:.2f} ({over:.1f}x soglia): i ponti "
                f"cross-scale sono troppo fasulli per essere semplicemente "
                f"rifatti. Il Regista riporta l'Attore alla fase expand per "
                f"ri-radicare l'esplorazione nel testo prima di ritentare."
            ),
        )

    # --- REPEAT: risultato compromesso, rifai la fase appena conclusa -------
    if over >= 1.5:
        return DirectorIntervention(
            after_phase=after_phase,
            target_phase=after_phase,
            control=CONTROL_REPEAT,
            param_overrides=_overrides_for(after_phase),
            reasoning=(
                f"Divergenza {divergence:.2f} ({over:.1f}x soglia): la fase "
                f"'{after_phase}' ha prodotto un risultato troppo compromesso. "
                f"Il Regista la fa ri-eseguire con parametri corretti prima di "
                f"lasciar proseguire l'Attore."
            ),
        )

    # --- PROCEED + override: zoom-in lieve, correzione di parametri ---------
    if not target:
        return None
    overrides = _overrides_for(target)
    if not overrides:
        return None
    return DirectorIntervention(
        after_phase=after_phase,
        target_phase=target,
        control=CONTROL_PROCEED,
        param_overrides=overrides,
        reasoning=(
            f"Divergenza {divergence:.2f} appena sopra soglia: l'Attore puo' "
            f"proseguire, ma il Regista corregge i parametri di '{target}' "
            f"(zoom-in lieve)."
        ),
    )


def _overrides_for(phase: str) -> dict[str, Any]:
    """Parametri correttivi per una data fase. Dichiarati, deterministici.

    - revalidate: only_uncertain=False -> rivedi TUTTE le ipotesi, non solo
      le 'uncertain' (zoom-in sulla chiusura).
    - bridge: max_bridges=1 -> non allargare ancora il gap (zoom-out).
    - expand: nessun override di default (l'eventuale expand_top_n lo mette
      direttamente il ramo GOTO).
    """
    if phase == "revalidate":
        return {"only_uncertain": False}
    if phase == "bridge":
        return {"max_bridges": 1}
    return {}


# -----------------------------------------------------------------------------
# Il Regista.
# -----------------------------------------------------------------------------


class Director:
    """L7 Director. Osserva l'Attore fase dopo fase e decide se intromettersi.

    Uso tipico (dentro auto_explore):
        director = Director(client, ...)
        ...dopo la fase 'expand'...
        intervention = director.observe(ft, phase="expand")
        if intervention: applica intervention.param_overrides alla fase dopo
        ...
        report = director.finalize(ft)   # opzionale: racconto LLM
    """

    def __init__(
        self,
        client: LLMClient | None,
        *,
        llm_calls_dir: Path | None = None,
        telemetry_path: Path | None = None,
        silence_band: float = 1.5,
        divergence_threshold: float = 0.34,
        narrate: bool = True,
    ) -> None:
        """
        silence_band: soglia dell'INTEGRALE di deriva oltre cui il costo del
            silenzio e' considerato "scattato". 1.5 = mezza scala accumulata
            su piu' fasi (l'asse ha 9 scale, indici 0..8).
        divergence_threshold: frazione di cross-scale spurious oltre cui la
            divergenza e' considerata "scattata". 0.34 ~ un terzo dei ponti.
        narrate: se True, finalize() fa una chiamata LLM per raccontare
            l'auto-osservazione. Se client e' None, narrate viene ignorato.
        """
        self.client = client
        self.silence_band = silence_band
        self.divergence_threshold = divergence_threshold
        self.narrate = narrate and client is not None
        self._agent = (
            RoleAgent(
                client,
                role_name="L7_Director",
                role_prompt=DIRECTOR_PROMPT,
                out_dir=llm_calls_dir,
                max_output_tokens=budget("director"),
            )
            if self.narrate else None
        )
        self.telemetry_path = telemetry_path
        # stato accumulato lungo l'osservazione
        self._silence_integral: float = 0.0
        # Guardia anti-loop del pieno controllo: il Regista puo' ripetere e
        # tornare indietro, ma un budget finito di atti di regia impedisce
        # cicli infiniti. Oltre il budget, ogni intervento e' degradato a
        # 'proceed' (l'Attore prosegue comunque verso la chiusura).
        self.control_budget: int = 6
        self._control_acts: int = 0
        self.report = DirectorReport(
            silence_band=silence_band,
            divergence_threshold=divergence_threshold,
        )

    def budget_exhausted(self) -> bool:
        """True se il Regista ha consumato il budget di atti di regia.

        E' la rete di sicurezza del pieno controllo: senza, REPEAT e GOTO
        potrebbero generare un loop. Esaurito il budget, il Regista perde
        il potere di deviare il flusso e puo' solo osservare.
        """
        return self._control_acts >= self.control_budget

    def register_control_act(self) -> None:
        """Il loop di regia segnala che un atto di flusso (skip/repeat/goto/
        halt) e' stato eseguito. Consuma una unita' di budget."""
        self._control_acts += 1

    def observe(
        self,
        ft: FractalTriadResult,
        *,
        phase: str,
        trace: list[str] | None = None,
    ) -> DirectorIntervention | None:
        """Osserva l'Attore dopo una fase. Ritorna una correzione o None.

        E' il "voltarsi indietro a guardare se stesso mentre lavora": misura
        le tre condizioni, le accumula, e decide. Puro Python: nessun LLM.
        """
        trace = trace if trace is not None else []

        divergence = measure_scale_divergence(ft)
        step_cost = measure_silence_cost(ft)
        self._silence_integral += step_cost
        irreversible = is_phase_irreversible(phase)

        # Le tre condizioni. Servono TUTTE per rompere il silenzio.
        cond_divergence = divergence >= self.divergence_threshold
        cond_silence = self._silence_integral >= self.silence_band
        cond_irreversible = irreversible
        all_three = cond_divergence and cond_silence and cond_irreversible

        reading = DirectorReading(
            phase=phase,
            scale_divergence=round(divergence, 4),
            silence_cost=round(self._silence_integral, 4),
            is_irreversible=irreversible,
            intervened=all_three,
            note=self._reading_note(
                cond_divergence, cond_silence, cond_irreversible, all_three
            ),
        )
        self.report.readings.append(reading)
        trace.append(
            f"L7_Director: observe phase={phase} divergence={divergence:.2f} "
            f"silence_integral={self._silence_integral:.2f} "
            f"irreversible={irreversible} intervened={all_three}"
        )

        if not all_three:
            return None

        intervention = _build_intervention(
            phase, divergence, self._silence_integral, self.divergence_threshold
        )
        if intervention is None:
            trace.append(
                f"L7_Director: 3 condizioni scattate ma fase '{phase}' non ha "
                f"una correzione applicabile -> sola osservazione"
            )
            return None

        self.report.interventions.append(intervention)
        trace.append(
            f"L7_Director: INTERVENTO dopo '{phase}' control={intervention.control} "
            f"target='{intervention.target_phase}' "
            f"overrides={intervention.param_overrides}"
        )
        return intervention

    def finalize(self, ft: FractalTriadResult, *, trace: list[str] | None = None) -> DirectorReport:
        """Chiude l'osservazione e produce il DirectorReport.

        Se narrate=True, una chiamata LLM racconta l'auto-osservazione: e' la
        formalizzazione del distacco dell'Osservatore dall'Attore. Se la
        chiamata fallisce o narrate=False, il report resta valido con il solo
        summary deterministico.
        """
        trace = trace if trace is not None else []
        n_interv = len(self.report.interventions)
        self.report.summary = (
            f"Il Regista ha osservato {len(self.report.readings)} fasi "
            f"dell'Attore e si e' intromesso {n_interv} volta/e. "
            f"Soglie: divergenza>={self.divergence_threshold}, "
            f"costo_silenzio>={self.silence_band}."
        )

        if not self.narrate or self._agent is None:
            return self.report

        payload = _build_director_payload(self.report)
        raw, _meta = self._agent.run_json(
            payload, DIRECTOR_CONTRACT, trace, telemetry_path=self.telemetry_path
        )
        if isinstance(raw, dict) and "_parse_error" not in raw and "_llm_error" not in raw:
            narrazione = str(raw.get("auto_osservazione", "")).strip()
            if narrazione:
                self.report.summary = narrazione
            trace.append("L7_Director: narrazione LLM dell'auto-osservazione acquisita")
        else:
            trace.append("L7_Director: narrazione LLM fallita -> summary deterministico")
        return self.report

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _reading_note(
        cond_div: bool, cond_sil: bool, cond_irr: bool, all_three: bool
    ) -> str:
        if all_three:
            return ("Tutte e tre le condizioni scattate: divergenza di scala, "
                    "costo del silenzio oltre banda, fase irreversibile. Il "
                    "Regista rompe il silenzio.")
        scattate = []
        if cond_div:
            scattate.append("divergenza")
        if cond_sil:
            scattate.append("costo-silenzio")
        if cond_irr:
            scattate.append("irreversibilita'")
        if not scattate:
            return "Nessuna condizione scattata: l'Attore opera nei limiti."
        return (f"Condizioni parziali ({', '.join(scattate)}): NON sufficienti. "
                f"Una sola o due condizioni sarebbero orologeria. Il Regista "
                f"resta in osservazione.")


# -----------------------------------------------------------------------------
# Integrazione e render
# -----------------------------------------------------------------------------


def attach_director_report(ft: FractalTriadResult, report: DirectorReport) -> None:
    """Aggancia il DirectorReport al ft, in metadata, senza toccare il resto.

    Il ft non ha un campo dedicato (per non rompere la (de)serializzazione
    V15): il report vive in ft.trace come marcatori + qui in un attributo
    dinamico leggero. Il render lo legge da li'.
    """
    setattr(ft, "_director_report", report)


def render_director_md(report: DirectorReport) -> str:
    """Rende il DirectorReport in markdown leggibile (director_report.md)."""
    out: list[str] = ["# Relazione del Regista (L7 Director)\n"]
    out.append(report.summary + "\n")

    # Flusso reale eseguito dall'Attore sotto regia.
    out.append("\n## Flusso eseguito\n")
    if report.executed_phases:
        out.append("\n`" + " -> ".join(report.executed_phases) + "`\n")
    else:
        out.append("\n_(nessuna fase eseguita)_\n")
    if report.halted:
        out.append("\n**L'Attore e' stato fermato dal Regista prima della "
                    "chiusura naturale.**\n")

    out.append("\n## Letture per fase\n")
    if not report.readings:
        out.append("\n_(nessuna lettura)_\n")
    for r in report.readings:
        flag = "  ← INTERVENTO" if r.intervened else ""
        out.append(
            f"\n### fase: {r.phase}{flag}\n"
            f"- divergenza di scala: {r.scale_divergence}\n"
            f"- costo del silenzio (integrale): {r.silence_cost}\n"
            f"- fase irreversibile: {r.is_irreversible}\n"
            f"- {r.note}"
        )
    out.append("\n\n## Atti di regia\n")
    if not report.interventions:
        out.append("\n_(il Regista non si e' mai intromesso: l'Attore ha "
                    "operato nei limiti)_\n")
    for itv in report.interventions:
        verb = itv.control.upper()
        dest = (f" -> {itv.goto_phase}" if itv.control == "goto" else "")
        ovr = (f" con `{itv.param_overrides}`" if itv.param_overrides else "")
        out.append(
            f"\n- dopo **{itv.after_phase}**: **{verb}**{dest}{ovr}\n"
            f"  - {itv.reasoning}"
        )
    return "\n".join(out) + "\n"


# -----------------------------------------------------------------------------
# Prompt e contratto per la narrazione (la sola parte LLM, opzionale).
# -----------------------------------------------------------------------------


DIRECTOR_PROMPT = """Sei il REGISTA del Fractal Causal Engine.

Hai osservato l'ATTORE (la pipeline di esplorazione causale) mentre lavorava,
e a PIENO CONTROLLO ne hai governato il flusso: oltre a osservare, hai potuto
saltare, ripetere, far tornare indietro o fermare l'Attore.

Ricevi le tue stesse LETTURE (per ogni fase: divergenza di scala, costo del
silenzio, irreversibilita'), il FLUSSO realmente eseguito e i tuoi ATTI DI
REGIA (proceed/skip/repeat/goto/halt).

Il tuo compito: scrivere una breve AUTO-OSSERVAZIONE in italiano. NON una
nuova analisi causale: un resoconto di come l'Attore si e' mosso, di come tu
hai piegato il suo flusso, e perche'.

REGOLE
1. Parla in prima persona come Regista. Non inventare misure ne' atti: usa
   solo quelli forniti.
2. Distingui osservazione pura (non sei intervenuto) da atto di regia (hai
   deviato il flusso). Spiega che un atto di regia avviene SOLO quando tutte
   e tre le condizioni scattano insieme -- mai per una sola.
3. Se hai fermato l'Attore (halt), spiega che proseguire avrebbe prodotto
   solo rumore: il silenzio era preferibile al lavoro a vuoto.
4. Se non sei mai intervenuto, dillo: l'Attore ha operato nei limiti.
5. 4-8 frasi. Tono lucido, sobrio. Niente proclami.
6. Restituisci JSON valido come da OUTPUT_CONTRACT.
"""


DIRECTOR_CONTRACT: dict[str, Any] = {
    "auto_osservazione": "<4-8 frasi: come si e' mosso l'Attore, quando e perche' sei intervenuto>",
}


def _build_director_payload(report: DirectorReport) -> dict[str, Any]:
    """Serializza per l'LLM solo cio' che serve a raccontare l'osservazione."""
    return {
        "soglie": {
            "divergence_threshold": report.divergence_threshold,
            "silence_band": report.silence_band,
        },
        "flusso_eseguito": list(report.executed_phases),
        "attore_fermato": report.halted,
        "letture": [
            {
                "fase": r.phase,
                "divergenza_di_scala": r.scale_divergence,
                "costo_del_silenzio": r.silence_cost,
                "irreversibile": r.is_irreversible,
                "intromesso": r.intervened,
                "nota": r.note,
            }
            for r in report.readings
        ],
        "atti_di_regia": [
            {
                "dopo_fase": itv.after_phase,
                "verbo": itv.control,
                "destinazione": itv.goto_phase,
                "correzione": itv.param_overrides,
                "motivazione": itv.reasoning,
            }
            for itv in report.interventions
        ],
    }
