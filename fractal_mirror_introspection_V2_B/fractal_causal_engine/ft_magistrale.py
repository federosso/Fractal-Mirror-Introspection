"""ft_magistrale (V10.15.0).

MagistraleReportBuilder: produce una relazione finale sintetica e umana che
RACCONTA il FractalTriadResult senza aggiungere nuove validazioni.

UNA sola chiamata LLM, alimentata dai dati gia' validati:
  - items (incluse espansioni e bridge, con i loro epistemic_status)
  - same_scale_links validati da L2
  - cross_scale con i loro verdict ('genuine' | 'uncertain' | 'spurious')
  - vision (core_image, human_summary, epistemic_warning)
  - locked_reports (per le sintesi per scala)

OUTPUT: MagistraleReport (dataclass), formato testuale italiano.

REGOLA INDEROGABILE: il prompt al modello dichiara esplicitamente che NON
deve inventare nuove cause o nuovi effetti. Deve scegliere tra gli items
esistenti, raggruppandoli in ruoli causali (predispositions, triggers,
proximate_causes, bridge_mechanisms) e ruoli di effetto (direct, downstream,
interpretations, social_propagations).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .ft_model import (
    SCALE_DEPTH,
    ClassifiedItem,
    FractalTriadResult,
    MagistraleCones,
    MagistraleEffects,
    MagistraleReport,
    Nature,
    PredicateType,
)
from .llm import LLMClient, RoleAgent
from .ft_budget import budget


MAGISTRALE_PROMPT = """Sei l'ARCHITETTO MAGISTRALE del Fractal Causal Engine.

Ricevi un FRACTAL TRIAD RESULT gia' analizzato: items classificati con
scala e natura, link same-scale validati, ipotesi cross-scale con verdict,
visione globale e gap.

Il tuo compito: produrre una RELAZIONE MAGISTRALE in italiano, sintetica,
profonda, leggibile. Una lettura SISTEMICA del fenomeno.

REGOLE INDEROGABILI

1. NON inventare nuove cause o nuovi effetti. Scegli SOLO tra gli items
   forniti in input. Cita gli item per il loro testo (quote o
   generated_text), non per il loro id.

2. Distingui sempre OSSERVATO (epistemic_status=text_observed) da INFERITO
   (domain_knowledge, causal_model) e SPECULATIVO. Nel campo
   stato_epistemico esplicitalo apertamente.

3. Le ipotesi cross-scale con verdict='spurious' NON vanno usate come
   propagazione valida. Le 'uncertain' vanno presentate come dubbi aperti.
   Solo le 'genuine' (e i same_scale_links validati) sono propagazione
   solida.

4. Schema rigoroso (rispetta i nomi dei campi):
   - sintesi_magistrale: 3-6 frasi che raccontano il fenomeno come un
     insieme, con il linguaggio del paradigma (scale discrete, risonanza,
     propagazione, cono di osservazione).
   - cono_cause:
     * predispositions: condizioni di sfondo (context, state) che
       PREDISPONGONO. Liste di testi citati.
     * triggers: eventi innescanti (event, process_description) di
       nature=cause.
     * proximate_causes: cause prossime, vicine all'effetto.
     * bridge_mechanisms: meccanismi intermedi (nature=bridge o
       epistemic_status=causal_model) che fanno da ponte tra scale.
   - cono_effetti:
     * direct_effects: effetti diretti (nature=effect) alla stessa scala
       della causa prossima.
     * downstream_effects: effetti a valle, su scale piu' superficiali.
     * interpretations: items con nature=interpretation o
       epistemic_status=speculative (lettura simbolica, spirituale).
     * social_propagations: effetti su scala sociale o piu' alta.
   - propagazione_multi_scala: 2-4 frasi che descrivono il passaggio
     attraverso le scale (es. "atomico -> molecolare -> organismo ->
     sociale"). Cita le scale effettivamente popolate.
   - stato_epistemico: 2-3 frasi che dichiarano cosa e' osservato, cosa
     inferito, cosa speculativo.
   - verdetto_finale: 1-2 frasi che chiudono. Nessun proclama di certezza
     dove i dati non la consentono.

5. Tono: tecnico ma leggibile. Eviti il gergo da paper, ma anche la poesia
   vuota. Niente metafore se non chiariscono.

6. REGIME ESPLORATIVO (importante!): se la maggior parte degli items ha
   nature=context e epistemic_status non e' text_observed (cioe' il
   testo era definitorio e l'esplorazione viene quasi tutta da
   expansion/L3A), allora:
   - il cono_cause non sara' una catena temporale ma una rete di
     FATTORI COSTITUTIVI (cosa COMPONE / SOSTIENE il fenomeno);
   - il cono_effetti diventera' una rete di MANIFESTAZIONI ipotizzate
     (cosa POTREBBE EMERGERE su scale superficiali);
   - i bridge_mechanisms diventano la parte centrale: e' li' che si
     concentrano le teorie nominate (IIT, GWT, predictive coding,
     allostasi, ecc.) che spiegherebbero il passaggio tra scale.
   - dichiara apertamente nel campo stato_epistemico che la lettura e'
     interamente ESPLORATIVA, e che la mappa qui presentata e' una rete
     di ipotesi, non una catena causale dimostrata.
   - NON usare formule del tipo "il sistema dimostra che..." o "e' provato
     che..." nel verdetto_finale. Usa invece "una possibile lettura...",
     "tra le ipotesi attive...", "merita verifica empirica...".

7. Quando un item nasce da espansione o bridge e cita una teoria nota
   per nome proprio (Tononi/IIT, Friston/predictive coding, etc.),
   PRIVILEGIA quella citazione nel testo: l'utente vuole vedere dove la
   lettura si ancora.

8. STARTING_QUESTIONS: il payload puo' contenere un campo
   `starting_questions` con le domande di partenza poste nel testo (es.
   "Che cos'e' la coscienza?"). Queste NON sono fattori causali: non
   metterle MAI tra predisposizioni, trigger, cause prossime o effetti.
   Usale solo per orientare la sintesi -- la relazione e' una RISPOSTA
   esplorativa a quelle domande. Puoi richiamarle nella sintesi_magistrale
   ("Rispetto alle domande poste..."), mai nei coni.

9. Restituisci JSON valido come da OUTPUT_CONTRACT, in italiano.
"""


MAGISTRALE_CONTRACT: dict[str, Any] = {
    "sintesi_magistrale": "<3-6 frasi>",
    "cono_cause": {
        "predispositions": ["<testi citati>"],
        "triggers": ["<testi>"],
        "proximate_causes": ["<testi>"],
        "bridge_mechanisms": ["<testi>"],
    },
    "cono_effetti": {
        "direct_effects": ["<testi>"],
        "downstream_effects": ["<testi>"],
        "interpretations": ["<testi>"],
        "social_propagations": ["<testi>"],
    },
    "propagazione_multi_scala": "<2-4 frasi>",
    "stato_epistemico": "<2-3 frasi>",
    "verdetto_finale": "<1-2 frasi>",
}


class MagistraleReportBuilder:
    """Costruisce la relazione magistrale finale. Una chiamata LLM."""

    def __init__(
        self,
        client: LLMClient,
        *,
        llm_calls_dir: Path | None,
        telemetry_path: Path | None = None,
    ) -> None:
        self.agent = RoleAgent(
            client,
            role_name="L6_MagistraleReport",
            role_prompt=MAGISTRALE_PROMPT,
            out_dir=llm_calls_dir,
            max_output_tokens=budget("magistrale"),
        )
        self.telemetry_path = telemetry_path

    def build(self, ft: FractalTriadResult, *, trace: list[str]) -> MagistraleReport:
        if not ft.items:
            trace.append("L6_MagistraleReport: ft.items vuoto -> report vuoto degraded")
            return MagistraleReport(degraded=True, sintesi_magistrale="(nessun item)")

        payload = _build_payload(ft)
        raw, _meta = self.agent.run_json(
            payload, MAGISTRALE_CONTRACT, trace, telemetry_path=self.telemetry_path
        )

        if not isinstance(raw, dict) or "_parse_error" in raw or "_llm_error" in raw:
            trace.append("L6_MagistraleReport: LLM error or parse failure -> degraded")
            return MagistraleReport(
                degraded=True,
                sintesi_magistrale="(generazione fallita)",
            )

        report = _coerce_to_report(raw)
        ft.magistrale = report
        trace.append(
            f"L6_MagistraleReport: built, "
            f"predispositions={len(report.cono_cause.predispositions)} "
            f"triggers={len(report.cono_cause.triggers)} "
            f"direct_effects={len(report.cono_effetti.direct_effects)}"
        )
        return report


# -----------------------------------------------------------------------------
# Costruzione del payload: serializziamo SOLO cio' che serve al modello.
# -----------------------------------------------------------------------------


def _item_text(it: ClassifiedItem) -> str:
    """Testo da mostrare: quote se viene dal testo, generated_text se da espansione/bridge."""
    if it.quote:
        return it.quote
    gt = it.metadata.get("generated_text") if isinstance(it.metadata, dict) else None
    return str(gt or "")


def _build_payload(ft: FractalTriadResult) -> dict[str, Any]:
    items_payload: list[dict[str, Any]] = []
    starting_questions: list[str] = []
    for it in ft.items:
        is_question = it.predicate == PredicateType.QUESTION
        if is_question:
            # Le domande sono il quesito di partenza, NON fattori causali.
            # Le raccogliamo a parte cosi' il magistrale non le mette nei coni.
            txt = _item_text(it)
            if txt:
                starting_questions.append(txt)
            continue
        items_payload.append(
            {
                "id": it.id,
                "text": _item_text(it),
                "predicate": it.predicate.value,
                "nature": it.nature.value,
                "scale": it.scale,
                "epistemic_status": it.epistemic_status.value,
                "origin": (it.metadata or {}).get("origin", "text_observed"),
            }
        )

    same_scale_payload: list[dict[str, Any]] = []
    for r in ft.locked_reports:
        for lnk in r.same_scale_links:
            same_scale_payload.append(
                {
                    "scale": lnk.scale,
                    "cause_id": lnk.cause_item_id,
                    "effect_id": lnk.effect_item_id,
                    "rationale": lnk.rationale,
                    "confidence": lnk.confidence,
                }
            )

    cross_scale_payload: list[dict[str, Any]] = []
    for h in ft.cross_scale:
        cross_scale_payload.append(
            {
                "cause_id": h.cause_item_id,
                "effect_id": h.effect_item_id,
                "cause_scale": h.cause_scale,
                "effect_scale": h.effect_scale,
                "verdict": h.verdict,
                "reasoning": h.reasoning,
                "confidence": h.confidence,
            }
        )

    populated_scales = sorted(
        {it.scale for it in ft.items if it.scale in SCALE_DEPTH},
        key=lambda s: SCALE_DEPTH[s],
    )

    return {
        "task": "build_magistrale_report",
        "items": items_payload,
        "starting_questions": starting_questions,
        "same_scale_links": same_scale_payload,
        "cross_scale_hypotheses": cross_scale_payload,
        "vision": {
            "core_image": ft.vision.core_image,
            "human_summary": ft.vision.human_summary,
            "epistemic_warning": ft.vision.epistemic_warning,
            "dominant_domain": ft.vision.dominant_domain,
        },
        "populated_scales_top_to_deep": populated_scales,
    }


# -----------------------------------------------------------------------------
# Coercion del JSON LLM in MagistraleReport con difese.
# -----------------------------------------------------------------------------


def _as_str_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if x is not None and str(x).strip()]
    if v is None:
        return []
    return [str(v)]


def _coerce_to_report(raw: dict[str, Any]) -> MagistraleReport:
    cc_raw = raw.get("cono_cause") if isinstance(raw.get("cono_cause"), dict) else {}
    ce_raw = raw.get("cono_effetti") if isinstance(raw.get("cono_effetti"), dict) else {}

    cono_cause = MagistraleCones(
        predispositions=_as_str_list(cc_raw.get("predispositions")),
        triggers=_as_str_list(cc_raw.get("triggers")),
        proximate_causes=_as_str_list(cc_raw.get("proximate_causes")),
        bridge_mechanisms=_as_str_list(cc_raw.get("bridge_mechanisms")),
    )
    cono_effetti = MagistraleEffects(
        direct_effects=_as_str_list(ce_raw.get("direct_effects")),
        downstream_effects=_as_str_list(ce_raw.get("downstream_effects")),
        interpretations=_as_str_list(ce_raw.get("interpretations")),
        social_propagations=_as_str_list(ce_raw.get("social_propagations")),
    )
    return MagistraleReport(
        sintesi_magistrale=str(raw.get("sintesi_magistrale") or ""),
        cono_cause=cono_cause,
        cono_effetti=cono_effetti,
        propagazione_multi_scala=str(raw.get("propagazione_multi_scala") or ""),
        stato_epistemico=str(raw.get("stato_epistemico") or ""),
        verdetto_finale=str(raw.get("verdetto_finale") or ""),
        degraded=False,
    )


# -----------------------------------------------------------------------------
# Rendering: la relazione magistrale in markdown leggibile.
# -----------------------------------------------------------------------------


def render_magistrale_md(report: MagistraleReport) -> str:
    """Render del MagistraleReport in markdown. Funzione pura, niente LLM."""

    def _bullets(items: list[str]) -> str:
        if not items:
            return "_(nessuno)_"
        return "\n".join(f"- {x}" for x in items)

    out: list[str] = []
    out.append("# Relazione Magistrale\n")
    if report.degraded:
        out.append("> _Report degraded: generazione non completata._\n")
    out.append("## Sintesi\n")
    out.append((report.sintesi_magistrale or "_(vuota)_") + "\n")
    out.append("## Cono delle Cause\n")
    out.append("### Predisposizioni\n" + _bullets(report.cono_cause.predispositions) + "\n")
    out.append("### Trigger\n" + _bullets(report.cono_cause.triggers) + "\n")
    out.append("### Cause prossime\n" + _bullets(report.cono_cause.proximate_causes) + "\n")
    out.append("### Meccanismi di ponte\n" + _bullets(report.cono_cause.bridge_mechanisms) + "\n")
    out.append("## Cono degli Effetti\n")
    out.append("### Effetti diretti\n" + _bullets(report.cono_effetti.direct_effects) + "\n")
    out.append("### Effetti a valle\n" + _bullets(report.cono_effetti.downstream_effects) + "\n")
    out.append("### Interpretazioni\n" + _bullets(report.cono_effetti.interpretations) + "\n")
    out.append("### Propagazioni sociali\n" + _bullets(report.cono_effetti.social_propagations) + "\n")
    out.append("## Propagazione multi-scala\n")
    out.append((report.propagazione_multi_scala or "_(vuota)_") + "\n")
    out.append("## Stato epistemico\n")
    out.append((report.stato_epistemico or "_(vuota)_") + "\n")
    out.append("## Verdetto finale\n")
    out.append((report.verdetto_finale or "_(vuoto)_") + "\n")
    return "\n".join(out)
