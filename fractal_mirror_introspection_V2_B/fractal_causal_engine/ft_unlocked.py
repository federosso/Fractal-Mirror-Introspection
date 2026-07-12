"""L3.A -- Unlocked Domain Explorer.

Quattro micro-chiamate LLM separate, ciascuna con schema piccolo:
  A1. domain_knowledge        -- concetti adiacenti
  A2. causal_principles       -- principi sottostanti
  A3. cross_domain_analogies  -- analogie da altri domini
  A4. open_questions          -- domande aperte
  A5. (sintesi) global_synthesis -- core_image + human_summary + warning

Vantaggi:
- ogni chiamata e' breve, il modello locale la rispetta meglio;
- se una fallisce, le altre sopravvivono (degraded=True su quella parte);
- A5 riceve in input gli output veri di A1-A4, non li inventa.

Nessun fallback hardcoded LENR/incidente. Se tutto fallisce, UnlockedReport
ha sezioni vuote e degraded=True. Il fallback semantico per dominio specifico
era esattamente il problema che la V14 elimina.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .ft_model import (
    CausalPrinciple,
    ClassifiedItem,
    CrossDomainAnalogy,
    DomainConcept,
    EpistemicStatus,
    UnlockedReport,
)
from .llm import LLMClient, RoleAgent
from .ft_budget import budget


# -----------------------------------------------------------------------------
# Prompts (uno per micro-chiamata)
# -----------------------------------------------------------------------------

KNOWLEDGE_PROMPT = """Sei l'EXPLORER DI DOMINIO -- A1: domain_knowledge.

Il tuo ruolo: essere ESPLORATIVO e PROPOSITIVO. Il testo e' un seme;
il tuo lavoro e' allargare la sua mappa concettuale citando cio' che
lo circonda nel dominio della conoscenza, anche cose che il testo NON
nomina ma che sono adiacenti.

Vedi il testo e i claim probatori gia' estratti. Elenca concetti del
DOMINIO ADIACENTE che il testo evoca, anche solo implicitamente.

REGOLE
- Fino a 8 concetti (sii generoso quando il testo lo permette).
- Quando possibile, NOMINA esplicitamente teorie scientifiche, modelli
  formali, principi noti per nome proprio. Esempi:
    * coscienza -> Integrated Information Theory (Tononi), Global Workspace
      Theory (Dehaene-Baars), Predictive Coding (Friston), Higher-Order
      Theories, Attention Schema Theory (Graziano).
    * memoria -> Atkinson-Shiffrin, Working Memory (Baddeley), LTP/LTD,
      consolidamento ippocampale, schema theory.
    * patologia psicosomatica -> modello biopsicosociale (Engel), allostasi
      (Sterling), HPA axis, polyvagal theory (Porges).
    * fisica delle particelle -> Modello Standard, simmetrie di gauge,
      decadimento beta, equazione di Dirac.
  La citazione e' SEMPRE un'IPOTESI di lettura, mai una conferma.
- Ogni concetto va marcato come "domain_knowledge" (concetto/teoria
  ampiamente accettata nel dominio) o "causal_model" (meccanismo proposto,
  modello specifico).
- Mai validare. relation_to_input deve essere descrittiva, non causale.
- Concetti che il testo gia' contiene NON vanno qui (sono gia' nei claim
  probatori).
- Indica una scala suggerita tra: cosmologico, planetario, sociale,
  organismo, cellulare, molecolare, atomico, subatomico, fondamentale.

OUTPUT: JSON con campo `concepts`.
"""


PRINCIPLES_PROMPT = """Sei l'EXPLORER DI DOMINIO -- A2: causal_principles.

Il tuo ruolo: proporre meccanismi di lettura plausibili. Ipotesi, non
verita'. Apertura, non chiusura.

Vedi il testo e i claim. Elenca PRINCIPI CAUSALI generici che potrebbero
applicarsi al fenomeno. Sono modelli candidati, non prove.

REGOLE
- Fino a 5 principi (vai oltre 3 se il testo e' ricco di dimensioni).
- Quando possibile, ancora il principio a una formulazione NOTA con nome
  proprio: "principio di minima azione", "feedback negativo (cibernetica)",
  "rinforzo operante (Skinner)", "equilibrio di Nash", "principio di Hebb
  ('cells that fire together wire together')", "deriva genetica neutrale
  (Kimura)", "integrazione predittiva (Friston)", "embodied cognition
  (Varela-Thompson)", "principio di sovrapposizione (Schrodinger)",
  "complementarita' (Bohr)".
- Ogni principio ha name (breve, evocativo, possibilmente con autore tra
  parentesi) e description (1-2 frasi che spiegano in che senso si applica
  AL FENOMENO IN QUESTIONE -- non una definizione astratta del principio).
- Marca sempre status="causal_model".
- Non descrivere il testo. Descrivi un principio applicabile.
- Se due principi sono in tensione tra loro (es. emergenza vs riduzionismo),
  citali entrambi: la tensione e' parte dell'esplorazione.

OUTPUT: JSON con campo `principles`.
"""


ANALOGIES_PROMPT = """Sei l'EXPLORER DI DOMINIO -- A3: cross_domain_analogies.

Il tuo ruolo: aprire ponti tra il fenomeno e altri domini. Ogni analogia
e' un'ipotesi di lettura; insieme formano una rete di prospettive che
illumina il fenomeno da angoli diversi.

Proponi ANALOGIE da altri domini che possono illuminare il fenomeno senza
validarlo come equivalente.

REGOLE
- Fino a 4 analogie (sii generoso se i domini possibili sono vari).
- Domini DIVERSI tra loro, mai due dello stesso campo. Esempi di domini:
  biologia, ingegneria/cibernetica, musica/armonia, fisica, matematica,
  linguistica, mitologia/letteratura, sociologia, architettura, medicina,
  ecologia, cosmologia, economia, teoria dei giochi.
- Quando l'analogia ha un autore o un termine tecnico, citalo per nome
  proprio. Esempi:
    * coscienza come "spazio di lavoro globale" (Baars)
    * mente come "macchina inferenziale" (Helmholtz/Friston)
    * cervello come "ecosistema" (Edelman, neural Darwinism)
    * trauma come "memoria somatica" (van der Kolk)
    * energia LENR come "cavita' risonante elettromagnetica" (analogia con
      cavita' QED)
- warning: una frase breve che ricorda che e' analogia, non equivalenza,
  e PERCHE' (cosa l'analogia coglie + cosa NON coglie).
- Marca status="cross_domain_analogy".

OUTPUT: JSON con campo `analogies`.
"""


QUESTIONS_PROMPT = """Sei l'EXPLORER DI DOMINIO -- A4: open_questions.

Il tuo ruolo: incrinare le certezze e indicare dove il testo lascia
spazio. Le domande aperte sono il motore dell'esplorazione successiva.

Vedi il testo e i claim. Proponi DOMANDE che restano aperte e che, se
risposte, aiuterebbero a validare o falsificare la lettura del testo,
oppure a estendere la mappa concettuale.

REGOLE
- Fino a 6 domande.
- Domande CONCRETE, non retoriche. Una domanda buona indica un esperimento
  mentale, un'osservazione fattibile, una distinzione che il testo non fa.
- Niente domande generiche tipo "e' davvero cosi?".
- Almeno una domanda deve essere CROSS-SCALA quando il fenomeno ne ammette
  piu' d'una (es. "Come si lega il livello molecolare al livello del
  vissuto soggettivo?").
- Almeno una domanda deve essere FALSIFICATIVA quando possibile
  (es. "Quale osservazione confuterebbe l'ipotesi?").

OUTPUT: JSON con campo `questions`.
"""


SYNTHESIS_PROMPT = """Sei l'EXPLORER DI DOMINIO -- A5: global_synthesis.

Hai il testo, i claim probatori, la conoscenza di dominio esplorata, i
principi, le analogie e le domande aperte. Componi una visione globale
che SI APRA, non che chiuda.

REGOLE
- core_image: una sola frase metaforica forte (max 18 parole). E' la
  metafora che apre, non quella che riduce.
- human_summary: 3-6 frasi prose, leggibili, che integrano probatorio +
  esplorativo. Quando l'esplorativo ha citato teorie nominate, riprendile
  per nome -- l'utente apprezza vedere dove la lettura si ancora.
- epistemic_warning: una frase che ricorda cosa NON e' stato dimostrato
  dal testo e in che senso le ipotesi di L3.A sono ipotesi.
- dominant_domain: etichetta breve del dominio (es. "vissuto_traumatico",
  "coscienza_fenomenologica", "fisica_nucleare_alternativa"). Non
  inventare; scegli in base a cio' che hai visto.
- primary_lenses: 3-5 etichette di lettura primarie attivate dal testo
  (es. "neuroscienze_della_coscienza", "psicofisiologia_del_trauma",
  "cibernetica_dei_sistemi").
- blocked_lenses: 1-3 letture che sarebbero fuori posto applicare come
  prova (es. "diagnosi_clinica_individuale", "validazione_metafisica").

OUTPUT: JSON con campi core_image, human_summary, epistemic_warning,
dominant_domain, primary_lenses, blocked_lenses.
"""


# -----------------------------------------------------------------------------
# Contratti (uno per micro-chiamata, piccoli e specifici)
# -----------------------------------------------------------------------------

KNOWLEDGE_CONTRACT: dict[str, Any] = {
    "concepts": [
        {
            "concept": "<termine>",
            "relation_to_input": "<come si collega senza validare>",
            "status": "domain_knowledge|causal_model",
            "suggested_scale": "cosmologico|planetario|sociale|organismo|cellulare|molecolare|atomico|subatomico|fondamentale",
        }
    ]
}

PRINCIPLES_CONTRACT: dict[str, Any] = {
    "principles": [
        {"name": "<nome>", "description": "<1-2 frasi>", "status": "causal_model"}
    ]
}

ANALOGIES_CONTRACT: dict[str, Any] = {
    "analogies": [
        {
            "domain": "<es. biologia, musica>",
            "analogy": "<frase breve>",
            "warning": "<perche' resta analogia>",
            "status": "cross_domain_analogy",
        }
    ]
}

QUESTIONS_CONTRACT: dict[str, Any] = {"questions": ["<domanda concreta>"]}

SYNTHESIS_CONTRACT: dict[str, Any] = {
    "core_image": "<una frase metaforica>",
    "human_summary": "<3-5 frasi>",
    "epistemic_warning": "<una frase>",
    "dominant_domain": "<etichetta breve>",
    "primary_lenses": ["<lente>"],
    "blocked_lenses": ["<lente>"],
}


# -----------------------------------------------------------------------------
# Coercion helpers
# -----------------------------------------------------------------------------

def _coerce_status(value: Any, default: EpistemicStatus) -> EpistemicStatus:
    s = str(value or "").strip().lower()
    for st in EpistemicStatus:
        if st.value == s:
            return st
    return default


def _items_payload(items: list[ClassifiedItem]) -> list[dict[str, Any]]:
    return [
        {
            "quote": it.quote,
            "predicate": it.predicate.value,
            "nature": it.nature.value,
            "scale": it.scale,
        }
        for it in items
    ]


# -----------------------------------------------------------------------------
# UnlockedExplorer
# -----------------------------------------------------------------------------


class UnlockedExplorer:
    """L3.A -- esplorazione di dominio in 4 micro-chiamate + sintesi."""

    def __init__(
        self,
        client: LLMClient,
        *,
        llm_calls_dir: Path | None,
        telemetry_path: Path | None = None,
    ) -> None:
        self.client = client
        self.llm_calls_dir = llm_calls_dir
        self.telemetry_path = telemetry_path

    def run(
        self,
        text: str,
        items: list[ClassifiedItem],
        trace: list[str],
    ) -> UnlockedReport:
        items_payload = _items_payload(items)

        # ---- A1 domain_knowledge ----
        knowledge, ok1 = self._call(
            "L3A1_DomainKnowledge",
            KNOWLEDGE_PROMPT,
            {"input_text": text, "claims": items_payload},
            KNOWLEDGE_CONTRACT,
            trace,
            max_tokens=budget("l3a1_domain_knowledge"),
        )
        domain_knowledge: list[DomainConcept] = []
        if ok1:
            for c in (knowledge.get("concepts") or [])[:6]:
                if not isinstance(c, dict):
                    continue
                concept = str(c.get("concept") or "").strip()
                if not concept:
                    continue
                domain_knowledge.append(
                    DomainConcept(
                        concept=concept,
                        relation_to_input=str(c.get("relation_to_input") or "")[:240],
                        status=_coerce_status(c.get("status"), EpistemicStatus.DOMAIN_KNOWLEDGE),
                        suggested_scale=str(c.get("suggested_scale") or "").strip().lower(),
                        not_in_input=True,
                    )
                )

        # ---- A2 principles ----
        principles_raw, ok2 = self._call(
            "L3A2_CausalPrinciples",
            PRINCIPLES_PROMPT,
            {"input_text": text, "claims": items_payload},
            PRINCIPLES_CONTRACT,
            trace,
            max_tokens=budget("l3a2_causal_principles"),
        )
        principles: list[CausalPrinciple] = []
        if ok2:
            for p in (principles_raw.get("principles") or [])[:4]:
                if not isinstance(p, dict):
                    continue
                name = str(p.get("name") or "").strip()
                if not name:
                    continue
                principles.append(
                    CausalPrinciple(
                        name=name,
                        description=str(p.get("description") or "")[:300],
                        status=EpistemicStatus.CAUSAL_MODEL,
                    )
                )

        # ---- A3 analogies ----
        analogies_raw, ok3 = self._call(
            "L3A3_CrossDomainAnalogies",
            ANALOGIES_PROMPT,
            {"input_text": text, "claims": items_payload},
            ANALOGIES_CONTRACT,
            trace,
            max_tokens=budget("l3a3_cross_domain"),
        )
        analogies: list[CrossDomainAnalogy] = []
        if ok3:
            for a in (analogies_raw.get("analogies") or [])[:3]:
                if not isinstance(a, dict):
                    continue
                an = str(a.get("analogy") or "").strip()
                if not an:
                    continue
                analogies.append(
                    CrossDomainAnalogy(
                        domain=str(a.get("domain") or "")[:60],
                        analogy=an[:240],
                        warning=str(a.get("warning") or "")[:160],
                        status=EpistemicStatus.CROSS_DOMAIN_ANALOGY,
                    )
                )

        # ---- A4 open questions ----
        questions_raw, ok4 = self._call(
            "L3A4_OpenQuestions",
            QUESTIONS_PROMPT,
            {"input_text": text, "claims": items_payload},
            QUESTIONS_CONTRACT,
            trace,
            max_tokens=budget("l3a4_open_questions"),
        )
        open_questions: list[str] = []
        if ok4:
            for q in (questions_raw.get("questions") or [])[:5]:
                qs = str(q or "").strip()
                if qs:
                    open_questions.append(qs[:240])

        # ---- A5 synthesis (riceve gli output veri di A1-A4) ----
        synthesis_payload = {
            "input_text": text,
            "claims": items_payload,
            "domain_knowledge": [dc.__dict__ for dc in domain_knowledge],
            "principles": [pr.__dict__ for pr in principles],
            "analogies": [an.__dict__ for an in analogies],
            "open_questions": open_questions,
        }
        synthesis_raw, ok5 = self._call(
            "L3A5_GlobalSynthesis",
            SYNTHESIS_PROMPT,
            synthesis_payload,
            SYNTHESIS_CONTRACT,
            trace,
            max_tokens=budget("l3a5_global_synthesis"),
        )

        degraded_parts: list[str] = []
        if not ok1:
            degraded_parts.append("domain_knowledge")
        if not ok2:
            degraded_parts.append("causal_principles")
        if not ok3:
            degraded_parts.append("cross_domain_analogies")
        if not ok4:
            degraded_parts.append("open_questions")
        if not ok5:
            degraded_parts.append("global_synthesis")

        dominant_domain = ""
        if ok5:
            dominant_domain = str(synthesis_raw.get("dominant_domain") or "")[:80]

        report = UnlockedReport(
            domain=dominant_domain,
            domain_knowledge=domain_knowledge,
            causal_principles=principles,
            cross_domain_analogies=analogies,
            open_questions=open_questions,
            known_uncertainties=[],
            degraded=bool(degraded_parts),
            degraded_parts=degraded_parts,
        )
        # La sintesi globale (core_image, human_summary, ...) viene letta
        # poi dall'orchestratore: la conserviamo nella metadata? Per ora la
        # esponiamo via attributo dinamico per non sporcare lo schema base.
        # Sara' GlobalVision di L4 a usarla.
        report.metadata = {  # type: ignore[attr-defined]
            "synthesis_raw": synthesis_raw if ok5 else {},
            "synthesis_ok": ok5,
        }
        trace.append(
            f"L3A_Unlocked: knowledge={len(domain_knowledge)} principles={len(principles)} "
            f"analogies={len(analogies)} questions={len(open_questions)} degraded={degraded_parts}"
        )
        return report

    def _call(
        self,
        role_name: str,
        prompt: str,
        payload: dict[str, Any],
        contract: dict[str, Any],
        trace: list[str],
        max_tokens: int,
    ) -> tuple[dict[str, Any], bool]:
        agent = RoleAgent(
            self.client,
            role_name=role_name,
            role_prompt=prompt,
            out_dir=self.llm_calls_dir,
            max_output_tokens=max_tokens,
        )
        raw, meta = agent.run_json(payload, contract, trace, telemetry_path=self.telemetry_path)
        if not isinstance(raw, dict) or meta.get("parse_failed") or meta.get("llm_error"):
            return {}, False
        return raw, True
