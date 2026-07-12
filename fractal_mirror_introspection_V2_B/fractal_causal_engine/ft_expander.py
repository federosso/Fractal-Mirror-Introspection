"""L5 -- FractalExpander (V10.15.0).

Espansione frattale iterativa di un singolo ClassifiedItem.

Dato un item esistente nel risultato della pipeline (frutto di L1 o di una
espansione precedente), chiede all'LLM 4 figli, uno per ciascuna direzione
canonica del paradigma Fractal Triad:

  1. SAME_SCALE_CAUSE        -- causa orizzontale, stessa scala del padre
  2. SCALE_UP_PROPAGATION    -- propagazione su scala piu' superficiale
  3. SCALE_DOWN_MECHANISM    -- meccanismo su scala piu' profonda
  4. COHERENCE_BRIDGE        -- ponte cross-scale ragionato (non lineare)

Una sola chiamata LLM per espansione. Validator Python che riapplica i
contratti di V14: nature, scale canonica, predicate. Nessun bypass.

VINCOLI EREDITATI da V10.14.0:
- I figli sono nuovi ClassifiedItem ma con quote vuota (non vengono dal testo):
  il loro epistemic_status NON e' TEXT_OBSERVED. Lo status di default e':
    * SAME_SCALE_CAUSE       -> DOMAIN_KNOWLEDGE
    * SCALE_UP_PROPAGATION   -> DOMAIN_KNOWLEDGE
    * SCALE_DOWN_MECHANISM   -> CAUSAL_MODEL
    * COHERENCE_BRIDGE       -> CAUSAL_MODEL
- Le relazioni generate dall'espansione NON sono validate per prossimita':
  * la direzione SAME_SCALE_CAUSE produce un SameScaleLink (figlio -> padre)
    aggiunto al locked_report della scala;
  * le tre direzioni cross-scale producono CrossScaleHypothesis con verdict
    iniziale = 'uncertain'. Per promuoverle a 'genuine' bisogna passare
    da CrossScaleValidator (L3.B) in una passata successiva. Qui non si
    bypassa nulla.

L'expander e' STATELESS: prende un FractalTriadResult e ritorna un
ExpansionRecord. Sara' un nuovo modulo, ft_explorer.py, ad applicare il
record al ft (cosi' separiamo 'cosa l'LLM ha detto' da 'come integriamo').
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .ft_model import (
    SCALES_CANONICAL,
    SCALE_DEPTH,
    ClassifiedItem,
    EpistemicStatus,
    ExpansionChild,
    ExpansionDirection,
    ExpansionRecord,
    Nature,
    PredicateType,
    is_valid_scale,
)
from .llm import LLMClient, RoleAgent
from .text import stable_hash
from .ft_budget import budget


EXPANDER_PROMPT = """Sei l'ESPLORATORE FRATTALE del Fractal Causal Engine.

Ricevi un ITEM gia' classificato (con scale e nature). Il tuo compito e'
generare 4 ITEM figli che esplorano la realta' attorno al padre seguendo
le 4 direzioni canoniche del paradigma Fractal Triad.

REGOLE INDEROGABILI

1. Produci ESATTAMENTE 4 figli, uno per ciascuna direction:
   - same_scale_cause:     una CAUSA del padre, alla STESSA scala del padre.
                           Se il padre e' un CONCETTO DEFINITORIO senza
                           causalita' osservata (nature=context con
                           predicate=definition o claimed_property), allora
                           same_scale_cause diventa un CONCETTO FRATELLO
                           o COSTITUENTE alla stessa scala: cio' che NUTRE
                           o ALIMENTA il concetto. Mantieni nature=cause
                           (e' una causa formale, non temporale).
   - scale_up_propagation: una PROPAGAZIONE EMERGENTE, su scala PIU' SUPERFICIALE
                           (cioe' con indice piu' basso nell'asse canonico).
   - scale_down_mechanism: un MECCANISMO SOTTOSTANTE, su scala PIU' PROFONDA
                           (cioe' con indice piu' alto nell'asse canonico).
   - coherence_bridge:     un PONTE di coerenza che spiega come il fenomeno
                           si manifesta tra scale VICINE. La scala del ponte
                           DEVE essere quella del padre o al massimo UNA
                           scala adiacente (indice +1 o -1). Mai un salto
                           di 2 o piu' scale. DEVE essere un MECCANISMO
                           plausibile, non una metafora vuota e non una
                           interpretazione simbolica/spirituale: se la sola
                           cosa che colleghi le scale e' un significato
                           attribuito (es. "e' un segno del destino"),
                           NON e' un coherence_bridge -- limita il ponte
                           a un meccanismo concreto sulla scala adiacente.

2. L'asse canonico delle scale (dall'alto = superficie, verso il basso = radice):
   cosmologico (0), planetario (1), sociale (2), organismo (3), cellulare (4),
   molecolare (5), atomico (6), subatomico (7), fondamentale (8).
   "scale_up" = indice piu' basso; "scale_down" = indice piu' alto.

3. Per ogni figlio:
   - text: una frase BREVE (<= 20 parole) che descrive il concetto.
           Non e' una quote dal testo: e' una proposizione che TU formuli.
   - predicate: definition | process_description | claimed_property
                | event | state | comparison.
   - nature: cause | effect | context | bridge | interpretation.
     Vincoli forti:
       * same_scale_cause     -> nature = cause
       * scale_up_propagation -> nature = effect
       * scale_down_mechanism -> nature = bridge (meccanismo)
       * coherence_bridge     -> nature = bridge
   - scale: una delle 9 canoniche, coerente con la direction.
   - relation_to_parent: una frase BREVE che spiega come il figlio si lega
     al padre (causa di / propagazione di / meccanismo di / ponte tra ...).
   - confidence: 0.0 -- 1.0. Bassa se stai costruendo un'analogia.

4. NON inventare prove osservate. Stai generando CONOSCENZA DI DOMINIO o
   MODELLI CAUSALI plausibili attorno al padre. L'utente sapra' che e' un
   layer esplorativo, non una validazione.

5. Se il padre ha scale = "cosmologico", non puoi avere scale_up. Restituisci
   comunque il campo, ma con direction = scale_up_propagation e scale =
   "cosmologico" stessa, marcando confidence = 0 e relation_to_parent
   con "scala_up non disponibile, padre gia' al limite superiore".
   Simmetrico per "fondamentale" e scale_down.

VINCOLI FINALI
- Mai 0 figli, mai piu' di 4. Esattamente 4.
- Mai inventare scale fuori dalle 9 elencate.
- Restituisci JSON valido come da OUTPUT_CONTRACT.
"""


EXPANDER_CONTRACT: dict[str, Any] = {
    "children": [
        {
            "direction": "same_scale_cause|scale_up_propagation|scale_down_mechanism|coherence_bridge",
            "text": "<frase <=20 parole>",
            "predicate": "definition|process_description|claimed_property|event|state|comparison",
            "nature": "cause|effect|context|bridge|interpretation",
            "scale": "cosmologico|planetario|sociale|organismo|cellulare|molecolare|atomico|subatomico|fondamentale",
            "relation_to_parent": "<breve, perche' lega al padre>",
            "confidence": 0.0,
        }
    ]
}


MAX_CHILD_WORDS = 20


def _word_count(s: str) -> int:
    return len([w for w in re.split(r"\s+", (s or "").strip()) if w])


def _normalize_one_line(s: str) -> str:
    t = unicodedata.normalize("NFKD", s or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


# Status epistemico di default per direzione: niente TEXT_OBSERVED.
_DEFAULT_STATUS: dict[ExpansionDirection, EpistemicStatus] = {
    ExpansionDirection.SAME_SCALE_CAUSE: EpistemicStatus.DOMAIN_KNOWLEDGE,
    ExpansionDirection.SCALE_UP_PROPAGATION: EpistemicStatus.DOMAIN_KNOWLEDGE,
    ExpansionDirection.SCALE_DOWN_MECHANISM: EpistemicStatus.CAUSAL_MODEL,
    ExpansionDirection.COHERENCE_BRIDGE: EpistemicStatus.CAUSAL_MODEL,
}

# Nature forzata per direzione: protezione dura.
_FORCED_NATURE: dict[ExpansionDirection, Nature] = {
    ExpansionDirection.SAME_SCALE_CAUSE: Nature.CAUSE,
    ExpansionDirection.SCALE_UP_PROPAGATION: Nature.EFFECT,
    ExpansionDirection.SCALE_DOWN_MECHANISM: Nature.BRIDGE,
    ExpansionDirection.COHERENCE_BRIDGE: Nature.BRIDGE,
}


class FractalExpander:
    """L5 -- espansione frattale di un singolo item. Stateless."""

    def __init__(
        self,
        client: LLMClient,
        *,
        llm_calls_dir: Path | None,
        telemetry_path: Path | None = None,
    ) -> None:
        self.agent = RoleAgent(
            client,
            role_name="L5_FractalExpander",
            role_prompt=EXPANDER_PROMPT,
            out_dir=llm_calls_dir,
            max_output_tokens=budget("l5_expander"),
        )
        self.telemetry_path = telemetry_path

    def expand(
        self,
        parent: ClassifiedItem,
        *,
        original_text: str,
        trace: list[str],
        source_input_id: str = "input_001",
    ) -> ExpansionRecord:
        """Espande un solo item. Ritorna un ExpansionRecord (no side effects)."""
        if parent.scale not in SCALE_DEPTH:
            trace.append(f"L5_FractalExpander: parent scale invalid={parent.scale!r}, skip")
            return ExpansionRecord(
                parent_item_id=parent.id,
                degraded=True,
                notes=f"scale_invalid:{parent.scale}",
            )

        payload = {
            "task": "expand_item_into_four_children",
            "parent": {
                "id": parent.id,
                "quote": parent.quote,
                "predicate": parent.predicate.value,
                "nature": parent.nature.value,
                "scale": parent.scale,
            },
            "input_text_excerpt": (original_text or "")[:600],
            "scales_allowed": SCALES_CANONICAL,
        }
        raw, _meta = self.agent.run_json(
            payload, EXPANDER_CONTRACT, trace, telemetry_path=self.telemetry_path
        )
        children_raw = raw.get("children") if isinstance(raw, dict) else None
        if not isinstance(children_raw, list) or not children_raw:
            trace.append(f"L5_FractalExpander: no children for parent={parent.id} -> degraded")
            return ExpansionRecord(
                parent_item_id=parent.id,
                degraded=True,
                notes="no_children_in_output",
            )

        children_accepted: list[ExpansionChild] = []
        directions_seen: list[ExpansionDirection] = []
        rejected_reasons: list[str] = []
        parent_depth = SCALE_DEPTH[parent.scale]

        for i, raw_child in enumerate(children_raw):
            if not isinstance(raw_child, dict):
                rejected_reasons.append(f"#{i}:not_a_dict")
                continue

            # direction
            dir_str = str(raw_child.get("direction") or "").strip().lower()
            direction = None
            for d in ExpansionDirection:
                if d.value == dir_str:
                    direction = d
                    break
            if direction is None:
                rejected_reasons.append(f"#{i}:invalid_direction={dir_str!r}")
                continue
            if direction in directions_seen:
                rejected_reasons.append(f"#{i}:duplicate_direction={direction.value}")
                continue

            # text
            text = _normalize_one_line(str(raw_child.get("text") or ""))
            if not text:
                rejected_reasons.append(f"#{i}:empty_text")
                continue
            if _word_count(text) > MAX_CHILD_WORDS:
                rejected_reasons.append(f"#{i}:text_too_long={_word_count(text)}")
                continue

            # scale
            scale = str(raw_child.get("scale") or "").strip().lower()
            if not is_valid_scale(scale):
                rejected_reasons.append(f"#{i}:invalid_scale={scale!r}")
                continue

            # vincolo direzione/scale: protezione contro LLM che ignora la direction
            child_depth = SCALE_DEPTH[scale]
            if direction == ExpansionDirection.SAME_SCALE_CAUSE and child_depth != parent_depth:
                # forziamo scale = parent.scale
                scale = parent.scale
            elif direction == ExpansionDirection.SCALE_UP_PROPAGATION and child_depth >= parent_depth:
                # se padre e' gia' al limite (cosmologico), accetta come "degenere"
                if parent_depth == 0:
                    pass
                else:
                    rejected_reasons.append(
                        f"#{i}:scale_up_not_higher parent_depth={parent_depth} child_depth={child_depth}"
                    )
                    continue
            elif direction == ExpansionDirection.SCALE_DOWN_MECHANISM and child_depth <= parent_depth:
                if parent_depth == len(SCALES_CANONICAL) - 1:
                    pass
                else:
                    rejected_reasons.append(
                        f"#{i}:scale_down_not_deeper parent_depth={parent_depth} child_depth={child_depth}"
                    )
                    continue
            elif direction == ExpansionDirection.COHERENCE_BRIDGE:
                # Il ponte di coerenza deve restare VICINO al padre: stessa
                # scala o al massimo una scala adiacente. Senza questo vincolo
                # l'LLM puo' far saltare un'interpretazione (es. "sincronicita'
                # come segnale spirituale") fino a 'cosmologico', producendo
                # un bridge privo di meccanismo reale.
                if abs(child_depth - parent_depth) > 1:
                    # Forziamo alla scala adiacente piu' vicina, nella direzione
                    # che l'LLM aveva scelto (verso l'alto o verso il basso).
                    if child_depth > parent_depth:
                        clamped = min(parent_depth + 1, len(SCALES_CANONICAL) - 1)
                    else:
                        clamped = max(parent_depth - 1, 0)
                    rejected_reasons.append(
                        f"#{i}:coherence_bridge_scale_clamped da {scale} "
                        f"(depth={child_depth}) a {SCALES_CANONICAL[clamped]} (depth={clamped})"
                    )
                    scale = SCALES_CANONICAL[clamped]
                    child_depth = clamped

            # predicate (coerce)
            predicate = _coerce_predicate(raw_child.get("predicate"))

            # nature: FORZATA dalla direzione (niente liberta' all'LLM qui)
            nature = _FORCED_NATURE[direction]

            # relation_to_parent
            relation_to_parent = str(raw_child.get("relation_to_parent") or "")[:240]

            # confidence
            try:
                confidence = float(raw_child.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.0

            # epistemic status di default per direzione
            status = _DEFAULT_STATUS[direction]

            # nuovo item
            child_id = "exp_" + stable_hash(f"{parent.id}|{direction.value}|{text}", 10)
            child_item = ClassifiedItem(
                id=child_id,
                quote="",  # NON viene dal testo, e' generato dall'expander
                predicate=predicate,
                nature=nature,
                scale=scale,
                rationale=relation_to_parent,
                source_input_id=source_input_id,
                epistemic_status=status,
                metadata={
                    "origin": "expansion",
                    "parent_item_id": parent.id,
                    "direction": direction.value,
                    "generated_text": text,  # il testo generato vive qui, separato dalla quote
                },
            )

            children_accepted.append(
                ExpansionChild(
                    item=child_item,
                    direction=direction,
                    relation_to_parent=relation_to_parent,
                    confidence=confidence,
                )
            )
            directions_seen.append(direction)

        if not children_accepted:
            trace.append(
                f"L5_FractalExpander: parent={parent.id} no_accepted children, "
                f"reasons={rejected_reasons[:6]}"
            )
            return ExpansionRecord(
                parent_item_id=parent.id,
                degraded=True,
                notes="no_accepted_children;" + ";".join(rejected_reasons[:6]),
            )

        trace.append(
            f"L5_FractalExpander: parent={parent.id} accepted={len(children_accepted)} "
            f"directions={[c.direction.value for c in children_accepted]}"
        )
        if rejected_reasons:
            trace.append(
                f"L5_FractalExpander: parent={parent.id} rejected={len(rejected_reasons)} "
                f"first={rejected_reasons[:3]}"
            )

        return ExpansionRecord(
            parent_item_id=parent.id,
            direction_set=list(directions_seen),
            children=children_accepted,
            # links / cross_scale verranno popolati da ft_explorer al momento dell'integrazione
        )


def _coerce_predicate(value: Any) -> PredicateType:
    s = str(value or "").strip().lower()
    for pt in PredicateType:
        if pt.value == s:
            return pt
    return PredicateType.UNKNOWN
