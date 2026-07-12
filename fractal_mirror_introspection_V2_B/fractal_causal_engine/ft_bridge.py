"""ft_bridge (V10.15.0).

BridgeBuilder: dato un gap cross-scale (tra due ClassifiedItem su scale
diverse, separate da almeno una scala intermedia non popolata), chiede
all'LLM di proporre il NODO BRIDGE alla scala intermedia.

Il bridge_item generato e' un nuovo ClassifiedItem con:
  - nature = Nature.BRIDGE
  - scale  = gap_scale (la scala intermedia richiesta)
  - epistemic_status = EpistemicStatus.CAUSAL_MODEL
    (e' un meccanismo PROPOSTO, non un fatto osservato)

Effetti dell'integrazione del BridgeRecord:
  1. bridge_item aggiunto a ft.items;
  2. due CrossScaleHypothesis create:
       source -> bridge_item   (verdict='uncertain')
       bridge_item -> target   (verdict='uncertain')
     entrambe richiedono un giro di L3.B per essere promosse a 'genuine'.
  3. Il record viene appeso a ft.bridges.

NON si bypassa nulla: il bridge e' un MODELLO, non un'osservazione.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .ft_model import (
    SCALES_CANONICAL,
    SCALE_DEPTH,
    BridgeRecord,
    ClassifiedItem,
    CrossScaleHypothesis,
    EpistemicStatus,
    FractalTriadResult,
    Nature,
    PredicateType,
    is_valid_scale,
)
from .llm import LLMClient, RoleAgent
from .text import stable_hash
from .ft_budget import budget


BRIDGE_PROMPT = """Sei il BRIDGE BUILDER del Fractal Causal Engine.

Ricevi due ITEM su scale diverse (SORGENTE e DESTINAZIONE) e una SCALA
INTERMEDIA che sembra vuota. Il tuo compito e' proporre un nodo BRIDGE su
quella scala intermedia che renda meccanicamente plausibile il passaggio
SORGENTE -> BRIDGE -> DESTINAZIONE.

REGOLE INDEROGABILI

1. Il bridge e' un MECCANISMO PROPOSTO, non un fatto osservato. Stai
   costruendo un MODELLO CAUSALE plausibile. Sii onesto: se non hai
   un'ipotesi solida, esplicitalo nel campo reasoning.

2. text del bridge: una frase BREVE (<= 25 parole) che descrive il
   meccanismo intermedio.

3. predicate: scegli quello che meglio descrive il bridge:
   - process_description: "X agisce su Y"
   - state: "X e' in uno stato Y che rende possibile Z"
   - claimed_property: "X possiede la proprieta' P"
   Evita 'definition': un bridge non e' una definizione.

4. nature: SEMPRE 'bridge'. Non puo' essere altro.

5. scale: ESATTAMENTE la gap_scale richiesta. Non scegliere altre scale.

6. reasoning: 1-3 frasi che spiegano IL MECCANISMO. Non poesia, meccanica:
   "la sorgente, attraverso X, modifica una variabile Y alla scala
   intermedia, e questa variabile Y produce/innesca la destinazione".
   Se il meccanismo non e' noto in letteratura, scrivilo: "modello
   ipotetico, non documentato".

7. Se il gap richiesto e' assurdo (es. scala_intermedia non e' davvero
   tra sorgente e destinazione), restituisci comunque un bridge con
   confidence = 0 e reasoning che dichiara la difficolta'.

8. Restituisci JSON valido come da OUTPUT_CONTRACT.
"""


BRIDGE_CONTRACT: dict[str, Any] = {
    "bridge": {
        "text": "<frase <=25 parole>",
        "predicate": "process_description|state|claimed_property|event",
        "scale": "<una delle 9 canoniche, coincidente con gap_scale>",
        "reasoning": "<meccanismo proposto, 1-3 frasi>",
        "confidence": 0.0,
    }
}


MAX_BRIDGE_WORDS = 25


def _word_count(s: str) -> int:
    return len([w for w in re.split(r"\s+", (s or "").strip()) if w])


def _normalize_one_line(s: str) -> str:
    t = unicodedata.normalize("NFKD", s or "").strip()
    return re.sub(r"\s+", " ", t)


class BridgeBuilder:
    """Costruisce un nodo bridge tra due item su scale diverse. Stateless."""

    def __init__(
        self,
        client: LLMClient,
        *,
        llm_calls_dir: Path | None,
        telemetry_path: Path | None = None,
    ) -> None:
        self.agent = RoleAgent(
            client,
            role_name="L5_BridgeBuilder",
            role_prompt=BRIDGE_PROMPT,
            out_dir=llm_calls_dir,
            max_output_tokens=budget("bridge"),
        )
        self.telemetry_path = telemetry_path

    def build(
        self,
        source: ClassifiedItem,
        target: ClassifiedItem,
        gap_scale: str,
        *,
        trace: list[str],
        source_input_id: str = "input_001",
    ) -> BridgeRecord:
        """Costruisce un BridgeRecord. NON integra nel ft (lo fa integrate_bridge)."""
        # validazioni dure
        if not is_valid_scale(gap_scale):
            trace.append(f"L5_BridgeBuilder: invalid gap_scale={gap_scale!r}, skip")
            return BridgeRecord(
                source_item_id=source.id,
                target_item_id=target.id,
                gap_scale=gap_scale,
                bridge_item=_placeholder_bridge(source, target, gap_scale, source_input_id),
                mechanism_reasoning="invalid_gap_scale",
                degraded=True,
            )
        if source.scale == target.scale:
            trace.append(f"L5_BridgeBuilder: same scale ({source.scale}), no cross-scale bridge needed")
            return BridgeRecord(
                source_item_id=source.id,
                target_item_id=target.id,
                gap_scale=gap_scale,
                bridge_item=_placeholder_bridge(source, target, gap_scale, source_input_id),
                mechanism_reasoning="source_and_target_on_same_scale",
                degraded=True,
            )

        payload = {
            "task": "build_bridge_between_two_items",
            "source": {
                "id": source.id,
                "text": source.quote or source.metadata.get("generated_text", ""),
                "scale": source.scale,
                "nature": source.nature.value,
            },
            "target": {
                "id": target.id,
                "text": target.quote or target.metadata.get("generated_text", ""),
                "scale": target.scale,
                "nature": target.nature.value,
            },
            "gap_scale": gap_scale,
            "scales_allowed": SCALES_CANONICAL,
        }
        raw, _meta = self.agent.run_json(
            payload, BRIDGE_CONTRACT, trace, telemetry_path=self.telemetry_path
        )
        bridge_raw = raw.get("bridge") if isinstance(raw, dict) else None
        if not isinstance(bridge_raw, dict):
            trace.append(f"L5_BridgeBuilder: no bridge in output for {source.id}->{target.id} -> degraded")
            return BridgeRecord(
                source_item_id=source.id,
                target_item_id=target.id,
                gap_scale=gap_scale,
                bridge_item=_placeholder_bridge(source, target, gap_scale, source_input_id),
                mechanism_reasoning="llm_returned_no_bridge",
                degraded=True,
            )

        # parsing
        text = _normalize_one_line(str(bridge_raw.get("text") or ""))
        if not text or _word_count(text) > MAX_BRIDGE_WORDS:
            trace.append(
                f"L5_BridgeBuilder: bridge text invalid (len={_word_count(text)}) -> degraded"
            )
            return BridgeRecord(
                source_item_id=source.id,
                target_item_id=target.id,
                gap_scale=gap_scale,
                bridge_item=_placeholder_bridge(source, target, gap_scale, source_input_id),
                mechanism_reasoning="bridge_text_invalid",
                degraded=True,
            )
        scale_out = str(bridge_raw.get("scale") or "").strip().lower()
        if scale_out != gap_scale:
            # forziamo la gap_scale: il bridge VIVE su quella scala per definizione
            scale_out = gap_scale
        predicate = _coerce_predicate(bridge_raw.get("predicate"))
        reasoning = str(bridge_raw.get("reasoning") or "")[:480]
        try:
            confidence = float(bridge_raw.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        bridge_id = "brg_" + stable_hash(f"{source.id}|{target.id}|{gap_scale}|{text}", 10)
        bridge_item = ClassifiedItem(
            id=bridge_id,
            quote="",  # niente quote: il bridge NON viene dal testo
            predicate=predicate,
            nature=Nature.BRIDGE,
            scale=gap_scale,
            rationale=reasoning,
            source_input_id=source_input_id,
            epistemic_status=EpistemicStatus.CAUSAL_MODEL,
            metadata={
                "origin": "bridge",
                "bridge_source": source.id,
                "bridge_target": target.id,
                "gap_scale": gap_scale,
                "confidence": confidence,
                "generated_text": text,
            },
        )

        trace.append(
            f"L5_BridgeBuilder: built bridge {bridge_id} on {gap_scale} "
            f"({source.id}->{target.id}) conf={confidence:.2f}"
        )

        return BridgeRecord(
            source_item_id=source.id,
            target_item_id=target.id,
            gap_scale=gap_scale,
            bridge_item=bridge_item,
            mechanism_reasoning=reasoning,
            degraded=False,
        )


def integrate_bridge(ft: FractalTriadResult, record: BridgeRecord) -> BridgeRecord:
    """Applica un BridgeRecord al ft. Modifica ft IN-PLACE."""
    if record.degraded:
        ft.bridges.append(record)
        return record

    source = _find_item(ft.items, record.source_item_id)
    target = _find_item(ft.items, record.target_item_id)
    if source is None or target is None:
        record.degraded = True
        record.mechanism_reasoning += " | source_or_target_not_in_ft"
        ft.bridges.append(record)
        return record

    # 1. aggiungi il bridge_item
    ft.items.append(record.bridge_item)

    # 2. due cross-scale hypotheses uncertain: source -> bridge, bridge -> target
    conf = float(record.bridge_item.metadata.get("confidence", 0.0))
    hyp_src_to_bridge = CrossScaleHypothesis(
        id="csh_brg_" + stable_hash(f"{source.id}|{record.bridge_item.id}", 10),
        cause_item_id=source.id,
        effect_item_id=record.bridge_item.id,
        cause_scale=source.scale,
        effect_scale=record.bridge_item.scale,
        verdict="uncertain",
        reasoning=f"[bridge_in] {record.mechanism_reasoning}",
        confidence=conf,
    )
    hyp_bridge_to_tgt = CrossScaleHypothesis(
        id="csh_brg_" + stable_hash(f"{record.bridge_item.id}|{target.id}", 10),
        cause_item_id=record.bridge_item.id,
        effect_item_id=target.id,
        cause_scale=record.bridge_item.scale,
        effect_scale=target.scale,
        verdict="uncertain",
        reasoning=f"[bridge_out] {record.mechanism_reasoning}",
        confidence=conf,
    )
    ft.cross_scale.append(hyp_src_to_bridge)
    ft.cross_scale.append(hyp_bridge_to_tgt)
    record.cross_scale_added = [hyp_src_to_bridge, hyp_bridge_to_tgt]

    ft.bridges.append(record)
    return record


# -----------------------------------------------------------------------------
# Helpers privati
# -----------------------------------------------------------------------------


def _find_item(items, item_id: str) -> ClassifiedItem | None:
    for it in items:
        if it.id == item_id:
            return it
    return None


def _coerce_predicate(value: Any) -> PredicateType:
    s = str(value or "").strip().lower()
    for pt in PredicateType:
        if pt.value == s:
            return pt
    return PredicateType.UNKNOWN


def _placeholder_bridge(
    source: ClassifiedItem, target: ClassifiedItem, gap_scale: str, source_input_id: str
) -> ClassifiedItem:
    """Bridge placeholder usato quando l'espansione fallisce. Mai integrato."""
    return ClassifiedItem(
        id="brg_placeholder",
        quote="",
        predicate=PredicateType.UNKNOWN,
        nature=Nature.BRIDGE,
        scale=gap_scale if is_valid_scale(gap_scale) else "fondamentale",
        rationale="placeholder",
        source_input_id=source_input_id,
        epistemic_status=EpistemicStatus.SPECULATIVE_EXTENSION,
        metadata={"origin": "bridge_placeholder", "degraded": True},
    )
