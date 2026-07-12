"""L3.B -- Cross-Scale Validator.

Riceve gli orphans da L2 (item che non hanno trovato pair sulla loro scala).
Per ogni coppia (cause su scala_a, effect su scala_b != scala_a) candidata,
una chiamata LLM decide: genuine | spurious | uncertain, con reasoning.

Niente prossimita' testuale. Niente promozione automatica. Solo reasoning.

Per evitare l'esplosione combinatoria: max N candidati totali (default 8),
selezionati per distanza di scala crescente (le distanze piu' brevi prima).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .ft_model import (
    ClassifiedItem,
    CrossScaleHypothesis,
    LockedScaleReport,
    Nature,
    scale_distance,
)
from .llm import LLMClient, RoleAgent
from .text import stable_hash
from .ft_budget import budget


VALIDATOR_PROMPT = """Sei il CROSS-SCALE VALIDATOR.

Ti vengono presentate coppie (cause su scala_A, effect su scala_B) dove le
due scale sono DIVERSE. Per ogni coppia, decidi se il legame causale e':
  - "genuine": esiste un meccanismo plausibile che attraversa le scale;
  - "spurious": prossimita' testuale o tematica, non causalita';
  - "uncertain": insufficiente per decidere senza altre prove.

REGOLE
1. RAGIONA. Niente verdetti senza un reasoning esplicito.
2. La distanza di scala e' un segnale ma non un verdetto. Distanza 1 puo' essere
   spurious, distanza 5 puo' essere genuine.
3. Se la cause e' una definizione o una proprieta' dichiarata, e' molto
   probabile che NON sia causale, anche se la coppia "suona" plausibile.
4. confidence in [0,1]: quanto sei sicuro del verdetto, non del legame.

OUTPUT: JSON con campo `verdicts`, una entry per ogni candidato.
"""


def _contract() -> dict[str, Any]:
    return {
        "verdicts": [
            {
                "candidate_id": "<id passato in input>",
                "verdict": "genuine|spurious|uncertain",
                "reasoning": "<perche'>",
                "confidence": 0.0,
            }
        ]
    }


def _collect_orphan_items(
    items: list[ClassifiedItem],
    locked_reports: list[LockedScaleReport],
) -> tuple[list[ClassifiedItem], list[ClassifiedItem]]:
    """Restituisce (orphan_causes, orphan_effects)."""
    items_by_id = {it.id: it for it in items}
    orphan_ids: set[str] = set()
    for rep in locked_reports:
        for orph in rep.orphans:
            orphan_ids.add(orph.item_id)
    orph_items = [items_by_id[i] for i in orphan_ids if i in items_by_id]
    causes = [it for it in orph_items if it.nature == Nature.CAUSE]
    effects = [it for it in orph_items if it.nature == Nature.EFFECT]
    return causes, effects


def _select_candidates(
    causes: list[ClassifiedItem],
    effects: list[ClassifiedItem],
    max_candidates: int,
) -> list[tuple[ClassifiedItem, ClassifiedItem, int]]:
    """Genera coppie cross-scale (scala diversa) ordinate per distanza crescente."""
    pairs: list[tuple[ClassifiedItem, ClassifiedItem, int]] = []
    for c in causes:
        for e in effects:
            d = scale_distance(c.scale, e.scale)
            if d <= 0:
                continue  # same-scale: gia' visto da L2
            pairs.append((c, e, d))
    pairs.sort(key=lambda t: t[2])
    return pairs[:max_candidates]


class CrossScaleValidator:
    """L3.B -- valida o respinge ipotesi cross-scale via reasoning LLM."""

    def __init__(
        self,
        client: LLMClient,
        *,
        llm_calls_dir: Path | None,
        telemetry_path: Path | None = None,
        max_candidates: int = 8,
    ) -> None:
        self.client = client
        self.llm_calls_dir = llm_calls_dir
        self.telemetry_path = telemetry_path
        self.max_candidates = max_candidates

    def run(
        self,
        items: list[ClassifiedItem],
        locked_reports: list[LockedScaleReport],
        trace: list[str],
    ) -> list[CrossScaleHypothesis]:
        causes, effects = _collect_orphan_items(items, locked_reports)
        if not causes or not effects:
            trace.append(
                f"L3B_CrossScaleValidator: no_candidates causes={len(causes)} effects={len(effects)}"
            )
            return []

        candidates = _select_candidates(causes, effects, self.max_candidates)
        if not candidates:
            trace.append("L3B_CrossScaleValidator: only_same_scale_pairs -> skip")
            return []

        payload_candidates = []
        candidate_index: dict[str, tuple[ClassifiedItem, ClassifiedItem, int]] = {}
        for cause, effect, dist in candidates:
            cand_id = "cnd_" + stable_hash(f"{cause.id}|{effect.id}", 8)
            candidate_index[cand_id] = (cause, effect, dist)
            payload_candidates.append(
                {
                    "candidate_id": cand_id,
                    "cause": {
                        "id": cause.id,
                        "quote": cause.quote,
                        "predicate": cause.predicate.value,
                        "scale": cause.scale,
                    },
                    "effect": {
                        "id": effect.id,
                        "quote": effect.quote,
                        "predicate": effect.predicate.value,
                        "scale": effect.scale,
                    },
                    "scale_distance": dist,
                }
            )

        agent = RoleAgent(
            self.client,
            role_name="L3B_CrossScaleValidator",
            role_prompt=VALIDATOR_PROMPT,
            out_dir=self.llm_calls_dir,
            max_output_tokens=budget("l3b_crossscale_rilevatore"),
        )
        raw, meta = agent.run_json(
            {"candidates": payload_candidates},
            _contract(),
            trace,
            telemetry_path=self.telemetry_path,
        )
        verdicts_raw = raw.get("verdicts") if isinstance(raw, dict) else None
        if not isinstance(verdicts_raw, list):
            trace.append("L3B_CrossScaleValidator: no_verdicts_list -> empty result")
            return []

        out: list[CrossScaleHypothesis] = []
        for v in verdicts_raw:
            if not isinstance(v, dict):
                continue
            cand_id = str(v.get("candidate_id") or "").strip()
            if cand_id not in candidate_index:
                continue
            verdict = str(v.get("verdict") or "").strip().lower()
            if verdict not in {"genuine", "spurious", "uncertain"}:
                verdict = "uncertain"
            try:
                conf = float(v.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            cause, effect, _ = candidate_index[cand_id]
            out.append(
                CrossScaleHypothesis(
                    id="csh_" + stable_hash(cand_id, 10),
                    cause_item_id=cause.id,
                    effect_item_id=effect.id,
                    cause_scale=cause.scale,
                    effect_scale=effect.scale,
                    verdict=verdict,
                    reasoning=str(v.get("reasoning") or "")[:400],
                    confidence=conf,
                )
            )

        counts = {"genuine": 0, "spurious": 0, "uncertain": 0}
        for h in out:
            counts[h.verdict] = counts.get(h.verdict, 0) + 1
        trace.append(
            f"L3B_CrossScaleValidator: candidates={len(candidates)} verdicts="
            f"genuine={counts['genuine']} spurious={counts['spurious']} uncertain={counts['uncertain']}"
        )
        return out

    def run_on_hypotheses(
        self,
        hypotheses: list[CrossScaleHypothesis],
        items: list[ClassifiedItem],
        trace: list[str],
    ) -> list[CrossScaleHypothesis]:
        """Rivaluta ipotesi cross-scale GIA' ESISTENTI (es. generate da
        espansione o bridge), senza ricalcolarle da zero.

        A differenza di run(), che parte dagli items orfani e costruisce
        nuove coppie, questo metodo prende le ipotesi cosi' come sono e
        chiede all'LLM solo il verdetto aggiornato. Ritorna una NUOVA lista
        di ipotesi con verdict/reasoning/confidence aggiornati; gli id sono
        preservati (cosi' i record di espansione/bridge restano coerenti).
        """
        if not hypotheses:
            trace.append("L3B_revalidate: nessuna ipotesi in input")
            return []

        items_by_id = {it.id: it for it in items}

        payload_candidates = []
        hyp_by_candidate: dict[str, CrossScaleHypothesis] = {}
        for h in hypotheses:
            cause = items_by_id.get(h.cause_item_id)
            effect = items_by_id.get(h.effect_item_id)
            # testo: quote se text_observed, altrimenti generated_text
            def _txt(it: ClassifiedItem | None) -> str:
                if it is None:
                    return ""
                if it.quote:
                    return it.quote
                return str((it.metadata or {}).get("generated_text", ""))

            payload_candidates.append({
                "candidate_id": h.id,  # riusiamo l'id dell'ipotesi
                "cause": {
                    "id": h.cause_item_id,
                    "quote": _txt(cause),
                    "predicate": cause.predicate.value if cause else "unknown",
                    "scale": h.cause_scale,
                },
                "effect": {
                    "id": h.effect_item_id,
                    "quote": _txt(effect),
                    "predicate": effect.predicate.value if effect else "unknown",
                    "scale": h.effect_scale,
                },
                "scale_distance": scale_distance(h.cause_scale, h.effect_scale),
                "prior_reasoning": h.reasoning[:200],
            })
            hyp_by_candidate[h.id] = h

        agent = RoleAgent(
            self.client,
            role_name="L3B_CrossScaleValidator",
            role_prompt=VALIDATOR_PROMPT,
            out_dir=self.llm_calls_dir,
            max_output_tokens=budget("l3b_crossscale_validator"),
        )
        raw, _meta = agent.run_json(
            {"candidates": payload_candidates},
            _contract(),
            trace,
            telemetry_path=self.telemetry_path,
        )
        verdicts_raw = raw.get("verdicts") if isinstance(raw, dict) else None
        if not isinstance(verdicts_raw, list):
            trace.append("L3B_revalidate: no_verdicts_list -> ipotesi invariate (uncertain)")
            return list(hypotheses)  # invariate, non perse

        updated_by_id: dict[str, CrossScaleHypothesis] = {}
        for v in verdicts_raw:
            if not isinstance(v, dict):
                continue
            cand_id = str(v.get("candidate_id") or "").strip()
            base = hyp_by_candidate.get(cand_id)
            if base is None:
                continue
            verdict = str(v.get("verdict") or "").strip().lower()
            if verdict not in {"genuine", "spurious", "uncertain"}:
                verdict = "uncertain"
            try:
                conf = float(v.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            updated_by_id[cand_id] = CrossScaleHypothesis(
                id=base.id,
                cause_item_id=base.cause_item_id,
                effect_item_id=base.effect_item_id,
                cause_scale=base.cause_scale,
                effect_scale=base.effect_scale,
                verdict=verdict,
                reasoning=str(v.get("reasoning") or base.reasoning)[:400],
                confidence=conf,
            )

        # Ogni ipotesi non coperta dall'output resta com'era (non si perde)
        out: list[CrossScaleHypothesis] = []
        for h in hypotheses:
            out.append(updated_by_id.get(h.id, h))

        counts = {"genuine": 0, "spurious": 0, "uncertain": 0}
        for h in out:
            counts[h.verdict] = counts.get(h.verdict, 0) + 1
        trace.append(
            f"L3B_revalidate: evaluated={len(hypotheses)} verdicts="
            f"genuine={counts['genuine']} spurious={counts['spurious']} uncertain={counts['uncertain']}"
        )
        return out
