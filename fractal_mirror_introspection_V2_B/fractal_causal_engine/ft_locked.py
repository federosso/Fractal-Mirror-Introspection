"""L2 -- Locked Observers per scala.

Per ogni scala canonica in cui sono presenti item, valuta i legami same-scale
tra cause ed effetti, e segnala gli orfani (item senza pair sulla stessa scala).

Principio: la causalita' va validata SOLO same-scale. Le coppie cross-scale
non sono compito di L2; vengono inoltrate a L3.B come orfani.

Una chiamata LLM per scala "popolata" (almeno una cause + un effect, oppure
almeno 2 item che giustifichino una valutazione). Niente prompt cross-scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .ft_model import (
    ClassifiedItem,
    LockedScaleReport,
    Nature,
    Orphan,
    SameScaleLink,
)
from .llm import LLMClient, RoleAgent
from .text import stable_hash
from .ft_budget import budget


LOCKED_PROMPT = """Sei un LOCKED OBSERVER per la scala {scale}.

Vedi SOLO item che vivono alla scala {scale}. Non guardi altre scale.

Il tuo compito: tra gli item che hai, decidere se esiste un legame causale
genuino tra una cause e un effect sulla MEDESIMA scala. Niente cross-scale.

REGOLE
1. Un legame valido richiede: una cause e un effect entrambi su scala {scale}.
2. Definizioni (nature=context) NON sono cause, anche se compaiono prima di un effect.
3. Proprieta' dichiarate (nature=context o effect) NON sono cause: sono attributi.
4. Una sequenza testuale NON implica causalita'. Se non c'e' un meccanismo
   plausibile sulla stessa scala, il legame non e' valido.
5. Se un item non trova partner sulla stessa scala, segnalalo come orphan.
   Sara' L3.B a valutare se ha senso cross-scale.

OUTPUT
- same_scale_links: lista di legami validi (cause_item_id, effect_item_id).
- orphans: item che restano senza pair sulla stessa scala.

Restituisci JSON conforme al contratto. Niente prosa fuori dal JSON.
"""


def _contract() -> dict[str, Any]:
    return {
        "same_scale_links": [
            {
                "cause_item_id": "<id>",
                "effect_item_id": "<id>",
                "rationale": "<breve>",
                "confidence": 0.0,
            }
        ],
        "orphans": [
            {
                "item_id": "<id>",
                "reason": "<breve>",
            }
        ],
        "summary": "<breve, opzionale>",
    }


class LockedPerScale:
    """L2 -- coordina gli osservatori locked, uno per scala popolata."""

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

    def run(self, items: list[ClassifiedItem], trace: list[str]) -> list[LockedScaleReport]:
        # raggruppa per scala
        by_scale: dict[str, list[ClassifiedItem]] = {}
        for it in items:
            by_scale.setdefault(it.scale, []).append(it)

        reports: list[LockedScaleReport] = []
        for scale, items_in_scale in sorted(by_scale.items(), key=lambda kv: kv[0]):
            report = self._observe_scale(scale, items_in_scale, trace)
            reports.append(report)
        trace.append(f"L2_LockedPerScale: scales_processed={len(reports)}")
        return reports

    def _observe_scale(
        self,
        scale: str,
        items: list[ClassifiedItem],
        trace: list[str],
    ) -> LockedScaleReport:
        causes = [it for it in items if it.nature == Nature.CAUSE]
        effects = [it for it in items if it.nature == Nature.EFFECT]

        # Caso degenere: nessuna possibilita' di legame -> tutti orfani.
        if not causes or not effects:
            orphans = [
                Orphan(item_id=it.id, nature=it.nature, scale=it.scale, reason="no_pair_at_scale")
                for it in items
            ]
            trace.append(
                f"L2_LockedPerScale[{scale}]: items={len(items)} no_pair -> orphans={len(orphans)}"
            )
            return LockedScaleReport(
                scale=scale,
                same_scale_links=[],
                orphans=orphans,
                items_seen=[it.id for it in items],
                summary="Nessuna coppia cause-effect sulla scala.",
            )

        agent = RoleAgent(
            self.client,
            role_name=f"L2_Locked_{scale}",
            role_prompt=LOCKED_PROMPT.format(scale=scale),
            out_dir=self.llm_calls_dir,
            max_output_tokens=budget("l2_locked"),
        )
        payload = {
            "scale": scale,
            "items": [
                {
                    "id": it.id,
                    "quote": it.quote,
                    "predicate": it.predicate.value,
                    "nature": it.nature.value,
                    "rationale": it.rationale,
                }
                for it in items
            ],
        }
        raw, meta = agent.run_json(payload, _contract(), trace, telemetry_path=self.telemetry_path)

        links: list[SameScaleLink] = []
        valid_ids = {it.id for it in items}
        cause_ids = {it.id for it in causes}
        effect_ids = {it.id for it in effects}
        seen_link_keys: set[tuple[str, str]] = set()

        for raw_link in (raw.get("same_scale_links") or []) if isinstance(raw, dict) else []:
            if not isinstance(raw_link, dict):
                continue
            cid = str(raw_link.get("cause_item_id") or "").strip()
            eid = str(raw_link.get("effect_item_id") or "").strip()
            if cid not in cause_ids or eid not in effect_ids:
                continue
            if (cid, eid) in seen_link_keys:
                continue
            seen_link_keys.add((cid, eid))
            try:
                conf = float(raw_link.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            link_id = "lnk_" + stable_hash(f"{scale}:{cid}:{eid}", 10)
            links.append(
                SameScaleLink(
                    id=link_id,
                    scale=scale,
                    cause_item_id=cid,
                    effect_item_id=eid,
                    rationale=str(raw_link.get("rationale") or "")[:240],
                    confidence=conf,
                )
            )

        # Orphan: chi non compare in nessun link valido.
        linked_ids = {l.cause_item_id for l in links} | {l.effect_item_id for l in links}
        declared_orphans_raw = raw.get("orphans") if isinstance(raw, dict) else None
        orphans: list[Orphan] = []
        seen_orphan: set[str] = set()
        if isinstance(declared_orphans_raw, list):
            for orph in declared_orphans_raw:
                if not isinstance(orph, dict):
                    continue
                iid = str(orph.get("item_id") or "").strip()
                if iid not in valid_ids or iid in linked_ids or iid in seen_orphan:
                    continue
                item = next(it for it in items if it.id == iid)
                orphans.append(
                    Orphan(
                        item_id=iid,
                        nature=item.nature,
                        scale=scale,
                        reason=str(orph.get("reason") or "")[:160],
                    )
                )
                seen_orphan.add(iid)
        # Aggiungi automaticamente eventuali item non linkati e non gia' marcati.
        for it in items:
            if it.id in linked_ids or it.id in seen_orphan:
                continue
            orphans.append(
                Orphan(
                    item_id=it.id,
                    nature=it.nature,
                    scale=scale,
                    reason="not_linked_at_scale",
                )
            )
            seen_orphan.add(it.id)

        summary = str(raw.get("summary") or "")[:300] if isinstance(raw, dict) else ""
        trace.append(
            f"L2_LockedPerScale[{scale}]: items={len(items)} links={len(links)} orphans={len(orphans)}"
        )
        return LockedScaleReport(
            scale=scale,
            same_scale_links=links,
            orphans=orphans,
            items_seen=[it.id for it in items],
            summary=summary,
        )
