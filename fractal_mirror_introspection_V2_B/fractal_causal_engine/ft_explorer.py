"""ft_explorer (V10.15.0).

Applica un ExpansionRecord (prodotto da FractalExpander) a un
FractalTriadResult esistente. Compito separato dall'expander per tenere
distinte "cosa l'LLM ha detto" e "come integriamo nel modello".

Effetti dell'applicazione di un ExpansionRecord:

  1. I figli vengono aggiunti a ft.items (con metadata origin=expansion).
  2. Per la direzione SAME_SCALE_CAUSE:
       - viene aggiunto un SameScaleLink (cause=figlio, effect=padre) al
         LockedScaleReport della scala del padre. Se il report per quella
         scala non esiste, viene creato.
       - epistemic_status del figlio resta DOMAIN_KNOWLEDGE: il link
         e' un'ipotesi non un legame validato dal testo.
  3. Per le 3 direzioni cross-scale (scale_up, scale_down, coherence_bridge):
       - viene aggiunta una CrossScaleHypothesis con verdict='uncertain'
         e reasoning = relation_to_parent del figlio. Per promuoverla a
         'genuine' bisogna invocare L3B su quella ipotesi: questo modulo
         NON promuove nulla per prossimita'.
  4. L'ExpansionRecord (con i link/cross-scale aggiunti) viene appeso a
     ft.expansions per tracciabilita'.
  5. Il double_cone viene ricalcolato da L4 (FractalTriadOrchestrator) come
     dopo ogni modifica del set di items.

Vincoli mantenuti:
- Niente "promozione per prossimita'".
- Niente bypass di Nature: e' gia' forzata dall'expander.
- Niente promozione a TEXT_OBSERVED: i figli restano DOMAIN_KNOWLEDGE o
  CAUSAL_MODEL.
"""
from __future__ import annotations

from typing import Iterable

from .ft_model import (
    SCALES_CANONICAL,
    SCALE_DEPTH,
    ClassifiedItem,
    CrossScaleHypothesis,
    DoubleCone,
    ExpansionChild,
    ExpansionDirection,
    ExpansionRecord,
    FractalTriadResult,
    LockedScaleReport,
    Nature,
    SameScaleLink,
)
from .text import stable_hash


# Mappa direzione cross-scale -> ruolo della relazione, per chiarezza nel reasoning.
_CROSS_SCALE_DIRS = {
    ExpansionDirection.SCALE_UP_PROPAGATION,
    ExpansionDirection.SCALE_DOWN_MECHANISM,
    ExpansionDirection.COHERENCE_BRIDGE,
}


def integrate_expansion(ft: FractalTriadResult, record: ExpansionRecord) -> ExpansionRecord:
    """Applica un ExpansionRecord al FractalTriadResult.

    Modifica ft IN-PLACE: aggiunge items, link, cross-scale, e appende il
    record (arricchito con i link/cross-scale generati) a ft.expansions.

    Ritorna lo stesso record, arricchito.
    """
    if record.degraded and not record.children:
        # Espansione fallita: tracciamola comunque, ma niente da integrare.
        ft.expansions.append(record)
        return record

    parent = _find_item(ft.items, record.parent_item_id)
    if parent is None:
        # Non possiamo integrare un'espansione di un item che non e' nel ft.
        record.degraded = True
        record.notes = (record.notes + ";parent_not_in_ft" if record.notes else "parent_not_in_ft")
        ft.expansions.append(record)
        return record

    new_same_scale_links: list[SameScaleLink] = []
    new_cross_scale: list[CrossScaleHypothesis] = []

    for child in record.children:
        # 1. aggiungi il figlio come nuovo item
        ft.items.append(child.item)

        # 2. genera la relazione in base alla direction
        if child.direction == ExpansionDirection.SAME_SCALE_CAUSE:
            link = _make_same_scale_link(child, parent)
            new_same_scale_links.append(link)
            _attach_link_to_locked_report(ft, link, child.item)
        elif child.direction in _CROSS_SCALE_DIRS:
            hyp = _make_cross_scale_hypothesis(child, parent)
            new_cross_scale.append(hyp)
            ft.cross_scale.append(hyp)
        # (nessun else: l'expander e' garantito a coprire solo le 4 direzioni)

    record.same_scale_links_added = new_same_scale_links
    record.cross_scale_added = new_cross_scale
    ft.expansions.append(record)

    # 3. ricalcola il double_cone con i nuovi items
    ft.double_cone = _rebuild_double_cone(ft.items)

    return record


# -----------------------------------------------------------------------------
# Helpers privati
# -----------------------------------------------------------------------------


def _find_item(items: Iterable[ClassifiedItem], item_id: str) -> ClassifiedItem | None:
    for it in items:
        if it.id == item_id:
            return it
    return None


def _make_same_scale_link(child: ExpansionChild, parent: ClassifiedItem) -> SameScaleLink:
    link_id = "lnk_exp_" + stable_hash(f"{child.item.id}|{parent.id}", 10)
    return SameScaleLink(
        id=link_id,
        scale=parent.scale,
        cause_item_id=child.item.id,
        effect_item_id=parent.id,
        rationale=child.relation_to_parent or "espansione: causa orizzontale generata",
        confidence=child.confidence,
    )


def _make_cross_scale_hypothesis(
    child: ExpansionChild, parent: ClassifiedItem
) -> CrossScaleHypothesis:
    # Orientamento sorgente/destinazione in base alla direction:
    # - scale_down_mechanism: il meccanismo (figlio, scala profonda) e' la sorgente
    # - scale_up_propagation: il padre (scala profonda) e' la sorgente, la
    #   propagazione (figlio, scala superficiale) e' la destinazione
    # - coherence_bridge: trattata simbolicamente padre -> figlio
    if child.direction == ExpansionDirection.SCALE_DOWN_MECHANISM:
        cause_id = child.item.id
        effect_id = parent.id
        cause_scale = child.item.scale
        effect_scale = parent.scale
    else:
        cause_id = parent.id
        effect_id = child.item.id
        cause_scale = parent.scale
        effect_scale = child.item.scale

    hyp_id = "csh_exp_" + stable_hash(f"{cause_id}|{effect_id}|{child.direction.value}", 10)
    return CrossScaleHypothesis(
        id=hyp_id,
        cause_item_id=cause_id,
        effect_item_id=effect_id,
        cause_scale=cause_scale,
        effect_scale=effect_scale,
        verdict="uncertain",  # NIENTE 'genuine' qui. Va validato da L3.B.
        reasoning=(
            f"[{child.direction.value}] {child.relation_to_parent}".strip(" -")
        ),
        confidence=child.confidence,
    )


def _attach_link_to_locked_report(
    ft: FractalTriadResult, link: SameScaleLink, child_item: ClassifiedItem
) -> None:
    """Trova o crea il LockedScaleReport per la scala del link, e ci aggiunge il link."""
    report = None
    for r in ft.locked_reports:
        if r.scale == link.scale:
            report = r
            break
    if report is None:
        report = LockedScaleReport(scale=link.scale, summary="report creato da espansione")
        ft.locked_reports.append(report)
    report.same_scale_links.append(link)
    if child_item.id not in report.items_seen:
        report.items_seen.append(child_item.id)


def _rebuild_double_cone(items: list[ClassifiedItem]) -> DoubleCone:
    cone_c: dict[str, list[str]] = {}
    cone_e: dict[str, list[str]] = {}
    for it in items:
        if it.scale not in SCALE_DEPTH:
            continue
        if it.nature == Nature.CAUSE:
            cone_c.setdefault(it.scale, []).append(it.id)
        elif it.nature == Nature.EFFECT:
            cone_e.setdefault(it.scale, []).append(it.id)
    return DoubleCone(cone_of_causes=cone_c, cone_of_effects=cone_e)
