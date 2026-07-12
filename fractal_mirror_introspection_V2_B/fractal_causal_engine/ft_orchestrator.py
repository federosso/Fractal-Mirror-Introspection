"""L4 -- Orchestrator Fractal Triad.

Riceve i risultati dei livelli precedenti e li compone:
  - double_cone: cono delle cause (scale profonde) e cono degli effetti (scale superficiali)
  - vision: core_image, human_summary, epistemic_warning (gia' prodotti da L3A5)
  - final_report.md: documento leggibile

NESSUNA chiamata LLM in L4. La sintesi creativa e' stata fatta da L3A5;
qui c'e' solo composizione e formattazione.

NESSUN quality_fallback per prossimita'. La validazione e' gia' avvenuta:
- same-scale links validati da L2
- cross-scale ipotesi ragionate da L3.B
L'orchestratore filtra solo per verdict: scarta verdetti "spurious".
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .ft_model import (
    SCALES_CANONICAL,
    SCALE_DEPTH,
    ClassifiedItem,
    CrossScaleHypothesis,
    DoubleCone,
    FractalTriadResult,
    GlobalVision,
    LockedScaleReport,
    Nature,
    UnlockedReport,
)


class FractalTriadOrchestrator:
    """L4 -- compone l'output finale. Deterministico, niente LLM."""

    def compose(
        self,
        text: str,
        items: list[ClassifiedItem],
        locked_reports: list[LockedScaleReport],
        unlocked: UnlockedReport,
        cross_scale: list[CrossScaleHypothesis],
        trace: list[str],
    ) -> FractalTriadResult:
        double_cone = self._build_double_cone(items)
        vision = self._build_vision(unlocked)
        trace.append(
            f"L4_Orchestrator: items={len(items)} locked_scales={len(locked_reports)} "
            f"cross_scale_total={len(cross_scale)} "
            f"genuine={sum(1 for h in cross_scale if h.verdict == 'genuine')}"
        )
        return FractalTriadResult(
            items=items,
            locked_reports=locked_reports,
            unlocked=unlocked,
            cross_scale=cross_scale,
            double_cone=double_cone,
            vision=vision,
            trace=list(trace),
        )

    # -------------------------------------------------------------------------
    # Double cone
    # -------------------------------------------------------------------------

    def _build_double_cone(self, items: list[ClassifiedItem]) -> DoubleCone:
        cone_c: dict[str, list[str]] = {s: [] for s in SCALES_CANONICAL}
        cone_e: dict[str, list[str]] = {s: [] for s in SCALES_CANONICAL}
        for it in items:
            if it.scale not in SCALE_DEPTH:
                continue
            if it.nature == Nature.CAUSE:
                cone_c[it.scale].append(it.id)
            elif it.nature == Nature.EFFECT:
                cone_e[it.scale].append(it.id)
        # rimuovi scale vuote per pulizia
        cone_c = {s: ids for s, ids in cone_c.items() if ids}
        cone_e = {s: ids for s, ids in cone_e.items() if ids}
        return DoubleCone(cone_of_causes=cone_c, cone_of_effects=cone_e)

    # -------------------------------------------------------------------------
    # Vision (legge la sintesi di L3A5 dal metadata)
    # -------------------------------------------------------------------------

    def _build_vision(self, unlocked: UnlockedReport) -> GlobalVision:
        syn: dict[str, Any] = {}
        meta = getattr(unlocked, "metadata", None)
        if isinstance(meta, dict):
            syn = meta.get("synthesis_raw") or {}
        return GlobalVision(
            core_image=str(syn.get("core_image") or "")[:200],
            human_summary=str(syn.get("human_summary") or "")[:1200],
            epistemic_warning=str(syn.get("epistemic_warning") or "")[:300],
            dominant_domain=str(syn.get("dominant_domain") or unlocked.domain or "")[:80],
            primary_lenses=[str(x)[:60] for x in (syn.get("primary_lenses") or [])][:6],
            blocked_lenses=[str(x)[:60] for x in (syn.get("blocked_lenses") or [])][:6],
        )


# -----------------------------------------------------------------------------
# Render: final_report.md a partire da FractalTriadResult
# -----------------------------------------------------------------------------


def render_final_report_md(ft: FractalTriadResult, original_text: str = "") -> str:
    """Rende la relazione finale in markdown.

    V10.18.3 -- modifica A: se `original_text` e' fornito, il report si apre
    con una sezione 'Testo analizzato' che ne riporta il contenuto. Un report
    deve contenere cio' che ha analizzato: serve a chi lo rilegge a distanza
    di tempo e a chi lo riceve senza avere l'input sottomano.
    """
    items_by_id = {it.id: it for it in ft.items}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# Relazione Finale -- Fractal Causal Engine V10.18.3",
        "",
        f"*Generata il: {now}*",
        "",
    ]
    # ---- 0. Testo analizzato (modifica A) ----
    if original_text and original_text.strip():
        lines += ["## 0. Testo analizzato", ""]
        # blockquote per distinguerlo nettamente dall'analisi
        for row in original_text.strip().splitlines():
            lines.append(f"> {row}" if row.strip() else ">")
        lines += [""]
    lines += [
        "## 1. Visione",
        "",
    ]
    v = ft.vision
    if v.core_image:
        lines += [f"> **Immagine centrale:** {v.core_image}", ""]
    if v.human_summary:
        lines += [v.human_summary, ""]
    if v.epistemic_warning:
        lines += [f"**Avvertenza epistemica.** {v.epistemic_warning}", ""]

    # ---- 2. Doppio cono ----
    lines += ["## 2. Doppio cono", "", "### 2.1 Cono delle cause"]
    if not ft.double_cone.cone_of_causes:
        lines += ["", "_Nessuna cause classificata._"]
    else:
        # dal piu' profondo al piu' superficiale
        for scale in sorted(ft.double_cone.cone_of_causes.keys(), key=lambda s: -SCALE_DEPTH[s]):
            ids = ft.double_cone.cone_of_causes[scale]
            lines += ["", f"**Scala `{scale}`**"]
            for iid in ids:
                it = items_by_id.get(iid)
                if it:
                    lines.append(f"- {it.quote} _(predicate={it.predicate.value})_")
    lines += ["", "### 2.2 Cono degli effetti"]
    if not ft.double_cone.cone_of_effects:
        lines += ["", "_Nessun effect classificato._"]
    else:
        for scale in sorted(ft.double_cone.cone_of_effects.keys(), key=lambda s: SCALE_DEPTH[s]):
            ids = ft.double_cone.cone_of_effects[scale]
            lines += ["", f"**Scala `{scale}`**"]
            for iid in ids:
                it = items_by_id.get(iid)
                if it:
                    lines.append(f"- {it.quote} _(predicate={it.predicate.value})_")

    # ---- 3. Legami same-scale validati (L2) ----
    lines += ["", "## 3. Legami same-scale (Locked)"]
    any_link = False
    for rep in ft.locked_reports:
        if not rep.same_scale_links:
            continue
        any_link = True
        lines += ["", f"### Scala `{rep.scale}`"]
        for link in rep.same_scale_links:
            c = items_by_id.get(link.cause_item_id)
            e = items_by_id.get(link.effect_item_id)
            if c and e:
                lines.append(
                    f"- {c.quote} → {e.quote}  _(confidence={link.confidence:.2f})_"
                )
                if link.rationale:
                    lines.append(f"  - {link.rationale}")
    if not any_link:
        lines += ["", "_Nessun legame same-scale validato dai Locked._"]

    # ---- 4. Ipotesi cross-scale ragionate (L3.B) ----
    lines += ["", "## 4. Ipotesi cross-scale"]
    if not ft.cross_scale:
        lines += ["", "_Nessuna ipotesi cross-scale prodotta._"]
    else:
        # V10.18.3 -- modifica B: deduplica. Espansioni e bridge possono
        # generare la stessa ipotesi (stessa coppia di item, stesso verdetto)
        # piu' volte; renderle tutte appesantisce il report senza aggiungere
        # informazione. Si tiene la prima occorrenza di ogni coppia.
        seen_pairs: set[tuple[str, str]] = set()
        unique_hyps = []
        for h in ft.cross_scale:
            key = (h.cause_item_id, h.effect_item_id)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            unique_hyps.append(h)
        n_dup = len(ft.cross_scale) - len(unique_hyps)
        for h in sorted(unique_hyps, key=lambda x: {"genuine": 0, "uncertain": 1, "spurious": 2}.get(x.verdict, 3)):
            c = items_by_id.get(h.cause_item_id)
            e = items_by_id.get(h.effect_item_id)
            if not (c and e):
                continue
            badge = {"genuine": "validata", "spurious": "respinta", "uncertain": "incerta"}.get(h.verdict, h.verdict)
            lines.append(
                f"- **[{badge}]** `{c.scale}` → `{e.scale}`  ({h.confidence:.2f})"
            )
            lines.append(f"  - {c.quote} → {e.quote}")
            if h.reasoning:
                lines.append(f"  - {h.reasoning}")
        if n_dup > 0:
            lines += ["", f"_({n_dup} ipotesi duplicate omesse dalla visualizzazione.)_"]

    # ---- 5. Esplorazione di dominio (L3.A) ----
    u = ft.unlocked
    lines += ["", "## 5. Esplorazione di dominio (Unlocked)"]
    if not u or (not u.domain_knowledge and not u.causal_principles and not u.cross_domain_analogies and not u.open_questions):
        lines += ["", "_Nessuna esplorazione di dominio prodotta._"]
    else:
        if u.domain and u.domain != "":
            lines += ["", f"**Dominio dominante:** `{u.domain}`"]
        if u.domain_knowledge:
            lines += ["", "### 5.1 Conoscenza di dominio"]
            for c in u.domain_knowledge:
                tag = c.status.value
                scale_tag = f" `{c.suggested_scale}`" if c.suggested_scale else ""
                lines.append(f"- **{c.concept}** `{tag}`{scale_tag} -- {c.relation_to_input}")
        if u.causal_principles:
            lines += ["", "### 5.2 Principi causali"]
            for p in u.causal_principles:
                lines.append(f"- **{p.name}** `{p.status.value}` -- {p.description}")
        if u.cross_domain_analogies:
            lines += ["", "### 5.3 Analogie cross-dominio"]
            for a in u.cross_domain_analogies:
                lines.append(f"- **{a.domain}** -- {a.analogy} `{a.status.value}`")
                if a.warning:
                    lines.append(f"  - {a.warning}")
        if u.open_questions:
            lines += ["", "### 5.4 Domande aperte"]
            for q in u.open_questions:
                lines.append(f"- {q}")
        if u.degraded:
            lines += [
                "",
                f"_Esplorazione parzialmente degradata. Parti mancanti: {', '.join(u.degraded_parts)}._",
            ]

    # ---- 6. Inventario items ----
    lines += ["", "## 6. Inventario items (probatorio)", ""]
    if not ft.items:
        lines += ["_Nessun item classificato._"]
    else:
        for it in ft.items:
            lines.append(
                f"- `{it.scale}` _{it.nature.value}_ ({it.predicate.value}) -- {it.quote}"
            )

    # ---- 7. Filtri percettivi ----
    lines += ["", "## 7. Filtri percettivi per la conversazione"]
    if v.primary_lenses:
        lines.append(f"- Lenti primarie: {', '.join(v.primary_lenses)}")
    if v.blocked_lenses:
        lines.append(f"- Lenti bloccate: {', '.join(v.blocked_lenses)}")
    if v.dominant_domain:
        lines.append(f"- Dominio dominante: `{v.dominant_domain}`")

    lines += ["", "---", "", "_Pipeline V10.14.0: L0 DomainRouter -> L1 Classifier -> L2 Locked-per-scala -> L3A UnlockedExplorer -> L3B CrossScaleValidator -> L4 Orchestrator._"]
    return "\n".join(lines) + "\n"
