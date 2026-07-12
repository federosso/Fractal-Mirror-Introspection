"""Test smoke V10.15.0.

Verifica i contratti dei 3 nuovi moduli e dell'integrazione:
  - FractalExpander produce esattamente 4 figli con direzioni coerenti;
  - integrate_expansion produce SameScaleLink per same_scale_cause e
    CrossScaleHypothesis 'uncertain' per le 3 cross-scale;
  - BridgeBuilder produce un bridge sulla gap_scale richiesta, con
    nature=BRIDGE e epistemic_status=CAUSAL_MODEL;
  - integrate_bridge aggiunge 2 cross-scale 'uncertain' (in/out);
  - MagistraleReportBuilder produce un MagistraleReport non vuoto e i
    coni vengono popolati dai dati esistenti senza nuove validazioni;
  - ExplorerSession round-trip (serialize/deserialize) preserva i dati.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fractal_causal_engine.ft_bridge import BridgeBuilder, integrate_bridge
from fractal_causal_engine.ft_expander import FractalExpander
from fractal_causal_engine.ft_explorer import integrate_expansion
from fractal_causal_engine.ft_magistrale import MagistraleReportBuilder, render_magistrale_md
from fractal_causal_engine.ft_model import (
    SCALE_DEPTH,
    ClassifiedItem,
    EpistemicStatus,
    ExpansionDirection,
    FractalTriadResult,
    Nature,
    PredicateType,
)
from fractal_causal_engine.ft_session import ExplorerSession, deserialize_ft, serialize_ft
from fractal_causal_engine.llm import LLMClient, LLMConfig


def _client() -> LLMClient:
    return LLMClient(LLMConfig(mock=True))


def _seed_ft() -> FractalTriadResult:
    ft = FractalTriadResult()
    parent = ClassifiedItem(
        id="itm_parent",
        quote="tamponamento al semaforo",
        predicate=PredicateType.EVENT,
        nature=Nature.CAUSE,
        scale="organismo",
        epistemic_status=EpistemicStatus.TEXT_OBSERVED,
    )
    other = ClassifiedItem(
        id="itm_other",
        quote="colpo di frusta",
        predicate=PredicateType.EVENT,
        nature=Nature.EFFECT,
        scale="atomico",
        epistemic_status=EpistemicStatus.TEXT_OBSERVED,
    )
    ft.items.extend([parent, other])
    return ft


# -----------------------------------------------------------------------------
# Expander
# -----------------------------------------------------------------------------


def test_expander_produces_four_children_with_distinct_directions(tmp_path: Path) -> None:
    ft = _seed_ft()
    expander = FractalExpander(_client(), llm_calls_dir=tmp_path)
    trace: list[str] = []
    record = expander.expand(ft.items[0], original_text="testo originale", trace=trace)

    assert record.degraded is False
    assert len(record.children) == 4
    dirs = [c.direction for c in record.children]
    assert set(dirs) == set(ExpansionDirection)

    # vincolo nature forzata per direzione
    nature_by_dir = {c.direction: c.item.nature for c in record.children}
    assert nature_by_dir[ExpansionDirection.SAME_SCALE_CAUSE] == Nature.CAUSE
    assert nature_by_dir[ExpansionDirection.SCALE_UP_PROPAGATION] == Nature.EFFECT
    assert nature_by_dir[ExpansionDirection.SCALE_DOWN_MECHANISM] == Nature.BRIDGE
    assert nature_by_dir[ExpansionDirection.COHERENCE_BRIDGE] == Nature.BRIDGE


def test_expander_children_are_not_text_observed(tmp_path: Path) -> None:
    """I figli dell'expander non sono fatti dal testo: epistemic != TEXT_OBSERVED."""
    ft = _seed_ft()
    expander = FractalExpander(_client(), llm_calls_dir=tmp_path)
    record = expander.expand(ft.items[0], original_text="testo", trace=[])
    for child in record.children:
        assert child.item.epistemic_status != EpistemicStatus.TEXT_OBSERVED
        assert child.item.quote == ""
        assert child.item.metadata.get("origin") == "expansion"


def test_expander_respects_scale_axis(tmp_path: Path) -> None:
    """scale_up ha indice piu' basso, scale_down ha indice piu' alto del padre."""
    ft = _seed_ft()
    expander = FractalExpander(_client(), llm_calls_dir=tmp_path)
    record = expander.expand(ft.items[0], original_text="testo", trace=[])
    parent_depth = SCALE_DEPTH["organismo"]
    by_dir = {c.direction: c.item.scale for c in record.children}
    assert SCALE_DEPTH[by_dir[ExpansionDirection.SAME_SCALE_CAUSE]] == parent_depth
    assert SCALE_DEPTH[by_dir[ExpansionDirection.SCALE_UP_PROPAGATION]] < parent_depth
    assert SCALE_DEPTH[by_dir[ExpansionDirection.SCALE_DOWN_MECHANISM]] > parent_depth


# -----------------------------------------------------------------------------
# Explorer integration
# -----------------------------------------------------------------------------


def test_integrate_expansion_produces_same_scale_link_and_uncertain_cross_scale(tmp_path: Path) -> None:
    ft = _seed_ft()
    parent_id = ft.items[0].id
    expander = FractalExpander(_client(), llm_calls_dir=tmp_path)
    record = expander.expand(ft.items[0], original_text="testo", trace=[])

    integrate_expansion(ft, record)

    # 4 nuovi items aggiunti
    assert len(ft.items) == 2 + 4
    # 1 same-scale link
    org_report = next(r for r in ft.locked_reports if r.scale == "organismo")
    assert len(org_report.same_scale_links) == 1
    same_link = org_report.same_scale_links[0]
    assert same_link.effect_item_id == parent_id  # il padre e' l'effetto
    # 3 cross-scale 'uncertain'
    assert len(ft.cross_scale) == 3
    for h in ft.cross_scale:
        assert h.verdict == "uncertain"


def test_integrate_expansion_updates_double_cone(tmp_path: Path) -> None:
    ft = _seed_ft()
    expander = FractalExpander(_client(), llm_calls_dir=tmp_path)
    record = expander.expand(ft.items[0], original_text="testo", trace=[])
    integrate_expansion(ft, record)
    # cono cause include la same_scale_cause su organismo
    assert "organismo" in ft.double_cone.cone_of_causes
    assert len(ft.double_cone.cone_of_causes["organismo"]) >= 2  # padre + figlio cause


# -----------------------------------------------------------------------------
# Bridge
# -----------------------------------------------------------------------------


def test_bridge_builder_produces_bridge_on_gap_scale(tmp_path: Path) -> None:
    ft = _seed_ft()
    builder = BridgeBuilder(_client(), llm_calls_dir=tmp_path)
    src, tgt = ft.items[0], ft.items[1]  # organismo -> atomico
    record = builder.build(src, tgt, "molecolare", trace=[])

    assert record.degraded is False
    assert record.bridge_item.scale == "molecolare"
    assert record.bridge_item.nature == Nature.BRIDGE
    assert record.bridge_item.epistemic_status == EpistemicStatus.CAUSAL_MODEL
    assert record.bridge_item.metadata.get("origin") == "bridge"


def test_bridge_builder_rejects_same_scale(tmp_path: Path) -> None:
    """Se source e target sono sulla stessa scala, il bridge e' degraded."""
    ft = _seed_ft()
    # cambio scala del second item a "organismo" come il primo
    ft.items[1].scale = "organismo"
    builder = BridgeBuilder(_client(), llm_calls_dir=tmp_path)
    record = builder.build(ft.items[0], ft.items[1], "molecolare", trace=[])
    assert record.degraded is True


def test_integrate_bridge_adds_two_uncertain_cross_scale(tmp_path: Path) -> None:
    ft = _seed_ft()
    builder = BridgeBuilder(_client(), llm_calls_dir=tmp_path)
    record = builder.build(ft.items[0], ft.items[1], "molecolare", trace=[])
    integrate_bridge(ft, record)

    assert len(ft.items) == 3  # i 2 originali + bridge
    assert len(ft.cross_scale) == 2  # in + out
    for h in ft.cross_scale:
        assert h.verdict == "uncertain"


# -----------------------------------------------------------------------------
# Magistrale
# -----------------------------------------------------------------------------


def test_magistrale_builder_populates_cones_from_existing_items(tmp_path: Path) -> None:
    ft = _seed_ft()
    # Aggiungiamo un'interpretazione e un'espansione per coprire piu' campi
    ft.items.append(ClassifiedItem(
        id="itm_interp",
        quote="lettura simbolica",
        predicate=PredicateType.CLAIMED_PROPERTY,
        nature=Nature.INTERPRETATION,
        scale="sociale",
        epistemic_status=EpistemicStatus.SPECULATIVE_EXTENSION,
    ))
    builder = MagistraleReportBuilder(_client(), llm_calls_dir=tmp_path)
    report = builder.build(ft, trace=[])

    assert report.degraded is False
    assert report.sintesi_magistrale
    # i triggers dovrebbero includere il tamponamento (nature=cause, predicate=event)
    assert any("tamponamento" in t for t in report.cono_cause.triggers)
    # gli effetti diretti dovrebbero includere "colpo di frusta"
    assert any("colpo di frusta" in t for t in report.cono_effetti.direct_effects)
    # propagazione multi-scala deve menzionare le scale presenti
    assert "organismo" in report.propagazione_multi_scala
    assert "atomico" in report.propagazione_multi_scala


def test_magistrale_renders_markdown_with_all_sections(tmp_path: Path) -> None:
    ft = _seed_ft()
    builder = MagistraleReportBuilder(_client(), llm_calls_dir=tmp_path)
    report = builder.build(ft, trace=[])
    md = render_magistrale_md(report)
    for section in [
        "# Relazione Magistrale",
        "## Sintesi",
        "## Cono delle Cause",
        "### Predisposizioni",
        "### Trigger",
        "### Cause prossime",
        "### Meccanismi di ponte",
        "## Cono degli Effetti",
        "## Propagazione multi-scala",
        "## Stato epistemico",
        "## Verdetto finale",
    ]:
        assert section in md, f"manca sezione: {section!r}"


# -----------------------------------------------------------------------------
# Session round-trip
# -----------------------------------------------------------------------------


def test_session_round_trip_preserves_items_and_expansion(tmp_path: Path) -> None:
    ft = _seed_ft()
    expander = FractalExpander(_client(), llm_calls_dir=tmp_path)
    record = expander.expand(ft.items[0], original_text="testo", trace=[])
    integrate_expansion(ft, record)
    bridge_builder = BridgeBuilder(_client(), llm_calls_dir=tmp_path)
    brec = bridge_builder.build(ft.items[0], ft.items[1], "molecolare", trace=[])
    integrate_bridge(ft, brec)

    data = serialize_ft(ft, "testo originale", call_counter=42)
    ft2, txt, n = deserialize_ft(data)

    assert txt == "testo originale"
    assert n == 42
    assert len(ft2.items) == len(ft.items)
    assert len(ft2.expansions) == 1
    assert len(ft2.bridges) == 1
    # verifichiamo gli enum sono stati riportati a oggetti enum, non stringhe
    for it in ft2.items:
        assert isinstance(it.nature, Nature)
        assert isinstance(it.predicate, PredicateType)
        assert isinstance(it.epistemic_status, EpistemicStatus)


def test_session_full_workflow(tmp_path: Path) -> None:
    """End-to-end: open -> expand -> bridge -> revalidate -> magistrale."""
    client = _client()
    # input testuale minimo
    input_text = "Un automobilista viene tamponato; riporta un lieve colpo di frusta."
    text_file = tmp_path / "input.txt"
    text_file.write_text(input_text, encoding="utf-8")

    out = tmp_path / "session_out"
    sess = ExplorerSession.analyze(client, out, input_text)
    assert (out / "session.json").exists()
    assert len(sess.ft.items) > 0

    # expand sul primo item
    first_id = sess.ft.items[0].id
    rec = sess.expand(first_id)
    assert len(rec.children) == 4

    # bridge se ci sono due item su scale diverse
    items_by_scale = {it.scale: it for it in sess.ft.items if it.metadata.get("origin", "text_observed") == "text_observed"}
    if len(items_by_scale) >= 2:
        scales = list(items_by_scale.keys())
        src = items_by_scale[scales[0]]
        tgt = items_by_scale[scales[1]]
        # scegliamo come gap una scala "intermedia" diversa
        gap = "molecolare" if "molecolare" not in (src.scale, tgt.scale) else "cellulare"
        sess.bridge(src.id, tgt.id, gap)

    # revalidate
    stats = sess.revalidate_cross(only_uncertain=True)
    assert "evaluated" in stats

    # magistrale
    report = sess.magistrale()
    assert (out / "magistrale_report.md").exists()
    assert (out / "final_report.md").exists()
    assert report.sintesi_magistrale


# -----------------------------------------------------------------------------
# Ergonomia V10.15.1: resolve_item_ref + auto_explore
# -----------------------------------------------------------------------------


def test_resolve_item_ref_by_index_prefix_and_full_id(tmp_path: Path) -> None:
    """resolve_item_ref deve accettare indice numerico, prefisso unico, o ID completo."""
    text = "Un automobilista viene tamponato; riporta colpo di frusta."
    sess = ExplorerSession.analyze(_client(), tmp_path / "out", text)
    assert sess.ft.items, "la pipeline deve produrre almeno un item"

    # indice numerico [1]
    rows = sess.list_items()
    first_id = rows[0]["id"]
    assert sess.resolve_item_ref("1").id == first_id

    # prefisso unico
    prefix = first_id[:6]
    assert sess.resolve_item_ref(prefix).id == first_id

    # ID completo
    assert sess.resolve_item_ref(first_id).id == first_id


def test_resolve_item_ref_raises_on_unknown(tmp_path: Path) -> None:
    text = "Un automobilista viene tamponato."
    sess = ExplorerSession.analyze(_client(), tmp_path / "out", text)
    with pytest.raises(KeyError):
        sess.resolve_item_ref("itm_nonesistente_xyz")
    with pytest.raises(KeyError):
        sess.resolve_item_ref("9999")  # indice fuori range


def test_auto_explore_one_shot_produces_full_session(tmp_path: Path) -> None:
    """auto_explore deve produrre espansioni, bridge, e una magistrale senza chiedere ID."""
    text = (
        "Un automobilista viene tamponato; riporta un colpo di frusta. "
        "Nei mesi successivi sviluppa una fobia della guida."
    )
    out = tmp_path / "out_auto"
    sess = ExplorerSession.analyze(_client(), out, text)
    n_before = len(sess.ft.items)

    stats = sess.auto_explore(
        expand_top_n=2, build_bridges=True, max_bridges=2,
        do_revalidate=True, do_magistrale=True,
    )

    assert stats["magistrale"] is True
    assert stats["expanded"] >= 0
    assert stats["bridges_built"] >= 0
    # ci sono nuovi items (figli + eventuali bridge)
    assert len(sess.ft.items) >= n_before
    # file finali creati
    assert (out / "magistrale_report.md").exists()
    assert (out / "final_report.md").exists()
    # la sessione e' stata salvata
    assert (out / "session.json").exists()


def test_auto_explore_no_magistrale_skips_report(tmp_path: Path) -> None:
    """Con do_magistrale=False, niente magistrale_report.md."""
    text = "Un automobilista viene tamponato; riporta colpo di frusta."
    out = tmp_path / "out_no_mag"
    sess = ExplorerSession.analyze(_client(), out, text)
    sess.auto_explore(expand_top_n=1, build_bridges=False, do_revalidate=False, do_magistrale=False)
    assert not (out / "magistrale_report.md").exists()
    assert sess.ft.magistrale is None


# -----------------------------------------------------------------------------
# V10.16.0 -- Fix #3: bridge fallback su items expansion quando text_observed
# sono monoscala. Verifica che il fallback si attivi e produca bridge reali.
# -----------------------------------------------------------------------------


def test_bridge_fallback_uses_expansion_items_when_text_observed_monoscale(tmp_path: Path) -> None:
    """Quando i text_observed sono tutti sulla stessa scala, auto_explore deve
    usare gli items expansion (che vivono su scale diverse) per costruire
    almeno un bridge cross-scale. Senza questo, i testi monoscala (es. LENR)
    producono '0 bridges'.
    """
    ft = FractalTriadResult()
    # Tutti text_observed su atomico (caso LENR)
    for i, nat in enumerate([Nature.CAUSE, Nature.EFFECT, Nature.CONTEXT]):
        ft.items.append(ClassifiedItem(
            id=f"itm_obs_{i}",
            quote=f"frase osservata {i}",
            predicate=PredicateType.PROCESS_DESCRIPTION,
            nature=nat,
            scale="atomico",
            epistemic_status=EpistemicStatus.TEXT_OBSERVED,
            metadata={"origin": "text_observed"},
        ))

    # Costruiamo una ExplorerSession a partire dal ft, senza passare per
    # ExplorerSession.analyze (che richiede l'LLM per la pipeline V14).
    out = tmp_path / "out_mono"
    out.mkdir(parents=True, exist_ok=True)
    sess = ExplorerSession(out, _client(), ft, original_text="testo monoscala", call_counter=0)

    # 1. espandi -> il mock genera figli su molecolare/cellulare/subatomico
    sess.expand("itm_obs_0")
    sess.expand("itm_obs_1")

    # ora ci sono items expansion su scale diverse
    scales = {it.scale for it in sess.ft.items
              if (it.metadata or {}).get("origin") == "expansion"}
    assert len(scales) >= 2, f"expansion items dovrebbero essere multiscala: {scales}"

    # 2. auto_explore con expand=0 e bridge attivo: deve usare il fallback
    stats = sess.auto_explore(
        expand_top_n=0, build_bridges=True, max_bridges=3,
        do_revalidate=False, do_magistrale=False,
    )
    assert stats["bridges_built"] >= 1, (
        f"il fallback su items expansion doveva produrre almeno un bridge, "
        f"invece bridges_built={stats['bridges_built']}"
    )


def test_bridge_fallback_does_not_trigger_when_text_observed_already_cross_scale(tmp_path: Path) -> None:
    """Se i text_observed coprono gia' >= 2 scale, il fallback NON deve
    attivarsi: bridge devono essere costruiti tra i text_observed (piu' affidabili).
    """
    ft = FractalTriadResult()
    ft.items.append(ClassifiedItem(
        id="itm_obs_atomico", quote="osservato atomico",
        predicate=PredicateType.EVENT, nature=Nature.CAUSE,
        scale="atomico", epistemic_status=EpistemicStatus.TEXT_OBSERVED,
        metadata={"origin": "text_observed"},
    ))
    ft.items.append(ClassifiedItem(
        id="itm_obs_sociale", quote="osservato sociale",
        predicate=PredicateType.STATE, nature=Nature.EFFECT,
        scale="sociale", epistemic_status=EpistemicStatus.TEXT_OBSERVED,
        metadata={"origin": "text_observed"},
    ))

    out = tmp_path / "out_cs"
    out.mkdir(parents=True, exist_ok=True)
    sess = ExplorerSession(out, _client(), ft, original_text="testo cross-scale", call_counter=0)

    stats = sess.auto_explore(
        expand_top_n=0, build_bridges=True, max_bridges=3,
        do_revalidate=False, do_magistrale=False,
    )
    assert stats["bridges_built"] == 1
    # il bridge generato e' tra i due text_observed: source/target hanno
    # entrambi origin='text_observed'
    assert len(sess.ft.bridges) == 1
    src_id = sess.ft.bridges[0].source_item_id
    tgt_id = sess.ft.bridges[0].target_item_id
    src = next(it for it in sess.ft.items if it.id == src_id)
    tgt = next(it for it in sess.ft.items if it.id == tgt_id)
    assert (src.metadata or {}).get("origin") == "text_observed"
    assert (tgt.metadata or {}).get("origin") == "text_observed"


def test_find_cross_scale_pairs_filters_by_distance(tmp_path: Path) -> None:
    """L'helper _find_cross_scale_pairs ritorna solo coppie con distance >= 2."""
    from fractal_causal_engine.ft_session import _find_cross_scale_pairs
    a = ClassifiedItem(id="a", quote="a", predicate=PredicateType.EVENT,
                       nature=Nature.CAUSE, scale="atomico")
    b = ClassifiedItem(id="b", quote="b", predicate=PredicateType.EVENT,
                       nature=Nature.EFFECT, scale="molecolare")  # dist 1 da atomico
    c = ClassifiedItem(id="c", quote="c", predicate=PredicateType.EVENT,
                       nature=Nature.EFFECT, scale="sociale")     # dist 4 da atomico
    pairs = _find_cross_scale_pairs([a, b, c])
    pair_ids = [(p[0].id, p[1].id) for p in pairs]
    # (a,b) dist 1: scartata. (a,c) dist 4: tenuta. (b,c) dist 3: tenuta.
    assert ("a", "b") not in pair_ids
    assert ("a", "c") in pair_ids
    assert ("b", "c") in pair_ids


# -----------------------------------------------------------------------------
# V10.16.1 -- Fix A: auto_explore espande CONTEXT quando non ci sono CAUSE/EFFECT.
# E' il caso dei testi puramente definitori (es. "Cos'e' la coscienza?").
# -----------------------------------------------------------------------------


def test_auto_explore_falls_back_on_context_when_no_cause_effect(tmp_path: Path) -> None:
    """Se gli unici text_observed sono context (testo definitorio),
    auto_explore deve comunque espandere -- usando i context come semi --
    invece di fermarsi a 0 espansioni.
    """
    ft = FractalTriadResult()
    # Solo 2 context, niente cause/effect: scenario "Cos'e' la coscienza?"
    ft.items.append(ClassifiedItem(
        id="itm_def1", quote="Che cos'e' la coscienza",
        predicate=PredicateType.DEFINITION, nature=Nature.CONTEXT,
        scale="sociale", epistemic_status=EpistemicStatus.TEXT_OBSERVED,
        metadata={"origin": "text_observed"},
    ))
    ft.items.append(ClassifiedItem(
        id="itm_def2", quote="come la si puo' descrivere",
        predicate=PredicateType.PROCESS_DESCRIPTION, nature=Nature.CONTEXT,
        scale="sociale", epistemic_status=EpistemicStatus.TEXT_OBSERVED,
        metadata={"origin": "text_observed"},
    ))

    out = tmp_path / "out_def"
    out.mkdir(parents=True, exist_ok=True)
    sess = ExplorerSession(out, _client(), ft, original_text="Che cos'e' la coscienza?", call_counter=0)

    stats = sess.auto_explore(
        expand_top_n=2, build_bridges=False,
        do_revalidate=False, do_magistrale=False,
    )
    # Senza il fallback su context, expanded=0; con il fallback, expanded=2.
    assert stats["expanded"] == 2, (
        f"auto_explore doveva espandere i context come semi, invece expanded={stats['expanded']}"
    )
    # Ogni context ha generato 4 figli (max), totale items >= 2 (osservati) + 8 (figli).
    assert len(sess.ft.items) >= 10


def test_auto_explore_prefers_cause_effect_when_available(tmp_path: Path) -> None:
    """Se ci sono cause/effect, NON deve toccare i context: i cause/effect
    sono semi piu' robusti.
    """
    ft = FractalTriadResult()
    ft.items.append(ClassifiedItem(
        id="itm_ctx", quote="contesto generale",
        predicate=PredicateType.DEFINITION, nature=Nature.CONTEXT,
        scale="sociale", epistemic_status=EpistemicStatus.TEXT_OBSERVED,
        metadata={"origin": "text_observed"},
    ))
    ft.items.append(ClassifiedItem(
        id="itm_cau", quote="trigger osservato",
        predicate=PredicateType.EVENT, nature=Nature.CAUSE,
        scale="organismo", epistemic_status=EpistemicStatus.TEXT_OBSERVED,
        metadata={"origin": "text_observed"},
    ))

    out = tmp_path / "out_mix"
    out.mkdir(parents=True, exist_ok=True)
    sess = ExplorerSession(out, _client(), ft, original_text="mixed", call_counter=0)
    sess.auto_explore(
        expand_top_n=1, build_bridges=False,
        do_revalidate=False, do_magistrale=False,
    )

    # Cerca le expansion: il parent_item_id deve essere itm_cau (cause), NON itm_ctx.
    assert len(sess.ft.expansions) == 1
    assert sess.ft.expansions[0].parent_item_id == "itm_cau"


# -----------------------------------------------------------------------------
# V10.16.2 -- Fix C: revalidate_cross NON deve perdere le ipotesi cross-scale
# generate da expand/bridge. Bug presente da V15: il vecchio revalidate
# ricalcolava da zero e azzerava ft.cross_scale.
# -----------------------------------------------------------------------------


def test_revalidate_does_not_lose_expansion_bridge_hypotheses(tmp_path: Path) -> None:
    """Dopo expand + bridge ci sono N ipotesi cross-scale 'uncertain'.
    revalidate_cross deve RIVALUTARLE (cambiando verdict) senza perderne
    nessuna. Bug storico: ne perdeva tutte.
    """
    text = (
        "Un automobilista viene tamponato; riporta un colpo di frusta. "
        "Nei mesi successivi sviluppa una fobia della guida."
    )
    out = tmp_path / "out_reval"
    sess = ExplorerSession.analyze(_client(), out, text)

    # espandi un paio di item e costruisci bridge -> genera cross_scale 'uncertain'
    sess.auto_explore(
        expand_top_n=2, build_bridges=True, max_bridges=3,
        do_revalidate=False, do_magistrale=False,
    )
    n_before = len(sess.ft.cross_scale)
    assert n_before > 0, "expand+bridge dovevano generare ipotesi cross-scale"

    # ora revalidate: il numero NON deve diminuire (le ipotesi non si perdono)
    stats = sess.revalidate_cross(only_uncertain=True)
    n_after = len(sess.ft.cross_scale)

    assert n_after == n_before, (
        f"revalidate ha perso ipotesi: prima={n_before}, dopo={n_after}"
    )
    assert stats["evaluated"] == n_before
    # la somma dei verdetti deve coprire tutte le ipotesi valutate
    assert stats["genuine"] + stats["spurious"] + stats["uncertain"] == n_before


def test_revalidate_preserves_hypothesis_ids(tmp_path: Path) -> None:
    """Gli id delle ipotesi devono essere preservati dopo la revalidate,
    altrimenti i record di espansione/bridge perdono il riferimento.
    """
    text = "Un automobilista viene tamponato; riporta colpo di frusta."
    out = tmp_path / "out_ids"
    sess = ExplorerSession.analyze(_client(), out, text)
    sess.auto_explore(
        expand_top_n=2, build_bridges=True, max_bridges=2,
        do_revalidate=False, do_magistrale=False,
    )
    ids_before = {h.id for h in sess.ft.cross_scale}
    sess.revalidate_cross(only_uncertain=True)
    ids_after = {h.id for h in sess.ft.cross_scale}
    assert ids_before == ids_after, "gli id delle ipotesi cross-scale devono restare invariati"


# -----------------------------------------------------------------------------
# V10.16.3 -- Fix #2: espansione ricorsiva multi-livello.
# -----------------------------------------------------------------------------


def test_recursive_expansion_reaches_multiple_levels(tmp_path: Path) -> None:
    """Con expand_depth=3, auto_explore deve espandere su 3 livelli:
    gli item osservati, poi i loro figli, poi i nipoti.
    """
    text = (
        "Un automobilista viene tamponato; riporta un colpo di frusta. "
        "Nei mesi successivi sviluppa una fobia della guida."
    )
    out = tmp_path / "out_rec"
    sess = ExplorerSession.analyze(_client(), out, text)
    n_items_before = len(sess.ft.items)

    stats = sess.auto_explore(
        expand_top_n=2, expand_depth=3, expand_children_per_level=2,
        build_bridges=False, do_revalidate=False, do_magistrale=False,
    )

    by_level = stats["expand_by_level"]
    # Devono esserci 3 livelli L1, L2, L3
    assert "L1" in by_level and "L2" in by_level and "L3" in by_level, (
        f"attesi 3 livelli di espansione, trovati: {list(by_level)}"
    )
    # L2 e L3 ri-espandono al massimo expand_children_per_level item
    assert by_level["L2"] <= 2
    assert by_level["L3"] <= 2
    # Con 3 livelli di espansione gli items crescono ben oltre il livello 1
    assert len(sess.ft.items) > n_items_before + 4


def test_expand_depth_1_is_single_level(tmp_path: Path) -> None:
    """Con expand_depth=1 (default) deve esserci solo il livello L1."""
    text = "Un automobilista viene tamponato; riporta colpo di frusta."
    out = tmp_path / "out_d1"
    sess = ExplorerSession.analyze(_client(), out, text)
    stats = sess.auto_explore(
        expand_top_n=2, expand_depth=1,
        build_bridges=False, do_revalidate=False, do_magistrale=False,
    )
    by_level = stats["expand_by_level"]
    assert "L1" in by_level
    assert "L2" not in by_level, "expand_depth=1 non deve produrre un livello L2"


def test_recursive_expansion_skips_coherence_bridges(tmp_path: Path) -> None:
    """Nei livelli oltre il primo, i coherence_bridge NON vanno ri-espansi
    (sono ponti, non concetti seminali). Verifichiamo che i parent dei
    livelli >1 non siano mai item nati come coherence_bridge.
    """
    text = "Un automobilista viene tamponato; riporta colpo di frusta."
    out = tmp_path / "out_nocb"
    sess = ExplorerSession.analyze(_client(), out, text)
    sess.auto_explore(
        expand_top_n=2, expand_depth=3, expand_children_per_level=3,
        build_bridges=False, do_revalidate=False, do_magistrale=False,
    )
    # Raccogli gli id di tutti gli item nati come coherence_bridge
    coherence_bridge_ids = {
        it.id for it in sess.ft.items
        if (it.metadata or {}).get("direction") == "coherence_bridge"
    }
    # Nessuna expansion deve avere come parent un coherence_bridge
    for exp in sess.ft.expansions:
        assert exp.parent_item_id not in coherence_bridge_ids, (
            f"un coherence_bridge ({exp.parent_item_id}) e' stato ri-espanso, "
            f"ma non dovrebbe"
        )


# -----------------------------------------------------------------------------
# V10.16.3 -- Fix #3: il coherence_bridge non puo' saltare di 2+ scale.
# -----------------------------------------------------------------------------


def test_coherence_bridge_scale_is_clamped_to_adjacent(tmp_path: Path) -> None:
    """Un coherence_bridge generato dall'expander deve vivere sulla scala
    del padre o al massimo una adiacente. Mai un salto di 2+ scale.
    """
    ft = FractalTriadResult()
    # padre su 'organismo' (depth 3)
    parent = ClassifiedItem(
        id="itm_parent_org", quote="un evento a livello organismo",
        predicate=PredicateType.EVENT, nature=Nature.CAUSE,
        scale="organismo", epistemic_status=EpistemicStatus.TEXT_OBSERVED,
        metadata={"origin": "text_observed"},
    )
    ft.items.append(parent)

    from fractal_causal_engine.ft_expander import FractalExpander
    from fractal_causal_engine.ft_model import SCALE_DEPTH
    expander = FractalExpander(_client(), llm_calls_dir=tmp_path)
    record = expander.expand(parent, original_text="testo", trace=[])

    parent_depth = SCALE_DEPTH["organismo"]
    for child in record.children:
        if child.direction == ExpansionDirection.COHERENCE_BRIDGE:
            child_depth = SCALE_DEPTH[child.item.scale]
            assert abs(child_depth - parent_depth) <= 1, (
                f"coherence_bridge su scala {child.item.scale} (depth {child_depth}) "
                f"troppo lontano dal padre organismo (depth {parent_depth})"
            )


# -----------------------------------------------------------------------------
# V10.16.3 -- Fix #1: gli item-domanda (predicate=question) restano fuori
# dai coni del magistrale.
# -----------------------------------------------------------------------------


def test_question_items_excluded_from_magistrale_cones(tmp_path: Path) -> None:
    """Gli item con predicate=question non devono finire nei coni del
    magistrale: sono il quesito di partenza, non fattori causali.
    """
    from fractal_causal_engine.ft_magistrale import _build_payload
    ft = FractalTriadResult()
    ft.items.append(ClassifiedItem(
        id="itm_q", quote="Che cos'e' la coscienza?",
        predicate=PredicateType.QUESTION, nature=Nature.CONTEXT,
        scale="sociale", epistemic_status=EpistemicStatus.TEXT_OBSERVED,
        metadata={"origin": "text_observed"},
    ))
    ft.items.append(ClassifiedItem(
        id="itm_real", quote="un fatto osservato",
        predicate=PredicateType.EVENT, nature=Nature.CAUSE,
        scale="organismo", epistemic_status=EpistemicStatus.TEXT_OBSERVED,
        metadata={"origin": "text_observed"},
    ))
    payload = _build_payload(ft)
    # la domanda finisce in starting_questions, NON in items
    item_ids = {it["id"] for it in payload["items"]}
    assert "itm_q" not in item_ids, "l'item-domanda non deve stare tra gli items del payload"
    assert "itm_real" in item_ids
    assert any("coscienza" in q for q in payload["starting_questions"])
