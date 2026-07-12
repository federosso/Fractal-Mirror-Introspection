"""Test del Regista (L7 Director), V10.17.0.

Verifica le tre condizioni, la regola "tutte e tre o niente intromissione",
la modulazione dei parametri e la (non) regressione su auto_explore senza
Regista.
"""
from __future__ import annotations

from fractal_causal_engine.ft_director import (
    ACTOR_PHASES,
    Director,
    is_phase_irreversible,
    measure_scale_divergence,
    measure_silence_cost,
    render_director_md,
    _build_intervention,
    _next_phase,
)
from fractal_causal_engine.ft_model import (
    ClassifiedItem,
    CrossScaleHypothesis,
    EpistemicStatus,
    FractalTriadResult,
    Nature,
    PredicateType,
)


def _item(iid: str, scale: str, origin: str, nature: Nature = Nature.CAUSE) -> ClassifiedItem:
    return ClassifiedItem(
        id=iid, quote="q", predicate=PredicateType.EVENT,
        nature=nature, scale=scale, metadata={"origin": origin},
    )


def _csh(hid: str, verdict: str) -> CrossScaleHypothesis:
    return CrossScaleHypothesis(
        id=hid, cause_item_id="a", effect_item_id="b",
        cause_scale="atomico", effect_scale="cosmologico",
        verdict=verdict, reasoning="r",
    )


# --- Condizione 1: divergenza di scala ---------------------------------------


def test_divergence_zero_when_no_cross_scale():
    ft = FractalTriadResult()
    assert measure_scale_divergence(ft) == 0.0


def test_divergence_is_fraction_of_spurious():
    ft = FractalTriadResult()
    ft.cross_scale = [_csh("h1", "spurious"), _csh("h2", "spurious"),
                      _csh("h3", "genuine")]
    assert abs(measure_scale_divergence(ft) - (2 / 3)) < 1e-9


# --- Condizione 2: costo del silenzio ----------------------------------------


def test_silence_cost_zero_without_generated_items():
    ft = FractalTriadResult()
    ft.items = [_item("o1", "atomico", "text_observed")]
    assert measure_silence_cost(ft) == 0.0


def test_silence_cost_is_centroid_drift():
    ft = FractalTriadResult()
    # observed: baricentro su atomico (depth 6)
    ft.items = [_item("o1", "atomico", "text_observed")]
    # generated: baricentro su cosmologico (depth 0) -> deriva = 6
    ft.items.append(_item("g1", "cosmologico", "expansion"))
    assert measure_silence_cost(ft) == 6.0


# --- Condizione 3: irreversibilita' ------------------------------------------


def test_irreversible_only_after_bridge():
    # osservare dopo 'bridge' modula 'revalidate' = ultima prima della chiusura
    assert is_phase_irreversible("bridge") is True
    assert is_phase_irreversible("expand") is False
    assert is_phase_irreversible("revalidate") is False
    assert is_phase_irreversible("magistrale") is False


def test_next_phase_order():
    assert _next_phase("expand") == "bridge"
    assert _next_phase("bridge") == "revalidate"
    assert _next_phase("magistrale") == ""


# --- La regola: tutte e tre o niente -----------------------------------------


def _ft_high_divergence_high_drift() -> FractalTriadResult:
    ft = FractalTriadResult()
    ft.items = [
        _item("o1", "molecolare", "text_observed"),
        _item("g1", "cosmologico", "expansion"),
    ]
    ft.cross_scale = [_csh("h1", "spurious"), _csh("h2", "spurious")]
    return ft


def test_no_intervention_when_only_two_conditions():
    """Dopo 'expand': divergenza e costo del silenzio scattano, ma la fase
    NON e' irreversibile. Due su tre non bastano: niente intromissione."""
    ft = _ft_high_divergence_high_drift()
    d = Director(None, silence_band=1.0, divergence_threshold=0.5, narrate=False)
    intervention = d.observe(ft, phase="expand")
    assert intervention is None
    assert d.report.readings[-1].intervened is False


def test_intervention_when_all_three_conditions():
    """Dopo 'bridge': le tre condizioni scattano insieme -> atto di regia.

    Con divergenza 1.0 e soglia 0.5 (over=2.0), il verbo e' REPEAT: la fase
    bridge va rifatta perche' troppo compromessa.
    """
    ft = _ft_high_divergence_high_drift()
    d = Director(None, silence_band=1.0, divergence_threshold=0.5, narrate=False)
    intervention = d.observe(ft, phase="bridge")
    assert intervention is not None
    assert intervention.control == "repeat"
    assert intervention.after_phase == "bridge"
    assert d.report.readings[-1].intervened is True


def test_no_intervention_when_all_quiet():
    """ft senza spurious e senza deriva: il Regista resta in silenzio."""
    ft = FractalTriadResult()
    ft.items = [_item("o1", "atomico", "text_observed")]
    ft.cross_scale = [_csh("h1", "genuine")]
    d = Director(None, silence_band=1.0, divergence_threshold=0.5, narrate=False)
    assert d.observe(ft, phase="bridge") is None


# --- Integrale di deriva: accumulo lungo le fasi -----------------------------


def test_silence_cost_accumulates_across_phases():
    """Il costo del silenzio e' un integrale: si accumula osservazione dopo
    osservazione, non e' una soglia istantanea."""
    ft = _ft_high_divergence_high_drift()
    d = Director(None, silence_band=100.0, divergence_threshold=0.5, narrate=False)
    d.observe(ft, phase="expand")
    first = d.report.readings[-1].silence_cost
    d.observe(ft, phase="bridge")
    second = d.report.readings[-1].silence_cost
    assert second > first  # l'integrale cresce


# --- _build_intervention: gradazione dei verbi di regia ----------------------


def test_build_intervention_proceed_when_just_over_threshold():
    # divergenza appena sopra soglia (1.2x) -> PROCEED + override
    itv = _build_intervention("expand", divergence=0.41, silence_cost=0.5,
                              divergence_threshold=0.34)
    assert itv is not None
    assert itv.control == "proceed"
    assert itv.target_phase == "bridge"
    assert itv.param_overrides == {"max_bridges": 1}


def test_build_intervention_repeat_when_well_over_threshold():
    # divergenza ~1.8x soglia -> REPEAT della fase appena conclusa
    itv = _build_intervention("bridge", divergence=0.6, silence_cost=1.0,
                              divergence_threshold=0.34)
    assert itv is not None
    assert itv.control == "repeat"
    assert itv.target_phase == "bridge"


def test_build_intervention_goto_when_extreme():
    # divergenza ~2.6x soglia ma silenzio sotto la soglia di halt -> GOTO expand
    itv = _build_intervention("bridge", divergence=0.9, silence_cost=1.0,
                              divergence_threshold=0.34)
    assert itv is not None
    assert itv.control == "goto"
    assert itv.goto_phase == "expand"


def test_build_intervention_halt_when_extreme_and_drifted():
    # divergenza estrema + costo del silenzio estremo -> HALT
    itv = _build_intervention("bridge", divergence=0.9, silence_cost=3.5,
                              divergence_threshold=0.34)
    assert itv is not None
    assert itv.control == "halt"


# --- guardia anti-loop -------------------------------------------------------


def test_budget_exhaustion():
    d = Director(None, narrate=False)
    d.control_budget = 2
    assert d.budget_exhausted() is False
    d.register_control_act()
    assert d.budget_exhausted() is False
    d.register_control_act()
    assert d.budget_exhausted() is True


# --- Render ------------------------------------------------------------------


def test_render_director_md_contains_sections():
    ft = _ft_high_divergence_high_drift()
    d = Director(None, silence_band=1.0, divergence_threshold=0.5, narrate=False)
    d.observe(ft, phase="bridge")
    md = render_director_md(d.finalize(ft))
    assert "Relazione del Regista" in md
    assert "Letture per fase" in md
    assert "Atti di regia" in md
    assert "Flusso eseguito" in md


# --- Non regressione ---------------------------------------------------------


def test_actor_phases_order_stable():
    assert ACTOR_PHASES == ["expand", "bridge", "revalidate", "magistrale"]


# --- Integrazione: il motore _run_directed con pieno controllo ---------------


def test_run_directed_full_control_on_mock():
    """session-observe end-to-end con backend mock: verifica che il motore di
    regia esegua, produca il flusso e non vada in loop."""
    import tempfile
    from pathlib import Path
    from fractal_causal_engine.llm import LLMClient, LLMConfig
    from fractal_causal_engine.ft_session import ExplorerSession

    client = LLMClient(LLMConfig(mock=True))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "obs"
        text = ("La coscienza e' un fenomeno. La materia e' un substrato. "
                "L'organismo percepisce. Le cellule comunicano.")
        sess = ExplorerSession.analyze(client, out, text, source_input_id="i1")
        stats = sess.observe_with_director(narrate=False)
        # il flusso deve essere stato eseguito e tracciato
        assert stats["director_executed_phases"]
        assert "expand" in stats["director_executed_phases"]
        # nessun loop: il numero di fasi eseguite e' limitato
        assert len(stats["director_executed_phases"]) <= 24
        # il report del Regista esiste su disco
        assert (out / "director_report.md").exists()


def test_run_directed_respects_disabled_phases():
    """Le fasi disabilitate dai flag --no-* non vengono eseguite."""
    import tempfile
    from pathlib import Path
    from fractal_causal_engine.llm import LLMClient, LLMConfig
    from fractal_causal_engine.ft_session import ExplorerSession

    client = LLMClient(LLMConfig(mock=True))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "obs"
        sess = ExplorerSession.analyze(
            client, out, "La coscienza e' un fenomeno complesso.", source_input_id="i1"
        )
        stats = sess.observe_with_director(
            build_bridges=False, do_revalidate=False, narrate=False
        )
        executed = stats["director_executed_phases"]
        # bridge e revalidate marcati come disabled, non eseguiti
        assert "bridge:disabled" in executed
        assert "revalidate:disabled" in executed
        assert "bridge" not in executed
        assert "revalidate" not in executed


# --- Motore di regia: verbi di flusso con un Director deterministico --------


class _ScriptedDirector:
    """Director finto: emette atti di regia da uno script prefissato, per
    testare il motore _run_directed in isolamento dalle 3 misure."""

    def __init__(self, script):
        # script: dict {phase -> DirectorIntervention | None}
        self.script = script
        self.control_budget = 10
        self._acts = 0
        from fractal_causal_engine.ft_model import DirectorReport
        self.report = DirectorReport()
        self.narrate = False

    def observe(self, ft, *, phase, trace=None):
        return self.script.get(phase)

    def budget_exhausted(self):
        return self._acts >= self.control_budget

    def register_control_act(self):
        self._acts += 1

    def finalize(self, ft, *, trace=None):
        return self.report


def _scripted_run(script, *, enabled=None, budget=10):
    import tempfile
    from pathlib import Path
    from fractal_causal_engine.llm import LLMClient, LLMConfig
    from fractal_causal_engine.ft_session import ExplorerSession

    client = LLMClient(LLMConfig(mock=True))
    td = tempfile.mkdtemp()
    sess = ExplorerSession.analyze(
        client, Path(td) / "o", "La coscienza e' un fenomeno.", source_input_id="i1"
    )
    director = _ScriptedDirector(script)
    director.control_budget = budget
    phases = enabled if enabled is not None else {"expand", "bridge", "revalidate", "magistrale"}
    params = {"expand_top_n": 2, "expand_depth": 1, "expand_children_per_level": 2,
              "max_bridges": 2, "only_uncertain": True}
    sess._run_directed(director, params, enabled_phases=phases)
    return director.report.executed_phases, director.report.halted


def test_motor_halt_stops_actor():
    from fractal_causal_engine.ft_model import DirectorIntervention
    from fractal_causal_engine.ft_director import CONTROL_HALT
    script = {"expand": DirectorIntervention(
        after_phase="expand", target_phase="magistrale", control=CONTROL_HALT)}
    executed, halted = _scripted_run(script)
    assert halted is True
    assert executed == ["expand"]   # niente bridge/revalidate/magistrale


def test_motor_skip_skips_next_phase():
    from fractal_causal_engine.ft_model import DirectorIntervention
    from fractal_causal_engine.ft_director import CONTROL_SKIP
    script = {"expand": DirectorIntervention(
        after_phase="expand", target_phase="bridge", control=CONTROL_SKIP)}
    executed, halted = _scripted_run(script)
    assert "bridge:skipped" in executed
    assert "bridge" not in executed
    assert "revalidate" in executed and "magistrale" in executed


def test_motor_repeat_reexecutes_phase():
    from fractal_causal_engine.ft_model import DirectorIntervention
    from fractal_causal_engine.ft_director import CONTROL_REPEAT
    # ripeti 'expand' una volta sola: lo script si esaurisce dopo il primo giro
    seen = {"n": 0}

    class _OnceRepeat(_ScriptedDirector):
        def observe(self, ft, *, phase, trace=None):
            if phase == "expand" and seen["n"] == 0:
                seen["n"] += 1
                from fractal_causal_engine.ft_model import DirectorIntervention
                from fractal_causal_engine.ft_director import CONTROL_REPEAT
                return DirectorIntervention(
                    after_phase="expand", target_phase="expand", control=CONTROL_REPEAT)
            return None

    import tempfile
    from pathlib import Path
    from fractal_causal_engine.llm import LLMClient, LLMConfig
    from fractal_causal_engine.ft_session import ExplorerSession
    client = LLMClient(LLMConfig(mock=True))
    sess = ExplorerSession.analyze(
        client, Path(tempfile.mkdtemp()) / "o",
        "La coscienza e' un fenomeno.", source_input_id="i1")
    d = _OnceRepeat({})
    sess._run_directed(d, {"expand_top_n": 2, "expand_depth": 1,
                           "expand_children_per_level": 2, "max_bridges": 2,
                           "only_uncertain": True},
                       enabled_phases={"expand", "bridge", "revalidate", "magistrale"})
    # 'expand' compare due volte nel flusso eseguito
    assert d.report.executed_phases.count("expand") == 2


def test_motor_budget_degrades_control_to_proceed():
    """Con budget 0, anche un HALT scriptato viene degradato a proceed: il
    motore prosegue fino alla chiusura."""
    from fractal_causal_engine.ft_model import DirectorIntervention
    from fractal_causal_engine.ft_director import CONTROL_HALT
    script = {"expand": DirectorIntervention(
        after_phase="expand", target_phase="magistrale", control=CONTROL_HALT)}
    executed, halted = _scripted_run(script, budget=0)
    # HALT degradato: l'Attore non e' stato fermato
    assert halted is False
    assert "magistrale" in executed
