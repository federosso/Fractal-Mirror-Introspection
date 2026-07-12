"""ft_session (V10.15.0).

Sessione interattiva persistente sul filesystem. Permette di:

  - analyze: prima passata della pipeline V14 su un testo (crea ft_analysis.json
             e session.json).
  - list:    elenca gli item correnti, con scale, nature e stato epistemico.
  - expand:  espande un singolo item (chiama FractalExpander + integrate_expansion).
  - bridge:  costruisce un bridge tra due item (BridgeBuilder + integrate_bridge).
  - magistrale: genera la relazione magistrale finale.
  - revalidate-cross: rilancia L3.B (CrossScaleValidator) sulle ipotesi
                       'uncertain' generate da espansioni e bridge per
                       provare a promuoverle a 'genuine'.

Persistenza: session.json contiene il FractalTriadResult serializzato + il
testo originale + il counter delle chiamate. Ogni comando legge, modifica
e salva.

NESSUN comando bypassa i guard di V14: predicate, nature, scale canonica,
zoom coherence, verdict cross-scale.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .ft_bridge import BridgeBuilder, integrate_bridge
from .ft_crossscale import CrossScaleValidator
from .ft_director import Director, attach_director_report, render_director_md
from .ft_expander import FractalExpander
from .ft_explorer import integrate_expansion
from .ft_magistrale import MagistraleReportBuilder, render_magistrale_md
from .ft_model import (
    SCALES_CANONICAL,
    SCALE_DEPTH,
    BridgeRecord,
    ClassifiedItem,
    CrossScaleHypothesis,
    DoubleCone,
    EpistemicStatus,
    ExpansionChild,
    ExpansionDirection,
    ExpansionRecord,
    FractalTriadResult,
    GlobalVision,
    LockedScaleReport,
    MagistraleCones,
    MagistraleEffects,
    MagistraleReport,
    Nature,
    Orphan,
    PredicateType,
    SameScaleLink,
    UnlockedReport,
    DomainConcept,
    CausalPrinciple,
    CrossDomainAnalogy,
)
from .ft_orchestrator import render_final_report_md
from .ft_pipeline import FractalTriadPipeline
from .llm import LLMClient


SESSION_FILENAME = "session.json"


def _find_cross_scale_pairs(items: list[ClassifiedItem]) -> list[tuple[ClassifiedItem, ClassifiedItem, int]]:
    """Trova tutte le coppie di items con scale_distance >= 2.

    Ritorna (a, b, distance). L'orientamento src->target lo decide il chiamante.
    """
    pairs: list[tuple[ClassifiedItem, ClassifiedItem, int]] = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            dist = abs(SCALE_DEPTH[a.scale] - SCALE_DEPTH[b.scale])
            if dist >= 2:
                pairs.append((a, b, dist))
    return pairs


# =============================================================================
# (De)serializzazione robusta del FractalTriadResult
# =============================================================================


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def serialize_ft(ft: FractalTriadResult, original_text: str, call_counter: int) -> dict[str, Any]:
    return {
        "version": "10.15.0",
        "original_text": original_text,
        "call_counter": call_counter,
        "ft": {
            "items": [_to_jsonable(i) for i in ft.items],
            "locked_reports": [_to_jsonable(r) for r in ft.locked_reports],
            "unlocked": _to_jsonable(ft.unlocked) if ft.unlocked else None,
            "cross_scale": [_to_jsonable(h) for h in ft.cross_scale],
            "double_cone": _to_jsonable(ft.double_cone),
            "vision": _to_jsonable(ft.vision),
            "trace": list(ft.trace),
            "expansions": [_to_jsonable(e) for e in ft.expansions],
            "bridges": [_to_jsonable(b) for b in ft.bridges],
            "magistrale": _to_jsonable(ft.magistrale) if ft.magistrale else None,
        },
    }


def deserialize_ft(data: dict[str, Any]) -> tuple[FractalTriadResult, str, int]:
    ft_raw = data.get("ft", {})
    ft = FractalTriadResult()
    ft.items = [_item_from_dict(d) for d in ft_raw.get("items", [])]
    ft.locked_reports = [_locked_from_dict(d) for d in ft_raw.get("locked_reports", [])]
    ft.unlocked = _unlocked_from_dict(ft_raw.get("unlocked"))
    ft.cross_scale = [_csh_from_dict(d) for d in ft_raw.get("cross_scale", [])]
    ft.double_cone = _dc_from_dict(ft_raw.get("double_cone") or {})
    ft.vision = _vision_from_dict(ft_raw.get("vision") or {})
    ft.trace = list(ft_raw.get("trace") or [])
    ft.expansions = [_expansion_from_dict(d) for d in ft_raw.get("expansions", [])]
    ft.bridges = [_bridge_from_dict(d) for d in ft_raw.get("bridges", [])]
    ft.magistrale = _magistrale_from_dict(ft_raw.get("magistrale"))
    return ft, data.get("original_text", ""), int(data.get("call_counter", 0))


def _item_from_dict(d: dict) -> ClassifiedItem:
    return ClassifiedItem(
        id=d["id"],
        quote=d.get("quote", ""),
        predicate=PredicateType(d.get("predicate", "unknown")),
        nature=Nature(d.get("nature", "context")),
        scale=d.get("scale", ""),
        rationale=d.get("rationale", ""),
        source_input_id=d.get("source_input_id", ""),
        epistemic_status=EpistemicStatus(d.get("epistemic_status", "text_observed")),
        metadata=d.get("metadata") or {},
    )


def _locked_from_dict(d: dict) -> LockedScaleReport:
    return LockedScaleReport(
        scale=d["scale"],
        same_scale_links=[
            SameScaleLink(
                id=x["id"],
                scale=x["scale"],
                cause_item_id=x["cause_item_id"],
                effect_item_id=x["effect_item_id"],
                rationale=x.get("rationale", ""),
                confidence=x.get("confidence", 0.0),
            )
            for x in d.get("same_scale_links", [])
        ],
        orphans=[
            Orphan(
                item_id=x["item_id"],
                nature=Nature(x.get("nature", "context")),
                scale=x.get("scale", ""),
                reason=x.get("reason", ""),
            )
            for x in d.get("orphans", [])
        ],
        items_seen=list(d.get("items_seen") or []),
        summary=d.get("summary", ""),
    )


def _unlocked_from_dict(d):
    if not d:
        return None
    return UnlockedReport(
        domain=d.get("domain", ""),
        domain_knowledge=[
            DomainConcept(
                concept=x.get("concept", ""),
                relation_to_input=x.get("relation_to_input", ""),
                status=EpistemicStatus(x.get("status", "domain_knowledge")),
                suggested_scale=x.get("suggested_scale", ""),
                not_in_input=bool(x.get("not_in_input", True)),
            )
            for x in d.get("domain_knowledge", [])
        ],
        causal_principles=[
            CausalPrinciple(
                name=x.get("name", ""),
                description=x.get("description", ""),
                status=EpistemicStatus(x.get("status", "causal_model")),
            )
            for x in d.get("causal_principles", [])
        ],
        cross_domain_analogies=[
            CrossDomainAnalogy(
                domain=x.get("domain", ""),
                analogy=x.get("analogy", ""),
                warning=x.get("warning", ""),
                status=EpistemicStatus(x.get("status", "cross_domain_analogy")),
            )
            for x in d.get("cross_domain_analogies", [])
        ],
        open_questions=list(d.get("open_questions") or []),
        known_uncertainties=list(d.get("known_uncertainties") or []),
        degraded=bool(d.get("degraded", False)),
        degraded_parts=list(d.get("degraded_parts") or []),
    )


def _csh_from_dict(d: dict) -> CrossScaleHypothesis:
    return CrossScaleHypothesis(
        id=d["id"],
        cause_item_id=d["cause_item_id"],
        effect_item_id=d["effect_item_id"],
        cause_scale=d["cause_scale"],
        effect_scale=d["effect_scale"],
        verdict=d.get("verdict", "uncertain"),
        reasoning=d.get("reasoning", ""),
        confidence=d.get("confidence", 0.0),
    )


def _dc_from_dict(d: dict) -> DoubleCone:
    return DoubleCone(
        cone_of_causes={k: list(v) for k, v in (d.get("cone_of_causes") or {}).items()},
        cone_of_effects={k: list(v) for k, v in (d.get("cone_of_effects") or {}).items()},
    )


def _vision_from_dict(d: dict) -> GlobalVision:
    return GlobalVision(
        core_image=d.get("core_image", ""),
        human_summary=d.get("human_summary", ""),
        epistemic_warning=d.get("epistemic_warning", ""),
        dominant_domain=d.get("dominant_domain", ""),
        primary_lenses=list(d.get("primary_lenses") or []),
        blocked_lenses=list(d.get("blocked_lenses") or []),
    )


def _expansion_from_dict(d: dict) -> ExpansionRecord:
    return ExpansionRecord(
        parent_item_id=d["parent_item_id"],
        direction_set=[ExpansionDirection(x) for x in d.get("direction_set", [])],
        children=[
            ExpansionChild(
                item=_item_from_dict(c["item"]),
                direction=ExpansionDirection(c["direction"]),
                relation_to_parent=c.get("relation_to_parent", ""),
                confidence=c.get("confidence", 0.0),
            )
            for c in d.get("children", [])
        ],
        same_scale_links_added=[
            SameScaleLink(
                id=x["id"],
                scale=x["scale"],
                cause_item_id=x["cause_item_id"],
                effect_item_id=x["effect_item_id"],
                rationale=x.get("rationale", ""),
                confidence=x.get("confidence", 0.0),
            )
            for x in d.get("same_scale_links_added", [])
        ],
        cross_scale_added=[_csh_from_dict(x) for x in d.get("cross_scale_added", [])],
        degraded=bool(d.get("degraded", False)),
        notes=d.get("notes", ""),
    )


def _bridge_from_dict(d: dict) -> BridgeRecord:
    return BridgeRecord(
        source_item_id=d["source_item_id"],
        target_item_id=d["target_item_id"],
        gap_scale=d["gap_scale"],
        bridge_item=_item_from_dict(d["bridge_item"]),
        mechanism_reasoning=d.get("mechanism_reasoning", ""),
        cross_scale_added=[_csh_from_dict(x) for x in d.get("cross_scale_added", [])],
        degraded=bool(d.get("degraded", False)),
    )


def _magistrale_from_dict(d):
    if not d:
        return None
    cc = d.get("cono_cause") or {}
    ce = d.get("cono_effetti") or {}
    return MagistraleReport(
        sintesi_magistrale=d.get("sintesi_magistrale", ""),
        cono_cause=MagistraleCones(
            predispositions=list(cc.get("predispositions") or []),
            triggers=list(cc.get("triggers") or []),
            proximate_causes=list(cc.get("proximate_causes") or []),
            bridge_mechanisms=list(cc.get("bridge_mechanisms") or []),
        ),
        cono_effetti=MagistraleEffects(
            direct_effects=list(ce.get("direct_effects") or []),
            downstream_effects=list(ce.get("downstream_effects") or []),
            interpretations=list(ce.get("interpretations") or []),
            social_propagations=list(ce.get("social_propagations") or []),
        ),
        propagazione_multi_scala=d.get("propagazione_multi_scala", ""),
        stato_epistemico=d.get("stato_epistemico", ""),
        verdetto_finale=d.get("verdetto_finale", ""),
        degraded=bool(d.get("degraded", False)),
    )


# =============================================================================
# La sessione esploratore
# =============================================================================


class ExplorerSession:
    """Sessione esploratore persistente. Apertura: load_or_create."""

    def __init__(
        self,
        out_dir: Path,
        client: LLMClient,
        ft: FractalTriadResult,
        original_text: str,
        call_counter: int = 0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.llm_calls_dir = self.out_dir / "llm_calls"
        self.llm_calls_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.out_dir / "telemetry.jsonl"
        self.client = client
        self.ft = ft
        self.original_text = original_text
        self.call_counter = call_counter

    # ----- Persistenza ------------------------------------------------------

    @classmethod
    def load(cls, out_dir: Path, client: LLMClient) -> "ExplorerSession":
        path = Path(out_dir) / SESSION_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"session non trovata in {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        ft, original_text, call_counter = deserialize_ft(data)
        return cls(Path(out_dir), client, ft, original_text, call_counter)

    def save(self) -> None:
        payload = serialize_ft(self.ft, self.original_text, self.call_counter)
        (self.out_dir / SESSION_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ----- Comandi ----------------------------------------------------------

    @classmethod
    def analyze(
        cls,
        client: LLMClient,
        out_dir: Path,
        text: str,
        *,
        source_input_id: str = "input_001",
        max_cross_scale: int = 8,
    ) -> "ExplorerSession":
        """Comando: analyze. Prima passata della pipeline V14 + apertura sessione."""
        pipeline = FractalTriadPipeline(client, out_dir=out_dir, max_cross_scale_candidates=max_cross_scale)
        ft = pipeline.run(text, source_input_id=source_input_id)
        pipeline.write_outputs(ft, original_text=text)
        sess = cls(Path(out_dir), client, ft, text, call_counter=0)
        sess.save()
        return sess

    def expand(self, item_id: str) -> ExpansionRecord:
        """Comando: expand <item_id>."""
        parent = self._require_item(item_id)
        expander = FractalExpander(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
        )
        trace: list[str] = [self._stamp("expand", item_id)]
        record = expander.expand(parent, original_text=self.original_text, trace=trace)
        integrate_expansion(self.ft, record)
        self.ft.trace.extend(trace)
        self.call_counter += 1
        self.save()
        return record

    def bridge(self, source_id: str, target_id: str, gap_scale: str) -> BridgeRecord:
        """Comando: bridge <source_id> <target_id> <gap_scale>."""
        source = self._require_item(source_id)
        target = self._require_item(target_id)
        builder = BridgeBuilder(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
        )
        trace: list[str] = [self._stamp("bridge", f"{source_id}->{target_id}@{gap_scale}")]
        record = builder.build(source, target, gap_scale, trace=trace)
        integrate_bridge(self.ft, record)
        self.ft.trace.extend(trace)
        self.call_counter += 1
        self.save()
        return record

    def magistrale(self) -> MagistraleReport:
        """Comando: magistrale. Genera la relazione finale + salva su disco."""
        builder = MagistraleReportBuilder(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
        )
        trace: list[str] = [self._stamp("magistrale", "")]
        report = builder.build(self.ft, trace=trace)
        self.ft.trace.extend(trace)
        self.call_counter += 1
        # render markdown
        md = render_magistrale_md(report)
        (self.out_dir / "magistrale_report.md").write_text(md, encoding="utf-8")
        # aggiorna anche final_report.md riusando l'orchestrator render
        (self.out_dir / "final_report.md").write_text(
            render_final_report_md(self.ft, original_text=self.original_text),
            encoding="utf-8",
        )
        self.save()
        return report

    def revalidate_cross(self, *, only_uncertain: bool = True) -> dict[str, int]:
        """Comando: revalidate-cross.

        Rivaluta le ipotesi cross-scale GIA' PRESENTI nel ft (generate da
        expand/bridge, oppure dalla pipeline V14), chiedendo a L3.B un
        verdetto aggiornato {genuine | spurious | uncertain}.

        IMPORTANTE: a differenza della pipeline V14, qui NON si ricalcolano
        le ipotesi da zero. Si rivalutano quelle esistenti e si conservano
        i loro id, cosi' i record di espansione/bridge restano coerenti.
        Le ipotesi non rivalutate (es. gia' genuine, con only_uncertain=True)
        NON vengono perse: restano invariate.
        """
        validator = CrossScaleValidator(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
        )
        trace: list[str] = [self._stamp("revalidate_cross", "")]

        # Partizione: cosa rivalutare, cosa lasciare invariato.
        to_eval: list[CrossScaleHypothesis] = []
        keep_as_is: list[CrossScaleHypothesis] = []
        for h in self.ft.cross_scale:
            if only_uncertain and h.verdict != "uncertain":
                keep_as_is.append(h)
            else:
                to_eval.append(h)

        if not to_eval:
            trace.append("L3B_revalidate: nessuna ipotesi da rivalutare")
            self.ft.trace.extend(trace)
            self.call_counter += 1
            self.save()
            return {"evaluated": 0, "genuine": 0, "spurious": 0, "uncertain": 0}

        # Rivaluta le ipotesi esistenti SENZA ricalcolarle.
        updated = validator.run_on_hypotheses(to_eval, self.ft.items, trace)

        # Ricomponi: invariate + rivalutate. Le 'spurious' restano nella
        # lista ma marcate (il magistrale le ignora come propagazione valida).
        self.ft.cross_scale = keep_as_is + updated

        stats = {
            "evaluated": len(updated),
            "genuine": sum(1 for h in updated if h.verdict == "genuine"),
            "spurious": sum(1 for h in updated if h.verdict == "spurious"),
            "uncertain": sum(1 for h in updated if h.verdict == "uncertain"),
        }
        trace.append(f"L3B_revalidate: stats={stats}")
        self.ft.trace.extend(trace)
        self.call_counter += 1
        self.save()
        return stats

    # ----- Liste e pretty-print --------------------------------------------

    def list_items(self) -> list[dict[str, Any]]:
        """Comando: list. Ritorna struttura ordinata per scala."""
        rows: list[dict[str, Any]] = []
        for it in self.ft.items:
            rows.append(
                {
                    "id": it.id,
                    "scale": it.scale,
                    "scale_depth": SCALE_DEPTH.get(it.scale, -1),
                    "nature": it.nature.value,
                    "predicate": it.predicate.value,
                    "epistemic_status": it.epistemic_status.value,
                    "origin": (it.metadata or {}).get("origin", "text_observed"),
                    "text": it.quote or (it.metadata or {}).get("generated_text", ""),
                }
            )
        rows.sort(key=lambda r: (r["scale_depth"], r["nature"]))
        return rows

    def render_list_md(self) -> str:
        rows = self.list_items()
        out = ["# Item correnti (V10.15.0 session)\n"]
        if not rows:
            return out[0] + "\n_(nessun item)_\n"
        # Raggruppa per scala
        by_scale: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_scale.setdefault(r["scale"], []).append(r)
        for scale in sorted(by_scale.keys(), key=lambda s: SCALE_DEPTH.get(s, 99)):
            out.append(f"\n## scala: {scale}\n")
            for r in by_scale[scale]:
                out.append(
                    f"- `{r['id']}` | **{r['nature']}** | _{r['predicate']}_ | "
                    f"[{r['origin']}/{r['epistemic_status']}] {r['text']}"
                )
        return "\n".join(out) + "\n"

    # ----- helpers ----------------------------------------------------------

    def _require_item(self, item_id: str) -> ClassifiedItem:
        for it in self.ft.items:
            if it.id == item_id:
                return it
        raise KeyError(f"item_id non trovato: {item_id}")

    def _stamp(self, cmd: str, arg: str) -> str:
        return f"[{datetime.now().isoformat(timespec='seconds')}] SESSION_CMD={cmd} arg={arg}"

    # ----- V10.15.1: scorciatoie per ergonomia ------------------------------

    def resolve_item_ref(self, ref: str) -> ClassifiedItem:
        """Risolve un riferimento item: ID completo, ID parziale, o indice [N].

        Permette all'utente di scrivere '3' (indice in list_items()) oppure
        'itm_28ae' (prefisso) invece dell'ID completo. Match deterministico:
        - se ref e' un intero N: usa rows[N-1]['id'];
        - altrimenti: prima cerca match esatto, poi prefisso unico.
        """
        ref = (ref or "").strip()
        if not ref:
            raise KeyError("riferimento vuoto")
        rows = self.list_items()
        # 1. indice numerico
        if ref.isdigit():
            idx = int(ref)
            if not (1 <= idx <= len(rows)):
                raise KeyError(f"indice fuori range: {idx} (max {len(rows)})")
            return self._require_item(rows[idx - 1]["id"])
        # 2. match esatto
        for it in self.ft.items:
            if it.id == ref:
                return it
        # 3. prefisso unico
        candidates = [it for it in self.ft.items if it.id.startswith(ref)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise KeyError(
                f"prefisso ambiguo '{ref}', match: {[c.id for c in candidates[:5]]}"
            )
        raise KeyError(f"item non trovato: {ref}")

    def auto_explore(
        self,
        *,
        expand_top_n: int = 3,
        expand_depth: int = 1,
        expand_children_per_level: int = 2,
        build_bridges: bool = True,
        max_bridges: int = 3,
        do_revalidate: bool = True,
        do_magistrale: bool = True,
        progress=None,
        director=None,
    ) -> dict[str, Any]:
        """One-shot: espande i top-N item osservati, costruisce bridge automatici
        sui gap cross-scale, rivalida, e genera la magistrale.

        director: oggetto Director opzionale (L7). Se passato, dopo OGNI fase
        eseguita il Regista osserva il ft e puo' restituire una correzione dei
        parametri della fase successiva (DirectorIntervention). Senza director,
        il comportamento e' identico alle versioni precedenti: l'Attore segue
        la corrente. Le correzioni del Regista modulano solo i parametri
        locali di questa chiamata; non vengono persistiti nel ft.

        Strategia di selezione (deterministica, niente euristiche segrete):
        - expand: i primi N item con origin='text_observed' E nature in
          {cause, effect}, ordinati per scale_depth (dal piu' profondo al
          piu' superficiale). Se non ci sono cause/effect (testo definitorio),
          ricade sui context.
        - expand_depth: profondita' dell'espansione. 1 = solo gli item
          osservati. 2 = ri-espande anche i figli piu' promettenti. 3 = i
          nipoti, e cosi' via. E' il parametro che da' la profondita'
          multi-scala "vera" (esplorazione a piu' livelli).
        - expand_children_per_level: a ogni livello oltre il primo, quanti
          figli del livello precedente ri-espandere (i piu' promettenti per
          confidence). Tiene sotto controllo l'esplosione combinatoria.
        - bridge: prende le coppie (item_a, item_b) con scale_distance >= 2.
          La gap_scale e' la scala canonica esattamente a meta'.

        progress: callable opzionale (msg: str) -> None, per log live.
        """
        def _say(msg: str) -> None:
            if progress is not None:
                progress(msg)

        stats: dict[str, Any] = {
            "expanded": 0,
            "expand_degraded": 0,
            "expand_by_level": {},
            "bridges_built": 0,
            "bridge_degraded": 0,
            "revalidate": None,
            "magistrale": False,
            "director_interventions": 0,
        }

        # Override locali che il Regista (L7) puo' iniettare osservando l'Attore
        # fra una fase e l'altra. Vuoto = nessuna correzione, l'Attore segue la
        # corrente. Vedi ft_director.Director.observe.
        director_overrides: dict[str, Any] = {}

        def _consult_director(phase: str) -> None:
            """Fa osservare il Regista dopo 'phase' e raccoglie l'eventuale
            correzione per la fase successiva. No-op se director is None."""
            if director is None:
                return
            trace: list[str] = []
            intervention = director.observe(self.ft, phase=phase, trace=trace)
            self.ft.trace.extend(trace)
            if intervention is not None:
                director_overrides.update(intervention.param_overrides)
                stats["director_interventions"] += 1
                _say(f"[regista] intromissione dopo '{phase}': "
                     f"modulo '{intervention.target_phase}' con "
                     f"{intervention.param_overrides}")
            else:
                _say(f"[regista] osservo dopo '{phase}': nessuna intromissione")

        # 1. EXPAND (multi-livello)
        # LIVELLO 1 -- seme: cause/effect osservati, oppure context (fallback
        # per i testi definitori). E' la radice dell'esplorazione.
        observed_causal = [
            it for it in self.ft.items
            if (it.metadata or {}).get("origin", "text_observed") == "text_observed"
            and it.nature in (Nature.CAUSE, Nature.EFFECT)
            and it.scale in SCALE_DEPTH
        ]
        observed_causal.sort(key=lambda it: -SCALE_DEPTH[it.scale])

        if observed_causal:
            seed_items = observed_causal[:expand_top_n]
            _say(f"[auto] livello 1: espando {len(seed_items)} item osservati "
                 f"cause/effect (top {expand_top_n})")
        else:
            observed_context = [
                it for it in self.ft.items
                if (it.metadata or {}).get("origin", "text_observed") == "text_observed"
                and it.nature == Nature.CONTEXT
                and it.scale in SCALE_DEPTH
            ]
            observed_context.sort(key=lambda it: -SCALE_DEPTH[it.scale])
            seed_items = observed_context[:expand_top_n]
            _say(f"[auto] testo definitorio (niente cause/effect osservati): "
                 f"livello 1 espando {len(seed_items)} item context come semi")

        # Espande una lista di item, ritorna i figli generati (ExpansionChild).
        def _expand_level(items_to_expand: list, level: int) -> list:
            produced_children: list = []
            for it in items_to_expand:
                _say(f"[auto] L{level} expand item {it.id} ({it.scale}/{it.nature.value})")
                rec = self.expand(it.id)
                if rec.degraded and not rec.children:
                    stats["expand_degraded"] += 1
                else:
                    stats["expanded"] += 1
                    produced_children.extend(rec.children)
            stats["expand_by_level"][f"L{level}"] = len(items_to_expand)
            return produced_children

        # Livello 1
        children = _expand_level(seed_items, level=1)

        # LIVELLI 2..expand_depth -- ri-espande i figli piu' promettenti.
        # "Promettenti" = confidence piu' alta. Escludiamo i coherence_bridge
        # (sono ponti, non concetti da rilanciare) per evitare derive.
        for level in range(2, max(1, expand_depth) + 1):
            if not children:
                _say(f"[auto] L{level}: nessun figlio da ri-espandere, stop")
                break
            candidates = [
                c for c in children
                if c.direction != ExpansionDirection.COHERENCE_BRIDGE
            ]
            candidates.sort(key=lambda c: -c.confidence)
            next_seed = [c.item for c in candidates[:expand_children_per_level]]
            if not next_seed:
                _say(f"[auto] L{level}: nessun figlio promettente, stop")
                break
            _say(f"[auto] livello {level}: ri-espando {len(next_seed)} figli "
                 f"piu' promettenti (di {len(children)} disponibili)")
            children = _expand_level(next_seed, level=level)

        # --- Il Regista osserva l'Attore dopo la fase EXPAND ---
        # (solo se la fase expand e' stata effettivamente eseguita in questa
        #  chiamata, cioe' se c'erano semi da espandere)
        if seed_items:
            _consult_director("expand")

        # 2. BRIDGE: coppie a distanza >= 2.
        # Strategia:
        # - prima si cercano coppie tra item text_observed (pi\u00f9 robuste);
        # - se non si trovano coppie a distanza >= 2 (es. il testo \u00e8 monoscala),
        #   si fa fallback aggiungendo gli item nati dall'espansione (origin in
        #   'expansion' o 'bridge'), che possono vivere su scale diverse e
        #   quindi creare gap reali. Il bridge resta marcato in modo onesto:
        #   il bridge_item ha gi\u00e0 epistemic_status=CAUSAL_MODEL, quindi
        #   l'utente sa che e' una proposta meccanica, non una catena osservata.
        if build_bridges:
            # Il Regista puo' aver ridotto max_bridges osservando la deriva.
            effective_max_bridges = director_overrides.get("max_bridges", max_bridges)
            if effective_max_bridges != max_bridges:
                _say(f"[regista] max_bridges modulato: {max_bridges} -> "
                     f"{effective_max_bridges}")
            text_observed_items = [
                it for it in self.ft.items
                if (it.metadata or {}).get("origin", "text_observed") == "text_observed"
                and it.scale in SCALE_DEPTH
            ]
            bridge_candidates = _find_cross_scale_pairs(text_observed_items)
            used_fallback = False
            if not bridge_candidates and build_bridges:
                # Fallback: estendiamo agli item generati per espansione.
                _say("[auto] nessuna coppia text_observed cross-scale; "
                     "fallback su items expansion/bridge per costruire i ponti")
                extended_items = text_observed_items + [
                    it for it in self.ft.items
                    if (it.metadata or {}).get("origin") in ("expansion", "bridge")
                    and it.scale in SCALE_DEPTH
                ]
                bridge_candidates = _find_cross_scale_pairs(extended_items)
                used_fallback = bool(bridge_candidates)
            # ordina per distanza decrescente (gap piu' grandi prima)
            bridge_candidates.sort(key=lambda t: -t[2])
            bridge_candidates = bridge_candidates[:effective_max_bridges]
            mode = "(fallback expansion)" if used_fallback else "(text_observed)"
            _say(f"[auto] costruisco {len(bridge_candidates)} bridge sui gap cross-scale {mode}")
            for a, b, dist in bridge_candidates:
                # sorgente = scala piu' profonda, target = piu' superficiale
                if SCALE_DEPTH[a.scale] > SCALE_DEPTH[b.scale]:
                    src, tgt = a, b
                else:
                    src, tgt = b, a
                # gap_scale = scala canonica esattamente a meta' tra src e tgt
                mid_depth = (SCALE_DEPTH[src.scale] + SCALE_DEPTH[tgt.scale]) // 2
                gap_scale = SCALES_CANONICAL[mid_depth]
                _say(f"[auto] bridge {src.id}({src.scale}) -> {tgt.id}({tgt.scale}) @ {gap_scale}")
                rec = self.bridge(src.id, tgt.id, gap_scale)
                if rec.degraded:
                    stats["bridge_degraded"] += 1
                else:
                    stats["bridges_built"] += 1

            # --- Il Regista osserva l'Attore dopo la fase BRIDGE ---
            _consult_director("bridge")

        # 3. REVALIDATE
        if do_revalidate:
            # Il Regista puo' aver forzato only_uncertain=False per rivedere
            # TUTTE le ipotesi prima della chiusura (zoom-in sulla chiusura).
            effective_only_uncertain = director_overrides.get("only_uncertain", True)
            if not effective_only_uncertain:
                _say("[regista] revalidate estesa a tutte le ipotesi "
                     "(only_uncertain forzato a False)")
            _say("[auto] revalidate cross-scale")
            stats["revalidate"] = self.revalidate_cross(
                only_uncertain=effective_only_uncertain
            )

        # 4. MAGISTRALE
        if do_magistrale:
            _say("[auto] genero relazione magistrale")
            self.magistrale()
            stats["magistrale"] = True

        return stats

    # =========================================================================
    # V10.17.1 -- Pieno controllo del Regista.
    #
    # Le 4 fasi dell'Attore estratte in metodi RIESEGUIBILI, piu' un motore di
    # esecuzione (_run_directed) che le orchestra sotto la regia del Director.
    # A differenza di auto_explore (sequenza cablata), qui il flusso e' deciso
    # fase per fase: il Regista puo' saltare, ripetere, tornare indietro o
    # fermare l'Attore. auto_explore resta intatto: session-auto e i test V14
    # non sono toccati.
    # =========================================================================

    def _phase_expand(self, params: dict[str, Any], progress) -> dict[str, Any]:
        """Fase 'expand' isolata e rieseguibile. Ritorna stats parziali.

        Riusa la stessa strategia di selezione semi di auto_explore: cause/
        effect osservati, fallback su context per i testi definitori. Il
        multi-livello (expand_depth) e' qui mantenuto per parita' funzionale.
        """
        def _say(m: str) -> None:
            if progress:
                progress(m)

        expand_top_n = params.get("expand_top_n", 3)
        expand_depth = params.get("expand_depth", 1)
        expand_children_per_level = params.get("expand_children_per_level", 2)
        local = {"expanded": 0, "expand_degraded": 0, "expand_by_level": {}}

        observed_causal = [
            it for it in self.ft.items
            if (it.metadata or {}).get("origin", "text_observed") == "text_observed"
            and it.nature in (Nature.CAUSE, Nature.EFFECT)
            and it.scale in SCALE_DEPTH
        ]
        observed_causal.sort(key=lambda it: -SCALE_DEPTH[it.scale])
        if observed_causal:
            seed_items = observed_causal[:expand_top_n]
            _say(f"[fase expand] livello 1: {len(seed_items)} item cause/effect")
        else:
            observed_context = [
                it for it in self.ft.items
                if (it.metadata or {}).get("origin", "text_observed") == "text_observed"
                and it.nature == Nature.CONTEXT
                and it.scale in SCALE_DEPTH
            ]
            observed_context.sort(key=lambda it: -SCALE_DEPTH[it.scale])
            seed_items = observed_context[:expand_top_n]
            _say(f"[fase expand] testo definitorio: {len(seed_items)} item context come semi")

        def _expand_level(items_to_expand: list, level: int) -> list:
            produced: list = []
            for it in items_to_expand:
                _say(f"[fase expand] L{level} expand {it.id} ({it.scale}/{it.nature.value})")
                rec = self.expand(it.id)
                if rec.degraded and not rec.children:
                    local["expand_degraded"] += 1
                else:
                    local["expanded"] += 1
                    produced.extend(rec.children)
            local["expand_by_level"][f"L{level}"] = len(items_to_expand)
            return produced

        children = _expand_level(seed_items, level=1)
        for level in range(2, max(1, expand_depth) + 1):
            if not children:
                break
            candidates = [
                c for c in children
                if c.direction != ExpansionDirection.COHERENCE_BRIDGE
            ]
            candidates.sort(key=lambda c: -c.confidence)
            next_seed = [c.item for c in candidates[:expand_children_per_level]]
            if not next_seed:
                break
            _say(f"[fase expand] livello {level}: ri-espando {len(next_seed)} figli")
            children = _expand_level(next_seed, level=level)
        return local

    def _phase_bridge(self, params: dict[str, Any], progress) -> dict[str, Any]:
        """Fase 'bridge' isolata e rieseguibile. Ritorna stats parziali."""
        def _say(m: str) -> None:
            if progress:
                progress(m)

        max_bridges = params.get("max_bridges", 3)
        local = {"bridges_built": 0, "bridge_degraded": 0}

        text_observed_items = [
            it for it in self.ft.items
            if (it.metadata or {}).get("origin", "text_observed") == "text_observed"
            and it.scale in SCALE_DEPTH
        ]
        bridge_candidates = _find_cross_scale_pairs(text_observed_items)
        used_fallback = False
        if not bridge_candidates:
            _say("[fase bridge] nessuna coppia text_observed cross-scale; fallback")
            extended = text_observed_items + [
                it for it in self.ft.items
                if (it.metadata or {}).get("origin") in ("expansion", "bridge")
                and it.scale in SCALE_DEPTH
            ]
            bridge_candidates = _find_cross_scale_pairs(extended)
            used_fallback = bool(bridge_candidates)
        bridge_candidates.sort(key=lambda t: -t[2])
        bridge_candidates = bridge_candidates[:max_bridges]
        mode = "(fallback expansion)" if used_fallback else "(text_observed)"
        _say(f"[fase bridge] costruisco {len(bridge_candidates)} bridge {mode}")
        for a, b, dist in bridge_candidates:
            if SCALE_DEPTH[a.scale] > SCALE_DEPTH[b.scale]:
                src, tgt = a, b
            else:
                src, tgt = b, a
            mid_depth = (SCALE_DEPTH[src.scale] + SCALE_DEPTH[tgt.scale]) // 2
            gap_scale = SCALES_CANONICAL[mid_depth]
            _say(f"[fase bridge] {src.id}({src.scale}) -> {tgt.id}({tgt.scale}) @ {gap_scale}")
            rec = self.bridge(src.id, tgt.id, gap_scale)
            if rec.degraded:
                local["bridge_degraded"] += 1
            else:
                local["bridges_built"] += 1
        return local

    def _phase_revalidate(self, params: dict[str, Any], progress) -> dict[str, Any]:
        """Fase 'revalidate' isolata e rieseguibile. Ritorna stats parziali."""
        def _say(m: str) -> None:
            if progress:
                progress(m)
        only_uncertain = params.get("only_uncertain", True)
        if not only_uncertain:
            _say("[fase revalidate] estesa a TUTTE le ipotesi (only_uncertain=False)")
        else:
            _say("[fase revalidate] solo ipotesi 'uncertain'")
        return {"revalidate": self.revalidate_cross(only_uncertain=only_uncertain)}

    def _phase_magistrale(self, params: dict[str, Any], progress) -> dict[str, Any]:
        """Fase 'magistrale' isolata. E' la chiusura: non si ripete."""
        def _say(m: str) -> None:
            if progress:
                progress(m)
        _say("[fase magistrale] genero relazione magistrale")
        self.magistrale()
        return {"magistrale": True}

    def _run_directed(
        self,
        director: "Director",
        params: dict[str, Any],
        *,
        enabled_phases: set[str],
        progress=None,
    ) -> dict[str, Any]:
        """Motore di esecuzione governato dal Regista (pieno controllo).

        Esegue una fase alla volta. Dopo ogni fase fa osservare il Regista;
        in base al verbo di regia (proceed/skip/repeat/goto/halt) decide
        quale fase eseguire dopo. La guardia anti-loop del Director limita
        gli atti di flusso: esaurito il budget, ogni deviazione e' degradata
        a 'proceed' e l'Attore prosegue verso la chiusura.

        params: i parametri base di tutte le fasi. param_overrides del Regista
        vi vengono fusi sopra, in modo non distruttivo per le fasi seguenti.
        enabled_phases: le fasi che il chiamante ha abilitato (i flag --no-*).
        """
        from .ft_director import (
            ACTOR_PHASES, CLOSING_PHASE,
            CONTROL_GOTO, CONTROL_HALT, CONTROL_PROCEED, CONTROL_REPEAT, CONTROL_SKIP,
        )

        def _say(m: str) -> None:
            if progress:
                progress(m)

        phase_fns = {
            "expand": self._phase_expand,
            "bridge": self._phase_bridge,
            "revalidate": self._phase_revalidate,
            "magistrale": self._phase_magistrale,
        }
        stats: dict[str, Any] = {
            "expanded": 0, "expand_degraded": 0, "expand_by_level": {},
            "bridges_built": 0, "bridge_degraded": 0,
            "revalidate": None, "magistrale": False,
            "director_interventions": 0,
        }
        run_params = dict(params)              # mutabile: gli override vi si fondono
        idx = 0                                 # indice nella sequenza canonica
        steps_done = 0
        MAX_STEPS = 24                          # tetto assoluto, oltre la guardia budget

        while idx < len(ACTOR_PHASES) and steps_done < MAX_STEPS:
            phase = ACTOR_PHASES[idx]
            steps_done += 1

            # fase disabilitata dai flag del chiamante (--no-bridges, ecc.)
            if phase not in enabled_phases:
                _say(f"[regia] fase '{phase}' disabilitata, salto")
                director.report.executed_phases.append(f"{phase}:disabled")
                idx += 1
                continue

            # --- esegui la fase ---
            _say(f"[regia] eseguo fase '{phase}'")
            local = phase_fns[phase](run_params, progress)
            for k, v in local.items():
                if k == "expand_by_level":
                    stats[k].update(v)
                elif isinstance(stats.get(k), int) and isinstance(v, int):
                    stats[k] += v
                else:
                    stats[k] = v
            director.report.executed_phases.append(phase)

            # la chiusura termina sempre il loop
            if phase == CLOSING_PHASE:
                break

            # --- il Regista osserva e decide il flusso ---
            trace: list[str] = []
            intervention = director.observe(self.ft, phase=phase, trace=trace)
            self.ft.trace.extend(trace)

            if intervention is None:
                _say(f"[regia] Regista osserva dopo '{phase}': prosegue l'ordine naturale")
                idx += 1
                continue

            stats["director_interventions"] += 1
            control = intervention.control

            # budget esaurito: ogni deviazione degradata a proceed
            if control != CONTROL_PROCEED and director.budget_exhausted():
                _say(f"[regia] budget di regia esaurito: '{control}' degradato a proceed")
                control = CONTROL_PROCEED

            # applica gli override di parametro (validi per qualsiasi verbo)
            if intervention.param_overrides:
                run_params.update(intervention.param_overrides)
                _say(f"[regia] parametri aggiornati: {intervention.param_overrides}")

            # --- esegui il verbo di regia ---
            if control == CONTROL_PROCEED:
                _say(f"[regia] PROCEED: continuo con la fase successiva")
                idx += 1
            elif control == CONTROL_SKIP:
                director.register_control_act()
                skipped = ACTOR_PHASES[idx + 1] if idx + 1 < len(ACTOR_PHASES) else "(nessuna)"
                _say(f"[regia] SKIP: salto la fase '{skipped}'")
                director.report.executed_phases.append(f"{skipped}:skipped")
                idx += 2
            elif control == CONTROL_REPEAT:
                director.register_control_act()
                _say(f"[regia] REPEAT: ri-eseguo la fase '{phase}'")
                # idx invariato: la prossima iterazione rifa' la stessa fase
            elif control == CONTROL_GOTO:
                director.register_control_act()
                dest = intervention.goto_phase
                if dest in ACTOR_PHASES:
                    _say(f"[regia] GOTO: salto a '{dest}'")
                    idx = ACTOR_PHASES.index(dest)
                else:
                    _say(f"[regia] GOTO con destinazione ignota '{dest}': proseguo")
                    idx += 1
            elif control == CONTROL_HALT:
                director.register_control_act()
                _say(f"[regia] HALT: il Regista ferma l'Attore")
                director.report.halted = True
                break
            else:
                idx += 1

        if steps_done >= MAX_STEPS:
            _say("[regia] raggiunto il tetto assoluto di passi: arresto di sicurezza")
            director.report.halted = True
        return stats

    def observe_with_director(
        self,
        *,
        expand_top_n: int = 3,
        expand_depth: int = 1,
        expand_children_per_level: int = 2,
        build_bridges: bool = True,
        max_bridges: int = 3,
        do_revalidate: bool = True,
        do_magistrale: bool = True,
        silence_band: float = 1.5,
        divergence_threshold: float = 0.34,
        control_budget: int = 6,
        narrate: bool = True,
        progress=None,
    ) -> dict[str, Any]:
        """One-shot CON il Regista (L7) a PIENO CONTROLLO.

        L'Attore non segue piu' una catena cablata: ogni fase e' un metodo
        rieseguibile, e il Regista, fra una fase e l'altra, decide il flusso
        (proceed/skip/repeat/goto/halt). I guard di V14 restano intatti: il
        Regista governa l'ESECUZIONE delle fasi, non riscrive il ft ne'
        bypassa predicate/nature/scale/zoom-coherence.

        control_budget: numero massimo di atti di flusso (skip/repeat/goto/
        halt) prima che il Regista perda il potere di deviare e possa solo
        osservare. E' la rete anti-loop del pieno controllo.

        narrate: se True, il Regista chiude con una chiamata LLM che racconta
        l'auto-osservazione.
        """
        director = Director(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
            silence_band=silence_band,
            divergence_threshold=divergence_threshold,
            narrate=narrate,
        )
        director.control_budget = control_budget

        enabled_phases = {"expand", "magistrale"}
        if build_bridges:
            enabled_phases.add("bridge")
        if do_revalidate:
            enabled_phases.add("revalidate")
        if not do_magistrale:
            enabled_phases.discard("magistrale")

        params = {
            "expand_top_n": expand_top_n,
            "expand_depth": expand_depth,
            "expand_children_per_level": expand_children_per_level,
            "max_bridges": max_bridges,
            "only_uncertain": True,
        }

        stats = self._run_directed(
            director, params, enabled_phases=enabled_phases, progress=progress
        )

        trace: list[str] = [self._stamp("director_finalize", "")]
        report = director.finalize(self.ft, trace=trace)
        self.ft.trace.extend(trace)
        attach_director_report(self.ft, report)
        self.call_counter += 1

        (self.out_dir / "director_report.md").write_text(
            render_director_md(report), encoding="utf-8"
        )
        self.save()

        stats["director_report"] = "director_report.md"
        stats["director_readings"] = len(report.readings)
        stats["director_executed_phases"] = list(report.executed_phases)
        stats["director_halted"] = report.halted
        return stats
