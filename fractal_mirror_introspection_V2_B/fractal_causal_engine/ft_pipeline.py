"""Pipeline driver V10.14.0.

L0 DomainRouter (riuso) -> L1 Classifier -> L2 Locked-per-scala
                       -> L3A UnlockedExplorer -> L3B CrossScaleValidator
                       -> L4 Orchestrator.

Output:
- FractalTriadResult (dataclass, serializzabile)
- final_report.md
- ft_analysis.json (dump completo strutturato)
- llm_calls/*.json (gia' scritti da RoleAgent)
- trace.md (riepilogo dei livelli)

NON tocca l'orchestratore legacy: vive in parallelo. Il CLI puo' scegliere
con un flag --pipeline=v14 quale invocare.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .ft_classifier import Classifier
from .ft_crossscale import CrossScaleValidator
from .ft_locked import LockedPerScale
from .ft_model import FractalTriadResult
from .ft_orchestrator import FractalTriadOrchestrator, render_final_report_md
from .ft_unlocked import UnlockedExplorer
from .llm import LLMClient, sweep_orphan_records


# Sostituisce il vecchio domain_router con una euristica minima sui set di
# termini canonici. Ritorna solo il dominio dominante: tutto il resto
# (filtri, lenti, ecc.) era infrastruttura V13.1 morta e non viene riportata.
_DOMAIN_TERMS: dict[str, set[str]] = {
    "scientific_energy_matter": {
        "lenr", "fusione fredda", "fusione a bassa energia", "fissione",
        "trasmutazione", "metamorfosi", "cavitazione", "materia", "energia",
        "scorie", "radioattive", "radioattivo", "temperatura", "nucleare",
        "reticolo", "fononi", "coulomb", "palladio", "nichel", "deuterio",
        "idrogeno",
    },
    "human_experience_or_mixed": {
        "paura", "ansia", "fobia", "panico", "trauma", "impotenza",
        "memoria emotiva", "evitamento", "ipervigilanza", "anima",
        "simbol", "spiritual", "senso",
    },
}


def _diagnose_dominant_domain(text: str) -> str:
    """Heuristica minima sul dominio del testo. Solo per il campo vision."""
    t = (text or "").lower()
    scored = {
        name: sum(1 for term in terms if term in t)
        for name, terms in _DOMAIN_TERMS.items()
    }
    name, score = max(scored.items(), key=lambda kv: kv[1])
    if score >= 3:
        return name
    if score >= 1:
        return name + "_weak"
    return "generic_or_mixed"


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


class FractalTriadPipeline:
    """Driver dei 5 livelli. Una sola passata per un testo."""

    def __init__(
        self,
        client: LLMClient,
        out_dir: str | Path,
        *,
        max_cross_scale_candidates: int = 8,
    ) -> None:
        self.client = client
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.llm_calls_dir = self.out_dir / "llm_calls"
        self.llm_calls_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.out_dir / "telemetry.jsonl"
        self.max_cross_scale_candidates = max_cross_scale_candidates

    def run(self, text: str, *, source_input_id: str = "input_001") -> FractalTriadResult:
        trace: list[str] = []
        started = datetime.now().isoformat(timespec="seconds")
        trace.append(f"PIPELINE_START at={started}")

        # V10.17.2.1: pulizia dei record orfani di un run precedente morto a
        # meta' (status='started' mai chiuso). Una volta sola, qui all'avvio,
        # prima che gli agent partano.
        orphans = sweep_orphan_records(self.llm_calls_dir)
        if orphans:
            trace.append(
                f"SWEEP_ORPHANS: rimossi {len(orphans)} record incompleti di "
                f"un run precedente: {', '.join(orphans)}"
            )

        # L0 -- diagnosi minima del dominio (solo per il campo vision)
        dominant_domain = _diagnose_dominant_domain(text)
        trace.append(f"L0_DomainDiagnose: dominant={dominant_domain}")

        # L1 -- classifier
        classifier = Classifier(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
        )
        items, _l1_meta = classifier.run(text, source_input_id=source_input_id, trace=trace)

        # L2 -- locked per scala
        locked = LockedPerScale(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
        )
        locked_reports = locked.run(items, trace)

        # L3A -- unlocked explorer (4 micro + sintesi)
        explorer = UnlockedExplorer(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
        )
        unlocked = explorer.run(text, items, trace)

        # L3B -- cross-scale validator
        validator = CrossScaleValidator(
            self.client,
            llm_calls_dir=self.llm_calls_dir,
            telemetry_path=self.telemetry_path,
            max_candidates=self.max_cross_scale_candidates,
        )
        cross_scale = validator.run(items, locked_reports, trace)

        # L4 -- orchestrator deterministico
        orch = FractalTriadOrchestrator()
        ft = orch.compose(text, items, locked_reports, unlocked, cross_scale, trace)

        finished = datetime.now().isoformat(timespec="seconds")
        ft.trace.append(f"PIPELINE_END at={finished}")
        return ft

    def write_outputs(self, ft: FractalTriadResult, *, original_text: str = "") -> None:
        # ft_analysis.json: dump strutturato
        payload = {
            "items": [_to_jsonable(it) for it in ft.items],
            "locked_reports": [_to_jsonable(r) for r in ft.locked_reports],
            "unlocked": _to_jsonable(ft.unlocked) if ft.unlocked else None,
            "cross_scale": [_to_jsonable(h) for h in ft.cross_scale],
            "double_cone": _to_jsonable(ft.double_cone),
            "vision": _to_jsonable(ft.vision),
            "trace": list(ft.trace),
        }
        (self.out_dir / "ft_analysis.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # final_report.md -- include il testo analizzato (modifica A V10.18.3)
        (self.out_dir / "final_report.md").write_text(
            render_final_report_md(ft, original_text=original_text), encoding="utf-8"
        )
        # trace.md
        trace_md = "# Trace V10.14.0\n\n" + "\n".join(f"- {l}" for l in ft.trace) + "\n"
        (self.out_dir / "trace.md").write_text(trace_md, encoding="utf-8")
