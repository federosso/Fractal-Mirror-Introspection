"""Lettura tematica (V10.19.0). Fork concettuale del motore causale.

Invece di cercare catene di causa->effetto, genera OSSERVAZIONI da quattro
lenti diverse. Pensato per testi non argomentativi -- diari, dialoghi, testi
simbolici, religiosi -- dove la griglia causale del motore principale
forzerebbe una struttura che il testo non ha (ed e' cio' che faceva andare
in loop il Classifier sul libro di canalizzazioni).

LE QUATTRO LENTI
----------------
- simbolica     : immagini, metafore, archetipi -- inclusi i simboli del
                  sacro, del trascendente, dell'energia.
- strutturale   : come e' organizzato il discorso, come costruisce la sua
                  autorita' e il suo ritmo.
- relazionale   : le voci e gli interlocutori, chi parla a chi, le posizioni
                  reciproche -- inclusa la relazione fra l'umano e cio' che
                  il testo presenta come non-umano.
- esperienziale : il vissuto interiore riferito dal testo -- stati, percezioni,
                  esperienza -- incluso il piano metafisico, energetico,
                  extrasensoriale, spirituale.

ONESTA' EPISTEMICA
------------------
Una lente RILEVA e MAPPA come un testo costruisce il suo discorso. NON si
pronuncia sulla verita' di cio' che il testo afferma. Su un diario di
canalizzazione, la lente esperienziale osserva COME il testo descrive il
contatto con le guide -- non se quel contatto sia reale. La lente legge il
testo, non il referente del testo. Ogni prompt lo dichiara esplicitamente.

La sintesi finale e' PLURALE: non un verdetto ("il testo e' X") ma un
riepilogo di angolazioni ("da quattro lenti il testo mostra X, Y, Z").
Riusa la struttura observer + ricorsione del motore, cambia solo la lente.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .ft_model import (
    THEMATIC_LENSES,
    Observation,
    ThematicMotif,
    ThematicReading,
)
from .llm import LLMClient, RoleAgent
from .io_utils import write_text


# -----------------------------------------------------------------------------
# Prompt delle lenti. Uno per lente; tutti condividono la clausola di onesta'.
# -----------------------------------------------------------------------------

_HONESTY_CLAUSE = """
REGOLA DI ONESTA' (vincolante): osservi COME il testo costruisce il suo
discorso. NON ti pronunci sulla verita' di cio' che il testo afferma. Se il
testo parla di entita', energie o piani spirituali, osservi COME ne parla --
con quale linguaggio, quali immagini, quale struttura -- non SE siano reali.
Leggi il testo, non il referente del testo. Niente verdetti di verita'.
"""

_LENS_PROMPTS: dict[str, str] = {
    "simbolica": """Sei la LENTE SIMBOLICA di un sistema di lettura tematica.
Osservi immagini, metafore, archetipi e simboli del testo -- inclusi i
simboli del sacro, del trascendente, dell'energia, del divino.
Per ogni elemento simbolico rilevante produci un'osservazione: cosa e'
l'immagine, che cosa evoca nel testo, dove e come compare.
""" + _HONESTY_CLAUSE,

    "strutturale": """Sei la LENTE STRUTTURALE di un sistema di lettura tematica.
Osservi come e' ORGANIZZATO il discorso: le sue parti, il ritmo, le forme
ricorrenti (premesse, dialoghi, formule, intestazioni), come il testo
costruisce la propria autorita' e coerenza interna.
Produci osservazioni sulla forma e l'architettura del testo.
""" + _HONESTY_CLAUSE,

    "relazionale": """Sei la LENTE RELAZIONALE di un sistema di lettura tematica.
Osservi le VOCI e gli interlocutori del testo: chi parla, a chi, con quale
posizione reciproca; come si configurano i rapporti -- inclusa la relazione
fra l'umano e cio' che il testo presenta come non-umano (guide, entita',
voci). Produci osservazioni sulle relazioni e le posizioni di parola.
""" + _HONESTY_CLAUSE,

    "esperienziale": """Sei la LENTE ESPERIENZIALE di un sistema di lettura tematica.
Osservi il VISSUTO INTERIORE che il testo riferisce: stati d'animo,
percezioni, esperienze soggettive -- incluso il piano metafisico, energetico,
extrasensoriale e spirituale cosi' come il testo lo descrive.
Produci osservazioni su come il testo racconta l'esperienza interiore.
""" + _HONESTY_CLAUSE,
}

_OBSERVATION_CONTRACT: dict[str, Any] = {
    "observations": [
        {
            "focus": "<il punto del testo osservato, breve>",
            "note": "<l'osservazione: cosa nota la lente>",
            "evidence": "<breve citazione o riferimento dal testo>",
            "salience": "<numero 0..1: quanto e' centrale>",
        }
    ]
}

# Prompt e contratto per la sintesi plurale finale.
_SYNTHESIS_PROMPT = """Sei il SINTETIZZATORE di un sistema di lettura tematica.
Ricevi le osservazioni raccolte da quattro lenti (simbolica, strutturale,
relazionale, esperienziale) su un testo.

Il tuo compito: una SINTESI PLURALE. NON un verdetto, NON una conclusione su
cosa il testo "sia" o se cio' che afferma sia vero. Un riepilogo onesto delle
angolazioni: cosa ciascuna lente ha messo in luce, dove le lenti convergono,
dove offrono lo stesso testo da prospettive diverse.
Scrivi 6-12 frasi, in italiano, in prosa piana. Niente giudizi di verita'.
"""

_SYNTHESIS_CONTRACT: dict[str, Any] = {
    "synthesis": "<6-12 frasi: sintesi plurale delle quattro lenti>",
    "motifs": [
        {
            "name": "<nome breve del motivo ricorrente>",
            "lens": "<lente che lo ha colto>",
            "occurrences": ["<punto 1>", "<punto 2>"],
            "transformation": "<come il motivo cambia attraverso il testo, se cambia>",
        }
    ],
}


# -----------------------------------------------------------------------------
# Il motore di lettura tematica.
# -----------------------------------------------------------------------------


class ThematicReader:
    """Legge un testo attraverso le quattro lenti tematiche.

    Uso:
        reader = ThematicReader(client, out_dir)
        reading = reader.read(text)         # ThematicReading
    """

    def __init__(
        self,
        client: LLMClient,
        out_dir: str | Path,
        *,
        max_output_tokens: int = 1200,
    ) -> None:
        self.client = client
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.llm_calls_dir = self.out_dir / "llm_calls"
        self.llm_calls_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.out_dir / "telemetry.jsonl"
        self.max_output_tokens = max_output_tokens

    def _run_lens(
        self, lens: str, text: str, trace: list[str]
    ) -> list[Observation]:
        """Esegue una lente sul testo. Ritorna le sue osservazioni.

        Se la lente fallisce (JSON non valido, errore LLM) ritorna lista
        vuota e annota nel trace: una lente persa non blocca le altre.
        """
        agent = RoleAgent(
            self.client,
            role_name=f"L_Thematic_{lens}",
            role_prompt=_LENS_PROMPTS[lens],
            out_dir=self.llm_calls_dir,
            max_output_tokens=self.max_output_tokens,
        )
        payload = {"lens": lens, "text": text}
        raw, _meta = agent.run_json(
            payload, _OBSERVATION_CONTRACT, trace,
            telemetry_path=self.telemetry_path,
        )
        observations: list[Observation] = []
        if not isinstance(raw, dict) or "observations" not in raw:
            trace.append(f"L_Thematic_{lens}: nessuna osservazione valida")
            return observations
        for o in raw.get("observations", []):
            if not isinstance(o, dict):
                continue
            try:
                sal = float(o.get("salience", 0.5))
            except (TypeError, ValueError):
                sal = 0.5
            observations.append(
                Observation(
                    lens=lens,
                    focus=str(o.get("focus", "")).strip(),
                    note=str(o.get("note", "")).strip(),
                    evidence=str(o.get("evidence", "")).strip(),
                    salience=max(0.0, min(1.0, sal)),
                )
            )
        trace.append(f"L_Thematic_{lens}: {len(observations)} osservazioni")
        return observations

    def _synthesize(
        self, observations: list[Observation], trace: list[str]
    ) -> tuple[str, list[ThematicMotif]]:
        """Sintesi plurale + motivi ricorrenti dalle osservazioni."""
        if not observations:
            return "Nessuna osservazione raccolta: sintesi non disponibile.", []
        agent = RoleAgent(
            self.client,
            role_name="L_Thematic_Synthesis",
            role_prompt=_SYNTHESIS_PROMPT,
            out_dir=self.llm_calls_dir,
            max_output_tokens=self.max_output_tokens,
        )
        payload = {
            "observations": [
                {"lens": o.lens, "focus": o.focus, "note": o.note}
                for o in observations
            ]
        }
        raw, _meta = agent.run_json(
            payload, _SYNTHESIS_CONTRACT, trace,
            telemetry_path=self.telemetry_path,
        )
        synthesis = ""
        motifs: list[ThematicMotif] = []
        if isinstance(raw, dict):
            synthesis = str(raw.get("synthesis", "")).strip()
            for m in raw.get("motifs", []) or []:
                if not isinstance(m, dict):
                    continue
                occ = m.get("occurrences", [])
                motifs.append(
                    ThematicMotif(
                        name=str(m.get("name", "")).strip(),
                        lens=str(m.get("lens", "")).strip(),
                        occurrences=[str(x) for x in occ] if isinstance(occ, list) else [],
                        transformation=str(m.get("transformation", "")).strip(),
                    )
                )
        if not synthesis:
            synthesis = "Sintesi non disponibile (il modello non l'ha prodotta)."
        trace.append(
            f"L_Thematic_Synthesis: sintesi prodotta, {len(motifs)} motivi"
        )
        return synthesis, motifs

    def read(
        self,
        text: str,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> ThematicReading:
        """Legge `text` con le quattro lenti e produce una ThematicReading."""
        say = progress if progress is not None else (lambda _m: None)
        trace: list[str] = []
        reading = ThematicReading()

        if not text.strip():
            reading.notes.append("Testo vuoto: nessuna lettura.")
            return reading

        for lens in THEMATIC_LENSES:
            say(f"[tematica] lente '{lens}' ...")
            obs = self._run_lens(lens, text, trace)
            reading.observations.extend(obs)
            if not obs:
                reading.notes.append(f"Lente '{lens}': nessuna osservazione.")

        say("[tematica] sintesi plurale delle quattro lenti ...")
        synthesis, motifs = self._synthesize(reading.observations, trace)
        reading.synthesis = synthesis
        reading.motifs = motifs

        # trace su disco, coerente col resto del motore
        write_text("\n".join(trace) + "\n", self.out_dir / "thematic_trace.txt")
        say(f"[tematica] completata: {len(reading.observations)} osservazioni, "
            f"{len(reading.motifs)} motivi")
        return reading


# -----------------------------------------------------------------------------
# Render.
# -----------------------------------------------------------------------------


def render_thematic_md(reading: ThematicReading, original_text: str = "") -> str:
    """Rende la lettura tematica in markdown."""
    lines: list[str] = ["# Lettura tematica -- Fractal Triad V10.19.0", ""]

    if original_text and original_text.strip():
        lines += ["## 0. Testo analizzato", ""]
        for row in original_text.strip().splitlines():
            lines.append(f"> {row}" if row.strip() else ">")
        lines += [""]

    lines += [
        "## 1. Sintesi plurale", "",
        "_Non un verdetto: un riepilogo delle angolazioni delle quattro lenti._",
        "",
        reading.synthesis or "_(sintesi non disponibile)_",
        "",
    ]

    lines += ["## 2. Osservazioni per lente", ""]
    for lens in THEMATIC_LENSES:
        obs = reading.by_lens(lens)
        lines.append(f"### Lente {lens}")
        if not obs:
            lines += ["", "_Nessuna osservazione._", ""]
            continue
        lines.append("")
        for o in sorted(obs, key=lambda x: -x.salience):
            lines.append(f"- **{o.focus}** (salienza {o.salience:.2f})")
            lines.append(f"  - {o.note}")
            if o.evidence:
                lines.append(f"  - _testo:_ {o.evidence}")
        lines.append("")

    lines += ["## 3. Motivi ricorrenti", ""]
    if not reading.motifs:
        lines += ["_Nessun motivo ricorrente individuato._", ""]
    else:
        for m in reading.motifs:
            lines.append(f"- **{m.name}** _(lente {m.lens})_")
            if m.occurrences:
                lines.append(f"  - ricorre in: {', '.join(m.occurrences)}")
            if m.transformation:
                lines.append(f"  - trasformazione: {m.transformation}")
        lines.append("")

    if reading.notes:
        lines += ["## Note", ""]
        for n in reading.notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)
