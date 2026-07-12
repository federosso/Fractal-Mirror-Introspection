"""L1 -- Classifier.

Riceve il testo grezzo. Produce una lista di ClassifiedItem:
  - quote letterale (<=25 parole),
  - predicate (definition / process_description / claimed_property / event / ...),
  - nature (cause / effect / context / bridge / interpretation),
  - scale (una delle 9 canoniche).

Una sola chiamata LLM. Validator Python che forza i vincoli duri:
  - quote presente nel testo (case-insensitive, normalizzato);
  - quote <= 25 parole, altrimenti il claim e' scartato;
  - scale appartiene a SCALES_CANONICAL;
  - nature appartiene all'enum.

Niente fallback hardcoded per dominio. Se il modello fallisce, restituiamo
una lista vuota con trace esplicito. Mai inventare il dominio dalle keywords.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .ft_model import (
    SCALES_CANONICAL,
    ClassifiedItem,
    EpistemicStatus,
    Nature,
    PredicateType,
    is_valid_scale,
)
from .llm import LLMClient, RoleAgent
from .text import stable_hash
from .ft_budget import budget


CLASSIFIER_PROMPT = """Sei il CLASSIFICATORE del Fractal Causal Engine.

Il tuo compito: spezzare il testo in PROPOSIZIONI MINIME e classificare ognuna su due assi.

REGOLE INDEROGABILI
1. Una proposizione = una sola affermazione. Mai due predicati uniti. Mai elenchi compatti.
2. quote: COPIA LETTERALMENTE dal testo, massimo 25 parole. Se la frase originale e' piu' lunga, spezzala e ripeti la classificazione su ciascun pezzo.
3. Non aggiungere conoscenza esterna. Non interpretare. Solo cio' che il testo afferma.
4. predicate -- cosa fa la frase nel testo:
   - definition: "X e' il termine per indicare Y", "con X si intende Y", "X si definisce come Y"
   - process_description: "X agisce", "X stimola Y", "X genera Y" (in un contesto osservato/osservabile)
   - claimed_property: "X ha la proprieta' P", "X non ha P", "X e' senza P"
   - event: "e' successo X" (azione singola, datata o no)
   - state: "X e' nello stato Y" (stativo)
   - comparison: "X a differenza di Y...", "diversamente da X..."
   - question: una DOMANDA. "Che cos'e' X?", "come si fa Y?", "perche' Z?".
     Una domanda NON e' un'asserzione: non afferma nulla, chiede. Va
     SEMPRE marcata predicate=question e nature=context (e' il quesito di
     partenza, non un fattore causale).
5. nature -- la natura causale di cio' che la frase dice:
   - cause: condizione, stimolo, evento originante REALMENTE OSSERVATO o REALMENTE INNESCANTE
   - effect: manifestazione, esito, conseguenza OSSERVATA
   - context: cornice, sfondo, definizione, proprieta' che NON innesca nulla nel testo;
     anche le DOMANDE (predicate=question) sono sempre context
   - bridge: passaggio intermedio possibile, non causa autonoma
   - interpretation: lettura simbolica/spirituale/significato attribuito
6. scale -- la scala ontologica del REFERENTE della frase, da queste 9:
   cosmologico, planetario, sociale, organismo, cellulare, molecolare, atomico, subatomico, fondamentale.
   Scegli la scala dove vive il fenomeno descritto, non la scala dell'autore.

REGOLA CRUCIALE -- GLOSSE DEFINITORIE
Se la frase contiene una SUBORDINATA RELATIVA o FINALE dentro una DEFINIZIONE
del soggetto (struttura: "X = Y che fa Z al fine di W"), la subordinata
NON e' una catena causale osservata. E' una GLOSSA DEFINITORIA: descrive
COSA E' il soggetto, non un evento accaduto. Verbi come "stimolare",
"generare", "produrre", "al fine di", "attraverso", "mediante" dentro una
definizione restano parte della DEFINIZIONE: emetti UN solo item con
predicate=definition o predicate=process_description e nature=context.

NON spezzare mai una glossa definitoria in cause+effect anche se contiene
verbi causali. Lo si fa solo se il testo asserisce un'osservazione reale
(es. "Tizio stimola la materia e questa genera energia" come fatto osservato).

ESEMPIO 1 -- DEFINIZIONE CON GLOSSA SUBORDINATA (NON spezzare in causa+effetto):
TESTO: "Con l'acronimo LENR si definiscono una serie di fenomeni - tra i
quali fusione fredda - che stimolano la materia al fine di estrarre energia."
ITEMS:
  - quote: "Con l'acronimo LENR si definiscono una serie di fenomeni"
    predicate: definition, nature: context, scale: atomico
  - quote: "tra i quali fusione fredda"
    predicate: definition, nature: context, scale: subatomico
  - quote: "che stimolano la materia al fine di estrarre energia"
    predicate: process_description, nature: context, scale: atomico
    (NB: questa e' la GLOSSA DEFINITORIA del soggetto LENR, NON un evento
    osservato. Non emettere un secondo item "estrarre energia" con
    nature=effect: e' parte della stessa definizione.)

ESEMPIO 2 -- COMPARAZIONE DIFFERENZIALE (NON spezzare in causa+effetto):
TESTO: "A differenza della fissione, le LENR generano energia senza scorie radioattive."
ITEMS:
  - quote: "A differenza della fissione"
    predicate: comparison, nature: context, scale: atomico
  - quote: "le LENR generano energia senza scorie radioattive"
    predicate: claimed_property, nature: context, scale: atomico
    (NB: "generano" qui descrive una PROPRIETA' DICHIARATA di LENR rispetto
    al confronto, non un evento osservato.)

ESEMPIO 3 -- CAUSALITA' OSSERVATA (questa SI si spezza):
TESTO: "Dopo il tamponamento ho avuto un colpo di frusta e da allora ho fobia delle macchine."
ITEMS:
  - quote: "Dopo il tamponamento"
    predicate: event, nature: cause, scale: organismo
  - quote: "ho avuto un colpo di frusta"
    predicate: event, nature: effect, scale: organismo
  - quote: "da allora ho fobia delle macchine"
    predicate: state, nature: effect, scale: organismo

REGOLA DI DENSITA' CAUSALE
Se l'intero testo e' DEFINITORIO (esempio: voce di dizionario, paragrafo
introduttivo che dice "X e' Y", comparazioni differenziali) e NON contiene
una sequenza temporale o causale osservata, allora TUTTI gli item devono
avere nature=context. Nessun item cause/effect. Il magistrale lavorera'
con le ESPANSIONI per costruire ipotesi causali; non e' tuo compito
forzare causalita' dove il testo non la afferma.

VINCOLI FINALI
- Mai claim con quote > 25 parole. Spezza.
- Mai inventare scale fuori dalle 9 elencate.
- Se non sei sicuro della nature, scegli context.
- Restituisci JSON valido come da OUTPUT_CONTRACT.
"""


CLASSIFIER_CONTRACT: dict[str, Any] = {
    "items": [
        {
            "quote": "<copia letterale, <=25 parole>",
            "predicate": "definition|process_description|claimed_property|event|state|comparison|question",
            "nature": "cause|effect|context|bridge|interpretation",
            "scale": "cosmologico|planetario|sociale|organismo|cellulare|molecolare|atomico|subatomico|fondamentale",
            "rationale": "<breve, perche' questa classificazione>",
        }
    ]
}


MAX_WORDS = 25


def _normalize(text: str) -> str:
    """Normalizza per match: minuscole, NFKD, niente accenti, whitespace collassato."""
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _quote_in_text(quote: str, text: str) -> bool:
    """True se la quote appare nel testo (case-insensitive, senza accenti)."""
    if not quote:
        return False
    return _normalize(quote) in _normalize(text)


def _word_count(quote: str) -> int:
    return len([w for w in re.split(r"\s+", (quote or "").strip()) if w])


class Classifier:
    """L1 -- estrae ClassifiedItem dal testo. Una chiamata LLM, validatori puri."""

    def __init__(
        self,
        client: LLMClient,
        *,
        llm_calls_dir: Path | None,
        telemetry_path: Path | None = None,
    ) -> None:
        self.agent = RoleAgent(
            client,
            role_name="L1_Classifier",
            role_prompt=CLASSIFIER_PROMPT,
            out_dir=llm_calls_dir,
            max_output_tokens=budget("l1_classifier"),
        )
        self.telemetry_path = telemetry_path

    def run(self, text: str, source_input_id: str, trace: list[str]) -> tuple[list[ClassifiedItem], dict[str, Any]]:
        payload = {
            "task": "classify_text_into_minimal_propositions",
            "input_text": text,
            "scales_allowed": SCALES_CANONICAL,
        }
        raw, meta = self.agent.run_json(payload, CLASSIFIER_CONTRACT, trace, telemetry_path=self.telemetry_path)
        items_raw = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items_raw, list):
            trace.append("L1_Classifier: no items list in output -> empty result")
            return [], {"call_meta": meta, "rejected": 0, "accepted": 0}

        accepted: list[ClassifiedItem] = []
        rejected: list[dict[str, Any]] = []
        for i, raw_item in enumerate(items_raw):
            if not isinstance(raw_item, dict):
                rejected.append({"index": i, "reason": "not_a_dict"})
                continue
            quote = str(raw_item.get("quote") or "").strip().strip("\"")
            if not quote:
                rejected.append({"index": i, "reason": "empty_quote"})
                continue
            if _word_count(quote) > MAX_WORDS:
                rejected.append({"index": i, "reason": "quote_too_long", "words": _word_count(quote)})
                continue
            if not _quote_in_text(quote, text):
                rejected.append({"index": i, "reason": "quote_not_in_text", "quote": quote[:60]})
                continue
            scale = str(raw_item.get("scale") or "").strip().lower()
            if not is_valid_scale(scale):
                rejected.append({"index": i, "reason": "invalid_scale", "scale": scale})
                continue
            predicate = self._coerce_predicate(raw_item.get("predicate"))
            nature = self._coerce_nature(raw_item.get("nature"))
            item_id = "itm_" + stable_hash(f"{source_input_id}|{i}|{quote}", 10)
            accepted.append(
                ClassifiedItem(
                    id=item_id,
                    quote=quote,
                    predicate=predicate,
                    nature=nature,
                    scale=scale,
                    rationale=str(raw_item.get("rationale") or "")[:240],
                    source_input_id=source_input_id,
                    epistemic_status=EpistemicStatus.TEXT_OBSERVED,
                )
            )
        trace.append(
            f"L1_Classifier: accepted={len(accepted)} rejected={len(rejected)} "
            f"(reasons={[r['reason'] for r in rejected[:5]]})"
        )
        return accepted, {"call_meta": meta, "accepted": len(accepted), "rejected": rejected}

    @staticmethod
    def _coerce_predicate(value: Any) -> PredicateType:
        s = str(value or "").strip().lower()
        for pt in PredicateType:
            if pt.value == s:
                return pt
        return PredicateType.UNKNOWN

    @staticmethod
    def _coerce_nature(value: Any) -> Nature:
        s = str(value or "").strip().lower()
        for n in Nature:
            if n.value == s:
                return n
        return Nature.CONTEXT
