from __future__ import annotations

import json
import socket
import time
from datetime import datetime
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import write_json, write_jsonl_line
from .json_utils import JSONParseError, extract_json


# V10.19.2 -- header HTTP di base per ogni richiesta.
# Lo User-Agent esplicito e' NECESSARIO: alcuni endpoint protetti da
# Cloudflare (es. Groq) bloccano il default 'Python-urllib/3.x' come bot
# generico, rispondendo 403 / error code 1010. Un User-Agent normale
# supera il filtro.
_BASE_HTTP_HEADERS: dict[str, str] = {
    "Content-Type": "application/json",
    "User-Agent": "fractal-causal-engine/0.10.19.3",
}
from .text import short


SYSTEM_JSON = (
    "Sei un agente estremamente disciplinato del sistema FRACTAL EXPLORER / Fractal Causal Engine. "
    "Il tuo unico compito è restituire JSON valido conforme allo schema fornito, senza markdown e senza testo extra. "
    "Devi distinguere sempre text_observed, causal_deep_zoom, bridge_hypothesis, interpretive_hypothesis, speculative e rejected. "
    "Puoi inferire attraverso lo zoom frattale, ma devi dichiarare il grado epistemico dell'inferenza. "
    "Non confondere osservato, inferito, ipotizzato e validato. Non trasformare una correlazione in causa validata."
)


# V10.17.2: quante volte ritentare una chiamata troncata, raddoppiando ogni
# volta num_predict. 2 retry => budget fino a 4x quello iniziale. Oltre, si
# accetta il contenuto parziale e si marca status=truncated.
MAX_TRUNCATION_RETRIES: int = 2


@dataclass
class LLMConfig:
    backend: str = "ollama"
    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    llamacpp_url: str = "http://127.0.0.1:8080"
    # V10.18.3: backend 'groq' -- API OpenAI-compatible, per la "prova del 9"
    # (un modello grande via API, per distinguere limiti del framework da
    # limiti del modello locale). groq_url e groq_api_key sono usati solo
    # quando backend == 'groq'.
    groq_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: str = ""
    temperature: float = 0.1
    top_p: float = 0.9
    timeout_seconds: int = 600
    num_predict: int = 700
    num_ctx: int | None = None
    num_gpu: int | None = 100
    keep_alive: str = "15m"
    mock: bool = False


@dataclass
class ChatResult:
    """Esito di una chiamata al backend, con il segnale di troncamento.

    V10.17.2: prima chat() ritornava solo il contenuto (str) e il
    finish_reason del backend andava perso. Cosi' un JSON troncato per
    esaurimento di num_predict passava inosservato (il parser robusto
    recuperava gli oggetti completi e scartava in silenzio l'ultimo monco).

    finish_reason normalizzato:
      - "stop"   -> generazione completa, il modello ha chiuso da solo;
      - "length" -> TRONCATA: il modello ha colpito il tetto di token;
      - "error"  -> il backend ha segnalato un problema;
      - ""       -> backend che non espone il campo (lo trattiamo come stop).
    truncated e' la lettura comoda: finish_reason == "length".
    """
    content: str
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _normalize_finish(raw_reason: str) -> str:
    """Normalizza il finish_reason dei diversi backend a un vocabolario unico.

    llama.cpp/OpenAI usano 'length'/'stop'; Ollama usa 'load'/'stop' e per il
    troncamento da limite token riporta 'length'. Qualsiasi valore ignoto o
    vuoto viene trattato come 'stop' (completo): meglio non gridare al
    troncamento senza prova.
    """
    r = (raw_reason or "").strip().lower()
    if r in ("length", "max_tokens", "max_output_tokens"):
        return "length"
    if r in ("error",):
        return "error"
    return "stop"


_JSON_DIRECTIVE = "Rispondi esclusivamente con JSON valido."


def _inject_json_directive(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Aggiunge la direttiva 'rispondi in JSON' SENZA violare i template severi.

    I template di chat piu' recenti (Qwen3.5, Gemma-3/4) impongono che il
    messaggio 'system' stia in PRIMA posizione e ne ammettono uno solo:
    appendere un secondo system in coda fa fallire il server con
    'System message must be at the beginning' (HTTP 400) e azzera l'output.
    Llama-3.1 invece lo tollerava: di qui la differenza di comportamento tra
    modelli. Soluzione compatibile con tutti: se esiste gia' un system in
    testa, fondi la direttiva li'; altrimenti inseriscine uno nuovo in testa.
    """
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        base = msgs[0].get("content", "")
        if _JSON_DIRECTIVE not in base:
            msgs[0]["content"] = (base.rstrip() + "\n" + _JSON_DIRECTIVE).strip()
        return msgs
    return [{"role": "system", "content": _JSON_DIRECTIVE}, *msgs]


def sweep_orphan_records(llm_calls_dir: Path) -> list[str]:
    """Rimuove i record orfani lasciati da un run precedente morto a meta'.

    V10.17.2.1. Un record orfano e' un file con status='started': scritto al
    momento del START di una chiamata e mai chiuso, perche' il processo e'
    stato interrotto (crash, Ctrl-C, llama.cpp caduto, .bat sbagliato...).

    Va chiamata UNA volta all'avvio del run, prima che gli agent partano --
    non per-agent, o un agent cancellerebbe il record di un altro ancora vivo.
    Ritorna i nomi dei file rimossi (per loggarli nel trace).

    Nota: rimuove solo 'started'. I record completi -- anche 'llm_error' o
    'parse_failed' -- restano: sono esiti, non spazzatura, e raccontano la
    storia del run.
    """
    if not (llm_calls_dir and llm_calls_dir.exists()):
        return []
    removed: list[str] = []
    for p in sorted(llm_calls_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("status") == "started":
            try:
                p.unlink()
                removed.append(p.name)
            except OSError:
                pass
    return removed


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = config.model

    def chat(self, messages: list[dict[str, str]], *, format_json: bool = True, num_predict: int | None = None) -> str:
        """Retrocompatibile: ritorna solo il contenuto. Per il segnale di
        troncamento usare chat_ex()."""
        return self.chat_ex(messages, format_json=format_json, num_predict=num_predict).content

    def chat_ex(self, messages: list[dict[str, str]], *, format_json: bool = True, num_predict: int | None = None) -> ChatResult:
        """Come chat(), ma ritorna ChatResult con il finish_reason del backend.

        E' la via per accorgersi del troncamento (finish_reason == 'length').
        """
        if self.config.mock:
            return ChatResult(content=self._mock_response(messages), finish_reason="stop")
        if self.config.backend == "llamacpp":
            return self._chat_llamacpp(messages, format_json=format_json, num_predict=num_predict)
        if self.config.backend == "groq":
            return self._chat_groq(messages, format_json=format_json, num_predict=num_predict)
        return self._chat_ollama(messages, format_json=format_json, num_predict=num_predict)

    def health_check(self) -> dict[str, Any]:
        if self.config.mock:
            return {"ok": True, "mock": True, "backend": self.config.backend, "model": self.model}
        if self.config.backend == "llamacpp":
            url = self.config.llamacpp_url.rstrip("/") + "/v1/models"
            headers = dict(_BASE_HTTP_HEADERS)
        elif self.config.backend == "groq":
            url = self.config.groq_url.rstrip("/") + "/models"
            headers = dict(_BASE_HTTP_HEADERS)
            headers["Authorization"] = f"Bearer {self.config.groq_api_key}"
        else:
            url = self.config.base_url.rstrip("/") + "/api/tags"
            headers = dict(_BASE_HTTP_HEADERS)
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=min(self.config.timeout_seconds, 30)) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return {"ok": True, "backend": self.config.backend, "model": self.model, "body": body}
        except Exception as exc:
            return {"ok": False, "backend": self.config.backend, "model": self.model, "error": str(exc)}

    def _chat_ollama(self, messages: list[dict[str, str]], *, format_json: bool = True, num_predict: int | None = None) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": num_predict if num_predict is not None else self.config.num_predict,
            },
            "keep_alive": self.config.keep_alive,
        }
        if self.config.num_ctx:
            payload["options"]["num_ctx"] = self.config.num_ctx
        if self.config.num_gpu is not None:
            payload["options"]["num_gpu"] = self.config.num_gpu
        if format_json:
            payload["format"] = "json"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/api/chat",
            data=data,
            headers=dict(_BASE_HTTP_HEADERS),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "") or ""
            # Ollama: done_reason == "length" quando ha esaurito num_predict,
            # "stop" quando il modello ha chiuso da solo.
            done_reason = body.get("done_reason", "") or ""
            return ChatResult(content=content, finish_reason=_normalize_finish(done_reason))
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"Timeout Ollama dopo {self.config.timeout_seconds}s sul modello {self.model}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Impossibile contattare Ollama su {self.config.base_url}: {exc}") from exc

    def _chat_llamacpp(self, messages: list[dict[str, str]], *, format_json: bool = True, num_predict: int | None = None) -> ChatResult:
        safe_messages = _inject_json_directive(messages) if format_json else list(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": safe_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": num_predict if num_predict is not None else self.config.num_predict,
            "stream": False,
        }
        if format_json:
            payload["response_format"] = {"type": "json_object"}
        url = self.config.llamacpp_url.rstrip("/") + "/v1/chat/completions"
        return self._post_openai_compatible(
            url, payload, label="llama.cpp",
            allow_retry_without_response_format=format_json,
        )

    def _chat_groq(self, messages: list[dict[str, str]], *, format_json: bool = True, num_predict: int | None = None) -> ChatResult:
        """Chiamata a Groq -- API OpenAI-compatible. V10.18.3.

        Stesso protocollo di llama.cpp (/v1/chat/completions); l'unica
        differenza e' l'header Authorization con la API key. Riusa lo stesso
        POST generico.
        """
        if not self.config.groq_api_key:
            raise RuntimeError(
                "Backend 'groq' selezionato ma manca la API key. Passala con "
                "--groq-api-key o nella variabile d'ambiente GROQ_API_KEY."
            )
        safe_messages = _inject_json_directive(messages) if format_json else list(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": safe_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": num_predict if num_predict is not None else self.config.num_predict,
            "stream": False,
        }
        if format_json:
            payload["response_format"] = {"type": "json_object"}
        url = self.config.groq_url.rstrip("/") + "/chat/completions"
        return self._post_openai_compatible(
            url, payload, label="Groq",
            allow_retry_without_response_format=format_json,
            extra_headers={"Authorization": f"Bearer {self.config.groq_api_key}"},
        )

    def _post_openai_compatible(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        label: str,
        allow_retry_without_response_format: bool,
        extra_headers: dict[str, str] | None = None,
    ) -> ChatResult:
        """POST a un endpoint /chat/completions OpenAI-compatible.

        Condiviso da llama.cpp e Groq: stesso protocollo, cambiano solo URL e
        header. `label` serve solo per messaggi d'errore leggibili.

        V10.19.3: retry automatico su HTTP 429 (rate limit). Legge il campo
        'retry-after' dall'header della risposta (Groq lo valorizza in secondi
        decimali). Se assente usa backoff esponenziale: 5s, 10s, 20s.
        Massimo MAX_RATE_LIMIT_RETRIES tentativi; oltre, rilancia.
        """
        MAX_RATE_LIMIT_RETRIES: int = 4
        RATE_LIMIT_BASE_WAIT: float = 5.0

        headers = dict(_BASE_HTTP_HEADERS)
        if extra_headers:
            headers.update(extra_headers)
        data = json.dumps(payload).encode("utf-8")

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break  # successo: esci dal loop
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
                    if attempt >= MAX_RATE_LIMIT_RETRIES:
                        raise RuntimeError(f"{label} HTTP 429. Dettaglio: {detail}") from exc
                    # calcola wait: usa retry-after dall'header se presente
                    retry_after = exc.headers.get("retry-after") if exc.headers else None
                    try:
                        #wait = float(retry_after) + 1.0
                        wait = float(retry_after) + 5.0
                    except (TypeError, ValueError):
                        wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                    time.sleep(wait)
                    continue
                # errore non-429: gestione esistente
                detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
                # alcuni modelli/endpoint non supportano response_format: riprova senza
                if allow_retry_without_response_format and "response_format" in payload:
                    payload = dict(payload)
                    payload.pop("response_format", None)
                    return self._post_openai_compatible(
                        url, payload, label=label,
                        allow_retry_without_response_format=False,
                        extra_headers=extra_headers,
                    )
                raise RuntimeError(f"{label} HTTP {exc.code}. Dettaglio: {detail}") from exc
            except (TimeoutError, socket.timeout) as exc:
                raise RuntimeError(f"Timeout {label} dopo {self.config.timeout_seconds}s sul modello {self.model}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Impossibile contattare {label} su {url}: {exc}") from exc
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"] or ""
            finish = choice.get("finish_reason", "") or ""
            return ChatResult(content=content, finish_reason=_normalize_finish(finish))
        except Exception as exc:
            raise RuntimeError(f"Risposta {label} inattesa: {body}") from exc

    def _mock_response(self, messages: list[dict[str, str]]) -> str:
        content = "\n".join(m.get("content", "") for m in messages)
        lower = content.lower()
        try:
            payload_start = content.index("PAYLOAD_JSON=") + len("PAYLOAD_JSON=")
            contract_start = content.index("\nOUTPUT_CONTRACT=")
            payload = json.loads(content[payload_start:contract_start].strip())
        except Exception:
            payload = {}

        # ----------------------------------------------------------------
        # V10.14.0 -- mock generico per i ruoli FT.
        # Niente dominio hardcoded: il mock fa il minimo plausibile in modo
        # da esercitare la pipeline; il dominio specifico va al modello vero.
        # ----------------------------------------------------------------
        if "l1_classifier" in lower:
            return self._mock_l1_classifier(payload)
        if "l2_locked_" in lower:
            return self._mock_l2_locked(payload)
        if "l3a1_domainknowledge" in lower:
            return json.dumps({"concepts": [
                {"concept": "concetto_dominio_1", "relation_to_input": "evocato dal testo, non dimostrato", "status": "domain_knowledge", "suggested_scale": "organismo"},
                {"concept": "concetto_dominio_2", "relation_to_input": "adiacente al fenomeno", "status": "causal_model", "suggested_scale": "molecolare"},
            ]}, ensure_ascii=False)
        if "l3a2_causalprinciples" in lower:
            return json.dumps({"principles": [
                {"name": "Principio mock A", "description": "Descrizione generica del principio.", "status": "causal_model"},
                {"name": "Principio mock B", "description": "Secondo principio applicabile.", "status": "causal_model"},
            ]}, ensure_ascii=False)
        if "l3a3_crossdomainanalogies" in lower:
            return json.dumps({"analogies": [
                {"domain": "biologia", "analogy": "analogia mock biologica", "warning": "analogia, non equivalenza", "status": "cross_domain_analogy"},
                {"domain": "musica", "analogy": "risonanza come modello", "warning": "semplificazione concettuale", "status": "cross_domain_analogy"},
            ]}, ensure_ascii=False)
        if "l3a4_openquestions" in lower:
            return json.dumps({"questions": [
                "Quale evidenza servirebbe per validare il legame?",
                "Quali passaggi intermedi non sono osservati?",
            ]}, ensure_ascii=False)
        if "l3a5_globalsynthesis" in lower:
            return json.dumps({
                "core_image": "Mock synthesis: figura centrale ridotta.",
                "human_summary": "Sintesi mock: il testo presenta dei claim, l'esplorazione di dominio resta indicativa, non probatoria.",
                "epistemic_warning": "Mock: nessuna prova esterna e' stata cercata. Il dominio reale richiede LLM vero.",
                "dominant_domain": "mock_generic",
                "primary_lenses": ["lente_a", "lente_b", "lente_c"],
                "blocked_lenses": ["lente_bloccata"],
            }, ensure_ascii=False)
        if "l3b_crossscalevalidator" in lower:
            cands = payload.get("candidates", []) if isinstance(payload, dict) else []
            verdicts = []
            for c in cands:
                if not isinstance(c, dict):
                    continue
                dist = c.get("scale_distance", 0)
                if dist == 1:
                    v = "uncertain"
                elif dist == 2:
                    v = "genuine"   # distanza media: il mock promuove (per testare la promozione)
                elif dist >= 3:
                    v = "spurious"
                else:
                    v = "uncertain"
                verdicts.append({
                    "candidate_id": c.get("candidate_id"),
                    "verdict": v,
                    "reasoning": f"mock: distanza={dist}, verdetto prudente.",
                    "confidence": 0.5,
                })
            return json.dumps({"verdicts": verdicts}, ensure_ascii=False)
        # ----------------------------------------------------------------
        # V10.15.0 mocks
        # ----------------------------------------------------------------
        if "l5_fractalexpander" in lower:
            return self._mock_l5_expander(payload)
        if "l5_bridgebuilder" in lower:
            return self._mock_l5_bridge(payload)
        if "l6_magistralereport" in lower:
            return self._mock_l6_magistrale(payload)
        # ----------------------------------------------------------------
        # V10.19.x -- mock per i ruoli della lettura tematica.
        # ----------------------------------------------------------------
        if "l_thematic_" in lower and "synthesis" not in lower:
            # una lente su un segmento: due osservazioni plausibili
            lens = payload.get("lens", "tematica") if isinstance(payload, dict) else "tematica"
            return json.dumps({"observations": [
                {"focus": f"elemento mock {lens} 1", "note": f"osservazione mock della lente {lens}.",
                 "evidence": "riferimento mock", "salience": 0.6},
                {"focus": f"elemento mock {lens} 2", "note": "seconda osservazione mock.",
                 "evidence": "altro riferimento", "salience": 0.4},
            ]}, ensure_ascii=False)
        if "l_thematic_synthesis" in lower:
            return json.dumps({
                "synthesis": "Sintesi mock plurale: le quattro lenti mostrano il testo da angolazioni diverse.",
                "motifs": [{"name": "motivo mock", "lens": "simbolica",
                            "occurrences": ["punto 1", "punto 2"],
                            "transformation": "trasformazione mock"}],
            }, ensure_ascii=False)
        if "l_thematicbook_opera" in lower:
            return json.dumps({
                "synthesis": "Sintesi mock dell'opera: riepilogo plurale delle quattro lenti sull'intero libro.",
            }, ensure_ascii=False)
        if "l_thematicbook_" in lower:
            return json.dumps({
                "synthesis": "Sintesi mock di lente sull'intero libro: temi e immagini ricorrenti.",
            }, ensure_ascii=False)
        return json.dumps({"rationale": "mock"}, ensure_ascii=False)

    # ----------------------------------------------------------------
    # Mock helpers V10.14.0
    # ----------------------------------------------------------------
    def _mock_l1_classifier(self, payload: dict) -> str:
        """Mock generico: spezza il testo per punto/virgola, alterna nature.

        Riconosce le DOMANDE (frasi che finiscono con '?'): le marca
        predicate=question, nature=context -- coerente col fix V10.16.3.
        """
        import re as _re
        text = str(payload.get("input_text") or "")
        # split che PRESERVA il '?' come fine frase
        raw_pieces = _re.split(r"(?<=[.;?])|\s--\s|\s-\s", text)
        pieces = [p.strip() for p in raw_pieces if p.strip()]
        items = []
        natures_cycle = ["context", "cause", "effect", "context"]
        scales_cycle = ["organismo", "atomico", "molecolare", "sociale"]
        for i, p in enumerate(pieces[:8]):
            words = p.split()
            if len(words) > 25:
                p = " ".join(words[:25])
            is_question = p.rstrip().endswith("?")
            if is_question:
                predicate = "question"
                nature = "context"
            else:
                predicate = "process_description" if i % 3 == 1 else ("event" if i % 3 == 2 else "definition")
                nature = natures_cycle[i % len(natures_cycle)]
            items.append({
                "quote": p,
                "predicate": predicate,
                "nature": nature,
                "scale": scales_cycle[i % len(scales_cycle)],
                "rationale": "mock classification",
            })
        return json.dumps({"items": items}, ensure_ascii=False)

    def _mock_l2_locked(self, payload: dict) -> str:
        """Mock: lega la prima cause con il primo effect sulla scala."""
        items = payload.get("items", []) if isinstance(payload, dict) else []
        causes = [it for it in items if isinstance(it, dict) and it.get("nature") == "cause"]
        effects = [it for it in items if isinstance(it, dict) and it.get("nature") == "effect"]
        links = []
        if causes and effects:
            links.append({
                "cause_item_id": causes[0].get("id"),
                "effect_item_id": effects[0].get("id"),
                "rationale": "mock same-scale link",
                "confidence": 0.6,
            })
        orphans = []
        linked_ids = {links[0]["cause_item_id"], links[0]["effect_item_id"]} if links else set()
        for it in items:
            if isinstance(it, dict) and it.get("id") and it["id"] not in linked_ids:
                orphans.append({"item_id": it["id"], "reason": "mock no pair"})
        return json.dumps({"same_scale_links": links, "orphans": orphans, "summary": "mock locked scale"}, ensure_ascii=False)

    # ----------------------------------------------------------------
    # V10.15.0 mocks
    # ----------------------------------------------------------------
    def _mock_l5_expander(self, payload: dict) -> str:
        """Mock: produce 4 figli nelle 4 direzioni canoniche.

        Sceglie scale rispettando il vincolo direzione: same/up/down. Per
        evitare di sbagliare, calcoliamo gli indici sulla scala canonica
        del padre.
        """
        # Importiamo qui per evitare circolarita' al modulo
        scales = ["cosmologico", "planetario", "sociale", "organismo", "cellulare",
                  "molecolare", "atomico", "subatomico", "fondamentale"]
        parent = payload.get("parent", {}) if isinstance(payload, dict) else {}
        parent_scale = parent.get("scale", "organismo")
        try:
            depth = scales.index(parent_scale)
        except ValueError:
            depth = 3
        scale_up = scales[max(0, depth - 1)] if depth > 0 else parent_scale
        scale_down = scales[min(len(scales) - 1, depth + 1)] if depth < len(scales) - 1 else parent_scale
        children = [
            {
                "direction": "same_scale_cause",
                "text": f"mock_same_scale_cause di '{parent.get('id','?')}'",
                "predicate": "process_description",
                "nature": "cause",
                "scale": parent_scale,
                "relation_to_parent": "mock: causa orizzontale generata",
                "confidence": 0.55,
            },
            {
                "direction": "scale_up_propagation",
                "text": f"mock_propagazione_su di '{parent.get('id','?')}'",
                "predicate": "claimed_property",
                "nature": "effect",
                "scale": scale_up,
                "relation_to_parent": "mock: propagazione verso scala superficiale",
                "confidence": 0.5,
            },
            {
                "direction": "scale_down_mechanism",
                "text": f"mock_meccanismo_giu di '{parent.get('id','?')}'",
                "predicate": "process_description",
                "nature": "bridge",
                "scale": scale_down,
                "relation_to_parent": "mock: meccanismo sottostante",
                "confidence": 0.5,
            },
            {
                "direction": "coherence_bridge",
                "text": f"mock_ponte_di_coerenza di '{parent.get('id','?')}'",
                "predicate": "state",
                "nature": "bridge",
                "scale": scale_up,
                "relation_to_parent": "mock: ponte di coerenza",
                "confidence": 0.4,
            },
        ]
        return json.dumps({"children": children}, ensure_ascii=False)

    def _mock_l5_bridge(self, payload: dict) -> str:
        """Mock: produce un bridge sulla gap_scale richiesta."""
        gap = payload.get("gap_scale", "molecolare") if isinstance(payload, dict) else "molecolare"
        source = payload.get("source", {}) if isinstance(payload, dict) else {}
        target = payload.get("target", {}) if isinstance(payload, dict) else {}
        src_t = source.get("text", "?")
        tgt_t = target.get("text", "?")
        return json.dumps({
            "bridge": {
                "text": f"meccanismo intermedio su {gap} che traduce '{src_t[:30]}' in '{tgt_t[:30]}'",
                "predicate": "process_description",
                "scale": gap,
                "reasoning": (
                    f"mock: la sorgente su {source.get('scale','?')} agisce su una variabile "
                    f"intermedia alla scala {gap}, che a sua volta produce il target su "
                    f"{target.get('scale','?')}. Modello ipotetico, non documentato."
                ),
                "confidence": 0.45,
            }
        }, ensure_ascii=False)

    def _mock_l6_magistrale(self, payload: dict) -> str:
        """Mock: smista gli items nei coni in base a nature/scale."""
        items = payload.get("items", []) if isinstance(payload, dict) else []
        def _t(item):
            return item.get("text") or item.get("id", "?")
        predispositions = [_t(it) for it in items if it.get("nature") == "context"][:6]
        triggers = [_t(it) for it in items if it.get("nature") == "cause" and it.get("predicate") in ("event", "process_description")][:6]
        proximate = [_t(it) for it in items if it.get("nature") == "cause"][:6]
        bridges = [_t(it) for it in items if it.get("nature") == "bridge"][:6]
        direct = [_t(it) for it in items if it.get("nature") == "effect"][:6]
        # downstream: effetti su scala piu' superficiale (depth piu' basso)
        scales = ["cosmologico", "planetario", "sociale", "organismo", "cellulare",
                  "molecolare", "atomico", "subatomico", "fondamentale"]
        scale_depth = {s: i for i, s in enumerate(scales)}
        causes_depths = [scale_depth.get(it.get("scale", ""), 99) for it in items if it.get("nature") == "cause"]
        cause_min_depth = min(causes_depths) if causes_depths else 99
        downstream = [_t(it) for it in items
                      if it.get("nature") == "effect"
                      and scale_depth.get(it.get("scale", ""), 99) < cause_min_depth][:6]
        interpretations = [_t(it) for it in items if it.get("nature") == "interpretation"][:6]
        social = [_t(it) for it in items if it.get("scale") in ("sociale", "planetario", "cosmologico") and it.get("nature") == "effect"][:6]
        scales_used = sorted({it.get("scale") for it in items if it.get("scale")}, key=lambda s: scale_depth.get(s, 99))
        return json.dumps({
            "sintesi_magistrale": (
                "Mock: il sistema ha mappato il fenomeno su piu' scale discrete. "
                "La propagazione attraversa il cono cause -> cono effetti con alcuni "
                "passaggi intermedi (bridge) ancora a stato ipotetico."
            ),
            "cono_cause": {
                "predispositions": predispositions,
                "triggers": triggers,
                "proximate_causes": proximate,
                "bridge_mechanisms": bridges,
            },
            "cono_effetti": {
                "direct_effects": direct,
                "downstream_effects": downstream,
                "interpretations": interpretations,
                "social_propagations": social,
            },
            "propagazione_multi_scala": (
                "Le scale popolate, dal piu' superficiale al piu' profondo, sono: "
                + ", ".join(scales_used) + "."
            ),
            "stato_epistemico": (
                "Mock: gli items con origine 'text_observed' sono osservati; "
                "quelli da espansione o bridge sono modelli causali o conoscenza di dominio. "
                "Le ipotesi cross-scale 'uncertain' restano aperte."
            ),
            "verdetto_finale": "Mock: lettura sistemica disponibile, validazione cross-scale parziale.",
        }, ensure_ascii=False)


class RoleAgent:
    def __init__(self, client: LLMClient, role_name: str, role_prompt: str, *, out_dir: Path | None = None, max_output_tokens: int | None = None) -> None:
        self.client = client
        self.role_name = role_name
        self.role_prompt = role_prompt.strip()
        self.out_dir = out_dir
        self.max_output_tokens = max_output_tokens

    def run_json(self, payload: dict[str, Any], output_contract: dict[str, Any], trace: list[str], telemetry_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = self._build_prompt(payload, output_contract)
        call_id = self._next_call_id()
        started = time.time()
        started_iso = datetime.now().isoformat(timespec="seconds")
        record = {
            "call_id": call_id,
            "role_name": self.role_name,
            "backend": self.client.config.backend,
            "model": self.client.model,
            "prompt_chars": len(prompt),
            "payload_chars": len(json.dumps(payload, ensure_ascii=False, default=str)),
            "max_output_tokens": self.max_output_tokens,
            "status": "started",
            "started_at": started_iso,
            "role_prompt": self.role_prompt,
            "payload": payload,
            "output_contract": output_contract,
            "prompt": prompt,
        }
        if self.out_dir:
            write_json(record, self.out_dir / f"{call_id}.json")
        trace.append(f"{self.role_name}: START call_id={call_id} prompt_chars={record['prompt_chars']} payload_chars={record['payload_chars']}")
        if telemetry_path:
            write_jsonl_line({"event": "actor_start", "call_id": call_id, "actor": self.role_name, "timestamp": started_iso, "prompt_chars": record["prompt_chars"]}, telemetry_path)
            write_jsonl_line({"event": "llm_call_start", "call_id": call_id, "role": self.role_name, "timestamp": started_iso, "prompt_chars": record["prompt_chars"]}, telemetry_path)
        messages = [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": prompt}]
        try:
            # --- V10.17.2: auto-retry su troncamento ------------------------
            # Se il backend riporta finish_reason='length' il JSON e' troncato
            # per esaurimento di num_predict. Rifacciamo la chiamata con il
            # budget di token raddoppiato, fino a MAX_TRUNCATION_RETRIES volte.
            base_tokens = self.max_output_tokens or self.client.config.num_predict
            attempts: list[dict[str, Any]] = []
            result = None
            tokens = base_tokens
            for attempt_idx in range(MAX_TRUNCATION_RETRIES + 1):
                result = self.client.chat_ex(
                    messages, format_json=True, num_predict=tokens
                )
                attempts.append({
                    "attempt": attempt_idx + 1,
                    "num_predict": tokens,
                    "finish_reason": result.finish_reason,
                    "response_chars": len(result.content),
                })
                if not result.truncated:
                    break
                # troncato: raddoppia e ritenta (se restano tentativi)
                if attempt_idx < MAX_TRUNCATION_RETRIES:
                    next_tokens = tokens * 2
                    trace.append(
                        f"{self.role_name}: TRUNCATED call_id={call_id} "
                        f"(num_predict={tokens}, finish_reason=length) -> "
                        f"retry con num_predict={next_tokens}"
                    )
                    if telemetry_path:
                        write_jsonl_line({
                            "event": "llm_call_truncated_retry", "call_id": call_id,
                            "role": self.role_name, "num_predict": tokens,
                            "next_num_predict": next_tokens,
                        }, telemetry_path)
                    tokens = next_tokens

            raw = result.content
            elapsed = time.time() - started
            finished_iso = datetime.now().isoformat(timespec="seconds")
            truncated_final = result.truncated
            record.update({
                "status": "raw_response_received",
                "raw_response": raw,
                "response_chars": len(raw),
                "elapsed_seconds": round(elapsed, 3),
                "finished_at": finished_iso,
                "finish_reason": result.finish_reason,
                "truncated": truncated_final,
                "truncation_attempts": attempts,
                "final_num_predict": tokens,
            })
            parsed = extract_json(raw)
            if not isinstance(parsed, dict):
                parsed = {"_raw_list": parsed}
            # status: 'truncated' se anche dopo i retry il JSON e' monco;
            # altrimenti 'parsed_dict' come sempre. Il troncamento non e' piu'
            # silenzioso: e' scritto nel record e gridato nel trace.
            final_status = "truncated" if truncated_final else "parsed_dict"
            record.update({"status": final_status, "parsed_json": parsed})
            if truncated_final:
                trace.append(
                    f"{self.role_name}: WARNING call_id={call_id} elapsed={elapsed:.2f}s "
                    f"response_chars={len(raw)} status=truncated "
                    f"-- JSON TRONCATO anche dopo {MAX_TRUNCATION_RETRIES} retry "
                    f"(num_predict finale={tokens}). Contenuto parziale recuperato: "
                    f"il modello ha colpito il tetto di token. Alza --num-predict."
                )
            else:
                retry_note = (f" (dopo {len(attempts)-1} retry)" if len(attempts) > 1 else "")
                trace.append(
                    f"{self.role_name}: END call_id={call_id} elapsed={elapsed:.2f}s "
                    f"response_chars={len(raw)} status=parsed_dict{retry_note}"
                )
            if telemetry_path:
                write_jsonl_line({"event": "actor_end", "call_id": call_id, "actor": self.role_name, "timestamp": finished_iso, "elapsed_seconds": round(elapsed, 3), "status": final_status}, telemetry_path)
                write_jsonl_line({"event": "llm_call_end", "call_id": call_id, "role": self.role_name, "timestamp": finished_iso, "elapsed_seconds": round(elapsed, 3), "status": final_status, "truncated": truncated_final}, telemetry_path)
            if self.out_dir:
                write_json(record, self.out_dir / f"{call_id}.json")
            return parsed, {"call_id": call_id, "path": str(self.out_dir / f"{call_id}.json") if self.out_dir else None, "elapsed_seconds": round(elapsed, 3), "truncated": truncated_final}
        except JSONParseError as exc:
            elapsed = time.time() - started
            finished_iso = datetime.now().isoformat(timespec="seconds")
            record.update({"status": "parse_failed", "parse_error": str(exc), "elapsed_seconds": round(elapsed, 3), "finished_at": finished_iso})
            trace.append(f"{self.role_name}: PARSE_FAILED call_id={call_id}: {exc}")
            if telemetry_path:
                write_jsonl_line({"event": "actor_end", "call_id": call_id, "actor": self.role_name, "timestamp": finished_iso, "elapsed_seconds": round(elapsed, 3), "status": "parse_failed"}, telemetry_path)
                write_jsonl_line({"event": "llm_call_parse_failed", "call_id": call_id, "role": self.role_name, "timestamp": finished_iso, "elapsed_seconds": round(elapsed, 3), "error": str(exc)}, telemetry_path)
            if self.out_dir:
                write_json(record, self.out_dir / f"{call_id}.json")
            return {"_parse_error": str(exc), "_raw_text": record.get("raw_response", "")}, {"call_id": call_id, "path": str(self.out_dir / f"{call_id}.json") if self.out_dir else None, "elapsed_seconds": round(elapsed, 3), "parse_failed": True}
        except Exception as exc:
            elapsed = time.time() - started
            finished_iso = datetime.now().isoformat(timespec="seconds")
            record.update({"status": "llm_error", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": round(elapsed, 3), "finished_at": finished_iso})
            trace.append(f"{self.role_name}: LLM_ERROR call_id={call_id}: {type(exc).__name__}: {exc}")
            if telemetry_path:
                write_jsonl_line({"event": "actor_end", "call_id": call_id, "actor": self.role_name, "timestamp": finished_iso, "elapsed_seconds": round(elapsed, 3), "status": "llm_error"}, telemetry_path)
                write_jsonl_line({"event": "llm_call_error", "call_id": call_id, "role": self.role_name, "timestamp": finished_iso, "elapsed_seconds": round(elapsed, 3), "error": str(exc)}, telemetry_path)
            if self.out_dir:
                write_json(record, self.out_dir / f"{call_id}.json")
            return {"_llm_error": str(exc)}, {"call_id": call_id, "path": str(self.out_dir / f"{call_id}.json") if self.out_dir else None, "elapsed_seconds": round(elapsed, 3), "llm_error": True}

    # Stati che indicano un record COMPLETO (la chiamata e' arrivata a una
    # conclusione, riuscita o no). 'started' NON e' qui: e' un record orfano,
    # scritto al START e mai chiuso -> il processo e' morto a meta'.
    _COMPLETED_STATUSES: tuple[str, ...] = (
        "parsed_dict", "truncated", "parse_failed", "llm_error",
    )

    def _next_call_id(self) -> str:
        """Assegna il prossimo call_id.

        V10.17.2.1: la numerazione si basa sui record COMPLETI, non sul mero
        conteggio dei file. Un record orfano (status='started', lasciato da un
        processo morto a meta') non sposta piu' la numerazione: il run che
        riparte riusa il numero del run precedente fallito invece di scalare
        all'infinito (era la causa di 0001/0002/0003 per la stessa chiamata).
        """
        if not (self.out_dir and self.out_dir.exists()):
            return f"0001_{self._role_slug()}"
        completed = 0
        for p in sorted(self.out_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                # file illeggibile/corrotto: lo trattiamo come non-completo
                continue
            if rec.get("status") in self._COMPLETED_STATUSES:
                completed += 1
        return f"{completed + 1:04d}_{self._role_slug()}"

    def _role_slug(self) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in self.role_name)[:80]

    def _build_prompt(self, payload: dict[str, Any], output_contract: dict[str, Any]) -> str:
        return (
            f"ROLE={self.role_name}\n"
            f"ROLE_INSTRUCTIONS={self.role_prompt}\n\n"
            "PAYLOAD_JSON=\n"
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
            "OUTPUT_CONTRACT=\n"
            f"{json.dumps(output_contract, ensure_ascii=False, indent=2)}\n\n"
            "Rispondi solo con un JSON conforme al contratto."
        )
