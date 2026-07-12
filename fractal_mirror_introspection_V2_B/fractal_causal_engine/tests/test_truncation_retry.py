"""Test del rilevamento troncamento + auto-retry (V10.17.2, opzione C).

Verifica che RoleAgent:
- rilevi finish_reason='length' e ritenti con num_predict raddoppiato;
- marchi status='truncated' se anche dopo i retry il JSON resta monco;
- scriva un WARNING visibile nel trace;
- non tocchi il percorso normale quando la chiamata e' completa.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fractal_causal_engine.llm import (
    ChatResult, LLMClient, LLMConfig, RoleAgent, _normalize_finish,
    MAX_TRUNCATION_RETRIES,
)


class _ScriptedClient(LLMClient):
    """Client finto: ritorna ChatResult da una coda prefissata, e registra
    con quale num_predict e' stato chiamato."""

    def __init__(self, results: list[ChatResult]):
        super().__init__(LLMConfig(mock=True))
        self._results = list(results)
        self.calls: list[int] = []   # num_predict di ogni chiamata

    def chat_ex(self, messages, *, format_json=True, num_predict=None):
        self.calls.append(num_predict)
        if self._results:
            return self._results.pop(0)
        return ChatResult(content='{"ok": true}', finish_reason="stop")


def _agent(client, out_dir):
    return RoleAgent(
        client, role_name="L1_Classifier", role_prompt="test",
        out_dir=out_dir, max_output_tokens=900,
    )


# --- normalizzazione ---------------------------------------------------------


def test_normalize_finish_variants():
    assert _normalize_finish("length") == "length"
    assert _normalize_finish("LENGTH") == "length"
    assert _normalize_finish("max_tokens") == "length"
    assert _normalize_finish("stop") == "stop"
    assert _normalize_finish("") == "stop"
    assert _normalize_finish("load") == "stop"
    assert _normalize_finish("error") == "error"


def test_chatresult_truncated_property():
    assert ChatResult("x", "length").truncated is True
    assert ChatResult("x", "stop").truncated is False
    assert ChatResult("x", "").truncated is False


# --- caso normale: nessun troncamento, nessun retry --------------------------


def test_no_retry_when_complete():
    client = _ScriptedClient([ChatResult('{"items": [1, 2]}', "stop")])
    with tempfile.TemporaryDirectory() as td:
        agent = _agent(client, Path(td))
        trace: list[str] = []
        parsed, meta = agent.run_json({}, {"items": []}, trace)
    assert client.calls == [900]                 # una sola chiamata
    assert meta["truncated"] is False
    assert parsed == {"items": [1, 2]}
    assert any("status=parsed_dict" in l for l in trace)
    assert not any("TRUNCATED" in l for l in trace)


# --- troncamento al 1o colpo, completo al 2o: retry riuscito -----------------


def test_retry_succeeds_on_second_attempt():
    client = _ScriptedClient([
        ChatResult('{"items": [1, 2', "length"),       # troncato
        ChatResult('{"items": [1, 2, 3]}', "stop"),     # completo
    ])
    with tempfile.TemporaryDirectory() as td:
        agent = _agent(client, Path(td))
        trace: list[str] = []
        parsed, meta = agent.run_json({}, {"items": []}, trace)
    # 2 chiamate: 900, poi raddoppiato a 1800
    assert client.calls == [900, 1800]
    assert meta["truncated"] is False
    assert parsed == {"items": [1, 2, 3]}
    assert any("TRUNCATED" in l and "retry con num_predict=1800" in l for l in trace)
    assert any("status=parsed_dict" in l and "retry" in l for l in trace)


# --- troncamento persistente: status=truncated + warning --------------------


def test_persistent_truncation_marks_status_and_warns():
    # tronca sempre: 1 iniziale + MAX_TRUNCATION_RETRIES retry, tutti 'length'
    trunc = ChatResult('{"items": [1', "length")
    client = _ScriptedClient([trunc] * (MAX_TRUNCATION_RETRIES + 1))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        agent = _agent(client, out)
        trace: list[str] = []
        parsed, meta = agent.run_json({}, {"items": []}, trace)
        # num_predict raddoppiato a ogni retry: 900, 1800, 3600
        assert client.calls == [900, 1800, 3600]
        assert meta["truncated"] is True
        # WARNING visibile nel trace
        assert any("WARNING" in l and "status=truncated" in l for l in trace)
        # il record su disco riporta status e dettagli
        rec = json.loads((out / "0001_L1_Classifier.json").read_text(encoding="utf-8"))
        assert rec["status"] == "truncated"
        assert rec["truncated"] is True
        assert rec["final_num_predict"] == 3600
        assert len(rec["truncation_attempts"]) == 3


# --- V10.17.2.1: sweep dei record orfani -------------------------------------


def test_sweep_removes_only_started_records():
    """sweep_orphan_records elimina i record status='started' (run morti a
    meta') e lascia intatti tutti i record completi, esiti compresi."""
    from fractal_causal_engine.llm import sweep_orphan_records
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "0001_L1.json").write_text(json.dumps({"status": "parsed_dict"}))
        (d / "0002_L1.json").write_text(json.dumps({"status": "started"}))
        (d / "0003_L1.json").write_text(json.dumps({"status": "llm_error"}))
        (d / "0004_L1.json").write_text(json.dumps({"status": "truncated"}))
        (d / "0005_L1.json").write_text(json.dumps({"status": "started"}))
        removed = sweep_orphan_records(d)
        assert sorted(removed) == ["0002_L1.json", "0005_L1.json"]
        survivors = sorted(p.name for p in d.iterdir())
        assert survivors == ["0001_L1.json", "0003_L1.json", "0004_L1.json"]


def test_next_call_id_ignores_orphans():
    """_next_call_id numera sui record COMPLETI: un orfano non sposta piu' la
    numerazione (era la causa di 0001/0002/0003 per la stessa chiamata)."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # 2 completi + 3 orfani
        (d / "0001_L1_Classifier.json").write_text(json.dumps({"status": "parsed_dict"}))
        (d / "0002_L1_Classifier.json").write_text(json.dumps({"status": "started"}))
        (d / "0003_L1_Classifier.json").write_text(json.dumps({"status": "started"}))
        (d / "0004_L1_Classifier.json").write_text(json.dumps({"status": "truncated"}))
        (d / "0005_L1_Classifier.json").write_text(json.dumps({"status": "started"}))
        client = _ScriptedClient([])
        agent = _agent(client, d)
        # 2 record completi -> il prossimo e' 0003, non 0006
        assert agent._next_call_id() == "0003_L1_Classifier"


def test_sweep_empty_or_missing_dir():
    """sweep su dir inesistente o vuota non esplode e ritorna lista vuota."""
    from fractal_causal_engine.llm import sweep_orphan_records
    assert sweep_orphan_records(Path("/nonexistent/xyz")) == []
    with tempfile.TemporaryDirectory() as td:
        assert sweep_orphan_records(Path(td)) == []
