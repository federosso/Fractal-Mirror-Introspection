"""Test del book runner (V10.18.0, passo 2).

Verifica manifest, checkpoint atomico, resume dopo crash, retry con backoff,
dead-letter. Usa il backend mock; gli errori sono iniettati con un client
fittizio per esercitare i rami di fallimento in modo deterministico.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fractal_causal_engine.ft_book_runner import (
    BookRunner, MANIFEST_FILENAME, _backoff_seconds, _is_transient,
)
from fractal_causal_engine.ft_model import BookManifest, SegmentRecord
from fractal_causal_engine.json_utils import read_json
from fractal_causal_engine.llm import LLMClient, LLMConfig


def _mock_client() -> LLMClient:
    return LLMClient(LLMConfig(mock=True))


class _AlwaysFailClient(LLMClient):
    """Client che solleva sempre un errore a livello backend. Esercita il
    degrado silenzioso: la pipeline cattura e produce un ft vuoto."""

    def chat_ex(self, messages, *, format_json=True, num_predict=None):
        raise RuntimeError("ValueError: contratto non rispettato")


def _long_book(n_paragraphs: int = 40) -> str:
    """Testo abbastanza lungo da produrre piu' segmenti con num_ctx piccolo."""
    return "\n\n".join(
        f"Paragrafo {i}: la causa produce un effetto osservabile. " * 6
        for i in range(n_paragraphs)
    )


def _no_sleep(_seconds: float) -> None:
    """sleep_fn iniettata: i test non aspettano davvero il backoff."""
    return None


# --- helper di base ----------------------------------------------------------


def test_is_transient_classification():
    assert _is_transient("Timeout llama.cpp dopo 1200s")
    assert _is_transient("Impossibile contattare Ollama")
    assert _is_transient("Connection refused")
    assert not _is_transient("JSONParseError: token inatteso")
    assert not _is_transient("ValueError: scala non canonica")


def test_backoff_is_exponential_with_cap():
    assert _backoff_seconds(1) == 4.0
    assert _backoff_seconds(2) == 8.0
    assert _backoff_seconds(3) == 16.0
    # tetto a 120s
    assert _backoff_seconds(10) == 120.0


# --- run completo (happy path) -----------------------------------------------


def test_full_run_all_segments_done():
    with tempfile.TemporaryDirectory() as td:
        runner = BookRunner(_mock_client(), td, num_ctx=2048, sleep_fn=_no_sleep)
        manifest = runner.run(_long_book(), book_id="libro_test")
        assert len(manifest.segments) > 1
        c = manifest.counts()
        assert c["done"] == len(manifest.segments)
        assert c["failed"] == 0
        # il manifest e' su disco
        assert (Path(td) / MANIFEST_FILENAME).exists()
        # ogni segmento done ha la sua sotto-cartella con una sessione
        for rec in manifest.segments:
            seg_dir = Path(td) / rec.out_dir
            assert (seg_dir / "session.json").exists()


def test_manifest_persisted_and_reloadable():
    with tempfile.TemporaryDirectory() as td:
        runner = BookRunner(_mock_client(), td, num_ctx=2048, sleep_fn=_no_sleep)
        runner.run(_long_book(), book_id="libro_test")
        raw = read_json(Path(td) / MANIFEST_FILENAME)
        assert raw["book_id"] == "libro_test"
        assert raw["segments"]
        assert all(s["status"] == "done" for s in raw["segments"])


# --- idempotenza: rilanciare non rifa' il lavoro -----------------------------


def test_rerun_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        runner = BookRunner(_mock_client(), td, num_ctx=2048, sleep_fn=_no_sleep)
        runner.run(_long_book(), book_id="libro_test")
        # secondo run: tutti gia' 'done', nessun lavoro
        msgs: list[str] = []
        manifest2 = runner.run(_long_book(), book_id="libro_test",
                               progress=msgs.append)
        assert manifest2.counts()["done"] == len(manifest2.segments)
        assert any("RESUME" in m for m in msgs)


# --- resume dopo crash -------------------------------------------------------


def test_resume_after_crash_continues_from_pending():
    """Simula un crash: il manifest ha alcuni segmenti 'done', uno 'running'
    (crash a meta') e altri 'pending'. Il resume completa il resto."""
    with tempfile.TemporaryDirectory() as td:
        runner = BookRunner(_mock_client(), td, num_ctx=2048, sleep_fn=_no_sleep)
        # primo run completo per ottenere un manifest valido
        manifest = runner.run(_long_book(), book_id="libro_test")
        n = len(manifest.segments)
        assert n >= 3
        # simula crash: forza il manifest a uno stato parziale
        manifest.segments[0].status = "done"
        manifest.segments[1].status = "running"   # era a meta'
        for r in manifest.segments[2:]:
            r.status = "pending"
        runner._checkpoint(manifest)
        # resume
        msgs: list[str] = []
        final = runner.run(_long_book(), book_id="libro_test", progress=msgs.append)
        # il 'running' e' stato ripristinato a pending e poi completato
        assert any("era 'running'" in m for m in msgs)
        assert final.counts()["done"] == n


# --- errore transitorio: retry con backoff, poi successo ---------------------
#
# La pipeline degrada internamente invece di sollevare: per simulare un
# guasto VERO (categoria 1 del runner) si fa sollevare analyze() stesso.

import fractal_causal_engine.ft_book_runner as _br_mod


class _AnalyzePatch:
    """Context manager: sostituisce ExplorerSession.analyze con una funzione
    che fallisce le prime `fail_times` chiamate, poi delega all'originale."""

    def __init__(self, fail_times: int, error: str):
        self.fail_times = fail_times
        self.error = error
        self.calls = 0
        self._orig = None

    def __enter__(self):
        from fractal_causal_engine.ft_session import ExplorerSession
        self._orig = ExplorerSession.analyze

        def fake_analyze(client, out_dir, text, **kw):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise RuntimeError(self.error)
            return self._orig(client, out_dir, text, **kw)

        _br_mod.ExplorerSession.analyze = staticmethod(fake_analyze)
        return self

    def __exit__(self, *exc):
        _br_mod.ExplorerSession.analyze = self._orig
        return False


def test_transient_error_retried_then_succeeds():
    with tempfile.TemporaryDirectory() as td:
        # il 1o tentativo del 1o segmento solleva un timeout, poi tutto ok
        with _AnalyzePatch(fail_times=1, error="Timeout llama.cpp dopo 1200s"):
            runner = BookRunner(_mock_client(), td, num_ctx=2048, max_retries=3,
                                sleep_fn=_no_sleep)
            msgs: list[str] = []
            manifest = runner.run(_long_book(8), book_id="libro_test",
                                  progress=msgs.append)
        assert any("transitorio" in m for m in msgs)
        assert any("backoff" in m for m in msgs)
        assert manifest.counts()["failed"] == 0       # arriva in fondo


def test_transient_error_exhausts_retries_then_dead_letter():
    with tempfile.TemporaryDirectory() as td:
        # fallisce sempre con un errore transitorio: esaurisce i retry
        with _AnalyzePatch(fail_times=999, error="Timeout llama.cpp"):
            runner = BookRunner(_mock_client(), td, num_ctx=2048, max_retries=3,
                                sleep_fn=_no_sleep)
            manifest = runner.run(_long_book(6), book_id="libro_test")
        failed = [r for r in manifest.segments if r.status == "failed"]
        assert failed
        # ha provato max_retries volte prima di arrendersi
        assert all(r.attempts == 3 for r in failed)
        assert all(r.dead_letter for r in failed)


# --- errore definitivo: dead-letter, niente retry ----------------------------


def test_permanent_error_goes_to_dead_letter():
    with tempfile.TemporaryDirectory() as td:
        with _AnalyzePatch(fail_times=999, error="ValueError: contratto rotto"):
            runner = BookRunner(_mock_client(), td, num_ctx=2048,
                                sleep_fn=_no_sleep)
            msgs: list[str] = []
            manifest = runner.run(_long_book(8), book_id="libro_test",
                                  progress=msgs.append)
        c = manifest.counts()
        assert c["failed"] == len(manifest.segments)
        assert all(r.dead_letter for r in manifest.segments if r.status == "failed")
        assert any("dead-letter" in m for m in msgs)


def test_permanent_error_not_retried():
    """Un errore definitivo non consuma i retry: un solo tentativo."""
    with tempfile.TemporaryDirectory() as td:
        with _AnalyzePatch(fail_times=999, error="ValueError: contratto rotto"):
            runner = BookRunner(_mock_client(), td, num_ctx=2048, max_retries=3,
                                sleep_fn=_no_sleep)
            manifest = runner.run(_long_book(6), book_id="libro_test")
        assert all(r.attempts == 1 for r in manifest.segments if r.status == "failed")


def test_empty_ft_outcome_is_failure():
    """Se la pipeline non solleva ma produce un ft vuoto (degrado
    silenzioso), il segmento e' fallito, non 'done'."""
    with tempfile.TemporaryDirectory() as td:
        runner = BookRunner(_AlwaysFailClient(LLMConfig(mock=True)), td,
                            num_ctx=2048, sleep_fn=_no_sleep)
        manifest = runner.run(_long_book(6), book_id="libro_test")
        # _AlwaysFailClient -> ogni agente fallisce -> ft vuoto -> failed
        assert manifest.counts()["failed"] == len(manifest.segments)


# --- halt-on-failure ---------------------------------------------------------


def test_halt_on_failure_stops_the_job():
    with tempfile.TemporaryDirectory() as td:
        with _AnalyzePatch(fail_times=999, error="ValueError: contratto rotto"):
            runner = BookRunner(_mock_client(), td, num_ctx=2048,
                                halt_on_failure=True, sleep_fn=_no_sleep)
            manifest = runner.run(_long_book(20), book_id="libro_test")
        c = manifest.counts()
        assert c["failed"] == 1
        assert c["pending"] == len(manifest.segments) - 1


# --- retry esplicito dei dead-letter -----------------------------------------


def test_retry_dead_letter_reenables_failed_segments():
    with tempfile.TemporaryDirectory() as td:
        # primo run: fallisce tutto
        with _AnalyzePatch(fail_times=999, error="ValueError: rotto"):
            bad = BookRunner(_mock_client(), td, num_ctx=2048, sleep_fn=_no_sleep)
            m1 = bad.run(_long_book(8), book_id="libro_test")
        assert m1.counts()["failed"] == len(m1.segments)
        # secondo run senza patch + retry_dead_letter: completano
        good = BookRunner(_mock_client(), td, num_ctx=2048, sleep_fn=_no_sleep)
        msgs: list[str] = []
        m2 = good.run(_long_book(8), book_id="libro_test",
                      retry_dead_letter=True, progress=msgs.append)
        assert any("ri-abilitato" in m for m in msgs)
        assert m2.counts()["done"] == len(m2.segments)


def test_dead_letter_skipped_without_retry_flag():
    with tempfile.TemporaryDirectory() as td:
        with _AnalyzePatch(fail_times=999, error="ValueError: rotto"):
            bad = BookRunner(_mock_client(), td, num_ctx=2048, sleep_fn=_no_sleep)
            bad.run(_long_book(6), book_id="libro_test")
        good = BookRunner(_mock_client(), td, num_ctx=2048, sleep_fn=_no_sleep)
        msgs: list[str] = []
        m2 = good.run(_long_book(6), book_id="libro_test", progress=msgs.append)
        assert m2.counts()["failed"] == len(m2.segments)
        assert any("in dead-letter, salto" in m for m in msgs)
