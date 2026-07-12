"""
esecutore.py — esecuzione dei run della Strada B in background.

Gemello di run_introspezione_loop_B.py sul lato parametri: stesso client,
stesso nucleo (specchio_del_modello_nucleo.md), stesso contratto, stessa
`esegui_loop`. Un solo worker in coda (coerente col llama-server --parallel 1).

Il progresso NON è simulato: il loop scrive un artefatto per livello nella
cartella del run, quindi lo stato osservabile È lo stato reale — `stato()`
guarda quali file 00…10 esistono e la coda della telemetry per l'attore in corso.
"""
from __future__ import annotations

import json
import pathlib
import queue
import threading
import time
import traceback
from datetime import datetime
from typing import Optional

from . import storico


class Esecutore:
    """`avvia(payload)` accoda un run e ritorna subito il run_id (che è anche il
    nome della cartella loopB_<timestamp>); `stato(run_id)` è consultabile in
    polling; a fine run gli artefatti sono già nello storico, per costruzione."""

    def __init__(self, *, nucleo_path: str, contratto_path: str,
                 backend: str, model: str,
                 num_predict: int, num_ctx: int, timeout: int,
                 modalita_default: str = "completo"):
        self.nucleo_path = nucleo_path
        self.contratto_path = contratto_path
        self.backend = backend
        self.model = model
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.modalita_default = modalita_default

        self._coda: "queue.Queue[dict]" = queue.Queue()
        self._jobs: dict[str, dict] = {}      # run_id -> job in memoria
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # -- client LLM (identico a run_introspezione_loop_B.py) -----------------
    def _nuovo_client(self):
        from fractal_causal_engine.llm import LLMClient, LLMConfig
        return LLMClient(LLMConfig(
            backend=self.backend, model=self.model,
            num_predict=self.num_predict, num_ctx=self.num_ctx,
            timeout_seconds=self.timeout,
        ))

    def health(self) -> dict:
        try:
            return self._nuovo_client().health_check()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # -- API pubblica ---------------------------------------------------------
    def avvia(self, payload: dict) -> str:
        """payload: {sonda, modalita?, _client?} — modalita: auto|completo|leggero."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"loopB_{stamp}"
        job = {
            "id": run_id,
            "stato": "in_coda",
            "creato": datetime.now().isoformat(timespec="seconds"),
            "sonda": payload.get("sonda", ""),
            "modalita": payload.get("modalita") or self.modalita_default,
            "errore": None,
            "payload": payload,
        }
        with self._lock:
            self._jobs[run_id] = job
        self._coda.put(job)
        return run_id

    def stato(self, run_id: str) -> Optional[dict]:
        """Stato reale del run: fase della coda + livelli già scritti su disco
        + attore LLM in corso (coda della telemetry). None se sconosciuto."""
        with self._lock:
            job = self._jobs.get(run_id)
        d = storico.percorso_run(run_id)
        if job is None and d is None:
            return None
        s = {
            "id": run_id,
            "stato": job["stato"] if job else ("completato" if (d / "10_azione.json").exists() else "incompleto"),
            "errore": (job or {}).get("errore"),
        }
        s.update(self._progresso(d))
        return s

    def occupato(self) -> bool:
        with self._lock:
            return any(j["stato"] in ("in_coda", "in_corso") for j in self._jobs.values())

    # -- progresso osservato dagli artefatti ----------------------------------
    def _progresso(self, d: Optional[pathlib.Path]) -> dict:
        n_tot = len(storico.LIVELLI)
        if d is None:
            return {"livelli_fatti": 0, "livelli_totali": n_tot,
                    "ultimo_livello": "", "attore_in_corso": ""}
        fatti, ultimo = 0, ""
        for nome, _, etichetta in storico.LIVELLI:
            if (d / nome).exists():
                fatti += 1
                ultimo = etichetta
        return {"livelli_fatti": fatti, "livelli_totali": n_tot,
                "ultimo_livello": ultimo,
                "attore_in_corso": self._attore_in_corso(d)}

    def _attore_in_corso(self, d: pathlib.Path) -> str:
        """L'ultimo actor_start senza il suo actor_end, dalla telemetry."""
        p = d / "trace" / "telemetry.jsonl"
        if not p.exists():
            return ""
        aperti: list[str] = []
        try:
            for riga in p.read_text(encoding="utf-8").splitlines():
                riga = riga.strip()
                if not riga:
                    continue
                e = json.loads(riga)
                if e.get("event") == "actor_start":
                    aperti.append(e.get("actor", ""))
                elif e.get("event") == "actor_end" and e.get("actor", "") in aperti:
                    aperti.remove(e.get("actor", ""))
        except Exception:
            return ""
        return aperti[-1] if aperti else ""

    # -- worker ----------------------------------------------------------------
    def _loop(self):
        while True:
            job = self._coda.get()
            with self._lock:
                job["stato"] = "in_corso"
                job["avviato"] = datetime.now().isoformat(timespec="seconds")
            t0 = time.time()
            try:
                self._esegui(job)
                with self._lock:
                    job["stato"] = "completato"
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    job["stato"] = "errore"
                    job["errore"] = f"{exc}\n\n{traceback.format_exc()}"
            finally:
                with self._lock:
                    job["durata_s"] = round(time.time() - t0, 1)
                self._coda.task_done()

    def _esegui(self, job: dict) -> None:
        import strada_b_loop as L
        client = job["payload"].get("_client") or self._nuovo_client()
        out_dir = str(storico.STORICO_DIR / job["id"])
        L.esegui_loop(
            job["sonda"],
            out_dir=out_dir,
            nucleo_path=self.nucleo_path,
            contratto_path=self.contratto_path,
            backend=self.backend,
            model=self.model,
            client=client,
            modalita=job["modalita"],
            storico_dir=str(storico.STORICO_DIR),
        )
