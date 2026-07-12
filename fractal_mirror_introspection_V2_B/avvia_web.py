"""
avvia_web.py — avvia l'interfaccia web della Strada B (Fractal · Specchio).

Gemello di run_introspezione_loop_B.py sul lato path/parametri: stesso bootstrap
del layout integrato, stesso backend/model, stesso nucleo (specchio_del_modello
_nucleo.md — fix dello slittamento di livello) e stessi parametri di generazione.
Non tocca il motore — lo riveste soltanto.

Uso:
  1. avvia il llama-server su 127.0.0.1:8080 (vedi _RUN_fractal_mirror_introspection_V2_B.txt)
  2. python avvia_web.py
  3. apri http://127.0.0.1:5000

I run finiscono in ./storico_introspezione/loopB_<timestamp>/ — le stesse cartelle
di run_introspezione_loop_B.py: la web li lancia, li mostra e li rilegge da lì.
"""
import sys
import pathlib

# --- bootstrap path del layout integrato (identico a run_introspezione_loop_B.py)
HERE = pathlib.Path(__file__).resolve().parent
SPECCHIO = HERE / "specchio_di_coscienza"
for p in (str(HERE), str(SPECCHIO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from web import storico
from web.esecutore import Esecutore
from web.app import crea_app

# === backend/model — ALLINEATI a run_introspezione_loop_B.py ================
BACKEND = "llamacpp"
MODEL = "local-model"

# --- parametri di generazione: ALLINEATI a run_introspezione_loop_B.py ------
NUM_PREDICT = 16000
NUM_CTX = 4096
TIMEOUT = 1800

# NUCLEO DEL MODELLO (fix dello slittamento di livello) — il contratto è condiviso.
NUCLEO = str(SPECCHIO / "specchio_del_modello_nucleo.md")
CONTRATTO = str(SPECCHIO / "specchio_di_coscienza_contratto_di_output.md")

# 'auto' | 'completo' | 'leggero' — preselezionata nella home, modificabile a ogni run
MODALITA_DEFAULT = "completo"

# --- host/porta del web server ----------------------------------------------
HOST = "127.0.0.1"
PORT = 5000

# --- storico: le stesse cartelle del driver da riga di comando ---------------
storico.STORICO_DIR = HERE / "storico_introspezione"


def main():
    esecutore = Esecutore(
        nucleo_path=NUCLEO, contratto_path=CONTRATTO,
        backend=BACKEND, model=MODEL,
        num_predict=NUM_PREDICT, num_ctx=NUM_CTX, timeout=TIMEOUT,
        modalita_default=MODALITA_DEFAULT,
    )
    app = crea_app(esecutore, backend=BACKEND, model=MODEL)

    h = esecutore.health()
    if h.get("ok"):
        print(f"· llama-server raggiunto su :8080 — backend {BACKEND}/{MODEL}")
    else:
        print("⚠  llama-server non raggiunto su 127.0.0.1:8080.")
        print(f"   L'interfaccia parte lo stesso; lo storico è consultabile. Dettaglio: {h.get('error','')}")

    print(f"· interfaccia su http://{HOST}:{PORT}   (run in ./storico_introspezione/)\n")
    # threaded=True: il worker dell'Esecutore gira in un thread separato e il
    # polling /stato deve poter rispondere mentre un run è in corso.
    app.run(host=HOST, port=PORT, threaded=True, debug=False)


if __name__ == "__main__":
    main()
