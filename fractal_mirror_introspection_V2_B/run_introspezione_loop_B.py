"""
run_introspezione_loop_B.py — driver della Strada B (riscrittura del loop chiuso).

Gemello di run_introspezione_loop.py, ma sul nuovo strada_b_loop:
bersaglio sul gesto (nucleo del modello + frame allo Specchio + must-reject per
referente), canale 1 sintattico, substrato per frase, gating, memoria, telos.

TODO 1 — la sonda ; TODO 2 — backend/model (identici alla A).
Poi: `python run_introspezione_loop_B.py`. Guarda report.md nella cartella
loopB_<timestamp>: verdetto, azione, telos, e il perché di ogni livello.
"""
import sys, pathlib, datetime

HERE = pathlib.Path(__file__).resolve().parent
SPECCHIO = HERE / "specchio_di_coscienza"
for p in (str(HERE), str(SPECCHIO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import strada_b_loop as L
from probes_introspezione import PROBES_BY_ID

# NUCLEO DEL MODELLO (fix dello slittamento di livello) — il contratto è condiviso.
NUCLEO = str(SPECCHIO / "specchio_del_modello_nucleo.md")
CONTRATTO = str(SPECCHIO / "specchio_di_coscienza_contratto_di_output.md")

# === TODO 1 — la sonda ======================================================
#SONDA = PROBES_BY_ID["confabulazione"]["sonda"]

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# LA DOMANDA AL MODELLO:
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

SONDA = "qual'è l'origine del decadimento radioattivo?"

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# === TODO 2 — backend/model =================================================
BACKEND = "llamacpp"
MODEL = "local-model"
NUM_PREDICT, NUM_CTX, TIMEOUT = 16000, 4096, 1800

# 'auto'     = gating: il loop pesante (Fractal+Specchio) si accende solo su anomalia
# 'completo' = forza sempre il loop pesante (comportamento della Strada A)
# 'leggero'  = solo canali economici (superficie + substrato)
MODALITA = "completo"     # per i primi test tienilo 'completo', poi passa ad 'auto'

from fractal_causal_engine.llm import LLMClient, LLMConfig
client = LLMClient(LLMConfig(backend="llamacpp", model=MODEL,
                             num_predict=NUM_PREDICT, num_ctx=NUM_CTX,
                             timeout_seconds=TIMEOUT))

_h = client.health_check()
if not _h.get("ok"):
    sys.exit("⚠  Nessun llama-server su :8080. Avvialo prima (vedi run_serie.py).")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
STORICO = HERE / "storico_introspezione"
OUT = str(STORICO / f"loopB_{stamp}")

esito = L.esegui_loop(
    SONDA, out_dir=OUT, nucleo_path=NUCLEO, contratto_path=CONTRATTO,
    backend=BACKEND, model=MODEL, client=client, modalita=MODALITA,
    storico_dir=str(STORICO))

c, a, g, t = esito["collasso"], esito["azione"], esito["gating"], esito["telos"]
print("=== SONDA ===\n" + SONDA)
print("\n=== MANIFESTAZIONE ===\n" + esito["manifestazione"])
print(f"\n=== GATING ===\nmodalità: {g.modalita}  ·  motivi: {'; '.join(g.motivi)}")
print(f"\n=== COLLASSO ===\nverdetto: {c.verdetto}  ·  regola: {c.regola}")
print(f"{c.motivazione}\nconfidenza: {c.confidenza}  ·  residuo(budget): {c.residuo}")
if c.frasi_deboli:
    print("frasi con substrato debole:")
    for d in c.frasi_deboli:
        print(f"  · (conf={d['confidenza']}) “{d['testo']}”")
if c.nota_memoria:
    print(f"memoria: {c.nota_memoria}")
print(f"\n=== TELOS ===\nconforme: {t.conforme}")
for v in t.verifiche:
    print(f"  · {v.regola} → {v.esito}" + (f" ({v.intervento})" if v.intervento else ""))
print(f"\n=== AZIONE ===\n{a.tipo} — {a.nota or 'nessuna nota'}")
print(f"\n· artefatti per livello scritti in: {OUT}")
print("· report.md ispezionabile lì dentro")
