"""
test_web_strada_b.py — test OFFLINE (zero token) del rivestimento web della Strada B.

Copre, su uno storico sintetico in tempdir (nessun run reale toccato):
  1. storico.elenco: sintesi dai file di livello, ordine inverso, run incompleto marcato
  2. storico.carica: ricomposizione completa + riepilogo trace (durate, retry)
  3. sicurezza: run_id e file grezzi fuori whitelist → None (niente path traversal)
  4. nota del gate: scrittura/rimozione di nota_gate.md
  5. esecutore.stato: progresso letto dagli artefatti reali + attore in corso dalla trace
  6. esecutore.avvia: accoda, il worker esegue (con _esegui iniettato) e chiude il job
  7. app: home / storico / run completo / run incompleto (pagina di attesa) /
     stato JSON / file grezzo → 200; heatmap e verdetto presenti nell'HTML

Esecuzione:  python test_web_strada_b.py
"""
import json
import pathlib
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
SPECCHIO = HERE / "specchio_di_coscienza"
for p in (str(HERE), str(SPECCHIO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from web import storico
from web.esecutore import Esecutore
from web.app import crea_app, _blocchi_heatmap, _md_minimo

ok = tot = 0


def check(nome: str, cond: bool, dettaglio: str = ""):
    global ok, tot
    tot += 1
    ok += bool(cond)
    print(("  ✓ " if cond else "  ✗ ") + nome + (f" — {dettaglio}" if dettaglio and not cond else ""))


# ---------------------------------------------------------------------------
# Storico sintetico: due run, uno completo e uno a metà
# ---------------------------------------------------------------------------

def scrivi_run(base: pathlib.Path, nome: str, *, completo: bool) -> pathlib.Path:
    d = base / nome
    (d / "trace").mkdir(parents=True)
    dati = {
        "00_manifestazione.json": {"sonda": "sonda di prova?", "manifestazione": "Prima frase certa. Seconda frase incerta."},
        "01_superficie.json": {"densita_hedge": 0.0, "densita_asserzione": 0.1,
                               "densita_copule_def": 0.2, "densita_condizionale": 0.0,
                               "quota_grassetti": 0.0, "ha_sezione_sintesi": False,
                               "conf_superficie": 0.7, "informativa": True},
        "02_corpo.json": {"affidabile": True, "n_token": 20, "confidenza_media": 0.8,
                          "confidenza_contenuto": 0.82, "entropia_media": 0.5,
                          "entropia_contenuto": 0.5, "quota_alta_conf": 0.6,
                          "quota_esitazione": 0.05, "n_punti_impegno": 10,
                          "quota_token_contenuto": 0.7,
                          "frasi": [
                              {"indice": 0, "testo": "Prima frase certa.", "n_token_contenuto": 4,
                               "confidenza": 0.95, "entropia": 0.1},
                              {"indice": 1, "testo": "Seconda frase incerta.", "n_token_contenuto": 4,
                               "confidenza": 0.55, "entropia": 1.4}],
                          "frasi_deboli": [{"indice": 1, "testo": "Seconda frase incerta.",
                                            "n_token_contenuto": 4, "confidenza": 0.55, "entropia": 1.4}],
                          "allineamento": "risposta_allineata", "ha_ragionamento": True,
                          "n_token_ragionamento": 12, "conf_ragionamento": 0.7,
                          "entropia_ragionamento": 0.9,
                          "testo_ragionamento": ("<|channel|>thought\n"
                              "1. **Analyze the request:** identify the stable claim first.\n"
                              "   * *Initial thought:* keep it simple."),
                          "motivo": ""},
        "03_gating.json": {"modalita": "completo", "richiesta": "completo",
                           "motivi": ["richiesto esplicitamente"]},
        "04_struttura_fractal.json": {"n_prop": 2, "quota_cause_effect": 0.5,
                                      "quota_speculativa": 0.0, "assertivita": 0.5,
                                      "nessi_totali": 0, "nessi_genuine": 0,
                                      "tenuta_nessi": 0.0, "disponibile": True, "fonte": "items"},
        "05_specchio_segnali.json": {"disponibile": True, "residuo": 0.2, "auto_deformazione": False},
    }
    if completo:
        dati.update({
            "06_must_reject.json": {"ventaglio_filtrato": {
                "tenuti": [{"testo": "la scelta del token X", "scala": "processo", "motivo": ""}],
                "rigettati": [{"testo": "il fenomeno descritto", "scala": "atomico",
                               "motivo": "referente = fenomeno descritto: fuori dal gesto"}]}},
            "07_memoria.json": {"disponibile": True, "n_run": 2, "media_conf": 0.8,
                                "dev_conf": 0.02, "media_entropia": 0.5, "dev_entropia": 0.05,
                                "z_conf": 1.0, "z_entropia": 0.0,
                                "substrato_vs_storia": "nella_norma", "motivo": ""},
            "08_collasso.json": {"verdetto": "coerente", "azione": "procedi_annotando",
                                 "confidenza": 0.82, "residuo": 0.24,
                                 "regola": "coerente_con_punti_deboli",
                                 "motivazione": "concordi nel globale, 1 frase debole",
                                 "modalita_loop": "completo", "conf_manifest": 0.7,
                                 "conf_struttura": 0.5, "specchio_residuo": 0.2,
                                 "specchio_deformazione": False, "conf_substrato": 0.82,
                                 "conf_presentata": 0.58, "corroborazione_specchio": "",
                                 "canali_controllati_attivi": 3, "nota_degrado": "",
                                 "frasi_deboli": [{"testo": "Seconda frase incerta.",
                                                   "confidenza": 0.55, "entropia": 1.4}],
                                 "nota_memoria": "substrato vs storia (2 run): nella_norma"},
            "09_telos.json": {"conforme": True, "verifiche": [
                {"regola": "R1: mai presentare certezza che il substrato non regge",
                 "esito": "conforme", "intervento": ""}]},
            "10_azione.json": {"tipo": "procedi_annotando", "output_finale": "Prima frase certa.",
                               "confidenza": 0.82, "nota": "annotate le frasi deboli"},
        })
    for nome_f, contenuto in dati.items():
        (d / nome_f).write_text(json.dumps(contenuto, ensure_ascii=False, indent=2), encoding="utf-8")
    if completo:
        (d / "11_specchio_lettura.md").write_text("**1 · Attributi**\nlettura di prova.", encoding="utf-8")
        (d / "report.md").write_text("# report di prova", encoding="utf-8")
    righe = [
        {"event": "actor_start", "call_id": "0001_A", "actor": "Attore_A"},
        {"event": "llm_call_truncated_retry", "call_id": "0001_A", "role": "Attore_A"},
        {"event": "actor_end", "call_id": "0001_A", "actor": "Attore_A",
         "elapsed_seconds": 10.5, "status": "ok"},
    ]
    if not completo:  # attore aperto → "in corso"
        righe.append({"event": "actor_start", "call_id": "0002_B", "actor": "Attore_B"})
    (d / "trace" / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(r) for r in righe) + "\n", encoding="utf-8")
    return d


tmp = tempfile.TemporaryDirectory()
BASE = pathlib.Path(tmp.name) / "storico_introspezione"
BASE.mkdir()
scrivi_run(BASE, "loopB_20260101_100000", completo=True)
scrivi_run(BASE, "loopB_20260101_110000", completo=False)
(BASE / "indice_memoria.jsonl").write_text(json.dumps({
    "timestamp": "2026-01-01T10:05:00", "out_dir": str(BASE / "loopB_20260101_100000"),
    "sonda": "sonda di prova?", "model": "modello-finto", "modalita": "completo",
    "verdetto": "coerente"}) + "\n", encoding="utf-8")
storico.STORICO_DIR = BASE

# ---------------------------------------------------------------------------
print("1. storico.elenco")
voci = storico.elenco()
check("due run, il più recente in cima", len(voci) == 2 and voci[0]["id"].endswith("110000"))
check("run a metà marcato incompleto", voci[0]["completo"] is False)
check("run completo: verdetto e metriche", voci[1]["verdetto"] == "coerente"
      and voci[1]["conf_substrato"] == 0.82 and voci[1]["n_frasi_deboli"] == 1)
check("model arricchito dall'indice", voci[1]["model"] == "modello-finto")

print("2. storico.carica + trace")
rec = storico.carica("loopB_20260101_100000")
check("record completo ricomposto", rec["completo"] and rec["collasso"]["regola"] == "coerente_con_punti_deboli")
check("lettura Specchio presente", "Attributi" in rec["lettura_specchio"])
a = rec["trace"]["attori"][0]
check("trace: durata, retry, stato", a["durata_s"] == 10.5 and a["retry_troncamento"] == 1 and a["stato"] == "ok")

print("3. sicurezza dei percorsi")
check("run_id con traversal → None", storico.percorso_run("../fuori") is None
      and storico.percorso_run("a/b") is None)
check("file fuori whitelist → None", storico.leggi_grezzo("loopB_20260101_100000", "nota_gate.md") is None)

print("4. nota del gate")
storico.aggiorna_nota_gate("loopB_20260101_100000", "vedo io")
check("nota scritta", (BASE / "loopB_20260101_100000" / "nota_gate.md").read_text(encoding="utf-8").strip() == "vedo io")
storico.aggiorna_nota_gate("loopB_20260101_100000", "")
check("nota vuota rimuove il file", not (BASE / "loopB_20260101_100000" / "nota_gate.md").exists())

print("5. esecutore.stato (progresso dagli artefatti)")
e = Esecutore.__new__(Esecutore)   # senza worker: testiamo la sola lettura
import threading
e._jobs, e._lock = {}, threading.Lock()
s = e.stato("loopB_20260101_110000")
check("run a metà: 6/11 livelli, incompleto", s["stato"] == "incompleto" and s["livelli_fatti"] == 6)
check("attore in corso dalla trace", s["attore_in_corso"] == "Attore_B")
check("run completo: stato completato", e.stato("loopB_20260101_100000")["stato"] == "completato")
check("run sconosciuto → None", e.stato("loopB_00000000_000000") is None)

print("6. esecutore.avvia (worker con _esegui iniettato)")
eseguiti = []
class EsecFinto(Esecutore):
    def __init__(self):
        super().__init__(nucleo_path="n", contratto_path="c", backend="b", model="m",
                         num_predict=1, num_ctx=1, timeout=1, modalita_default="auto")
    def _esegui(self, job):
        eseguiti.append((job["sonda"], job["modalita"]))
ef = EsecFinto()
rid = ef.avvia({"sonda": "prova?", "modalita": "leggero"})
for _ in range(50):
    if ef.stato(rid) and ef.stato(rid)["stato"] in ("completato", "errore"):
        break
    time.sleep(0.05)
check("job accodato ed eseguito", eseguiti == [("prova?", "leggero")])
check("run_id nel formato loopB_*", rid.startswith("loopB_"))
check("job chiuso senza errore", ef.stato(rid)["errore"] is None)

print("7. app: rotte e rendering")
class EsecLettura:
    modalita_default = "completo"
    def health(self): return {"ok": False, "error": "offline"}
    def occupato(self): return False
    def stato(self, rid): return e.stato(rid)
app = crea_app(EsecLettura(), backend="b", model="m")
cl = app.test_client()
check("home 200", cl.get("/").status_code == 200)
h_html = cl.get("/").data.decode("utf-8")
check("home: sonda libera PRIMA dei preset",
      0 < h_html.find('id="sonda"') < h_html.find('class="chip"'))
check("home: preset come chip che riempiono la sonda",
      'data-sonda="sonda' not in h_html and h_html.count('class="chip"') >= 1)
r_tabs = cl.get("/run/loopB_20260101_100000").data.decode("utf-8")
check("run: barra con 5 tab tematici", r_tabs.count('data-tab="') == 5
      and all(f'id="pan-{n}"' in r_tabs for n in
              ["verdetto", "manifestazione", "canali", "specchio", "chiusura"]))
check("run: 5 popover informativi con bottone ⓘ",
      r_tabs.count('class="info-btn"') == 5 and r_tabs.count('class="popover"') == 5
      and "Cosa stai osservando" in r_tabs)
check("run: canali a righe chiave-valore", r_tabs.count('class="kv"') >= 15)
check("niente collisione .tab: le tabelle usano .tabella",
      '<table class="tab"' not in r_tabs and 'class="tabella"' in r_tabs)
r_intro = cl.get("/introduzione")
check("pagina Introduzione 200 con i 4 canali", r_intro.status_code == 200
      and "quattro canali".encode() in r_intro.data
      and "Substrato".encode() in r_intro.data)
check("Introduzione nel menu", "Introduzione".encode() in cl.get("/").data)
# fix segnalati: niente sottolineatura/color-mix, niente manifestazione duplicata,
# note del verdetto etichettate
css = pathlib.Path("web/static/stile.css").read_text(encoding="utf-8")
check("heatmap: tinta precalcolata, niente color-mix né accento inferiore",
      "color-mix" not in css and "--cbg: rgba(" in r_tabs
      and "inset 0 -2px 0 var(--c)" not in css)
check("manifestazione non duplicata (heatmap = testo vero)",
      "manifestazione originale" not in r_tabs and "testo originale" in r_tabs)
check("verdetto: note etichettate (regola/memoria)",
      r_tabs.count('class="nota-lab"') >= 2 and ">regola<" in r_tabs
      and ">memoria<" in r_tabs)
check("storico 200", cl.get("/storico").status_code == 200)
r = cl.get("/run/loopB_20260101_100000")
check("run completo 200 con verdetto e heatmap", r.status_code == 200
      and b"coerente" in r.data and "Prima frase certa".encode() in r.data)
check("frase debole evidenziata", b'class="frase debole"' in r.data)
check("ragionamento nascosto: testo mostrato e formattato",
      "identify the stable claim first".encode() in r.data
      and "<strong>Analyze the request:</strong>".encode() in r.data)
check("marcatore di canale reso come badge",
      'class="canale-tag"'.encode() in r.data
      and "channel · thought".encode() in r.data
      and "&lt;|channel|&gt;".encode() not in r.data)
check("lista numerata e sotto-punto indentato",
      "<li>1. <strong>".encode() in r.data and 'li class="ind"'.encode() in r.data)
# retro-compatibilità: run vecchio senza testo_ragionamento → nota, non errore
d_old = BASE / "loopB_20260101_100000"
corpo_j = json.loads((d_old / "02_corpo.json").read_text(encoding="utf-8"))
del corpo_j["testo_ragionamento"]
(d_old / "02_corpo.json").write_text(json.dumps(corpo_j, ensure_ascii=False), encoding="utf-8")
r_old = cl.get("/run/loopB_20260101_100000")
check("run precedente al campo → nota di indisponibilità, pagina intatta",
      r_old.status_code == 200 and "testo non disponibile".encode() in r_old.data)
corpo_j["testo_ragionamento"] = ("<|channel|>thought\n"
    "1. **Analyze the request:** identify the stable claim first.\n"
    "   * *Initial thought:* keep it simple.")
(d_old / "02_corpo.json").write_text(json.dumps(corpo_j, ensure_ascii=False), encoding="utf-8")
r = cl.get("/run/loopB_20260101_110000")
check("run a metà → pagina di attesa col progresso", r.status_code == 200 and b"6/11" in r.data)
check("stato JSON 200", cl.get("/stato/loopB_20260101_100000").status_code == 200)
check("grezzo whitelisted 200", cl.get("/run/loopB_20260101_100000/grezzo/08_collasso.json").status_code == 200)
check("grezzo fuori whitelist 404", cl.get("/run/loopB_20260101_100000/grezzo/nota_gate.md").status_code == 404)
check("run inesistente 404", cl.get("/run/loopB_00000000_000000").status_code == 404)

print("9. export del run in un solo markdown")
r = cl.get("/run/loopB_20260101_100000/esporta.md")
check("rotta di export 200 con download", r.status_code == 200
      and 'attachment; filename="loopB_20260101_100000.md"' in r.headers.get("Content-Disposition", ""))
mdoc = r.data.decode("utf-8")
attesi = ["# Run introspettivo loopB_20260101_100000", "Come leggere questo documento",
          "## Sonda", "sonda di prova?", "## Verdetto del collasso", "coerente",
          "## Manifestazione", "Prima frase certa. Seconda frase incerta.",
          "## Ragionamento nascosto", "identify the stable claim first",
          "Profilo per frase", "| 0 | 0.95 |", "Frasi con substrato debole",
          "## Must-reject", "la scelta del token X", "referente = fenomeno descritto",
          "## Memoria", "nella_norma", "## Telos", "R1: mai presentare certezza",
          "## Lettura dello Specchio", "lettura di prova",
          "## Azione", "### Output finale consegnato", "## Costo del run", "Attore_A"]
mancanti = [a for a in attesi if a not in mdoc]
check("tutte le sezioni esplose presenti", not mancanti, f"mancano: {mancanti}")
check("nota del gate assente se non scritta", "Nota del gate umano" not in mdoc)
storico.aggiorna_nota_gate("loopB_20260101_100000", "osservazione del gate")
mdoc2 = cl.get("/run/loopB_20260101_100000/esporta.md").data.decode("utf-8")
check("nota del gate inclusa quando c'e'", "## Nota del gate umano" in mdoc2
      and "osservazione del gate" in mdoc2)
storico.aggiorna_nota_gate("loopB_20260101_100000", "")
# run a meta': esporta comunque, dichiarando le assenze
r_mid = cl.get("/run/loopB_20260101_110000/esporta.md")
mdoc_mid = r_mid.data.decode("utf-8")
check("run a meta': export 200 con assenze dichiarate", r_mid.status_code == 200
      and "Run incompleto" in mdoc_mid and "*(collasso non disponibile)*" in mdoc_mid)
check("run inesistente -> 404", cl.get("/run/loopB_00000000_000000/esporta.md").status_code == 404)
check("bottone di download nella pagina del run",
      "Scarica il run (.md)".encode() in cl.get("/run/loopB_20260101_100000").data)
# gate d'informatività della struttura: run vecchi senza campo → nel blend (default);
# informativa=false → badge "fuori dal blend"
check("run senza campo informativa → nessun badge (retro-compat)",
      "fuori dal blend".encode() not in cl.get("/run/loopB_20260101_100000").data)
s_j = json.loads((d_old / "04_struttura_fractal.json").read_text(encoding="utf-8"))
s_j["informativa"] = False
(d_old / "04_struttura_fractal.json").write_text(json.dumps(s_j), encoding="utf-8")
check("struttura non informativa → badge fuori dal blend",
      "non informativa: fuori dal blend".encode() in cl.get("/run/loopB_20260101_100000").data)
del s_j["informativa"]
(d_old / "04_struttura_fractal.json").write_text(json.dumps(s_j), encoding="utf-8")

print("10. helper di presentazione — heatmap sul testo vero")
corpo = json.loads((BASE / "loopB_20260101_100000" / "02_corpo.json").read_text(encoding="utf-8"))
man = "Prima frase certa. Seconda frase incerta."
blocchi = _blocchi_heatmap(man, corpo)
sp = [s for b in blocchi for s in b["spans"]]
check("un span per frase, allineati", len(sp) == 2 and sp[0]["conf"] == 0.95 and sp[1]["conf"] == 0.55)
check("frase debole marcata", sp[1]["debole"] and not sp[0]["debole"])

# allineamento col troncamento a 90 caratteri del motore (frasi = diagnostica)
lunga = "Questa è una frase molto lunga che il canale substrato tronca a novanta caratteri esatti nel suo profilo diagnostico."
corpo_tr = {"frasi": [{"indice": 0, "testo": lunga.strip()[:90], "n_token_contenuto": 20,
                       "confidenza": 0.9, "entropia": 0.2}]}
bl = _blocchi_heatmap(lunga, corpo_tr)
s0 = bl[0]["spans"][0]
check("frase piena riallineata al profilo troncato", s0["testo"].rstrip() == lunga
      and not s0["neutro"] and s0["conf"] == 0.9)

# struttura conservata + veste markdown ripulita + segmenti saltati neutri
man_md = ("### 1. Il Titolo\n\nUn **paragrafo** con enfasi.\n"
          "* voce di lista profilata.\n* voce saltata dal substrato.")
corpo_md = {"frasi": [
    {"indice": 0, "testo": "Un **paragrafo** con enfasi.", "n_token_contenuto": 5,
     "confidenza": 0.8, "entropia": 0.3},
    {"indice": 1, "testo": "* voce di lista profilata.", "n_token_contenuto": 5,
     "confidenza": 0.7, "entropia": 0.4}]}
bl = _blocchi_heatmap(man_md, corpo_md)
tipi = [b["tipo"] for b in bl]
check("blocchi: titolo, paragrafo, due voci di lista", tipi == ["titolo", "paragrafo", "lista", "lista"])
sp_par = bl[1]["spans"][0]
check("markdown ripulito nel display, match sul grezzo",
      sp_par["testo"].strip() == "Un paragrafo con enfasi." and sp_par["conf"] == 0.8)
check("marcatore di lista tolto, profilo agganciato",
      bl[2]["spans"][0]["testo"].startswith("voce di lista") and bl[2]["spans"][0]["conf"] == 0.7)
check("segmento saltato dal substrato → neutro, senza disallineare",
      bl[3]["spans"][0]["neutro"] and bl[0]["spans"][0]["neutro"])
check("md minimo: grassetto e lista", "<strong>x</strong>" in _md_minimo("**x**")
      and "<li>a</li>" in _md_minimo("* a"))

print("\n" + "=" * 60)
print(f"ESITO: {ok}/{tot} verdi")
tmp.cleanup()
sys.exit(0 if ok == tot else 1)
