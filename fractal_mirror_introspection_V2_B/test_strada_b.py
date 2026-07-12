"""
test_strada_b.py — test OFFLINE (zero token) della Strada B.

Copre, per iniezione (elicitor_lp + reader mock, client=None):
  1. superficie sintattica: testo assertivo SENZA avverbi → canale informativo
  2. substrato per frase: frase debole individuata e allineata
  3. gating: anomalia → completo; quiete → leggero; substrato assente → leggero
  4. must-reject per referente: processo tenuto, contenuto rigettato
  5. collasso: sopravvalutazione con punti deboli; coerente_con_punti_deboli
  6. estrazione Specchio sui vincoli di forma (massa = 0.NN, riga esatta)
  7. telos: R2 (residuo dichiarato) e R4 (degrado marcato) correggono l'azione
  8. memoria: baseline da storico finto, z-score e anomalia
  9. esegui_loop end-to-end offline: tutti gli artefatti scritti

Esecuzione:  python test_strada_b.py
"""
import json
import math
import sys
import tempfile
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SPECCHIO = HERE / "specchio_di_coscienza"
for p in (str(HERE), str(SPECCHIO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import strada_b_loop as L
import ponte_fractal_specchio as P

OK, KO = "✓", "✗"
esiti = []


def check(nome: str, cond: bool, extra: str = ""):
    esiti.append((nome, cond))
    print(f"  {OK if cond else KO} {nome}" + (f"  [{extra}]" if extra else ""))


def _tok(token: str, p: float, alt_p: float = None):
    """Costruisce un token logprob in forma OpenAI. p = prob del token scelto."""
    lp = math.log(p)
    alts = [{"token": token, "logprob": lp}]
    resto = alt_p if alt_p is not None else max(1e-6, (1.0 - p) * 0.9)
    alts.append({"token": "~", "logprob": math.log(max(resto, 1e-9))})
    return {"token": token, "logprob": lp, "top_logprobs": alts}


def _frase_tokens(parole: list[str], p: float, chiusura: str = ". "):
    toks = []
    for i, w in enumerate(parole):
        toks.append(_tok((" " if i else "") + w, p))
    toks.append(_tok(chiusura, 0.99))
    return toks


# ---------------------------------------------------------------------------
print("\n[1] Canale 1 — assertività sintattica senza avverbi")
testo_assertivo = (
    "L'origine del fenomeno risiede nella **instabilità intrinseca** del nucleo.\n"
    "Il decadimento è un meccanismo di rilascio di energia.\n"
    "In sintesi, la causa è la tendenza verso stati stabili."
)
s = L.misura_superficie(testo_assertivo)
check("testo definitorio → canale informativo", s.informativa)
check("conf_superficie > 0.5 (assertivo)", s.conf_superficie > 0.5, f"conf={s.conf_superficie}")
check("copule definitorie rilevate", s.densita_copule_def > 0, f"d={s.densita_copule_def}")

testo_prudente = ("Forse la causa potrebbe risiedere altrove. Direi che sembra "
                  "un fenomeno che dovrebbe dipendere da più fattori.")
s2 = L.misura_superficie(testo_prudente)
check("testo prudente → conf < 0.5", s2.conf_superficie < 0.5, f"conf={s2.conf_superficie}")

s3 = L.misura_superficie("Parigi Londra Roma Berlino Madrid")
check("testo neutro → NON informativo", not s3.informativa)

# ---------------------------------------------------------------------------
print("\n[2] Canale 4 — profilo per frase, token strutturali esclusi")
logprobs = []
logprobs += _frase_tokens(["Il", "nucleo", "instabile", "rilascia", "energia"], 0.95)
logprobs += [_tok("**", 0.999), _tok("#", 0.999), _tok("\n", 0.999)]      # struttura
logprobs += _frase_tokens(["la", "costante", "vale", "circa", "sette"], 0.35)  # frase debole
corpo = L.profilo_corpo(logprobs)
check("substrato affidabile", corpo.affidabile)
check("token strutturali esclusi dal contenuto", corpo.quota_token_contenuto < 1.0,
      f"quota={corpo.quota_token_contenuto}")
check("due frasi profilate", len(corpo.frasi) == 2, f"n={len(corpo.frasi)}")
check("frase debole individuata", len(corpo.frasi_deboli) == 1
      and "costante" in corpo.frasi_deboli[0].testo,
      f"deboli={[f.testo for f in corpo.frasi_deboli]}")
check("conf contenuto < conf grezza (markdown non gonfia)",
      corpo.confidenza_contenuto < corpo.confidenza_media,
      f"cont={corpo.confidenza_contenuto} vs grezza={corpo.confidenza_media}")

# ---------------------------------------------------------------------------
print("\n[2b] Segmentazione ragionamento nascosto / risposta")
RISPOSTA_IT = "Il nucleo instabile rilascia energia. In sintesi la causa è nota."
lp_misti = []
lp_misti += _frase_tokens(["Determine", "the", "key", "scientific", "principles"], 0.40, ".\n")
lp_misti += _frase_tokens(["The", "final", "answer", "should", "emphasize", "stability"], 0.45, ".\n")
lp_misti += _frase_tokens(["Il", "nucleo", "instabile", "rilascia", "energia"], 0.95, ". ")
lp_misti += _frase_tokens(["In", "sintesi", "la", "causa", "è", "nota"], 0.93, ".")
c_seg = L.profilo_corpo(lp_misti, manifestazione=RISPOSTA_IT)
check("ragionamento nascosto individuato", c_seg.ha_ragionamento
      and c_seg.allineamento == "risposta_allineata", c_seg.allineamento)
check("ragionamento profilato a parte (conf bassa)",
      0 < c_seg.conf_ragionamento < 0.6, f"conf_rag={c_seg.conf_ragionamento}")
check("metro sul solo segmento-risposta (conf alta)",
      c_seg.confidenza_contenuto > 0.9, f"conf={c_seg.confidenza_contenuto}")
check("le frasi profilate NON includono il ragionamento",
      all("Determine" not in f.testo and "answer" not in f.testo for f in c_seg.frasi)
      and len(c_seg.frasi) == 2, f"n_frasi={len(c_seg.frasi)}")
check("nessuna frase debole dal ragionamento", len(c_seg.frasi_deboli) == 0)
check("testo del ragionamento ricostruito INTERO",
      "Determine" in c_seg.testo_ragionamento
      and "emphasize stability" in c_seg.testo_ragionamento,
      f"testo_rag={c_seg.testo_ragionamento[:60]!r}")
check("il testo del ragionamento NON contiene la risposta",
      "Il nucleo" not in c_seg.testo_ragionamento, c_seg.testo_ragionamento[-40:])
c_no_rag = L.profilo_corpo(_frase_tokens(["Il", "nucleo", "instabile", "rilascia",
                                          "energia", "in", "eccesso"], 0.9, "."),
                           manifestazione="Il nucleo instabile rilascia energia in eccesso.")
check("senza ragionamento → nessun_ragionamento",
      c_no_rag.allineamento == "nessun_ragionamento" and not c_no_rag.ha_ragionamento)
check("senza ragionamento → testo_ragionamento vuoto", c_no_rag.testo_ragionamento == "")
c_nf = L.profilo_corpo(lp_misti, manifestazione="Testo completamente diverso e mai emesso dal modello qui.")
check("manifestazione non trovata → dichiarato, metro su tutto",
      c_nf.allineamento == "manifestazione_non_trovata" and not c_nf.ha_ragionamento)
c_bc = L.profilo_corpo(lp_misti)   # retro-compatibilità: nessuna manifestazione
check("senza manifestazione → non_verificato (comportamento precedente)",
      c_bc.allineamento == "non_verificato")

# ---------------------------------------------------------------------------
print("\n[3] Gating")
g1 = L.decidi_gating(s, corpo, richiesta="auto")
check("frase debole → completo", g1.modalita == "completo", "; ".join(g1.motivi))

logprobs_quieti = _frase_tokens(["Il", "nucleo", "instabile", "rilascia", "energia"], 0.95) \
    + _frase_tokens(["la", "fisica", "descrive", "questo", "processo"], 0.93)
corpo_quieto = L.profilo_corpo(logprobs_quieti)
g2 = L.decidi_gating(s, corpo_quieto, richiesta="auto")
check("nessuna anomalia → leggero", g2.modalita == "leggero", "; ".join(g2.motivi))

g3 = L.decidi_gating(s, L.profilo_corpo(None), richiesta="auto")
check("substrato assente → leggero (risparmio dichiarato)", g3.modalita == "leggero")
g4 = L.decidi_gating(s, corpo_quieto, richiesta="completo")
check("richiesta 'completo' vince sul gating", g4.modalita == "completo")

# ---------------------------------------------------------------------------
print("\n[4] must-reject per REFERENTE")
vent = P.Ventaglio(candidati=[
    P.Candidato(testo="Le frequenze apprese nei dati di training favoriscono spiegazioni standard.",
                nature="cause", scale="sociale", epistemic="causal_model"),
    P.Candidato(testo="Il personaggio da assistente didattico spinge il registro divulgativo.",
                nature="bridge", scale="fondamentale", epistemic="causal_model"),
    P.Candidato(testo="La forza nucleare forte governa la coesione dei quark nel nucleo.",
                nature="bridge", scale="fondamentale", epistemic="causal_model"),
    P.Candidato(testo="Il bilancio tra protoni e neutroni causa l'instabilità.",
                nature="cause", scale="atomico", epistemic="domain_knowledge"),
    P.Candidato(testo="Il decadimento è un effetto della probabilità quantistica del campo subatomico.",
                nature="cause", scale="subatomico", epistemic="causal_model"),
])
f = L.must_reject(vent)
check("candidati di processo tenuti (2)", len(f.tenuti) == 2,
      str([t['testo'][:30] for t in f.tenuti]))
check("fisica su scala 'fondamentale' RIGETTATA (fix collisione di scala)",
      any("quark" in r.testo for r in f.rigettati))
check("'probabilità quantistica' RIGETTATA (fix falso positivo lessicale)",
      any("probabilità quantistica" in r.testo for r in f.rigettati))
check("fenomeno descritto rigettato", len(f.rigettati) == 3)

# ---------------------------------------------------------------------------
print("\n[5] Collasso — regole")
sup_forte = L.misura_superficie(testo_assertivo)          # presentata alta
corpo_debole = L.profilo_corpo(_frase_tokens(
    ["la", "costante", "vale", "circa", "sette"], 0.35))  # substrato debole
c1 = L.collassa(sup_forte, L.StrutturaFractal(disponibile=False),
                L.SegnaliSpecchio(disponibile=False), corpo_debole)
check("presentata alta + substrato debole → sopravvalutazione",
      c1.verdetto == "contraddetto" and c1.regola == "sopravvalutazione")
check("i punti deboli entrano nel collasso", len(c1.frasi_deboli) == 1)

c2 = L.collassa(sup_forte, L.StrutturaFractal(disponibile=False),
                L.SegnaliSpecchio(disponibile=False), corpo)   # globale ok, 1 debole
check("concordi ma con frase debole → procedi_annotando",
      c2.regola == "coerente_con_punti_deboli" and c2.azione == "procedi_annotando",
      f"regola={c2.regola}")

c3 = L.collassa(L.misura_superficie("Parigi Londra Roma"),
                L.StrutturaFractal(disponibile=False),
                L.SegnaliSpecchio(disponibile=False), corpo_quieto)
check("nessun canale controllato → procedi_cauto", c3.regola == "scarto_non_valutabile")
c4 = L.collassa(sup_forte, L.StrutturaFractal(disponibile=False),
                L.SegnaliSpecchio(disponibile=False), L.profilo_corpo(None))
check("substrato illeggibile → astieni", c4.azione == "astieni")

c5 = L.collassa(sup_forte, L.StrutturaFractal(disponibile=False),
                L.SegnaliSpecchio(disponibile=False), corpo, ventaglio_vuoto=True)
check("ventaglio vuoto DICHIARATO nel degrado",
      "ventaglio vuoto" in c5.nota_degrado and "non-scelto" in c5.nota_degrado,
      c5.nota_degrado)
check("senza il flag la nota non compare", "ventaglio vuoto" not in c2.nota_degrado)

# --- gate d'informatività del canale struttura -------------------------------
class _FtFinto:  # 2 item non causali: il caso del run filosofico (n_prop=2)
    class _It:
        def __init__(self):
            from fractal_causal_engine.ft_model import Nature, EpistemicStatus
            self.nature = Nature.CONTEXT
            self.epistemic_status = EpistemicStatus.TEXT_OBSERVED
    items = [_It(), _It()]
    cross_scale = []
    unlocked = None
s_povera = L.struttura_fractal(_FtFinto())
check("2 proposizioni → struttura NON informativa",
      s_povera.disponibile and not s_povera.informativa and s_povera.assertivita == 0.0)

class _FtRicco(_FtFinto):
    items = [_FtFinto._It() for _ in range(8)]
s_ricca = L.struttura_fractal(_FtRicco())
check("8 proposizioni → struttura informativa", s_ricca.informativa)

# replay del run delfico: superficie ok + struttura 0.0 su 2 prop.
# PRIMA del gate: presentata 0.27 → impegno_disconosciuto (artefatto).
# COL gate: la struttura esce dal blend → presentata = superficie → concordi.
c6 = L.collassa(sup_forte, s_povera, L.SegnaliSpecchio(disponibile=False), corpo)
check("struttura non informativa FUORI dal blend (niente artefatto 0.27)",
      c6.conf_presentata == c6.conf_manifest and c6.regola != "impegno_disconosciuto",
      f"presentata={c6.conf_presentata}, regola={c6.regola}")
check("il degrado dichiara il gate",
      "struttura non informativa" in c6.nota_degrado and "2 proposizioni" in c6.nota_degrado,
      c6.nota_degrado)
c7 = L.collassa(sup_forte, s_ricca, L.SegnaliSpecchio(disponibile=False), corpo)
check("struttura informativa resta nel blend",
      c7.conf_presentata != c7.conf_manifest
      and "struttura non informativa" not in c7.nota_degrado)

# ---------------------------------------------------------------------------
print("\n[6] Estrazione Specchio sui vincoli di forma")
lettura_ok = ("**6 · Massa all'inatteso**\nmassa = 0.30\n\n"
              "**9 · Nota di auto-deformazione**\nLa lettura è elegante.\n"
              "auto-deformazione: presente\n")
sg = L.estrai_segnali_specchio(lettura_ok)
check("massa numerica estratta", sg.residuo == 0.30, f"residuo={sg.residuo}")
check("riga vincolata auto-deformazione", sg.auto_deformazione is True)
sg2 = L.estrai_segnali_specchio("auto-deformazione: assente")
check("assente → False", sg2.auto_deformazione is False)

# ---------------------------------------------------------------------------
print("\n[7] Telos — correzioni per regola")
az1 = L.applica_azione(c1, "testo")
t1, az1c = L.verifica_telos(c1, az1)
check("R2: residuo alto dichiarato nell'output",
      f"residuo={c1.residuo}" in az1c.output_finale)
check("R4: degrado marcato nell'output", "[CANALI DEGRADATI" in az1c.output_finale)
check("telos conforme dopo correzioni", t1.conforme)
check("R1 conforme (sopravvalutazione annotata)", any(
    v.regola.startswith("R1") and v.esito == "conforme" for v in t1.verifiche))

# ---------------------------------------------------------------------------
print("\n[8] Memoria — baseline da storico finto")
with tempfile.TemporaryDirectory() as td:
    st = pathlib.Path(td)
    for i, (cf, en) in enumerate([(0.90, 0.4), (0.88, 0.5), (0.91, 0.45), (0.89, 0.42)]):
        d = st / f"loop_2026070{i}_000000"
        d.mkdir()
        (d / "04_corpo.json").write_text(json.dumps(
            {"affidabile": True, "confidenza_media": cf, "entropia_media": en}),
            encoding="utf-8")
    mem = L.carica_memoria(st, corpo_debole)     # run corrente conf ~0.35
    check("baseline costruita (4 run)", mem.disponibile and mem.n_run == 4)
    check("run debole → anomalo_basso", mem.substrato_vs_storia == "anomalo_basso",
          f"z={mem.z_conf}")
    mem2 = L.carica_memoria(st, corpo_quieto)    # run corrente conf ~0.93
    check("run in firma → nella_norma o anomalo dichiarato",
          mem2.substrato_vs_storia in ("nella_norma", "anomalo_alto"),
          mem2.substrato_vs_storia)
    check("diagnosi dichiarata (dir e profili contati)",
          "4 cartelle" in mem.motivo and "4 profili" in mem.motivo, mem.motivo)
    # fix glob: anche le cartelle loopB_* (Strada B, 02_corpo.json) contano
    dB = st / "loopB_20260707_000000"
    dB.mkdir()
    (dB / "02_corpo.json").write_text(json.dumps(
        {"affidabile": True, "confidenza_contenuto": 0.90, "entropia_contenuto": 0.45}),
        encoding="utf-8")
    memB = L.carica_memoria(st, corpo_quieto)
    check("cartelle loopB_* incluse nella baseline (fix glob)",
          memB.n_run == 5, f"n_run={memB.n_run} · {memB.motivo}")
mem_ko = L.carica_memoria("/percorso/inesistente_xyz", corpo_quieto)
check("cartella inesistente → motivo esplicito",
      "inesistente" in mem_ko.motivo and mem_ko.n_run == 0, mem_ko.motivo)

# ---------------------------------------------------------------------------
print("\n[9] esegui_loop end-to-end OFFLINE (elicitor_lp + reader mock, client=None)")
MANIF = ("L'origine del decadimento radioattivo risiede nella instabilità del nucleo.\n"
         "Il decadimento è un meccanismo di rilascio di energia.\n"
         "In sintesi, la causa è la tendenza verso stati stabili.")


def _elicitor_lp(sonda, system):
    lp = []
    lp += _frase_tokens(["L'origine", "del", "decadimento", "risiede", "nella",
                         "instabilità", "del", "nucleo"], 0.92, ".\n")
    lp += _frase_tokens(["Il", "decadimento", "è", "un", "meccanismo",
                         "di", "rilascio"], 0.45, ".\n")     # frase debole → gating completo
    lp += _frase_tokens(["In", "sintesi", "la", "causa", "è", "la", "tendenza"], 0.90, ".")
    return MANIF, lp


def _reader_mock(input_composto, system):
    # verifica che il frame sia arrivato allo Specchio (fix slittamento)
    assert "OUTPUT PRODOTTO DA UN MODELLO LINGUISTICO" in input_composto, \
        "frame Specchio mancante nell'input"
    assert "Lo Specchio del Modello" in system, "nucleo del modello non montato"
    return ("**5 · Superposizione**\nP1: frequenze di training divulgative (peso alto)\n\n"
            "**6 · Massa all'inatteso**\nmassa = 0.25\n\n"
            "**9 · Nota di auto-deformazione**\nauto-deformazione: presente\n\n"
            "**10 · Consegna**\nSuperposizione aperta.")


with tempfile.TemporaryDirectory() as td:
    out = pathlib.Path(td) / "storico" / "loopB_test"

    foto_meta_run = {}
    def _reader_progressivo(input_composto, system):
        # fotografato DURANTE il run, alla fase Specchio: i livelli già scritti
        foto_meta_run["presenti"] = sorted(p.name for p in out.glob("*.json"))
        return _reader_mock(input_composto, system)

    res = L.esegui_loop(
        "sonda di prova?", out_dir=str(out),
        nucleo_path=str(SPECCHIO / "specchio_del_modello_nucleo.md"),
        contratto_path=str(SPECCHIO / "specchio_di_coscienza_contratto_di_output.md"),
        client=None, reader=_reader_progressivo, elicitor_lp=_elicitor_lp, modalita="auto")

    presenti = foto_meta_run.get("presenti", [])
    attesi_meta = ["00_manifestazione.json", "01_superficie.json", "02_corpo.json",
                   "03_gating.json", "04_struttura_fractal.json",
                   "06_must_reject.json", "07_memoria.json"]
    check("scrittura PROGRESSIVA: livelli economici già su disco alla fase Specchio",
          all(a in presenti for a in attesi_meta), f"presenti={presenti}")
    check("scrittura progressiva: il collasso NON esiste ancora alla fase Specchio",
          "08_collasso.json" not in presenti and "10_azione.json" not in presenti)

    check("gating auto → completo (frase debole)", res["gating"].modalita == "completo")
    check("Specchio letto e segnali estratti",
          res["specchio"].disponibile and res["specchio"].residuo == 0.25)
    check("collasso emesso", res["collasso"].verdetto in
          ("coerente", "contraddetto", "indeterminato"), res["collasso"].regola)
    check("residuo integra lo Specchio",
          abs(res["collasso"].residuo -
              round((1 - res["corpo"].confidenza_contenuto + 0.25) / 2, 3)) < 0.01,
          f"residuo={res['collasso'].residuo}")
    check("telos eseguito", len(res["telos"].verifiche) == 4)

    attesi = ["00_manifestazione.json", "01_superficie.json", "02_corpo.json",
              "03_gating.json", "04_struttura_fractal.json", "05_specchio_segnali.json",
              "06_must_reject.json", "07_memoria.json", "08_collasso.json",
              "09_telos.json", "10_azione.json", "11_specchio_lettura.md",
              "report.md", "GUIDA_interpretazione.md"]
    mancanti = [a for a in attesi if not (out / a).exists()]
    check("tutti gli artefatti scritti", not mancanti, f"mancanti={mancanti}")
    check("indice memoria aggiornato",
          (out.parent / "indice_memoria.jsonl").exists())
    check("trace elicitazione + specchio",
          (out / "trace" / "llm_calls" / "0000_Elicitazione.json").exists()
          and (out / "trace" / "llm_calls" / "9000_Specchio.json").exists())

# --- loop leggero end-to-end -------------------------------------------------
def _elicitor_quieto(sonda, system):
    lp = _frase_tokens(["L'origine", "risiede", "nella", "instabilità",
                        "del", "nucleo", "atomico"], 0.94, ".")
    lp += _frase_tokens(["In", "sintesi", "la", "causa", "è", "nota"], 0.93, ".")
    return MANIF, lp


with tempfile.TemporaryDirectory() as td:
    out = pathlib.Path(td) / "loopB_leggero"
    res = L.esegui_loop(
        "sonda di prova?", out_dir=str(out),
        nucleo_path=str(SPECCHIO / "specchio_del_modello_nucleo.md"),
        contratto_path=str(SPECCHIO / "specchio_di_coscienza_contratto_di_output.md"),
        client=None, reader=_reader_mock, elicitor_lp=_elicitor_quieto, modalita="auto")
    check("nessuna anomalia → loop LEGGERO (Specchio e Fractal spenti)",
          res["gating"].modalita == "leggero" and not res["specchio"].disponibile)
    check("verdetto comunque emesso (canali economici)",
          res["collasso"].verdetto == "coerente", res["collasso"].regola)

# ---------------------------------------------------------------------------
tot = len(esiti)
ok = sum(1 for _, c in esiti if c)
print(f"\n{'='*60}\nESITO: {ok}/{tot} verdi")
sys.exit(0 if ok == tot else 1)
