"""
test_integrazione_fractal.py — verifica gli interventi dell'integrazione
completa del fractal_causal_engine nella Strada B (handoff I-1..I-4):

  [1] I-1  genera() persiste la struttura completa (ft_analysis.json,
           final_report.md, trace.md) — e MAI la magistrale (invariante).
  [2] I-2  le chiamate L5 dell'expander entrano in trace/llm_calls.
  [3] I-3  esegui_loop scrive 04b_ventaglio.json (trust + candidati con
           provenienza) e must_reject propaga parent_id/confidence.
  [4] I-4  leggi_grezzo serve le llm_calls via pattern e respinge i path
           fuori whitelist (niente traversal).

Tutto offline: client mock del motore, reader/elicitor finti (stesso pattern
di test_ponte.py e test_strada_b.py).
"""
import json
import math
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SPECCHIO = HERE / "specchio_di_coscienza"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPECCHIO))

from fractal_causal_engine.llm import LLMClient, LLMConfig  # noqa: E402
import ponte_fractal_specchio as P                          # noqa: E402
import strada_b_loop as L                                   # noqa: E402

OK, KO = "\033[92mOK\033[0m", "\033[91mKO\033[0m"
esiti = []


def check(nome: str, cond: bool, extra: str = ""):
    esiti.append((nome, cond))
    print(f"  {OK if cond else KO} {nome}" + (f"  [{extra}]" if extra else ""))


def _tok(token: str, p: float):
    lp = math.log(p)
    alts = [{"token": token, "logprob": lp},
            {"token": "~", "logprob": math.log(max(1e-9, (1.0 - p) * 0.9))}]
    return {"token": token, "logprob": lp, "top_logprobs": alts}


def _frase_tokens(parole, p, chiusura=". "):
    toks = [_tok((" " if i else "") + w, p) for i, w in enumerate(parole)]
    toks.append(_tok(chiusura, 0.99))
    return toks


MANIF = ("Un uomo parla con voce ferma di un lutto recente, lessico tecnico e "
         "controllato; ma le mani gli tremano e la voce cede su una parola.")


# ---------------------------------------------------------------------------
print("\n[1] I-1 — genera() persiste la struttura completa, mai la magistrale")
with tempfile.TemporaryDirectory() as td:
    out = pathlib.Path(td)
    client = LLMClient(LLMConfig(mock=True))
    ft, records = P.genera(MANIF, client=client, top_n_espansioni=5,
                           out_dir=str(out))

    fa = out / "ft_analysis.json"
    check("ft_analysis.json scritto", fa.exists())
    check("final_report.md scritto", (out / "final_report.md").exists())
    check("trace.md scritto", (out / "trace.md").exists())
    if fa.exists():
        payload = json.loads(fa.read_text(encoding="utf-8"))
        attese = {"items", "locked_reports", "unlocked", "cross_scale",
                  "double_cone", "vision", "trace"}
        check("ft_analysis contiene le chiavi della struttura completa",
              attese.issubset(payload.keys()),
              f"mancanti={sorted(attese - set(payload.keys()))}")
        check("la magistrale NON è nel payload persistito",
              "magistrale" not in payload)
    check("invariante: ft.magistrale mai costruita", ft.magistrale is None)
    check("il report mostra la manifestazione VERA (non il frame)",
          MANIF[:40] in (out / "final_report.md").read_text(encoding="utf-8"))

    # -----------------------------------------------------------------------
    print("\n[2] I-2 — le chiamate L5 entrano nel trace della pipeline")
    calls = sorted(p.name for p in (out / "llm_calls").glob("*.json"))
    check("llm_calls popolata dalla pipeline", len(calls) > 0, f"n={len(calls)}")
    l5 = [c for c in calls if "L5_FractalExpander" in c]
    if records:
        check("chiamate L5_FractalExpander tracciate (espansioni avvenute)",
              len(l5) > 0, f"l5={l5}")
    else:
        # il mock può non produrre item espandibili: esercitiamo l'expander
        # a mano con gli stessi path della pipeline (come fa genera()).
        from fractal_causal_engine.ft_model import (ClassifiedItem, Nature,
                                                    PredicateType,
                                                    EpistemicStatus)
        from fractal_causal_engine.ft_expander import FractalExpander
        parent = ClassifiedItem(
            id="p1", quote="la voce cede su una parola",
            predicate=PredicateType.PROCESS_DESCRIPTION, nature=Nature.CAUSE,
            scale="organismo", epistemic_status=EpistemicStatus.TEXT_OBSERVED)
        FractalExpander(client, llm_calls_dir=out / "llm_calls",
                        telemetry_path=out / "telemetry.jsonl").expand(
            parent, original_text=MANIF, trace=[])
        l5 = [p.name for p in (out / "llm_calls").glob("*L5_FractalExpander*")]
        check("chiamate L5_FractalExpander tracciate (expander diretto)",
              len(l5) > 0, f"l5={l5}")


# ---------------------------------------------------------------------------
print("\n[3] I-3 — 04b_ventaglio.json e provenienza attraverso must_reject")

# 3a: must_reject propaga parent_id/confidence sia nei tenuti sia nei rigettati
v = P.Ventaglio(candidati=[
    P.Candidato(testo="frequenze di training sul lessico del lutto",
                nature="cause", scale="dominio",
                epistemic="domain_knowledge",
                parent_id="L3A1_DomainKnowledge", confidence=0.0),
    P.Candidato(testo="il nucleo instabile rilascia energia",
                nature="cause", scale="atomico",
                epistemic="causal_model",
                parent_id="it_004", confidence=0.72),
], trust="alta", trust_motivo="almeno un nesso cross-scale validato")
f = L.must_reject(v)
check("tenuto con provenienza (parent_id observer)",
      any(t.get("parent_id") == "L3A1_DomainKnowledge" for t in f.tenuti),
      f"tenuti={f.tenuti}")
check("rigettato con provenienza (parent_id espansione + confidence)",
      any(r.parent_id == "it_004" and r.confidence == 0.72 for r in f.rigettati))

# 3b: esegui_loop offline (client=None → Fractal non gira): 04b esiste comunque
#     e dichiara il ventaglio vuoto (vuoto dichiarato, non omesso).
def _elicitor_lp(sonda, system):
    m = ("L'origine del decadimento risiede nella instabilità del nucleo.\n"
         "Il decadimento è un meccanismo di rilascio.")
    lp = _frase_tokens(["L'origine", "del", "decadimento", "risiede"], 0.92, ".\n")
    lp += _frase_tokens(["Il", "decadimento", "è", "un", "meccanismo"], 0.45, ".")
    return m, lp


def _reader_mock(input_composto, system):
    return ("**6 · Massa all'inatteso**\nmassa = 0.25\n\n"
            "**9 · Nota di auto-deformazione**\nauto-deformazione: presente\n")


with tempfile.TemporaryDirectory() as td:
    out = pathlib.Path(td) / "storico" / "loopB_test"
    L.esegui_loop(
        "sonda di prova?", out_dir=str(out),
        nucleo_path=str(SPECCHIO / "specchio_del_modello_nucleo.md"),
        contratto_path=str(SPECCHIO / "specchio_di_coscienza_contratto_di_output.md"),
        client=None, reader=_reader_mock, elicitor_lp=_elicitor_lp,
        modalita="auto")
    vb = out / "04b_ventaglio.json"
    check("04b_ventaglio.json scritto nel loop completo", vb.exists())
    if vb.exists():
        d = json.loads(vb.read_text(encoding="utf-8"))
        check("04b: chiavi trust/trust_motivo/n_candidati/candidati",
              {"trust", "trust_motivo", "n_candidati", "candidati"} <= set(d),
              f"chiavi={sorted(d)}")
        check("04b: ventaglio vuoto DICHIARATO (client=None)",
              d["n_candidati"] == 0 and d["trust"] == "bassa")
    mrj = json.loads((out / "06_must_reject.json").read_text(encoding="utf-8"))
    filtrato = mrj.get("ventaglio_filtrato", {})
    check("06: schema retro-compatibile (tenuti/rigettati presenti)",
          {"tenuti", "rigettati"} <= set(filtrato))

    # -----------------------------------------------------------------------
    print("\n[4] I-4 — leggi_grezzo: pattern llm_calls sicuro")
    sys.path.insert(0, str(HERE / "web"))
    import storico as S
    S.STORICO_DIR = out.parent

    (out / "trace" / "llm_calls").mkdir(parents=True, exist_ok=True)
    (out / "trace" / "llm_calls" / "0000_Elicitazione.json").write_text(
        '{"call_id": "0000"}', encoding="utf-8")
    (out / "trace" / "ft_analysis.json").write_text(
        '{"items": []}', encoding="utf-8")

    check("llm_call raggiungibile via pattern",
          S.leggi_grezzo("loopB_test", "trace/llm_calls/0000_Elicitazione.json")
          is not None)
    check("ft_analysis raggiungibile via whitelist",
          S.leggi_grezzo("loopB_test", "trace/ft_analysis.json") is not None)
    check("04b raggiungibile via whitelist",
          S.leggi_grezzo("loopB_test", "04b_ventaglio.json") is not None)
    check("traversal respinto (../)",
          S.leggi_grezzo("loopB_test",
                         "trace/llm_calls/../../02_corpo.json") is None)
    check("nome fuori whitelist respinto",
          S.leggi_grezzo("loopB_test", "trace/llm_calls/evil.py") is None)

    rec = S.carica("loopB_test")
    check("carica(): ventaglio nel record", rec.get("ventaglio") is not None)
    check("carica(): fractal_analisi nel record",
          rec.get("fractal_analisi") is not None)
    check("carica(): indice llm_calls nel record",
          "0000_Elicitazione.json" in (rec.get("llm_calls") or []),
          f"llm_calls={rec.get('llm_calls')}")
    check("carica(): fractal_mappa lista (vuota con items=[])",
          rec.get("fractal_mappa") == [])


# ---------------------------------------------------------------------------
falliti = [n for n, c in esiti if not c]
print(f"\n{'='*60}\n{len(esiti) - len(falliti)}/{len(esiti)} verifiche passate")
if falliti:
    print("FALLITE:", *[f"\n  - {n}" for n in falliti])
    sys.exit(1)
print("Integrazione Fractal → Strada B: verificata.")
