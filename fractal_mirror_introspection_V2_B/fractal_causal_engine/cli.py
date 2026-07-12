"""CLI Fractal Causal Engine V10.17.0.

Comandi della sessione esploratore V15 + il Regista (L7) di V10.17.0.

Comandi disponibili:
  session-open        analizza un testo e apre una sessione esploratore
  session-list        elenca gli item correnti (numerati)
  session-expand      espande un item in 4 figli (FractalExpander)
  session-bridge      costruisce un bridge tra due item su gap_scale
  session-revalidate  rilancia L3.B sulle ipotesi 'uncertain'
  session-magistrale  genera la relazione magistrale finale
  session-auto        one-shot: open + expand + bridge + revalidate + magistrale
  session-observe     come session-auto, ma con il Regista (L7) che osserva
                      l'Attore fase per fase e ne corregge la traiettoria
  book-analyze        analizza un testo lungo (libro): segmentazione,
                      esecuzione resumabile per segmento, riduzione gerarchica
  theme-analyze       lettura tematica (quattro lenti) per testi non argomentativi
  theme-book-analyze  lettura tematica di un libro intero (segmentato, resumabile)
  session-shell       REPL interattiva (con indici corti [N] al posto di ID)

Tutti i comandi accettano --mock per smoke test offline; per il modello
reale, usare --backend (ollama|llamacpp) + --model + --base-url o
--llamacpp-url come prima.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm import LLMClient, LLMConfig
from .text import load_text_file


# =============================================================================
# Argparse
# =============================================================================


def _add_session_io(p: argparse.ArgumentParser) -> None:
    """Flag comuni a tutti i comandi session-*."""
    p.add_argument("--out", "-o", required=True,
                   help="Cartella sessione (contiene session.json e gli output)")
    p.add_argument("--backend", choices=["ollama", "llamacpp", "groq"], default="ollama",
                   help="Backend LLM (default: ollama; 'groq' = API OpenAI-compatible)")
    p.add_argument("--model", default="gemma3:4b",
                   help="Nome del modello (es. gemma3:4b, Hermes-3-Llama-3.1-8B.Q4_K_M.gguf, "
                        "llama-3.3-70b-versatile per groq)")
    p.add_argument("--base-url", default="http://localhost:11434",
                   help="URL base ollama (default: http://localhost:11434)")
    p.add_argument("--llamacpp-url", default="http://127.0.0.1:8080",
                   help="URL llama.cpp server (default: http://127.0.0.1:8080)")
    p.add_argument("--groq-url", default="https://api.groq.com/openai/v1",
                   help="URL base API Groq (default: https://api.groq.com/openai/v1)")
    p.add_argument("--groq-api-key", default="",
                   help="API key Groq. Se omessa, viene letta dalla variabile "
                        "d'ambiente GROQ_API_KEY")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout per ogni chiamata LLM in secondi (default: 600)")
    p.add_argument("--num-predict", type=int, default=900)
    p.add_argument("--num-ctx", type=int, default=None)
    p.add_argument("--num-gpu", type=int, default=100)
    p.add_argument("--keep-alive", default="15m")
    p.add_argument("--mock", action="store_true",
                   help="Usa risposte LLM mock (per smoke test, non per uso reale)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-causal",
        description="Fractal Causal Engine V10.16.0 -- esplorazione causale a piu' scale",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # session-open
    p = sub.add_parser("session-open",
                       help="Apre una sessione: prima analisi V14 del testo + salvataggio session.json")
    _add_session_io(p)
    p.add_argument("--input", "-i", required=True, help="File di input testuale")
    p.add_argument("--max-cross-scale", type=int, default=8)

    # session-list
    p = sub.add_parser("session-list", help="Elenca gli item della sessione (numerati per indice)")
    _add_session_io(p)

    # session-expand
    p = sub.add_parser("session-expand",
                       help="Espande un item in 4 figli frattali (cause, propagazione, meccanismo, bridge)")
    _add_session_io(p)
    p.add_argument("--item-id", required=True,
                   help="ID, prefisso unico (es. itm_28) o indice [N] (es. 3)")

    # session-bridge
    p = sub.add_parser("session-bridge",
                       help="Costruisce un bridge esplicito tra due item su una gap_scale intermedia")
    _add_session_io(p)
    p.add_argument("--source-id", required=True, help="ID, prefisso o indice item sorgente")
    p.add_argument("--target-id", required=True, help="ID, prefisso o indice item destinazione")
    p.add_argument("--gap-scale", required=True,
                   help="Scala intermedia (una delle 9 canoniche) dove vivra' il bridge")

    # session-revalidate
    p = sub.add_parser("session-revalidate",
                       help="Rilancia L3.B sulle ipotesi cross-scale 'uncertain'")
    _add_session_io(p)
    p.add_argument("--include-non-uncertain", action="store_true",
                   help="Rivaluta anche genuine/spurious (default: solo uncertain)")

    # session-magistrale
    p = sub.add_parser("session-magistrale",
                       help="Genera la relazione magistrale finale (markdown leggibile)")
    _add_session_io(p)

    # session-auto
    p = sub.add_parser("session-auto",
                       help="One-shot: open + espande top-N + bridge sui gap + revalidate + magistrale")
    _add_session_io(p)
    p.add_argument("--input", "-i", required=True, help="File di input testuale")
    p.add_argument("--max-cross-scale", type=int, default=8)
    p.add_argument("--expand-top-n", type=int, default=3,
                   help="Quanti item osservati espandere al livello 1 (default: 3)")
    p.add_argument("--expand-depth", type=int, default=1,
                   help="Profondita' dell'espansione: 1=solo osservati, 2=anche i figli, "
                        "3=anche i nipoti... (default: 1)")
    p.add_argument("--expand-children-per-level", type=int, default=2,
                   help="Quanti figli ri-espandere a ogni livello oltre il primo (default: 2)")
    p.add_argument("--max-bridges", type=int, default=3,
                   help="Quanti bridge costruire (default: 3)")
    p.add_argument("--no-bridges", action="store_true", help="Salta la costruzione dei bridge")
    p.add_argument("--no-revalidate", action="store_true", help="Salta la revalidate L3.B")
    p.add_argument("--no-magistrale", action="store_true", help="Salta la relazione magistrale")

    # session-observe (V10.17.0): one-shot CON il Regista (L7) attivo
    p = sub.add_parser(
        "session-observe",
        help="One-shot come session-auto, ma con il Regista (L7) che osserva "
             "l'Attore fase per fase e corregge la traiettoria",
    )
    _add_session_io(p)
    p.add_argument("--input", "-i", required=True, help="File di input testuale")
    p.add_argument("--max-cross-scale", type=int, default=8)
    p.add_argument("--expand-top-n", type=int, default=3,
                   help="Quanti item osservati espandere al livello 1 (default: 3)")
    p.add_argument("--expand-depth", type=int, default=1,
                   help="Profondita' dell'espansione (default: 1)")
    p.add_argument("--expand-children-per-level", type=int, default=2,
                   help="Quanti figli ri-espandere a ogni livello oltre il primo (default: 2)")
    p.add_argument("--max-bridges", type=int, default=3,
                   help="Quanti bridge costruire (default: 3)")
    p.add_argument("--no-bridges", action="store_true", help="Salta la costruzione dei bridge")
    p.add_argument("--no-revalidate", action="store_true", help="Salta la revalidate L3.B")
    p.add_argument("--no-magistrale", action="store_true", help="Salta la relazione magistrale")
    p.add_argument("--silence-band", type=float, default=1.5,
                   help="Soglia dell'integrale di deriva oltre cui il costo del "
                        "silenzio scatta (default: 1.5)")
    p.add_argument("--divergence-threshold", type=float, default=0.34,
                   help="Frazione di cross-scale spurious oltre cui la divergenza "
                        "di scala scatta (default: 0.34)")
    p.add_argument("--control-budget", type=int, default=6,
                   help="Numero massimo di atti di flusso del Regista (skip/"
                        "repeat/goto/halt) prima che perda il potere di deviare "
                        "e possa solo osservare. Rete anti-loop (default: 6)")
    p.add_argument("--no-narrate", action="store_true",
                   help="Non far raccontare al Regista l'auto-osservazione via LLM "
                        "(solo summary deterministico)")

    # book-analyze (V10.18.0): analisi di un testo lungo (libro) con
    # segmentazione, esecuzione resumabile e riduzione gerarchica.
    p = sub.add_parser(
        "book-analyze",
        help="Analizza un testo lungo (libro): lo segmenta, analizza ogni "
             "segmento in modo resumabile, e ne produce una lettura unitaria",
    )
    _add_session_io(p)
    p.add_argument("--input", "-i", required=True, help="File di input testuale (il libro)")
    p.add_argument("--book-id", default="book",
                   help="Identificatore del libro, usato nel manifest (default: book)")
    p.add_argument("--max-cross-scale", type=int, default=8)
    p.add_argument("--overlap-ratio", type=float, default=0.15,
                   help="Frazione del budget ripetuta come overlap fra segmenti "
                        "consecutivi (default: 0.15; 0 disattiva l'overlap)")
    p.add_argument("--per-segment-depth", choices=["base", "full"], default="base",
                   help="'base' = solo pipeline L0->L4 per segmento (consigliato); "
                        "'full' = anche expand/bridge/magistrale (default: base)")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Tentativi per segmento su errore transitorio, con "
                        "backoff esponenziale (default: 3)")
    p.add_argument("--halt-on-failure", action="store_true",
                   help="Ferma il job al primo segmento fallito invece di "
                        "proseguire (default: skip-and-continue + dead-letter)")
    p.add_argument("--retry-failed", action="store_true",
                   help="In un resume, ri-tenta i segmenti finiti in dead-letter "
                        "in un run precedente (usare dopo aver risolto la causa)")
    p.add_argument("--no-reduce", action="store_true",
                   help="Esegui solo la segmentazione e l'analisi, salta la "
                        "riduzione gerarchica finale")

    # theme-analyze (V10.19.0): lettura tematica -- quattro lenti invece
    # della griglia causale. Per testi non argomentativi (diari, dialoghi,
    # testi simbolici/spirituali).
    p = sub.add_parser(
        "theme-analyze",
        help="Lettura tematica di un testo: quattro lenti (simbolica, "
             "strutturale, relazionale, esperienziale) invece dell'analisi "
             "causale. Per testi non argomentativi.",
    )
    _add_session_io(p)
    p.add_argument("--input", "-i", required=True, help="File di input testuale")

    # theme-book-analyze (V10.19.2): lettura tematica di un LIBRO intero --
    # segmentazione + quattro lenti per segmento + riduzione tematica.
    # Resumabile, come book-analyze ma con le lenti invece della griglia causale.
    p = sub.add_parser(
        "theme-book-analyze",
        help="Lettura tematica di un libro intero: segmenta, applica le "
             "quattro lenti a ogni segmento (resumabile), riduce in una "
             "lettura tematica dell'opera.",
    )
    _add_session_io(p)
    p.add_argument("--input", "-i", required=True, help="File di input (il libro)")
    p.add_argument("--book-id", default="book",
                   help="Identificatore del libro, usato nel manifest (default: book)")
    p.add_argument("--overlap-ratio", type=float, default=0.15,
                   help="Overlap fra segmenti consecutivi (default: 0.15)")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Tentativi per segmento su errore transitorio (default: 3)")
    p.add_argument("--halt-on-failure", action="store_true",
                   help="Ferma il job al primo segmento fallito")
    p.add_argument("--retry-failed", action="store_true",
                   help="In un resume, ri-tenta i segmenti in dead-letter")
    p.add_argument("--no-reduce", action="store_true",
                   help="Esegui solo l'analisi per segmento, salta la riduzione")

    p = sub.add_parser("session-shell", help="REPL interattiva con indici corti [N]")
    _add_session_io(p)
    p.add_argument("--input", "-i", default=None,
                   help="Se passato, apre nuova sessione. Altrimenti riprende session.json esistente.")
    p.add_argument("--max-cross-scale", type=int, default=8)

    return parser


# =============================================================================
# Costruzione client LLM
# =============================================================================


def _make_client(args: argparse.Namespace) -> LLMClient:
    # la API key Groq: dal flag, oppure dalla variabile d'ambiente GROQ_API_KEY
    import os
    groq_key = getattr(args, "groq_api_key", "") or os.environ.get("GROQ_API_KEY", "")
    cfg = LLMConfig(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        llamacpp_url=args.llamacpp_url,
        groq_url=getattr(args, "groq_url", "https://api.groq.com/openai/v1"),
        groq_api_key=groq_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_seconds=args.timeout,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        num_gpu=args.num_gpu,
        keep_alive=args.keep_alive,
        mock=args.mock,
    )
    return LLMClient(cfg)


# =============================================================================
# Print helpers (timestamp e timer)
# =============================================================================


def _ts() -> str:
    """Timestamp HH:MM:SS per stampa a console."""
    return datetime.now().strftime("%H:%M:%S")


def _say(msg: str) -> None:
    """Stampa con timestamp."""
    print(f"[{_ts()}] {msg}")


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed in mm:ss o hh:mm:ss."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _print_items_table(rows: list) -> None:
    if not rows:
        print("  (nessun item)")
        return
    print(f"  {'#':>3}  {'id':<14}  {'scale':<12}  {'nat':<14}  {'pred':<22}  {'epist':<24}  {'orig':<14}  text")
    print("  " + "-" * 140)
    for i, r in enumerate(rows, start=1):
        print(
            f"  [{i:>2}] {r['id'][:14]:<14}  {r['scale'][:12]:<12}  {r['nature'][:14]:<14}  "
            f"{r['predicate'][:22]:<22}  {r['epistemic_status'][:24]:<24}  "
            f"{r['origin'][:14]:<14}  {r['text'][:60]}"
        )


# =============================================================================
# Runners
# =============================================================================


def run_session_open(args: argparse.Namespace) -> None:
    from .ft_session import ExplorerSession
    client = _make_client(args)
    inputs = load_text_file(args.input)
    if not inputs:
        raise SystemExit("Input vuoto.")
    raw = inputs[0]
    out = Path(args.out)
    _say(f"session-open: input='{args.input}' out='{out}'")
    _say(f"backend={args.backend} model={args.model} mock={args.mock}")

    t0 = time.perf_counter()
    _say("pipeline V14: L1 classifier -> L2 locked -> L3A unlocked -> L3B crossscale -> L4 orchestrator...")
    sess = ExplorerSession.analyze(
        client, out, raw.text,
        source_input_id=raw.id,
        max_cross_scale=args.max_cross_scale,
    )
    elapsed = time.perf_counter() - t0
    _say(f"pipeline V14 completata in {_fmt_elapsed(elapsed)}")
    _say(f"items iniziali: {len(sess.ft.items)}  locked_scales: {len(sess.ft.locked_reports)}  "
         f"cross_scale: {len(sess.ft.cross_scale)}")
    _print_items_table(sess.list_items())
    _say(f"session.json: {out / 'session.json'}")
    _say(f"final_report.md: {out / 'final_report.md'}")


def run_session_list(args: argparse.Namespace) -> None:
    from .ft_session import ExplorerSession
    client = _make_client(args)
    sess = ExplorerSession.load(Path(args.out), client)
    _say(f"session-list: out='{args.out}'  items={len(sess.ft.items)}")
    _print_items_table(sess.list_items())
    (Path(args.out) / "session_list.md").write_text(sess.render_list_md(), encoding="utf-8")
    _say(f"session_list.md: {Path(args.out) / 'session_list.md'}")


def run_session_expand(args: argparse.Namespace) -> None:
    from .ft_session import ExplorerSession
    client = _make_client(args)
    sess = ExplorerSession.load(Path(args.out), client)
    try:
        item = sess.resolve_item_ref(args.item_id)
    except KeyError as exc:
        _say(f"ERRORE: {exc}")
        raise SystemExit(2)
    _say(f"session-expand: item_id={item.id} ({item.scale}/{item.nature.value})")
    t0 = time.perf_counter()
    record = sess.expand(item.id)
    elapsed = time.perf_counter() - t0
    _say(f"expand completato in {_fmt_elapsed(elapsed)}")
    if record.degraded and not record.children:
        _say(f"DEGRADED: {record.notes}")
        return
    _say(f"figli generati: {len(record.children)}")
    for c in record.children:
        text = c.item.metadata.get("generated_text", "") if c.item.metadata else ""
        print(f"  - [{c.direction.value}] ({c.item.scale}/{c.item.nature.value}) "
              f"conf={c.confidence:.2f} :: {text}")
    _say(f"same_scale_links_added: {len(record.same_scale_links_added)}")
    _say(f"cross_scale_added: {len(record.cross_scale_added)} (tutti 'uncertain' fino a revalidate)")


def run_session_bridge(args: argparse.Namespace) -> None:
    from .ft_session import ExplorerSession
    client = _make_client(args)
    sess = ExplorerSession.load(Path(args.out), client)
    try:
        source = sess.resolve_item_ref(args.source_id)
        target = sess.resolve_item_ref(args.target_id)
    except KeyError as exc:
        _say(f"ERRORE: {exc}")
        raise SystemExit(2)
    _say(f"session-bridge: {source.id}({source.scale}) -> {target.id}({target.scale}) @ {args.gap_scale}")
    t0 = time.perf_counter()
    record = sess.bridge(source.id, target.id, args.gap_scale)
    elapsed = time.perf_counter() - t0
    _say(f"bridge completato in {_fmt_elapsed(elapsed)}")
    if record.degraded:
        _say(f"DEGRADED: {record.mechanism_reasoning}")
        return
    text = record.bridge_item.metadata.get("generated_text", "")
    _say(f"bridge generato: {record.bridge_item.id} ({record.bridge_item.scale})")
    print(f"    text: {text}")
    print(f"    reasoning: {record.mechanism_reasoning[:240]}")
    _say(f"cross_scale_added: {len(record.cross_scale_added)} (entrambe 'uncertain')")


def run_session_revalidate(args: argparse.Namespace) -> None:
    from .ft_session import ExplorerSession
    client = _make_client(args)
    sess = ExplorerSession.load(Path(args.out), client)
    only_uncertain = not args.include_non_uncertain
    _say(f"session-revalidate: only_uncertain={only_uncertain}")
    t0 = time.perf_counter()
    stats = sess.revalidate_cross(only_uncertain=only_uncertain)
    elapsed = time.perf_counter() - t0
    _say(f"revalidate completata in {_fmt_elapsed(elapsed)}")
    _say(f"stats: {stats}")


def run_session_magistrale(args: argparse.Namespace) -> None:
    from .ft_session import ExplorerSession
    client = _make_client(args)
    sess = ExplorerSession.load(Path(args.out), client)
    _say("session-magistrale: generazione relazione...")
    t0 = time.perf_counter()
    report = sess.magistrale()
    elapsed = time.perf_counter() - t0
    _say(f"magistrale completata in {_fmt_elapsed(elapsed)}")
    if report.degraded:
        _say("DEGRADED: generazione fallita")
    _say(f"magistrale_report.md: {Path(args.out) / 'magistrale_report.md'}")
    _say(f"final_report.md aggiornato: {Path(args.out) / 'final_report.md'}")
    print(f"  sintesi: {report.sintesi_magistrale[:280]}")


def run_session_auto(args: argparse.Namespace) -> None:
    """One-shot: analyze + auto_explore. Niente ID da passare a mano.

    Suddiviso in 5 fasi cronometrate separatamente, per dare un feedback
    chiaro su quale fase sta girando e quanto sta impiegando.
    """
    from .ft_session import ExplorerSession
    client = _make_client(args)
    inputs = load_text_file(args.input)
    if not inputs:
        raise SystemExit("Input vuoto.")
    raw = inputs[0]
    out = Path(args.out)
    overall_start = time.perf_counter()

    _say(f"session-auto: input='{args.input}' out='{out}'")
    _say(f"backend={args.backend} model={args.model} mock={args.mock}")

    # --- Fase 1: analyze ---
    _say("[1/5] pipeline V14: L1 -> L2 -> L3A -> L3B -> L4 ...")
    t0 = time.perf_counter()
    sess = ExplorerSession.analyze(
        client, out, raw.text,
        source_input_id=raw.id,
        max_cross_scale=args.max_cross_scale,
    )
    _say(f"[1/5] V14 completata in {_fmt_elapsed(time.perf_counter() - t0)}, "
         f"items={len(sess.ft.items)} locked_scales={len(sess.ft.locked_reports)} "
         f"cross_scale={len(sess.ft.cross_scale)}")
    _print_items_table(sess.list_items())

    stats: dict[str, Any] = {
        "expanded": 0, "expand_degraded": 0,
        "bridges_built": 0, "bridge_degraded": 0,
        "revalidate": None, "magistrale": False,
    }

    def _phase_progress(msg: str) -> None:
        _say(f"     {msg}")

    # --- Fase 2: expand ---
    depth_note = f" (profondita' {args.expand_depth})" if args.expand_depth > 1 else ""
    _say(f"[2/5] espansione frattale{depth_note} ...")
    t0 = time.perf_counter()
    expand_stats = sess.auto_explore(
        expand_top_n=args.expand_top_n,
        expand_depth=args.expand_depth,
        expand_children_per_level=args.expand_children_per_level,
        build_bridges=False, do_revalidate=False, do_magistrale=False,
        progress=_phase_progress,
    )
    stats["expanded"] = expand_stats["expanded"]
    stats["expand_degraded"] = expand_stats["expand_degraded"]
    by_level = expand_stats.get("expand_by_level", {})
    level_note = f" per livello={by_level}" if by_level else ""
    _say(f"[2/5] espansione completata in {_fmt_elapsed(time.perf_counter() - t0)}: "
         f"expanded={stats['expanded']}, degraded={stats['expand_degraded']}{level_note}")

    # --- Fase 3: bridge ---
    if not args.no_bridges:
        _say("[3/5] costruzione bridge sui gap cross-scale ...")
        t0 = time.perf_counter()
        bridge_stats = sess.auto_explore(
            expand_top_n=0, build_bridges=True, max_bridges=args.max_bridges,
            do_revalidate=False, do_magistrale=False,
            progress=_phase_progress,
        )
        stats["bridges_built"] = bridge_stats["bridges_built"]
        stats["bridge_degraded"] = bridge_stats["bridge_degraded"]
        _say(f"[3/5] bridge completati in {_fmt_elapsed(time.perf_counter() - t0)}: "
             f"built={stats['bridges_built']}, degraded={stats['bridge_degraded']}")
    else:
        _say("[3/5] bridge: SKIP (--no-bridges)")

    # --- Fase 4: revalidate ---
    if not args.no_revalidate:
        _say("[4/5] revalidate cross-scale ...")
        t0 = time.perf_counter()
        rev_stats = sess.auto_explore(
            expand_top_n=0, build_bridges=False,
            do_revalidate=True, do_magistrale=False,
            progress=_phase_progress,
        )
        stats["revalidate"] = rev_stats["revalidate"]
        _say(f"[4/5] revalidate completata in {_fmt_elapsed(time.perf_counter() - t0)}: "
             f"{stats['revalidate']}")
    else:
        _say("[4/5] revalidate: SKIP (--no-revalidate)")

    # --- Fase 5: magistrale ---
    if not args.no_magistrale:
        _say("[5/5] generazione relazione magistrale ...")
        t0 = time.perf_counter()
        mag_stats = sess.auto_explore(
            expand_top_n=0, build_bridges=False,
            do_revalidate=False, do_magistrale=True,
            progress=_phase_progress,
        )
        stats["magistrale"] = mag_stats["magistrale"]
        _say(f"[5/5] magistrale completata in {_fmt_elapsed(time.perf_counter() - t0)}")
    else:
        _say("[5/5] magistrale: SKIP (--no-magistrale)")

    total_elapsed = time.perf_counter() - overall_start
    _say("=" * 70)
    _say(f"TOTALE session-auto completato in {_fmt_elapsed(total_elapsed)}")
    _say(f"  expanded:      {stats['expanded']} (degraded: {stats['expand_degraded']})")
    _say(f"  bridges_built: {stats['bridges_built']} (degraded: {stats['bridge_degraded']})")
    if stats["revalidate"]:
        _say(f"  revalidate:    {stats['revalidate']}")
    _say(f"  magistrale:    {stats['magistrale']}")
    _say("Output:")
    _say(f"  - session.json          {out / 'session.json'}")
    _say(f"  - final_report.md       {out / 'final_report.md'}")
    if not args.no_magistrale:
        _say(f"  - magistrale_report.md  {out / 'magistrale_report.md'}")


def run_session_observe(args: argparse.Namespace) -> None:
    """One-shot con il Regista (L7) attivo.

    Come session-auto, ma l'esplorazione non e' una catena cieca: dopo ogni
    fase il Regista osserva l'Attore (le 3 misure: divergenza di scala, costo
    del silenzio, irreversibilita') e, se tutte e tre scattano insieme,
    corregge i parametri della fase successiva. Al termine scrive
    director_report.md con il racconto dell'auto-osservazione.
    """
    from .ft_session import ExplorerSession
    client = _make_client(args)
    inputs = load_text_file(args.input)
    if not inputs:
        raise SystemExit("Input vuoto.")
    raw = inputs[0]
    out = Path(args.out)
    overall_start = time.perf_counter()

    _say(f"session-observe: input='{args.input}' out='{out}'")
    _say(f"backend={args.backend} model={args.model} mock={args.mock}")
    _say(f"soglie Regista: silence_band={args.silence_band} "
         f"divergence_threshold={args.divergence_threshold}")

    # --- Fase 1: analyze (l'Attore parte) ---
    _say("[1/2] pipeline V14: L1 -> L2 -> L3A -> L3B -> L4 ...")
    t0 = time.perf_counter()
    sess = ExplorerSession.analyze(
        client, out, raw.text,
        source_input_id=raw.id,
        max_cross_scale=args.max_cross_scale,
    )
    _say(f"[1/2] V14 completata in {_fmt_elapsed(time.perf_counter() - t0)}, "
         f"items={len(sess.ft.items)} cross_scale={len(sess.ft.cross_scale)}")
    _print_items_table(sess.list_items())

    # --- Fase 2: auto_explore con il Regista ---
    _say("[2/2] esplorazione Attore+Regista (expand -> bridge -> revalidate "
         "-> magistrale, osservata) ...")
    t0 = time.perf_counter()
    stats = sess.observe_with_director(
        expand_top_n=args.expand_top_n,
        expand_depth=args.expand_depth,
        expand_children_per_level=args.expand_children_per_level,
        build_bridges=not args.no_bridges,
        max_bridges=args.max_bridges,
        do_revalidate=not args.no_revalidate,
        do_magistrale=not args.no_magistrale,
        silence_band=args.silence_band,
        divergence_threshold=args.divergence_threshold,
        control_budget=args.control_budget,
        narrate=not args.no_narrate,
        progress=lambda m: _say(f"     {m}"),
    )
    _say(f"[2/2] completata in {_fmt_elapsed(time.perf_counter() - t0)}")

    _say("=" * 70)
    _say(f"TOTALE session-observe completato in "
         f"{_fmt_elapsed(time.perf_counter() - overall_start)}")
    _say(f"  flusso eseguito:       "
         f"{' -> '.join(stats.get('director_executed_phases', []))}")
    _say(f"  expanded:              {stats.get('expanded')}")
    _say(f"  bridges_built:         {stats.get('bridges_built')}")
    _say(f"  revalidate:            {stats.get('revalidate')}")
    _say(f"  letture del Regista:   {stats.get('director_readings')}")
    _say(f"  atti di regia:         {stats.get('director_interventions')}")
    _say(f"  Attore fermato:        {stats.get('director_halted')}")
    _say("Output:")
    _say(f"  - session.json          {out / 'session.json'}")
    _say(f"  - final_report.md       {out / 'final_report.md'}")
    if not args.no_magistrale:
        _say(f"  - magistrale_report.md  {out / 'magistrale_report.md'}")
    _say(f"  - director_report.md    {out / 'director_report.md'}")


def run_book_analyze(args: argparse.Namespace) -> None:
    """Analisi di un testo lungo (libro): segmentazione, esecuzione
    resumabile per segmento, riduzione gerarchica finale.

    Resumabile per costruzione: se nella cartella --out esiste gia' un
    book_manifest.json, il job riprende da li' invece di ricominciare.
    """
    from .ft_book_runner import BookRunner
    from .ft_reducer import BookReducer

    client = _make_client(args)
    inputs = load_text_file(args.input)
    if not inputs:
        raise SystemExit("Input vuoto.")
    raw = inputs[0]
    out = Path(args.out)
    overall_start = time.perf_counter()

    # num_ctx: serve sia al client LLM sia al segmenter. Se non passato,
    # default conservativo 8192 (finestra nativa di un modello 8B).
    num_ctx = args.num_ctx if args.num_ctx else 8192

    _say(f"book-analyze: input='{args.input}' out='{out}' book-id='{args.book_id}'")
    _say(f"backend={args.backend} model={args.model} mock={args.mock}")
    _say(f"num_ctx={num_ctx} overlap={args.overlap_ratio} "
         f"depth={args.per_segment_depth} max_retries={args.max_retries}")
    if args.halt_on_failure:
        _say("politica errori: HALT al primo segmento fallito")
    else:
        _say("politica errori: skip-and-continue + dead-letter")

    # --- analisi resumabile per segmento ------------------------------------
    runner = BookRunner(
        client, out,
        num_ctx=num_ctx,
        overlap_ratio=args.overlap_ratio,
        per_segment_depth=args.per_segment_depth,
        max_retries=args.max_retries,
        halt_on_failure=args.halt_on_failure,
        max_cross_scale=args.max_cross_scale,
    )
    t0 = time.perf_counter()
    manifest = runner.run(
        raw.text,
        book_id=args.book_id,
        source_input_id=raw.id,
        retry_dead_letter=args.retry_failed,
        progress=lambda m: _say(f"  {m}"),
    )
    counts = manifest.counts()
    _say(f"[analisi] completata in {_fmt_elapsed(time.perf_counter() - t0)}: "
         f"done={counts['done']} failed={counts['failed']} "
         f"pending={counts['pending']}")

    # --- riduzione gerarchica -----------------------------------------------
    reduction = None
    if args.no_reduce:
        _say("[riduzione] saltata (--no-reduce)")
    elif counts["done"] == 0:
        _say("[riduzione] nessun segmento completato: riduzione saltata")
    else:
        t0 = time.perf_counter()
        reducer = BookReducer(client, out)
        reduction = reducer.reduce(manifest, progress=lambda m: _say(f"  {m}"))
        _say(f"[riduzione] completata in {_fmt_elapsed(time.perf_counter() - t0)}")

    _say("=" * 70)
    _say(f"TOTALE book-analyze completato in "
         f"{_fmt_elapsed(time.perf_counter() - overall_start)}")
    _say(f"  segmenti:    {len(manifest.segments)} "
         f"(done={counts['done']} failed={counts['failed']} "
         f"pending={counts['pending']})")
    if reduction is not None:
        _say(f"  capitoli:    {len(reduction.chapters)} sintetizzati")
    _say("Output:")
    _say(f"  - book_manifest.json    {out / 'book_manifest.json'}")
    _say(f"  - segments/             {out / 'segments'} "
         f"(una sotto-cartella per segmento)")
    if reduction is not None:
        _say(f"  - book_reduction.md     {out / 'book_reduction.md'}")
    if counts["failed"]:
        failed_ids = [r.id for r in manifest.segments if r.status == "failed"]
        _say(f"ATTENZIONE: {len(failed_ids)} segmenti in dead-letter: "
             f"{', '.join(failed_ids)}")
        _say("  Risolvi la causa e rilancia con --retry-failed per ritentarli.")
    if counts["pending"]:
        _say(f"NOTA: {counts['pending']} segmenti ancora 'pending'. "
             f"Rilancia lo stesso comando per riprendere.")


def run_theme_analyze(args: argparse.Namespace) -> None:
    """Lettura tematica di un testo: quattro lenti invece della griglia
    causale. Per testi non argomentativi -- diari, dialoghi, testi simbolici.
    """
    from .ft_thematic import ThematicReader, render_thematic_md

    client = _make_client(args)
    inputs = load_text_file(args.input)
    if not inputs:
        raise SystemExit("Input vuoto.")
    raw = inputs[0]
    out = Path(args.out)
    started = time.perf_counter()

    _say(f"theme-analyze: input='{args.input}' out='{out}'")
    _say(f"backend={args.backend} model={args.model} mock={args.mock}")
    _say("lettura tematica: lenti simbolica, strutturale, relazionale, esperienziale")

    reader = ThematicReader(client, out)
    reading = reader.read(raw.text, progress=lambda m: _say(f"  {m}"))

    md = render_thematic_md(reading, original_text=raw.text)
    (out / "thematic_report.md").write_text(md, encoding="utf-8")

    _say("=" * 70)
    _say(f"TOTALE theme-analyze completato in {_fmt_elapsed(time.perf_counter() - started)}")
    _say(f"  osservazioni: {len(reading.observations)}")
    _say(f"  motivi:       {len(reading.motifs)}")
    _say("Output:")
    _say(f"  - thematic_report.md   {out / 'thematic_report.md'}")
    if reading.notes:
        _say(f"  note: {len(reading.notes)} (vedi report)")


def run_theme_book_analyze(args: argparse.Namespace) -> None:
    """Lettura tematica di un libro intero: segmentazione + quattro lenti per
    segmento (resumabile) + riduzione tematica dell'opera.
    """
    from .ft_thematic_book import ThematicBookRunner, ThematicBookReducer

    client = _make_client(args)
    inputs = load_text_file(args.input)
    if not inputs:
        raise SystemExit("Input vuoto.")
    raw = inputs[0]
    out = Path(args.out)
    num_ctx = args.num_ctx if args.num_ctx else 8192
    overall_start = time.perf_counter()

    _say(f"theme-book-analyze: input='{args.input}' out='{out}' book-id='{args.book_id}'")
    _say(f"backend={args.backend} model={args.model} mock={args.mock}")
    _say(f"num_ctx={num_ctx} overlap={args.overlap_ratio} max_retries={args.max_retries}")
    _say("lettura tematica: lenti simbolica, strutturale, relazionale, esperienziale")

    # --- analisi resumabile per segmento ------------------------------------
    runner = ThematicBookRunner(
        client, out,
        num_ctx=num_ctx,
        overlap_ratio=args.overlap_ratio,
        max_retries=args.max_retries,
        halt_on_failure=args.halt_on_failure,
    )
    t0 = time.perf_counter()
    manifest = runner.run(
        raw.text,
        book_id=args.book_id,
        source_input_id=raw.id,
        retry_dead_letter=args.retry_failed,
        progress=lambda m: _say(f"  {m}"),
    )
    counts = manifest.counts()
    _say(f"[analisi] completata in {_fmt_elapsed(time.perf_counter() - t0)}: "
         f"done={counts['done']} failed={counts['failed']} pending={counts['pending']}")

    # --- riduzione tematica -------------------------------------------------
    reduction = None
    if args.no_reduce:
        _say("[riduzione] saltata (--no-reduce)")
    elif counts["done"] == 0:
        _say("[riduzione] nessun segmento completato: riduzione saltata")
    else:
        t0 = time.perf_counter()
        reducer = ThematicBookReducer(client, out)
        reduction = reducer.reduce(manifest, progress=lambda m: _say(f"  {m}"))
        _say(f"[riduzione] completata in {_fmt_elapsed(time.perf_counter() - t0)}")

    _say("=" * 70)
    _say(f"TOTALE theme-book-analyze completato in "
         f"{_fmt_elapsed(time.perf_counter() - overall_start)}")
    _say(f"  segmenti: {len(manifest.segments)} "
         f"(done={counts['done']} failed={counts['failed']} pending={counts['pending']})")
    if reduction is not None:
        _say(f"  osservazioni totali: {reduction.total_observations}")
    _say("Output:")
    _say(f"  - thematic_book_manifest.json   {out / 'thematic_book_manifest.json'}")
    _say(f"  - segments/                     {out / 'segments'}")
    if reduction is not None:
        _say(f"  - thematic_book_reduction.md    {out / 'thematic_book_reduction.md'}")
    if counts["failed"]:
        failed_ids = [r.id for r in manifest.segments if r.status == "failed"]
        _say(f"ATTENZIONE: {len(failed_ids)} segmenti in dead-letter: "
             f"{', '.join(failed_ids)}")
        _say("  Rilancia con --retry-failed per ritentarli.")
    if counts["pending"]:
        _say(f"NOTA: {counts['pending']} segmenti 'pending'. "
             f"Rilancia lo stesso comando per riprendere.")


SHELL_HELP = """\
Comandi (gli <ref> accettano indice [N], prefisso unico o ID completo):
  list                            elenca item correnti (numerati)
  expand <ref>                    espandi un item in 4 figli frattali
  bridge <ref_a> <ref_b> <scale>  costruisci un bridge cross-scale
  revalidate                      rilancia L3.B sulle 'uncertain'
  magistrale                      genera la relazione finale
  auto                            one-shot dentro la sessione corrente
  observe                         one-shot con il Regista (L7) attivo
  scales                          mostra le 9 scale canoniche
  help                            mostra questo aiuto
  quit                            esci
"""


def run_session_shell(args: argparse.Namespace) -> None:
    """REPL interattiva. Indici corti, niente ID lunghi."""
    from .ft_session import ExplorerSession
    from .ft_model import SCALES_CANONICAL
    client = _make_client(args)
    out_dir = Path(args.out)
    if args.input:
        _say(f"shell: apro nuova sessione in '{out_dir}' dal file '{args.input}'")
        inputs = load_text_file(args.input)
        if not inputs:
            raise SystemExit("Input vuoto.")
        raw = inputs[0]
        t0 = time.perf_counter()
        sess = ExplorerSession.analyze(
            client, out_dir, raw.text,
            source_input_id=raw.id,
            max_cross_scale=args.max_cross_scale,
        )
        _say(f"analisi V14 completata in {_fmt_elapsed(time.perf_counter() - t0)}")
    else:
        _say(f"shell: carico sessione esistente da '{out_dir}'")
        sess = ExplorerSession.load(out_dir, client)

    print(SHELL_HELP)
    _print_items_table(sess.list_items())

    while True:
        try:
            line = input("\nfractal> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd in ("quit", "exit", "q"):
                break
            if cmd in ("help", "?"):
                print(SHELL_HELP)
                continue
            if cmd == "scales":
                print("scale canoniche (top -> deep):")
                for i, s in enumerate(SCALES_CANONICAL):
                    print(f"  [{i}] {s}")
                continue
            if cmd == "list":
                _print_items_table(sess.list_items())
                continue

            t0 = time.perf_counter()  # comandi LLM
            if cmd == "expand":
                if len(parts) < 2:
                    print("uso: expand <ref>")
                    continue
                item = sess.resolve_item_ref(parts[1])
                _say(f"espando {item.id} ({item.scale}/{item.nature.value})")
                rec = sess.expand(item.id)
                if rec.degraded and not rec.children:
                    print(f"  DEGRADED: {rec.notes}")
                else:
                    for c in rec.children:
                        t = c.item.metadata.get("generated_text", "")
                        print(f"  + [{c.direction.value}] ({c.item.scale}/{c.item.nature.value}) "
                              f"conf={c.confidence:.2f} :: {t}")
            elif cmd == "bridge":
                if len(parts) < 4:
                    print("uso: bridge <ref_a> <ref_b> <gap_scale>")
                    continue
                a = sess.resolve_item_ref(parts[1])
                b = sess.resolve_item_ref(parts[2])
                gap = parts[3]
                if gap not in SCALES_CANONICAL:
                    print(f"  scala non canonica: {gap!r}. Usa 'scales'.")
                    continue
                _say(f"bridge {a.id}({a.scale}) -> {b.id}({b.scale}) @ {gap}")
                rec = sess.bridge(a.id, b.id, gap)
                if rec.degraded:
                    print(f"  DEGRADED: {rec.mechanism_reasoning}")
                else:
                    t = rec.bridge_item.metadata.get("generated_text", "")
                    print(f"  + bridge ({rec.bridge_item.scale}) :: {t}")
            elif cmd == "revalidate":
                stats = sess.revalidate_cross(only_uncertain=True)
                print(f"  stats: {stats}")
            elif cmd == "magistrale":
                report = sess.magistrale()
                print(f"  magistrale_report.md scritto in {out_dir / 'magistrale_report.md'}")
                if report.degraded:
                    print("  DEGRADED")
                else:
                    print(f"  sintesi: {report.sintesi_magistrale[:240]}")
            elif cmd == "auto":
                stats = sess.auto_explore(progress=lambda m: print(f"  {m}"))
                print(f"  stats: {stats}")
            elif cmd == "observe":
                stats = sess.observe_with_director(
                    progress=lambda m: print(f"  {m}")
                )
                print(f"  stats: {stats}")
                print(f"  director_report.md scritto in "
                      f"{out_dir / 'director_report.md'}")
            else:
                print(f"comando ignoto: {cmd!r}. Digita 'help'.")
                continue
            _say(f"   ({_fmt_elapsed(time.perf_counter() - t0)})")
        except KeyError as exc:
            print(f"  ERRORE: {exc}")
        except Exception as exc:
            print(f"  ERRORE inatteso ({type(exc).__name__}): {exc}")
    _say("shell: sessione salvata, arrivederci.")


# =============================================================================
# Main
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "session-open": run_session_open,
        "session-list": run_session_list,
        "session-expand": run_session_expand,
        "session-bridge": run_session_bridge,
        "session-revalidate": run_session_revalidate,
        "session-magistrale": run_session_magistrale,
        "session-auto": run_session_auto,
        "session-observe": run_session_observe,
        "book-analyze": run_book_analyze,
        "theme-analyze": run_theme_analyze,
        "theme-book-analyze": run_theme_book_analyze,
        "session-shell": run_session_shell,
    }
    runner = dispatch.get(args.command)
    if runner is None:
        parser.error(f"Comando non riconosciuto: {args.command}")
    runner(args)


if __name__ == "__main__":
    main(sys.argv[1:])
