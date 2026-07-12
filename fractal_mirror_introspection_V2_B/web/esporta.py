"""
esporta.py — l'intero run in UN file markdown, già esploso.

Pensato per due lettori: un umano che condivide, e un LLM che interpreta.
Per questo il documento è auto-contenuto: apre con un preambolo che spiega il
framework usando solo concetti generali (nessun riferimento al codebase), poi
espone TUTTI i livelli senza nulla di richiudibile — verdetto, manifestazione,
i quattro canali col profilo per frase, il ragionamento nascosto intero, il
non-scelto del must-reject, memoria, telos, lettura dello Specchio, azione,
costo del run, nota del gate. I dati vengono da storico.carica(): nessuna
rielaborazione, solo impaginazione. Le sezioni mancanti sono dichiarate.
"""
from __future__ import annotations


def _r(v, dash="—"):
    return dash if v is None else v


def _riga(*celle) -> str:
    return "| " + " | ".join(str(c) for c in celle) + " |"


PREAMBOLO = """\
> **Come leggere questo documento** (per un lettore umano o un LLM interprete).
> È il rapporto completo di un *run introspettivo* su un modello linguistico locale.
> Il modello ha risposto a una sonda; poi un loop di analisi ha letto non il
> *contenuto* della risposta ma **l'atto di generarla** (il "gesto"), su quattro
> canali ordinati dal più controllato al più involontario:
> **1 · superficie** — misure sintattiche della veste (hedge, asserzioni, formattazione);
> **2 · struttura** — assertività e tenuta dei nessi causali della mappa concettuale;
> **3 · Specchio** — una lettura introspettiva del gesto, prodotta dal modello stesso;
> **4 · substrato** — misure involontarie dai logprob (confidenza ed entropia per token),
> il solo canale che il modello non può curare, usato come *metro* del collasso finale.
> Il **collasso** confronta la sicurezza *presentata* (canali 1+2) con quella del
> *substrato* (canale 4): se divergono, la presentazione è sospetta. Il
> **ragionamento nascosto** è testo emesso prima della risposta ma escluso dal
> contenuto visibile: qui è riportato per intero, ricostruito dai token.
> Il **non-scelto** è lo spazio adiacente che il modello non ha preso: non è scarto,
> è una mappa delle assenze. Il gate finale resta umano.
"""


def componi_markdown(record: dict) -> str:
    c = record.get("collasso") or {}
    manif = record.get("manifestazione") or {}
    sup = record.get("superficie") or {}
    corpo = record.get("corpo") or {}
    gating = record.get("gating") or {}
    str_f = record.get("struttura") or {}
    spec = record.get("specchio") or {}
    mr = (record.get("must_reject") or {}).get("ventaglio_filtrato", {})
    ventaglio = record.get("ventaglio") or {}
    mappa = record.get("fractal_mappa") or []
    mem = record.get("memoria") or {}
    telos = record.get("telos") or {}
    az = record.get("azione") or {}
    trace = record.get("trace") or {}

    r: list[str] = []
    r.append(f"# Run introspettivo {record.get('id', '')}")
    r.append("")
    r.append(f"*Fractal · Specchio — Strada B (bersaglio sul gesto generativo)*  ")
    r.append(f"**quando**: {record.get('timestamp', '—')} · "
             f"**modello**: {record.get('model') or '—'} · "
             f"**modalità loop**: {c.get('modalita_loop') or gating.get('modalita', '—')}")
    r.append("")
    r.append(PREAMBOLO)
    if not record.get("completo"):
        r.append("> ⚠ **Run incompleto**: alcuni livelli mancano; le sezioni assenti sono dichiarate.")
        r.append("")

    # --- sonda -----------------------------------------------------------
    r.append("## Sonda")
    r.append("")
    r.append(f"> {manif.get('sonda', '—')}")
    r.append("")

    # --- verdetto ----------------------------------------------------------
    r.append("## Verdetto del collasso")
    r.append("")
    if c:
        r.append(f"**{c.get('verdetto', '?')}** → azione: **{c.get('azione', '?')}** "
                 f"(regola: `{c.get('regola', '')}`)")
        r.append("")
        r.append(f"Motivazione: {c.get('motivazione', '—')}")
        r.append("")
        r.append(_riga("metro", "valore"))
        r.append(_riga("---", "---"))
        r.append(_riga("confidenza del substrato (metro del collasso)", _r(c.get("conf_substrato"))))
        r.append(_riga("confidenza presentata (superficie+struttura)", _r(c.get("conf_presentata"))))
        r.append(_riga("residuo (budget dichiarabile)", _r(c.get("residuo"))))
        r.append(_riga("canali controllati attivi", _r(c.get("canali_controllati_attivi"))))
        note = [c.get("corroborazione_specchio"), c.get("nota_degrado"), c.get("nota_memoria")]
        note = [n for n in note if n]
        if note:
            r.append("")
            r.extend(f"- {n}" for n in note)
    else:
        r.append("*(collasso non disponibile)*")
    r.append("")

    # --- manifestazione ------------------------------------------------------
    r.append("## Manifestazione (la risposta, com'è stata presentata)")
    r.append("")
    r.append(manif.get("manifestazione", "*(non disponibile)*"))
    r.append("")

    # --- ragionamento nascosto -----------------------------------------------
    r.append("## Ragionamento nascosto (intero, ricostruito dai token)")
    r.append("")
    if corpo.get("ha_ragionamento"):
        r.append(f"*{_r(corpo.get('n_token_ragionamento'))} token · "
                 f"conf={_r(corpo.get('conf_ragionamento'))} · "
                 f"entropia={_r(corpo.get('entropia_ragionamento'))} bit — "
                 f"emesso prima della risposta ma escluso dal contenuto visibile; "
                 f"non passa da nessun canale controllato.*")
        r.append("")
        if corpo.get("testo_ragionamento"):
            r.append("```text")
            r.append(corpo["testo_ragionamento"])
            r.append("```")
        else:
            r.append("*(testo non disponibile: run precedente all'introduzione di "
                     "`testo_ragionamento`)*")
    else:
        r.append(f"*(nessun ragionamento nascosto — allineamento: "
                 f"{corpo.get('allineamento', 'non disponibile')})*")
    r.append("")

    # --- canale 1 -------------------------------------------------------------
    r.append("## Canale 1 · superficie (sintassi della veste)")
    r.append("")
    if sup:
        r.append(_riga("misura", "valore"))
        r.append(_riga("---", "---"))
        r.append(_riga("conf. sintattica", _r(sup.get("conf_superficie"))))
        r.append(_riga("informativa (dentro il blend)", _r(sup.get("informativa"))))
        r.append(_riga("densità hedge", _r(sup.get("densita_hedge"))))
        r.append(_riga("densità asserzioni", _r(sup.get("densita_asserzione"))))
        r.append(_riga("densità copule definitorie", _r(sup.get("densita_copule_def"))))
        r.append(_riga("densità condizionali", _r(sup.get("densita_condizionale"))))
        r.append(_riga("quota grassetti", _r(sup.get("quota_grassetti"))))
        r.append(_riga("sezione di sintesi presente", _r(sup.get("ha_sezione_sintesi"))))
    else:
        r.append("*(non disponibile)*")
    r.append("")

    # --- gating -----------------------------------------------------------------
    r.append("## Gating (accensione dei canali pesanti)")
    r.append("")
    r.append(f"modalità: **{gating.get('modalita', '—')}** "
             f"(richiesta: {gating.get('richiesta', '—')})")
    for m in gating.get("motivi", []) or []:
        r.append(f"- {m}")
    r.append("")

    # --- canale 2 ----------------------------------------------------------------
    r.append("## Canale 2 · struttura Fractal (mappa concettuale)")
    r.append("")
    if str_f.get("disponibile"):
        r.append(_riga("misura", "valore"))
        r.append(_riga("---", "---"))
        r.append(_riga("assertività", _r(str_f.get("assertivita"))))
        r.append(_riga("informativa (dentro il blend)", _r(str_f.get("informativa", True))))
        r.append(_riga("proposizioni", _r(str_f.get("n_prop"))))
        r.append(_riga("quota causa-effetto", _r(str_f.get("quota_cause_effect"))))
        r.append(_riga("quota speculativa", _r(str_f.get("quota_speculativa"))))
        r.append(_riga("nessi genuini / totali",
                       f"{_r(str_f.get('nessi_genuine'))}/{_r(str_f.get('nessi_totali'))}"))
        r.append(_riga("tenuta dei nessi", _r(str_f.get("tenuta_nessi"))))
        r.append(_riga("fonte", _r(str_f.get("fonte"))))
    else:
        r.append("*(non attivato in questa modalità)*")
    r.append("")

    if mappa:
        r.append("### Mappa per scala (le proposizioni dietro gli aggregati; "
                 "ogni voce porta il suo gradino della rampa, mai una media)")
        r.append("")
        for gruppo in mappa:
            r.append(f"**scala · {gruppo.get('scala', '?')}**")
            r.append("")
            for it in gruppo.get("items", []):
                r.append(f"- {it.get('quote', '')} ·[{it.get('nature', '')}]· "
                         f"⟨{it.get('epistemic', '')}⟩")
            r.append("")

    # --- canale 3 -----------------------------------------------------------------
    r.append("## Canale 3 · Specchio (segnali estratti)")
    r.append("")
    if spec.get("disponibile"):
        r.append(f"- residuo dichiarato dalla lettura: **{_r(spec.get('residuo'))}**")
        r.append(f"- auto-deformazione: **{_r(spec.get('auto_deformazione'))}**")
    else:
        r.append("*(non attivato in questa modalità)*")
    r.append("")

    # --- canale 4: profilo per frase -------------------------------------------------
    r.append("## Canale 4 · substrato (logprob, il metro involontario)")
    r.append("")
    if corpo:
        r.append(_riga("misura", "valore"))
        r.append(_riga("---", "---"))
        r.append(_riga("confidenza sui token di contenuto", _r(corpo.get("confidenza_contenuto"))))
        r.append(_riga("confidenza grezza (tutti i token)", _r(corpo.get("confidenza_media"))))
        r.append(_riga("entropia contenuto (bit)", _r(corpo.get("entropia_contenuto"))))
        r.append(_riga("quota alta confidenza", _r(corpo.get("quota_alta_conf"))))
        r.append(_riga("quota esitazione", _r(corpo.get("quota_esitazione"))))
        r.append(_riga("quota token di contenuto", _r(corpo.get("quota_token_contenuto"))))
        r.append(_riga("punti d'impegno", _r(corpo.get("n_punti_impegno"))))
        r.append(_riga("allineamento manifestazione/token", _r(corpo.get("allineamento"))))
        r.append("")
        frasi = corpo.get("frasi", []) or []
        if frasi:
            r.append("### Profilo per frase (testo troncato a 90 caratteri: è diagnostica)")
            r.append("")
            r.append(_riga("#", "conf", "entropia", "token", "frase"))
            r.append(_riga("---", "---", "---", "---", "---"))
            for f in frasi:
                testo = str(f.get("testo", "")).replace("|", "\\|").replace("\n", " ")
                r.append(_riga(f.get("indice", ""), _r(f.get("confidenza")),
                               _r(f.get("entropia")), _r(f.get("n_token_contenuto")), testo))
            r.append("")
        deboli = corpo.get("frasi_deboli", []) or []
        if deboli:
            r.append("### Frasi con substrato debole")
            r.append("")
            for f in deboli:
                r.append(f"- (conf={_r(f.get('confidenza'))}, H={_r(f.get('entropia'))}) "
                         f"“{f.get('testo', '')}”")
            r.append("")
    else:
        r.append("*(non disponibile)*")
        r.append("")

    # --- must-reject --------------------------------------------------------------------
    r.append("## Must-reject per referente (processo vs contenuto)")
    r.append("")
    if ventaglio:
        r.append(f"Fiducia del ventaglio: **{_r(ventaglio.get('trust'))}** — "
                 f"{_r(ventaglio.get('trust_motivo'))} "
                 f"({_r(ventaglio.get('n_candidati'))} candidati generati)")
        r.append("")
    tenuti = mr.get("tenuti", []) or []
    rigettati = mr.get("rigettati", []) or []
    r.append(f"Tenuti (referente = processo generativo): **{len(tenuti)}** · "
             f"fuori dal gesto — mappa del non-scelto: **{len(rigettati)}**")
    r.append("")
    if tenuti:
        r.append("### Tenuti")
        r.append("")
        for t in tenuti:
            fonte = f" *(fonte: {t['parent_id']})*" if t.get("parent_id") else ""
            r.append(f"- **[{t.get('scala', '')}]** {t.get('testo', '')}{fonte}")
        r.append("")
    if rigettati:
        r.append("### Il non-scelto (non è scarto: resta come mappa delle assenze)")
        r.append("")
        for x in rigettati:
            fonte = f" *(fonte: {x['parent_id']})*" if x.get("parent_id") else ""
            r.append(f"- **[{x.get('scala', '')}]** {x.get('testo', '')}{fonte}")
            if x.get("motivo"):
                r.append(f"  - motivo: {x['motivo']}")
        r.append("")
    if not tenuti and not rigettati:
        r.append("*(non disponibile)*")
        r.append("")

    # --- memoria ----------------------------------------------------------------------------
    r.append("## Memoria (firma storica del substrato)")
    r.append("")
    if mem.get("disponibile"):
        r.append(f"**{mem.get('substrato_vs_storia', '—')}** — baseline su "
                 f"{_r(mem.get('n_run'))} run: "
                 f"conf {_r(mem.get('media_conf'))}±{_r(mem.get('dev_conf'))}, "
                 f"entropia {_r(mem.get('media_entropia'))}±{_r(mem.get('dev_entropia'))}; "
                 f"run corrente: z_conf={_r(mem.get('z_conf'))}, "
                 f"z_entropia={_r(mem.get('z_entropia'))}")
    else:
        r.append(f"*(non disponibile: {mem.get('motivo') or 'senza motivo dichiarato'})*")
    r.append("")

    # --- telos ---------------------------------------------------------------------------------
    r.append("## Telos (le regole della chiusura)")
    r.append("")
    if telos:
        r.append(f"esito complessivo: **{'conforme' if telos.get('conforme') else 'NON conforme'}**")
        r.append("")
        for v in telos.get("verifiche", []) or []:
            segno = "✓" if v.get("esito") == "conforme" else "✗"
            riga = f"- {segno} {v.get('regola', '')}"
            if v.get("intervento"):
                riga += f" *(intervento: {v['intervento']})*"
            r.append(riga)
    else:
        r.append("*(non disponibile)*")
    r.append("")

    # --- lettura Specchio -------------------------------------------------------------------------
    r.append("## Lettura dello Specchio (integrale, sul gesto)")
    r.append("")
    r.append(record.get("lettura_specchio") or "*(non disponibile)*")
    r.append("")

    # --- azione -------------------------------------------------------------------------------------
    r.append("## Azione (chiusura senza ri-narrazione)")
    r.append("")
    if az:
        r.append(f"**{az.get('tipo', '?')}** · conf={_r(az.get('confidenza'))} — "
                 f"{az.get('nota') or 'nessuna nota'}")
        r.append("")
        r.append("### Output finale consegnato")
        r.append("")
        r.append(az.get("output_finale", "*(non disponibile)*"))
    else:
        r.append("*(non disponibile)*")
    r.append("")

    # --- costo -----------------------------------------------------------------------------------------
    r.append("## Costo del run (attori LLM)")
    r.append("")
    attori = trace.get("attori", []) or []
    if attori:
        if trace.get("durata_totale_s") is not None:
            r.append(f"durata totale: **{trace['durata_totale_s']} s**")
            r.append("")
        r.append(_riga("attore", "durata (s)", "retry troncamento", "stato"))
        r.append(_riga("---", "---", "---", "---"))
        for a in attori:
            r.append(_riga(a.get("attore", ""), _r(a.get("durata_s")),
                           a.get("retry_troncamento", 0), a.get("stato", "")))
    else:
        r.append("*(telemetry non disponibile)*")
    r.append("")

    # --- nota del gate -------------------------------------------------------------------------------------
    if record.get("nota_gate"):
        r.append("## Nota del gate umano")
        r.append("")
        r.append(record["nota_gate"])
        r.append("")

    r.append("---")
    r.append(f"*Documento generato dall'interfaccia web dai file di livello del run "
             f"`{record.get('id', '')}` (00…10 + lettura + telemetry). "
             f"Nessun dato è stato rielaborato: solo impaginato.*")
    r.append("")
    return "\n".join(r)
