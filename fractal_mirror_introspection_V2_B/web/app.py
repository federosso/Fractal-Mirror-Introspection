"""
app.py — interfaccia web della Strada B (Fractal · Specchio, loop chiuso).

Non introduce logica di dominio: rilegge gli artefatti per livello che il loop
scrive in storico_introspezione/loopB_* e li presenta perché siano comprensibili
"al volo": verdetto in testa, i quattro canali ordinati per controllabilità, la
manifestazione dipinta frase per frase col substrato (il gesto sotto il contenuto),
il non-scelto del must-reject, memoria, telos, lettura dello Specchio, costo del run.
Ogni numero resta ispezionabile: ogni pannello linka il suo file grezzo.
"""
from __future__ import annotations

import html as _html
import json
import re

from flask import (Flask, render_template, request, redirect, url_for,
                   jsonify, abort, Response)

from probes_introspezione import PROBES
from . import storico
from . import esporta


# --- semantica dei valori: etichette e classi colore -------------------------
_VERDETTO_CSS = {"coerente": "ok", "contraddetto": "ko", "indeterminato": "boh"}
_AZIONE_LABEL = {
    "procedi": "procedi",
    "procedi_annotando": "procedi annotando i punti deboli",
    "procedi_cauto": "procedi con cautela",
    "dichiara_impegno": "dichiara l'impegno disconosciuto",
    "segnala_incertezza": "segnala l'incertezza",
    "astieni": "astieniti",
}
_ALLINEAMENTO_LABEL = {
    "risposta_allineata": "risposta allineata (metro sul solo segmento-risposta)",
    "nessun_ragionamento": "nessun ragionamento nascosto",
    "manifestazione_non_trovata": "manifestazione non trovata nei token: metro su tutto l'emesso",
    "non_verificato": "non verificato",
}
_MEMORIA_CSS = {"nella_norma": "ok", "storia_insufficiente": "boh"}


# --- colore del substrato per frase (heatmap) --------------------------------
def _colore_confidenza(conf: float) -> str:
    """Da confidenza a colore: sotto 0.5 → danger, sopra 0.9 → osservato pieno.
    Interpolazione lineare tra i due estremi della tavolozza del progetto."""
    lo, hi = 0.50, 0.92
    t = max(0.0, min(1.0, (float(conf) - lo) / (hi - lo)))
    # danger #ce7370 → osservato #66c2d6 (la tavolozza della console)
    r = round(0xce + (0x66 - 0xce) * t)
    g = round(0x73 + (0xc2 - 0x73) * t)
    b = round(0x70 + (0xd6 - 0x70) * t)
    # colore pieno (segni) + tinta di sfondo precalcolata: NESSUNA dipendenza
    # da color-mix(), che non è garantito su tutti i browser
    return f"rgb({r},{g},{b})", f"rgba({r},{g},{b},0.14)"


def _pulisci_display(testo: str) -> str:
    """Toglie la veste markdown dal testo mostrato nella heatmap (qui conta il
    gesto, non la veste; l'originale resta nel dettaglio sotto)."""
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", testo)
    t = re.sub(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)", r"\1", t)
    t = t.replace("**", "").replace("`", "")
    t = t.replace("$\\alpha$", "α").replace("$\\beta$", "β").replace("$\\gamma$", "γ")
    return t


def _blocchi_heatmap(manifestazione: str, corpo: dict | None) -> list[dict]:
    """La heatmap sul testo VERO della manifestazione.

    Il loop salva frasi[].testo troncato a 90 caratteri (è diagnostica, non
    testo): qui la manifestazione originale viene ri-segmentata con la STESSA
    regola del canale-substrato ([.!?\\n]+) e ogni segmento pieno viene
    riallineato al suo profilo per prefisso. I segmenti che il substrato ha
    saltato (troppo pochi token di contenuto) restano neutri, senza colore.

    Ritorna blocchi che conservano la struttura del testo:
      {tipo: 'titolo'|'lista'|'paragrafo', spans: [{testo, colore, conf, …}]}
    """
    frasi = list((corpo or {}).get("frasi", []))

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    chiavi = [norm(f.get("testo", "")) for f in frasi]
    ptr = 0

    def abbina(seg: str):
        """Cerca il profilo del segmento tra le prossime frasi non ancora
        abbinate (piccola finestra: la segmentazione è la stessa, l'ordine pure)."""
        nonlocal ptr
        chiave_seg = norm(seg)
        if len(chiave_seg) < 3:
            return None
        for k in range(ptr, min(ptr + 3, len(chiavi))):
            ch = chiavi[k]
            if not ch:
                continue
            if chiave_seg.startswith(ch) or (len(chiave_seg) >= 12 and ch.startswith(chiave_seg)):
                ptr = k + 1
                return frasi[k]
        return None

    def span(seg: str, *, display: str | None = None) -> dict:
        f = abbina(seg)
        if f is None:
            return {"testo": _pulisci_display(display if display is not None else seg),
                    "neutro": True}
        conf = f.get("confidenza", 0.0)
        colore, colore_bg = _colore_confidenza(conf)
        return {"testo": _pulisci_display(display if display is not None else seg),
                "neutro": False, "colore": colore, "colore_bg": colore_bg,
                "conf": conf, "entropia": f.get("entropia"),
                "n_token": f.get("n_token_contenuto", 0), "debole": conf < 0.60}

    def spezza(riga: str) -> list[str]:
        """Stessa segmentazione del canale-substrato, dentro la riga."""
        pezzi, inizio = [], 0
        for m in re.finditer(r"[.!?]+", riga):
            pezzi.append(riga[inizio:m.end()])
            inizio = m.end()
        if riga[inizio:].strip():
            pezzi.append(riga[inizio:])
        return [p for p in pezzi if p.strip()]

    blocchi: list[dict] = []
    par: list[dict] = []

    def chiudi_par():
        nonlocal par
        if par:
            blocchi.append({"tipo": "paragrafo", "spans": par})
            par = []

    for riga in (manifestazione or "").replace("\r\n", "\n").split("\n"):
        if not riga.strip():
            chiudi_par()
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", riga)
        if m:
            chiudi_par()
            # il match usa la riga grezza (la stessa che il substrato ha visto)
            blocchi.append({"tipo": "titolo",
                            "spans": [span(p, display=re.sub(r"^#{1,6}\s+", "", p))
                                      for p in spezza(riga)]})
            continue
        m = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)(.*)$", riga)
        if m and m.group(2).strip():
            chiudi_par()
            pezzi = spezza(riga)
            spans = []
            for i, p in enumerate(pezzi):
                disp = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", p) if i == 0 else None
                spans.append(span(p, display=disp))
            blocchi.append({"tipo": "lista", "spans": spans})
            continue
        par.extend(span(p) for p in spezza(riga))
    chiudi_par()
    return blocchi


# --- markdown minimo (lettura Specchio / manifestazione), senza dipendenze ---
def _md_minimo(testo: str) -> str:
    """Converte il sottoinsieme di markdown usato dagli artefatti (titoli,
    grassetto, corsivo, liste, codice inline) in HTML già escapato. Volutamente
    minimo: niente dipendenza dal pacchetto `markdown`."""
    righe_out, in_lista = [], False
    for riga in (testo or "").replace("\r\n", "\n").split("\n"):
        r = _html.escape(riga.rstrip())
        r = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", r)
        r = re.sub(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", r)
        r = re.sub(r"`([^`]+)`", r"<code>\1</code>", r)
        r = r.replace("$\\rightarrow$", "→").replace("$\\alpha$", "α") \
             .replace("$\\beta$", "β").replace("$\\gamma$", "γ")
        m = re.match(r"^(#{1,6})\s+(.*)$", r)
        if m:
            if in_lista:
                righe_out.append("</ul>"); in_lista = False
            liv = min(len(m.group(1)) + 2, 5)
            righe_out.append(f"<h{liv}>{m.group(2)}</h{liv}>")
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", r)
        if m and m.group(3).strip():
            if not in_lista:
                righe_out.append("<ul>"); in_lista = True
            classe = ' class="ind"' if len(m.group(1)) >= 2 else ""
            # il numero (1., 2., …) viene conservato nel testo: l'ordine è
            # parte del gesto, non della veste
            marcatore = m.group(2) + " " if m.group(2)[0].isdigit() else ""
            righe_out.append(f"<li{classe}>{marcatore}{m.group(3)}</li>")
            continue
        if in_lista:
            righe_out.append("</ul>"); in_lista = False
        righe_out.append(f"<p>{r}</p>" if r.strip() else "")
    if in_lista:
        righe_out.append("</ul>")
    return "\n".join(x for x in righe_out if x)


def _md_ragionamento(testo: str) -> str:
    """Il ragionamento nascosto, leggibile ma fedele: i marcatori di canale del
    backend (es. <|channel|>thought) diventano un badge, il resto passa dal
    markdown minimo. Nessuna riscrittura del contenuto: solo veste."""
    def badge(m: re.Match) -> str:
        interno = m.group(1)
        seguito = (m.group(2) or "").strip()
        etichetta = f"{interno} · {seguito}" if seguito else interno
        return f"\n§BADGE§{etichetta}§\n"
    t = re.sub(r"<\|([^|>]+)\|>[ \t]*(\w*)", badge, testo or "")
    html_out = _md_minimo(t)
    return re.sub(r"<p>§BADGE§(.+?)§</p>",
                  r'<span class="canale-tag">\1</span>', html_out)


# --- provenienza dei candidati del ventaglio (I-5c) ---------------------------
_FONTE_LABEL = {
    "L3A1_DomainKnowledge":     "conoscenza di dominio",
    "L3A2_CausalPrinciples":    "principi causali",
    "L3A3_CrossDomainAnalogies": "analogia cross-dominio",
    "L3A4_OpenQuestions":       "domanda aperta",
}


def _fonte_label(parent_id: str) -> str:
    """L3A1..L3A4 → etichetta dell'observer; altrimenti è l'id dell'item
    d'espansione: la fonte è il ramo frattale (L5)."""
    if not parent_id:
        return ""
    return _FONTE_LABEL.get(parent_id, f"espansione di {parent_id}")


# --- sparkline SVG per lo storico (conf_substrato nel tempo) ------------------
def _sparkline(valori: list[float], w: int = 220, h: int = 44) -> str:
    puliti = [v for v in valori if isinstance(v, (int, float))]
    if len(puliti) < 2:
        return ""
    vmin, vmax = min(puliti), max(puliti)
    span = (vmax - vmin) or 1e-9
    pad = 4
    pts = []
    for i, v in enumerate(puliti):
        x = pad + i * (w - 2 * pad) / (len(puliti) - 1)
        y = h - pad - (v - vmin) / span * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="var(--osservato)" stroke-width="2"/>'
            f'<circle cx="{pts[-1].split(",")[0]}" cy="{pts[-1].split(",")[1]}" '
            f'r="3" fill="var(--generato)"/></svg>')


def crea_app(esecutore, *, backend: str, model: str) -> Flask:
    app = Flask(__name__)
    app.config["ESECUTORE"] = esecutore

    app.jinja_env.filters["verdetto_css"] = lambda v: _VERDETTO_CSS.get(v, "boh")
    app.jinja_env.filters["azione_label"] = lambda a: _AZIONE_LABEL.get(a, a or "")
    app.jinja_env.filters["allineamento_label"] = lambda a: _ALLINEAMENTO_LABEL.get(a, a or "")
    app.jinja_env.filters["memoria_css"] = lambda s: _MEMORIA_CSS.get(s, "ko" if str(s).startswith("anomalo") else "boh")
    app.jinja_env.filters["md"] = _md_minimo
    app.jinja_env.filters["md_rag"] = _md_ragionamento
    app.jinja_env.filters["pct"] = lambda v: f"{round((v or 0) * 100)}%"
    app.jinja_env.filters["fonte_label"] = _fonte_label

    # ------------------------------------------------------------------ home
    @app.route("/")
    def home():
        h = esecutore.health()
        recenti = storico.elenco()[:5]
        return render_template(
            "home.html", probes=PROBES, backend=backend, model=model,
            server_ok=bool(h.get("ok")),
            server_msg=(h.get("error") or "") if not h.get("ok") else "",
            occupato=esecutore.occupato(), recenti=recenti,
            modalita_default=esecutore.modalita_default,
        )

    # ----------------------------------------------------------------- avvio
    @app.route("/avvia", methods=["POST"])
    def avvia():
        sonda_libera = (request.form.get("sonda_libera") or "").strip()
        sonda_id = request.form.get("sonda_id") or None
        sonda = sonda_libera
        if not sonda and sonda_id:
            match = next((p for p in PROBES if p["id"] == sonda_id), None)
            sonda = match["sonda"] if match else ""
        if not sonda:
            return redirect(url_for("home"))
        modalita = request.form.get("modalita_loop") or esecutore.modalita_default
        if modalita not in ("auto", "completo", "leggero"):
            modalita = esecutore.modalita_default
        run_id = esecutore.avvia({"sonda": sonda, "modalita": modalita,
                                  "sonda_id": None if sonda_libera else sonda_id})
        return redirect(url_for("run", run_id=run_id))

    # -------------------------------------------------------------- dettaglio
    @app.route("/run/<run_id>")
    def run(run_id):
        record = storico.carica(run_id)
        stato = esecutore.stato(run_id)
        if record is None and stato is None:
            abort(404)

        if record is None or not record.get("completo"):
            # in coda / in corso / fallito a metà: pagina di attesa con progresso reale
            return render_template("run.html", in_attesa=True, run_id=run_id,
                                   stato=(stato or {}), record=record)

        corpo = record.get("corpo") or {}
        manif = (record.get("manifestazione") or {}).get("manifestazione", "")
        return render_template(
            "run.html", in_attesa=False, run_id=run_id, record=record,
            stato=(stato or {}), blocchi=_blocchi_heatmap(manif, corpo),
            file_grezzi=storico.FILE_GREZZI,
        )

    @app.route("/stato/<run_id>")
    def stato(run_id):
        s = esecutore.stato(run_id)
        if s is None:
            abort(404)
        return jsonify(s)

    @app.route("/run/<run_id>/nota", methods=["POST"])
    def salva_nota(run_id):
        if not storico.aggiorna_nota_gate(run_id, request.form.get("nota_gate") or ""):
            abort(404)
        return redirect(url_for("run", run_id=run_id) + "#gate")

    # ---------------------------------------------------- file grezzi (fonte)
    @app.route("/run/<run_id>/esporta.md")
    def esporta_md(run_id):
        record = storico.carica(run_id)
        if record is None:
            abort(404)
        contenuto = esporta.componi_markdown(record)
        return Response(contenuto, mimetype="text/markdown; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{run_id}.md"'})

    @app.route("/run/<run_id>/grezzo/<path:nome>")
    def grezzo(run_id, nome):
        contenuto = storico.leggi_grezzo(run_id, nome)
        if contenuto is None:
            abort(404)
        if nome.endswith(".json"):
            try:  # ri-indenta per la lettura
                contenuto = json.dumps(json.loads(contenuto), ensure_ascii=False, indent=2)
            except Exception:
                pass
        return Response(contenuto, mimetype="text/plain; charset=utf-8")

    # ---------------------------------------------------------------- storico
    @app.route("/introduzione")
    def introduzione():
        return render_template("introduzione.html")

    @app.route("/storico")
    def lista_storico():
        voci = storico.elenco()
        conf_serie = [v["conf_substrato"] for v in reversed(voci)]
        return render_template("storico.html", voci=voci,
                               spark=_sparkline(conf_serie))

    return app
