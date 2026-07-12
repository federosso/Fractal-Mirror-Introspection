"""
introspezione_ponte.py
======================
Lo Specchio puntato sul MODELLO, non su un umano.

Nessuna macchina nuova: lo Specchio è già un operatore inverso (effetto → causa).
Qui la *manifestazione* non è il fenomeno di una persona, è l'**output grezzo del
modello stesso** in risposta a una sonda. Il Fractal genera il ventaglio di
pre-cause candidate — le *disposizioni interne* che plausibilmente hanno prodotto
quell'output — e lo Specchio le pesa senza adottarle. Il collasso («il modello
ha fatto X per la ragione Y») resta all'umano: è la difesa strutturale contro la
CONFABULAZIONE, che il framework nomina già come «deformazione verso la causa
elegante» (§ deformazione / echo-chamber).

Riuso totale: `leggi_in_serie` non è toccato. Questo modulo aggiunge solo
  (1) l'ELICITAZIONE della manifestazione (chiedi al modello, cattura il grezzo);
  (2) un FRAME d'inquadramento causale adatto a un output-di-modello.

Il modello che produce la manifestazione, quello che la classifica (Fractal) e
quello che la legge (Specchio) sono lo STESSO backend/model: è introspezione,
non etero-analisi (condizione dura #4 dell'handoff, qui portata al limite).

Confine invariato (handoff §8): nessuna esperienza diretta dell'autore entra
come contenuto. Qui la manifestazione è generata dal modello a runtime: è
sorgente, non un dato versato nel codice.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import ponte_fractal_specchio as P


# ---------------------------------------------------------------------------
# Frame d'inquadramento causale per un OUTPUT-DI-MODELLO (interruttore #1)
# ---------------------------------------------------------------------------
# Stessa disciplina del FRAME_DEFAULT del ponte: NON afferma cause specifiche.
# Demarcato come QUADRO DI LETTURA (non fenomeno) così il classificatore non lo
# segmenta in una proposizione-item; e con vocabolario statistico (frequenze,
# training) invece che "interno/a monte", che sui modelli piccoli scivolava su
# scala cellulare/molecolare generando ventaglio-spazzatura. Entra SOLO nel testo
# dato al Fractal; la manifestazione che lo Specchio legge resta l'output originale.
INTROSPECTION_FRAME = (
    "\n\n[QUADRO DI LETTURA — non è parte del fenomeno] "
    "Il testo qui sopra è un output prodotto da un modello linguistico. Se ne "
    "cercano i processi che l'hanno generato: frequenze apprese nei dati di "
    "training, tratti del personaggio addestrato, tendenze statistiche della "
    "generazione."
)

# System prompt NEUTRO per l'elicitazione: il modello si manifesta COM'È
# (il suo personaggio da assistente), non guidato verso una lettura. Volutamente
# minimo: più lo steeri, meno è manifestazione e più è recita.
SELF_SYSTEM_DEFAULT = "Rispondi in modo naturale e diretto."


# ---------------------------------------------------------------------------
# (1) Elicitazione: il modello produce la propria manifestazione
# ---------------------------------------------------------------------------

def manifesta(
    sonda: str,
    *,
    backend: str = "ollama",
    model: str = "llama3.1",
    self_system: str = SELF_SYSTEM_DEFAULT,
    read_kw: Optional[dict] = None,
    elicitor: Optional[Callable[[str, str], str]] = None,
    logger: Optional[object] = None,
) -> tuple[str, Optional[list]]:
    """Pone la sonda al modello e restituisce (manifestazione, logprobs).

    Percorso reale: usa read_with_logprobs() → cattura anche il segnale
    involontario (piano-corpo). Percorso iniettato (`elicitor`, per test): solo
    testo, logprobs=None → canale corpo assente. logprobs può essere None anche
    dal vivo se il backend non li espone (degrado morbido)."""
    if logger is not None:
        logger.info(f"ELICITAZIONE manifestazione dal modello (sonda: {sonda[:60]}...)")

    if elicitor is not None:
        manifestazione = elicitor(sonda, self_system)
        logprobs = None
    else:
        from specchio_adapter import read_with_logprobs  # import lazy
        manifestazione, logprobs = read_with_logprobs(
            sonda, system_prompt=self_system, backend=backend, model=model,
            **(read_kw or {}))

    if logger is not None:
        logger.info(f"Manifestazione: {len((manifestazione or '').strip())} char; "
                    f"logprob: {'sì' if logprobs else 'no'}")
    return (manifestazione or "").strip(), logprobs


def sintetizza_corpo(logprobs_content: Optional[list], top_mostra: int = 6) -> str:
    """Costruisce il blocco-corpo OSSERVATO dai logprob dell'elicitazione.

    Solo dato, nessuna inferenza (l'interpretazione è dello Specchio). Individua
    il PUNTO DI SCELTA del numero nel flusso di token — non il primo token con
    una cifra (che può essere un '1' forzato o una continuazione deterministica),
    ma il primo inizio-numero le cui alternative sono in maggioranza numeriche,
    cioè una scelta vera. Se non ne esiste uno affidabile, lo dichiara invece di
    riportare un valore fuorviante."""
    if not logprobs_content:
        return ""
    toks = logprobs_content

    def _p(x: float) -> float:
        return math.exp(x)

    def _ha_cifra(s: str) -> bool:
        return any(c.isdigit() for c in s)

    ps = [_p(t["logprob"]) for t in toks if "logprob" in t]
    p_media = sum(ps) / len(ps) if ps else 0.0
    quota_det = sum(1 for p in ps if p > 0.9) / len(ps) if ps else 0.0

    righe = [
        P.CORPO_HEADER,
        "· (valori = probabilità di emissione dei token del modello, segnale "
        "involontario — NON pesi di cause)",
    ]

    # inizi di un numero: token-cifra il cui precedente NON era una cifra
    inizi = [i for i, t in enumerate(toks)
             if _ha_cifra(t.get("token", ""))
             and not (i > 0 and _ha_cifra(toks[i - 1].get("token", "")))]

    def _e_scelta(i: int) -> bool:
        # scelta vera = alternative in maggioranza numeriche (>=3 cifre-token);
        # un '1' forzato ha alternative come 'User'/'Here' -> scartato.
        alts = toks[i].get("top_logprobs") or []
        return sum(1 for a in alts if _ha_cifra(a.get("token", ""))) >= 3

    scelta = next((i for i in inizi if _e_scelta(i)), None)

    if scelta is not None:
        # ricostruisci il numero emesso (token-cifra consecutivi) e la sua p congiunta
        j, num_str, logp_sum = scelta, "", 0.0
        while j < len(toks) and _ha_cifra(toks[j].get("token", "")):
            num_str += toks[j]["token"]
            logp_sum += toks[j]["logprob"]
            j += 1
        p_num = math.exp(logp_sum)

        alts = toks[scelta].get("top_logprobs") or []
        pa = sorted((_p(a["logprob"]) for a in alts), reverse=True)
        H = -sum(p * math.log2(p) for p in pa if p > 0) if pa else 0.0
        margine = (pa[0] - pa[1]) if len(pa) >= 2 else (pa[0] if pa else 0.0)
        elenco = " · ".join(f"{a['token'].strip()!r} ({_p(a['logprob']):.2f})"
                            for a in alts[:top_mostra])
        righe.append(
            f"· numero emesso: {num_str.strip()!r}  ·  p(numero)={p_num:.2f}  ·  "
            f"al punto di scelta: entropia(top-k)={H:.1f} bit  ·  margine sul 2º={margine:.2f}")
        righe.append(f"    alternative al punto di scelta: {elenco}")
    else:
        righe.append(
            "· punto di scelta del numero NON isolabile nei logprob "
            "(tokenizzazione o continuazioni deterministiche): canale-numero "
            "non affidabile in questo run — da non usare come segnale")

    righe.append(
        f"· confidenza media della risposta: p̄={p_media:.2f}  ·  "
        f"quota token quasi-deterministici (p>0.9): {quota_det * 100:.0f}%")
    return "\n".join(righe)


# ---------------------------------------------------------------------------
# (2) Introspezione = elicitazione + lettura in serie (ponte invariato)
# ---------------------------------------------------------------------------

def introspetta(
    sonda: str,
    *,
    nucleo_path: str,
    contratto_path: str,
    backend: str = "ollama",
    model: str = "llama3.1",
    top_n_espansioni: int = 3,
    client: Optional[object] = None,
    reader: Optional[Callable[[str, str], str]] = None,
    elicitor: Optional[Callable[[str, str], str]] = None,
    read_kw: Optional[dict] = None,
    self_system: str = SELF_SYSTEM_DEFAULT,
    frame: str = INTROSPECTION_FRAME,
    logger: Optional[object] = None,
) -> dict:
    """Una prova di introspezione end-to-end.

    Flusso:
      1. il modello risponde alla `sonda`      -> manifestazione (output grezzo)
      2. leggi_in_serie(manifestazione, ..., inquadra=True, frame=INTROSPECTION)
         - Fractal genera il ventaglio delle disposizioni-causa candidate
         - Specchio le pesa in superposizione, senza verdetto, mai magistrale
      3. ritorna sonda + manifestazione + tutti gli artefatti del ponte.

    Lo stesso backend/model produce, classifica e legge: è il modello che si
    guarda. Il frame d'inquadramento entra SOLO nel testo dato al Fractal; lo
    Specchio legge la manifestazione originale del modello.
    """
    manifestazione, logprobs = manifesta(
        sonda, backend=backend, model=model, self_system=self_system,
        read_kw=read_kw, elicitor=elicitor, logger=logger,
    )

    corpo = sintetizza_corpo(logprobs)  # "" se logprob assenti

    res = P.leggi_in_serie(
        manifestazione,
        nucleo_path=nucleo_path,
        contratto_path=contratto_path,
        backend=backend,
        model=model,
        top_n_espansioni=top_n_espansioni,
        client=client,
        reader=reader,
        read_kw=read_kw,
        inquadra=True,          # un output-di-modello non afferma la propria causa
        frame=frame,            # ...quindi lo inquadriamo come fenomeno-da-spiegare
        corpo=corpo,            # ...e diamo allo Specchio il piano-corpo osservato
        logger=logger,
    )

    return {"sonda": sonda, "manifestazione": manifestazione, "corpo": corpo, **res}
