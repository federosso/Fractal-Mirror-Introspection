"""
Specchio di Coscienza — Piani bassi a freddo (Fase 3)

╔══════════════════════════════════════════════════════════════════════╗
║  DORMIENTE.  Non implementare né eseguire finché la Fase 4 non ha     ║
║  superato il gate.  Costruire questa macchina prima della validazione ║
║  è la mossa che il progetto vieta: si affinerebbe la risoluzione di   ║
║  una lettura che non si è ancora dimostrata corretta.                 ║
║                                                                       ║
║  Si attiva SOLO se la validazione mostra: metodo che regge            ║
║  (B traccia, E tace, F calibra) MA finezza del testo-solo grossa.     ║
╚══════════════════════════════════════════════════════════════════════╝

Quando si attiverà, questi estrattori calcoleranno la 'traiettoria attesa'
quantificata dai tre piani bassi, rendendo il residuo-spirito più pulito
(paradosso della fedeltà). L'interfaccia è già fissata; il calcolo no.

Pesatura: ogni piano è etichettato per volontarietà — è l'ordine in cui
lo specchio pesa (inversamente: il meno volontario pesa di più).
"""
from __future__ import annotations

GATE_APERTO = False  # → True solo dopo il superamento della Fase 4


def _gate():
    if not GATE_APERTO:
        raise RuntimeError(
            "Fase 3 dormiente: gate Fase 4 non superato. "
            "Valida il metodo prima di costruire la macchina fredda."
        )


# --- Estrattori per piano (stub: interfaccia fissata, calcolo da fare) -------

def feature_mente(testo: str) -> dict:
    """Strato curato (alta volontarietà, più mascherabile).
    Da calcolare: metriche di struttura linguistica — architettura del
    pensiero, coerenza, ricchezza/rigidità lessicale.
    """
    _gate()
    raise NotImplementedError("Fase 3 · mente")


def feature_emozione(audio_o_prosodia) -> dict:
    """Strato intermedio. Da calcolare: prosodia/affetto — carica, ritmo,
    tempismo e durata delle pause, incrinature della voce.
    """
    _gate()
    raise NotImplementedError("Fase 3 · emozione")


def feature_corpo(segnale_fisiologico) -> dict:
    """Strato involontario (bassa volontarietà, dove la verità trapela).
    Da calcolare: segnali fisiologici dove disponibili (es. HRV / coerenza
    cardiaca). È la firma meno controllabile.
    """
    _gate()
    raise NotImplementedError("Fase 3 · corpo")


# --- Iniezione (già reale: formatta, non calcola) ----------------------------

def inject_features(manifestation: str, features: dict) -> str:
    """Compone l'input dello specchio aggiungendo la traiettoria attesa
    quantificata. 'features' = {'corpo': {...}, 'emozione': {...}, 'mente': {...}}.
    Ordine di presentazione = ordine di peso (involontario prima).
    """
    if not features:
        return manifestation
    righe = ["[TRAIETTORIA ATTESA — piani bassi quantificati]"]
    for piano in ("corpo", "emozione", "mente"):  # dal meno al più volontario
        if piano in features:
            righe.append(f"· {piano}: {features[piano]}")
    return manifestation + "\n\n" + "\n".join(righe)
