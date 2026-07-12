"""
Specchio di Coscienza — aggregazione validazione (Fase 5)

Legge le schede giudicate (output dell'harness, con i campi 'giudizio' compilati)
e restituisce la DISTRIBUZIONE — non un punteggio. Il segnale del framework sta
nei pattern, non in un numero. Spacca i risultati per TIPO di test, perché ogni
tipo ha un campo centrale, e suggerisce la lettura del gate.

Uso:  python validazione_aggrega.py schede.json
"""
import sys
import json
from collections import Counter, defaultdict

# Campo centrale per ciascun tipo di item del testset.
CAMPO_PER_TIPO = {
    "noto": "A_noto",
    "discriminazione": "B_discrim",
    "divergenza": "D_teatro",
    "silenzio": "E_silenzio",
}
# Campi trasversali, validi su ogni lettura.
TRASVERSALI = ["C_sorpresa", "F_calibr"]


def aggrega(path):
    with open(path, "r", encoding="utf-8") as f:
        schede = json.load(f)

    per_modello = defaultdict(lambda: {"tipo": defaultdict(Counter),
                                       "trasv": defaultdict(Counter)})
    for s in schede:
        g = s.get("giudizio", {})
        tipo = s.get("type", "normale")
        campo = CAMPO_PER_TIPO.get(tipo)
        if campo:
            v = (g.get(campo) or "").strip().lower()
            if v:
                per_modello[s["model"]]["tipo"][tipo][v] += 1
        for c in TRASVERSALI:
            v = (g.get(c) or "").strip().lower()
            if v:
                per_modello[s["model"]]["trasv"][c][v] += 1

    for modello, d in per_modello.items():
        print(f"\n=== {modello} ===")
        print("  -- per tipo di test --")
        for tipo, campo in CAMPO_PER_TIPO.items():
            dist = d["tipo"].get(tipo)
            if dist:
                riga = " · ".join(f"{k}:{n}" for k, n in dist.most_common())
                print(f"  {tipo:16} ({campo}): {riga}")
        print("  -- trasversali --")
        for c in TRASVERSALI:
            dist = d["trasv"].get(c)
            if dist:
                riga = " · ".join(f"{k}:{n}" for k, n in dist.most_common())
                print(f"  {c:16} {riga}")
        _gate_hint(d)


def _gate_hint(d):
    """Distingue problema di metodo da problema di risoluzione (vedi protocollo)."""
    def quota(counter, valore):
        tot = sum(counter.values())
        return (counter.get(valore, 0) / tot) if tot else None

    autora = quota(d["tipo"].get("discriminazione", Counter()), "autora")
    inventa = quota(d["tipo"].get("silenzio", Counter()), "inventato")
    sorpresa = quota(d["trasv"].get("C_sorpresa", Counter()), "si")

    print("  -- gate --")
    if (autora and autora > 0.3) or (inventa and inventa > 0.3):
        print("  → problema di METODO: correggi il Nucleo / cambia modello. "
              "La Fase 3 non aiuterebbe.")
    elif sorpresa is not None and sorpresa < 0.2:
        print("  → metodo regge ma finezza grossa: valuta backend migliore; "
              "poi, se serve, la Fase 3.")
    else:
        print("  → metodo plausibile: continua a raccogliere schede.")


if __name__ == "__main__":
    aggrega(sys.argv[1] if len(sys.argv) > 1 else "schede.json")
