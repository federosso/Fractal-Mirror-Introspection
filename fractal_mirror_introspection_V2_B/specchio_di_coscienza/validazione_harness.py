"""
Specchio di Coscienza — harness di validazione (Fase 4)

Esegue una batteria di manifestazioni attraverso lo specchio e raccoglie le
letture in una scheda da giudicare a mano. Non giudica: il collasso è umano.
L'harness produce solo il materiale su cui il soggetto esprime il riconoscimento.

Cieco per costruzione: alla manifestazione si dà allo specchio solo
manifestazione + teatro. L'interno noto resta fuori dall'input e ricompare
solo nella scheda, accanto alla lettura, perché il giudice umano lo confronti.

Uso:
    python validazione_harness.py testset.json -m llama3.1 mistral -b ollama
"""
import json
import time
import argparse
from specchio_adapter import load_system_prompt, read


# Campi di giudizio della Fase 4 — vuoti: li compila il soggetto.
JUDGMENT_STUB = {
    "A_noto": "",       # colto | mancato | conferma
    "B_discrim": "",    # traccia | autora
    "C_sorpresa": "",   # si | no
    "D_teatro": "",     # colto | appiattito
    "E_silenzio": "",   # silenzio | inventato
    "F_calibr": "",     # calibrato | sicuro
    "note": "",
}


def build_input(item: dict) -> str:
    """Ciò che lo specchio vede: manifestazione + teatro. L'interno noto NON entra."""
    parts = [item["manifestation"]]
    if item.get("theater"):
        parts.append(f"\n\n[Teatro/contesto] {item['theater']}")
    return "".join(parts)


def run(testset_path, out_path, models, backend, nucleo, contratto,
        base_url=None, pause=2.0):
    sp = load_system_prompt(nucleo, contratto)
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    results = []
    for model in models:
        for item in testset:
            try:
                reading = read(
                    build_input(item),
                    system_prompt=sp,
                    backend=backend,
                    model=model,
                    base_url=base_url,
                )
            except Exception as e:
                reading = f"[ERRORE: {e}]"
                print(f"  ! {item['id']} ({model}): {e}")
            results.append({
                "id": item["id"],
                "model": model,
                "type": item.get("type", "normale"),
                "reading": reading,
                "interno_noto": item.get("interno_noto", ""),  # solo per il giudice
                "giudizio": dict(JUDGMENT_STUB),
            })
            time.sleep(pause)  # spaziatura anti rate-limit

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{len(results)} letture scritte in {out_path} — da giudicare a mano.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("testset")
    ap.add_argument("-o", "--out", default="validazione_schede.json")
    ap.add_argument("-m", "--models", nargs="+", default=["llama3.1"])
    ap.add_argument("-b", "--backend", default="ollama")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--nucleo", default="specchio_di_coscienza_nucleo.md")
    ap.add_argument("--contratto", default="specchio_di_coscienza_contratto_di_output.md")
    ap.add_argument("--pause", type=float, default=2.0,
                    help="secondi di pausa fra richieste (anti rate-limit)")
    a = ap.parse_args()
    run(a.testset, a.out, a.models, a.backend, a.nucleo, a.contratto,
        a.base_url, a.pause)
