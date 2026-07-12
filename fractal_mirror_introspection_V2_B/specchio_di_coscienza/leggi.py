"""
Specchio di Coscienza — lettura singola (V0 end-to-end)

Compone Nucleo + Contratto come system prompt, recupera contesto dal corpus
(se uno store è indicato) e legge una manifestazione. È l'entry point del V0.

Uso:
  python leggi.py "manifestazione..." --teatro "..." --corpus corpus -m llama3.1 -b ollama
"""
import argparse
from specchio_adapter import load_system_prompt, read

try:
    from specchio_rag import load_store, retrieve, format_context
    HAS_RAG = True
except Exception:
    HAS_RAG = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifestazione")
    ap.add_argument("--teatro", default=None)
    ap.add_argument("--corpus", default=None,
                    help="prefix dello store RAG; se assente, nessun recupero")
    ap.add_argument("-m", "--model", default="llama3.1")
    ap.add_argument("-b", "--backend", default="ollama")
    ap.add_argument("--base-url", default=None)
    # Embedding e chat sono servizi separati: parametri indipendenti per il recupero.
    ap.add_argument("--embed-backend", default="ollama")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--embed-base-url", default=None)
    a = ap.parse_args()

    sp = load_system_prompt(
        "specchio_di_coscienza_nucleo.md",
        "specchio_di_coscienza_contratto_di_output.md",
    )

    user = a.manifestazione
    if a.teatro:
        user += f"\n\n[Teatro/contesto] {a.teatro}"
    if a.corpus and HAS_RAG:
        store = load_store(a.corpus)
        hits = retrieve(a.manifestazione, store, k=5,
                        backend=a.embed_backend, model=a.embed_model,
                        base_url=a.embed_base_url)
        user = format_context(hits) + "\n\n[MANIFESTAZIONE DA LEGGERE]\n" + user

    # Consegna esplicita: impedisce ai modelli deboli di copiare il Contratto
    # invece di applicarlo.
    user += (
        "\n\n---\n"
        "Produci ORA una lettura di QUESTA manifestazione, nella forma del "
        "Contratto di output. Non riprodurre né tradurre le istruzioni: applicale. "
        "Rispondi in italiano."
    )

    print(read(user, system_prompt=sp, backend=a.backend,
               model=a.model, base_url=a.base_url))


if __name__ == "__main__":
    main()
