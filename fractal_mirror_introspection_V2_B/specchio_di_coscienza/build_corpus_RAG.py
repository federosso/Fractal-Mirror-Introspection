"""
Costruzione dello store RAG (Fase 2).

Embedding e chat sono servizi SEPARATI. Qui conta solo il servizio embeddings.
L'errore 'connection refused su :11434' significa che il servizio embeddings non
è attivo all'indirizzo atteso. Scegli una delle due opzioni e avvia il servizio.

Opzione A (consigliata) — Ollama per gli embedding, in parallelo a llama.cpp:
    ollama serve
    ollama pull nomic-embed-text
  Lascia i parametri sotto come sono (backend='ollama').

Opzione B — tutto llama.cpp: avvia un SECONDO server con un modello di embedding
  e il flag --embeddings, su una porta diversa da quella del chat:
    llama-server -m nomic-embed-text.gguf --embeddings --port 8081
  poi usa la configurazione 'B' qui sotto.
"""
from specchio_rag import build_store

manifest = [
    {'path': 'corpus_ancorato/*.txt',  'provenienza': 'ancorato'},
    {'path': 'corpus_inferenza/*.txt', 'provenienza': 'inferenza'},
]

# --- Opzione A: Ollama -------------------------------------------------------
build_store(
    manifest,
    prefix='corpus',
    backend='ollama',
    model='nomic-embed-text',
)

# --- Opzione B: llama.cpp dedicato (scommenta e commenta A) ------------------
# build_store(
#     manifest,
#     prefix='corpus',
#     backend='llamacpp',
#     base_url='http://localhost:8081/v1',
#     model='nomic-embed-text',   # il nome che il tuo llama-server espone
# )
