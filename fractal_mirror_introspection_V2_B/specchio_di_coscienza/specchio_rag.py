"""
Specchio di Coscienza — RAG (Fase 2)

Corpus → recupero contestuale, con la disciplina di provenienza del framework:
ogni chunk è marcato 'ancorato' (documentato, citabile) o 'inferenza'
(ricostruito). Lo specchio deve poter onorare la premessa esperienza-vs-
interpretazione: il recupero gliela porta etichettata.

Embeddings: endpoint locale compatibile (Ollama: /v1/embeddings).
Vector store: numpy su disco. Nessuna dipendenza pesante oltre numpy + requests.
"""
from __future__ import annotations
import os
import json
import glob
import numpy as np
import requests

from specchio_adapter import BACKENDS  # riuso dei preset dei vasi locali

PROVENIENZE = {"ancorato", "inferenza"}


# --- Embeddings --------------------------------------------------------------
def embed(texts, model="nomic-embed-text", backend="ollama",
          base_url=None, timeout=120) -> np.ndarray:
    """Vettorizza una lista di testi via endpoint /v1/embeddings compatibile."""
    url_base = base_url or BACKENDS[backend]["base_url"]
    url = f"{url_base}/embeddings"
    try:
        resp = requests.post(
            url, json={"model": model, "input": list(texts)}, timeout=timeout
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"\n[RAG] Nessun servizio di embedding su {url} (backend '{backend}').\n"
            f"      Embedding e chat sono servizi SEPARATI. Avvia un servizio di\n"
            f"      embedding e punta il build lì:\n"
            f"        A) Ollama:    ollama serve  +  ollama pull nomic-embed-text\n"
            f"        B) llama.cpp: llama-server -m <embed>.gguf --embeddings --port 8081\n"
            f"           e passa backend='llamacpp', base_url='http://localhost:8081/v1'\n"
        )
    vecs = [d["embedding"] for d in resp.json()["data"]]
    return np.array(vecs, dtype=np.float32)


# --- Ingestione e chunking ---------------------------------------------------
def _chunk(text, size=800, overlap=120):
    text = text.strip()
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def ingest(manifest, size=800, overlap=120):
    """manifest: lista di {'path': ..., 'provenienza': 'ancorato'|'inferenza'}.
    'path' può essere un file o un glob. Solo .txt/.md (converti PDF/trascrizioni
    a testo prima). Restituisce (chunks, meta) allineati per indice.
    """
    chunks, meta = [], []
    for src in manifest:
        prov = src["provenienza"]
        if prov not in PROVENIENZE:
            raise ValueError(f"provenienza non valida: {prov!r}; usa {PROVENIENZE}")
        for path in sorted(glob.glob(src["path"])):
            with open(path, "r", encoding="utf-8") as f:
                for j, c in enumerate(_chunk(f.read(), size, overlap)):
                    chunks.append(c)
                    meta.append({"source": os.path.basename(path),
                                 "provenienza": prov, "idx": j})
    return chunks, meta


# --- Store su disco ----------------------------------------------------------
def build_store(manifest, prefix="corpus", **kw):
    chunks, meta = ingest(manifest, **{k: kw[k] for k in ("size", "overlap") if k in kw})
    emb = embed(chunks, **{k: kw[k] for k in ("model", "backend", "base_url") if k in kw})
    np.savez_compressed(f"{prefix}.npz", emb=emb)
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "meta": meta}, f, ensure_ascii=False)
    print(f"Store '{prefix}': {len(chunks)} chunk, dim {emb.shape[1]}.")
    return prefix


def load_store(prefix="corpus"):
    emb = np.load(f"{prefix}.npz")["emb"]
    with open(f"{prefix}.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    return emb, d["chunks"], d["meta"]


# --- Recupero ----------------------------------------------------------------
def _cos(q, M):
    q = q / (np.linalg.norm(q) + 1e-9)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return Mn @ q


def retrieve(query, store, k=5, **kw):
    emb, chunks, meta = store
    qv = embed([query], **{kk: kw[kk] for kk in ("model", "backend", "base_url") if kk in kw})[0]
    scores = _cos(qv, emb)
    top = np.argsort(-scores)[:k]
    return [{"chunk": chunks[i], "score": float(scores[i]), **meta[i]} for i in top]


def format_context(hits) -> str:
    """Formatta i chunk recuperati per l'iniezione, etichettati per provenienza,
    così lo specchio non spaccia inferenza per sapere ancorato."""
    blocchi = []
    for h in hits:
        tag = "sapere ancorato" if h["provenienza"] == "ancorato" else "inferenza plausibile"
        blocchi.append(f"[CONTESTO — {tag} · {h['source']}]\n{h['chunk']}")
    return "\n\n".join(blocchi)


if __name__ == "__main__":
    # Esempio: costruzione store + recupero.
    manifest = [
        {"path": "corpus_ancorato/*.txt", "provenienza": "ancorato"},
        {"path": "corpus_inferenza/*.txt", "provenienza": "inferenza"},
    ]
    build_store(manifest, prefix="corpus")
    store = load_store("corpus")
    for h in retrieve("una query di prova", store, k=3):
        print(f"{h['score']:.3f} [{h['provenienza']}] {h['source']}: {h['chunk'][:60]}...")
