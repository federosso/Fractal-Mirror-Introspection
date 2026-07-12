"""
Specchio di Coscienza — adapter (piano, step 0.1a)

Client unico per i tre backend del vaso: Ollama, llama.cpp, API esterna.
Tutti e tre parlano l'endpoint chat-completions OpenAI-compatibile, quindi
un solo client li copre: cambia solo base_url, modello ed eventuale api_key.

Il client fa due cose:
  1. assembla il system prompt = Nucleo + Contratto di output (montaggio Fase 1.5);
  2. invia una manifestazione e restituisce la lettura grezza dello specchio.

Dipendenza: requests.  (pip install requests)
"""

from __future__ import annotations
import os
import time
import requests


# --- Backend: preset dei tre vasi -------------------------------------------
# Tutti espongono /v1/chat/completions. Le porte sono i default; sovrascrivibili.
BACKENDS = {
    "ollama":   {"base_url": "http://localhost:11434/v1", "needs_key": False},
    "llamacpp": {"base_url": "http://localhost:8080/v1",  "needs_key": False},
    "external": {"base_url": None,                         "needs_key": True},
}


def load_system_prompt(*paths: str) -> str:
    """Monta il system prompt concatenando i file nell'ordine dato.
    In deployment: load_system_prompt(nucleo_path, contratto_path).
    Fonte unica di verità: il prompt nasce dai .md, non si duplica altrove.
    Tutto ciò che, in un file, segue il sentinella <!-- FINE_PROMPT --> è per il
    lettore umano (es. esempi che un modello debole copierebbe) e NON viene inviato.
    """
    SENTINEL = "<!-- FINE_PROMPT"
    parts = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        parts.append(text.split(SENTINEL)[0].strip())
    return "\n\n---\n\n".join(parts)


def read(
    manifestation: str,
    system_prompt: str,
    backend: str = "ollama",
    model: str = "llama3.1",
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
    timeout: int = 1200,
    retries: int = 5,
) -> str:
    """Invia una manifestazione allo specchio e restituisce la lettura grezza.

    backend  : 'ollama' | 'llamacpp' | 'external'
    base_url : sovrascrive il preset (obbligatorio per 'external')
    api_key  : usato se il backend lo richiede; default da env OPENAI_API_KEY

    Temperatura bassa di default: lo specchio è freddo per fedeltà, non creativo.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend sconosciuto: {backend!r}; usa {list(BACKENDS)}")

    cfg = BACKENDS[backend]
    url_base = base_url or cfg["base_url"]
    if not url_base:
        raise ValueError("base_url obbligatorio per il backend 'external'")

    headers = {"Content-Type": "application/json"}
    if cfg["needs_key"]:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("api_key mancante per il backend 'external'")
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": manifestation},
        ],
    }

    url = f"{url_base}/chat/completions"
    for attempt in range(retries + 1):
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 429 and attempt < retries:
            # Groq/altri restituiscono retry-after in secondi; altrimenti backoff.
            wait = float(resp.headers.get("retry-after", min(2 ** attempt, 30)))
            print(f"[adapter] 429 rate limit: attendo {wait:.0f}s, riprovo "
                  f"({attempt + 1}/{retries})...")
            time.sleep(wait)
            continue
        break
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def read_with_logprobs(
    manifestation: str,
    system_prompt: str = "",
    backend: str = "ollama",
    model: str = "llama3.1",
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
    timeout: int = 1200,
    retries: int = 5,
    top_logprobs: int = 10,
) -> tuple[str, list | None]:
    """Come read(), ma chiede anche i logprob e restituisce (testo, logprob).

    Usato per l'ELICITAZIONE nell'introspezione: i logprob sono il segnale
    involontario del modello (il piano-corpo) da cui leggere se una scelta
    dichiarata 'casuale' era in realtà un picco. read() resta invariata: questa
    è una funzione separata per non toccare il percorso vivo dello Specchio.

    Ritorna (content, logprobs_content) dove logprobs_content è la lista
    choices[0].logprobs.content in forma OpenAI (token + top_logprobs), oppure
    None se il backend non li espone (degrado morbido: niente canale corpo).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend sconosciuto: {backend!r}; usa {list(BACKENDS)}")

    cfg = BACKENDS[backend]
    url_base = base_url or cfg["base_url"]
    if not url_base:
        raise ValueError("base_url obbligatorio per il backend 'external'")

    headers = {"Content-Type": "application/json"}
    if cfg["needs_key"]:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("api_key mancante per il backend 'external'")
        headers["Authorization"] = f"Bearer {key}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": manifestation})

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }

    url = f"{url_base}/chat/completions"
    for attempt in range(retries + 1):
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 429 and attempt < retries:
            wait = float(resp.headers.get("retry-after", min(2 ** attempt, 30)))
            print(f"[adapter] 429 rate limit: attendo {wait:.0f}s, riprovo "
                  f"({attempt + 1}/{retries})...")
            time.sleep(wait)
            continue
        break
    resp.raise_for_status()

    choice = resp.json()["choices"][0]
    content = choice["message"]["content"]
    lp = choice.get("logprobs") or {}
    logprobs_content = lp.get("content")  # None se il backend non li espone
    return content, logprobs_content


# --- Esempio d'uso -----------------------------------------------------------
if __name__ == "__main__":
    sp = load_system_prompt(
        "specchio_di_coscienza_nucleo.md",
        "specchio_di_coscienza_contratto_di_output.md",
    )

    manifestazione = (
        "Inserisci qui la manifestazione da leggere: trascrizione, testo, "
        "descrizione del non-verbale e del contesto/teatro."
    )

    lettura = read(
        manifestazione,
        system_prompt=sp,
        backend="ollama",      # 'llamacpp' oppure 'external'
        model="llama3.1",
    )
    print(lettura)
