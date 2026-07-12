# Installazione da zero — Fractal Mirror Introspection

Guida completa: ambiente Python, motore causale, GPU/CUDA, server llama.cpp, avvio e verifica.
Testata su Windows; le varianti Linux sono indicate dove differiscono.

---

## 0 · Prerequisiti

| Cosa | Requisito |
|---|---|
| Python | **3.10 o superiore** (consigliato 3.11+) |
| GPU (consigliata) | NVIDIA con **driver aggiornato** (`nvidia-smi` deve funzionare). Il toolkit CUDA completo NON è necessario: i binari precompilati di llama.cpp includono/riportano il runtime |
| RAM/VRAM | i modelli di riferimento (7–8B quantizzati Q3/IQ3) girano in ~5–6 GB di VRAM; senza GPU llama.cpp funziona su CPU, solo più lento |
| Disco | ~5–10 GB per i modelli GGUF |

---

## 1 · Ambiente Python

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

## 2 · Dipendenze del progetto

```bash
cd fractal_mirror_introspection_V2_B
pip install -r requirements.txt
```

Fine. Il Fractal Causal Engine **non si installa**: come tutto il resto del
codice, è sorgente semplice nella radice del progetto (`fractal_causal_engine/`)
e viene importato direttamente. Nessun `pip install -e`, nessun package da
registrare, nessun problema di versioni installate in precedenza: il codice che
gira è sempre e solo quello nella cartella.

Verifica (dalla radice del progetto):
```bash
python -c "import fractal_causal_engine; print(fractal_causal_engine.__file__)"
```
Il path stampato deve puntare a `fractal_causal_engine/__init__.py` dentro il progetto.

## 3 · llama.cpp (server di inferenza, con CUDA)

### Windows
Opzione A — winget (semplice):
```powershell
winget install llama.cpp
```
Opzione B — binari precompilati CUDA: dalla pagina Releases di
https://github.com/ggml-org/llama.cpp scaricare l'archivio
`llama-<build>-bin-win-cuda-cu12.x-x64.zip` (e, se richiesto, il pacchetto
`cudart-llama-bin-win-cu12.x-x64.zip` da estrarre nella stessa cartella).
Serve solo il driver NVIDIA aggiornato.

### Linux
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON     # senza GPU: omettere il flag
cmake --build build --config Release -j
# binario: build/bin/llama-server
```

### Modello GGUF
Scaricare un GGUF instruct 7–8B (es. da Hugging Face) in una cartella modelli, es.:
- `Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf`
- `qwen2.5-7b-instruct-q3_k_m.gguf`

> Nota dai test del progetto: la famiglia **Qwen 3.5 non restituisce JSON**
> conforme allo schema del Fractal (L0–L4 classifica zero item) → evitarla.

## 4 · Avvio del server llama.cpp

> ⚠️ **Contesto**: per evitare troncamenti usare `--ctx-size 16384` (o 32768).
> Il contesto reale per backend llamacpp è il `--ctx-size` del server;
> `NUM_CTX` nei driver vale solo per backend ollama.

```bash
llama-server -m "C:\modelli\Llama_3_1_8B\Meta-Llama-3.1-8B-Instruct-IQ3_M.gguf" ^
  --host 127.0.0.1 --port 8080 --ctx-size 16384 --parallel 1 ^
  --n-gpu-layers 999 --batch-size 256 --ubatch-size 64 --flash-attn on --threads 4
```
(Linux: stesso comando con `\` come continuazione riga; senza GPU rimuovere `--n-gpu-layers`.)

Verifica: `http://127.0.0.1:8080/health` deve rispondere `{"status":"ok"}`.

## 5 · Avvio di FMI

```bash
# A) run da riga di comando (la sonda è la variabile SONDA nel file)
python run_introspezione_loop_B.py

# B) interfaccia web → http://127.0.0.1:5000
python avvia_web.py
```

Gli artefatti di ogni run finiscono in `storico_introspezione/loopB_<timestamp>/`.

## 6 · Verifica dell'installazione (offline, zero token)

```bash
python test_strada_b.py             # 66 verifiche
python test_web_strada_b.py         # 59 verifiche
python test_integrazione_fractal.py # 24 verifiche
python test_ponte.py                # invarianti dell'handoff
```

Tutte verdi = installazione corretta. I test non chiamano il modello: si può
eseguire questa verifica anche senza server llama.cpp attivo.

## 7 · Problemi comuni

| Sintomo | Causa / rimedio |
|---|---|
| `ModuleNotFoundError: fractal_causal_engine` | Script lanciato fuori dalla radice del progetto: eseguire i comandi da `fractal_mirror_introspection_V2_B/` |
| `ModuleNotFoundError: flask` (o `requests`) | `pip install -r requirements.txt` non eseguito nel venv attivo |
| Residui di vecchie installazioni del motore | (solo macchine con install precedenti) `pip uninstall fractal-causal-engine`: il sorgente locale ha comunque la precedenza negli script |
| Output troncati / JSON incompleti | `--ctx-size` troppo basso: portarlo a 16384 (o 32768) |
| Ventaglio Fractal vuoto (0 item in L0–L4) | Il modello non rispetta lo schema JSON (es. famiglia Qwen 3.5): cambiare modello |
| `connection refused` su 8080 | Server llama.cpp non avviato o porta diversa da quella configurata nei driver |
| GPU non usata | Driver NVIDIA non aggiornato o build llama.cpp senza CUDA: `nvidia-smi` durante l'inferenza deve mostrare il processo |
