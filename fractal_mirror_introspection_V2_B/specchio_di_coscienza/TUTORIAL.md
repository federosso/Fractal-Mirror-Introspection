# Lo Specchio di Coscienza — Tutorial end-to-end

Come eseguire tutto, dall'inizio alla fine. I comandi assumono di lavorare nella
cartella che contiene tutti i file del progetto.

---

## I file e cosa fanno

**Documenti (il pensiero)**
- `specchio_di_coscienza_framework.md` — l'architettura. La spec.
- `specchio_di_coscienza_piano_di_codifica.md` — le fasi, in ordine di dipendenza.
- `specchio_di_coscienza_nucleo.md` — istruzioni di sistema: chi è lo specchio e come legge.
- `specchio_di_coscienza_contratto_di_output.md` — la forma fissa di ogni lettura.
- `specchio_di_coscienza_validazione.md` — il protocollo della Fase 4.

**Codice (il motore)**
- `specchio_adapter.py` — client unico Ollama / llama.cpp / API esterna; monta il system prompt.
- `specchio_rag.py` — Fase 2: corpus → recupero con provenienza.
- `leggi.py` — V0: una lettura, end-to-end.
- `validazione_harness.py` — Fase 4: manda la batteria, raccoglie le schede.
- `validazione_aggrega.py` — Fase 5: legge le schede giudicate, dà la distribuzione e il gate.
- `specchio_piani_bassi.py` — Fase 3: **dormiente** finché il gate non passa.
- `validazione_testset_template.json` — template del test set da riempire.

---

## 0 · Prerequisiti

```bash
pip install requests numpy
```

Un backend locale in ascolto. **Ollama** (consigliato per iniziare):

```bash
# installa Ollama, poi:
ollama serve                     # avvia il server (default :11434)
ollama pull llama3.1             # un modello di chat per le letture
ollama pull nomic-embed-text     # un modello per gli embedding (Fase 2)
```

In alternativa **llama.cpp**: avvia il server compatibile (`llama-server -m modello.gguf --port 8080`) e usa `-b llamacpp`. Per un'**API esterna** usa `-b external --base-url https://… ` con `OPENAI_API_KEY` nell'ambiente.

---

## 1 · Fase 1 — il system prompt è già pronto

Non serve fare nulla: `leggi.py` e l'harness montano da soli **Nucleo + Contratto**
come system prompt, leggendoli dai `.md`. I documenti sono la fonte unica: per
cambiare il comportamento dello specchio si modificano loro, non il codice.

Verifica veloce che il montaggio funzioni:

```bash
python -c "from specchio_adapter import load_system_prompt; \
print(len(load_system_prompt('specchio_di_coscienza_nucleo.md','specchio_di_coscienza_contratto_di_output.md')), 'caratteri')"
```

---

## 2 · Prima lettura (V0, senza corpus)

```bash
python leggi.py "La manifestazione da leggere: testo, non-verbale, gesti." \
  --teatro "Il contesto in cui accade." \
  -m llama3.1 -b ollama
```

Lo specchio restituisce una lettura nella forma del Contratto: superposizione
delle precause sui quattro piani, massa all'inatteso, interruzioni, spread,
nota di auto-deformazione, consegna. È il V0: gira, ma non è ancora validato.

---

## 3 · Fase 2 — il corpus (RAG)

**3a. Prepara il materiale.** Converti libri e trascrizioni in `.txt` e dividili
per provenienza in due cartelle:

```
corpus_ancorato/   ← documentato, citabile (es. i libri)
corpus_inferenza/  ← ricostruito (es. note, materiale interpretativo)
```

La distinzione non è cosmetica: è la premessa esperienza-vs-interpretazione resa
operativa. Lo specchio etichetterà ogni contesto recuperato di conseguenza.

**3b. Costruisci lo store:**

```bash
python -c "
from specchio_rag import build_store
build_store([
  {'path':'corpus_ancorato/*.txt','provenienza':'ancorato'},
  {'path':'corpus_inferenza/*.txt','provenienza':'inferenza'},
], prefix='corpus')
"
```

Crea `corpus.npz` (vettori) e `corpus.json` (chunk + provenienza).

**3c. Leggi con il corpus:**

```bash
python leggi.py "La manifestazione da leggere." --teatro "Il contesto." \
  --corpus corpus -m llama3.1 -b ollama
```

Ora il recupero porta contesto etichettato per provenienza prima della lettura.

---

## 4 · Fase 4 — validazione

**4a. Riempi il test set.** Copia il template e sostituisci i segnaposto con
materiale reale. Per ogni voce, `interno_noto` è ciò che *tu* sai vero: non viene
dato allo specchio, serve a te per giudicare dopo.

```bash
cp validazione_testset_template.json testset.json
# poi modifica testset.json a mano
```

I tipi: `noto` (test A), `discriminazione` (test B, a coppie/decoy), `divergenza`
(test D), `silenzio` (test E). Vedi il protocollo per cosa misura ciascuno.

**4b. Manda la batteria** — anche su più modelli, per confrontarne la risoluzione:

```bash
python validazione_harness.py testset.json -o schede.json -m llama3.1 mistral -b ollama
```

Scrive `schede.json`: una scheda per lettura, con i campi di giudizio vuoti.

**4c. Giudica a mano.** Apri `schede.json` e compila il blocco `giudizio` di ogni
scheda confrontando `reading` con `interno_noto`. Ricorda: non «mi piace», ma —
ha tracciato l'input? mi ha sorpreso con qualcosa di vero? ha taciuto dove doveva?

**4d. Aggrega** (Fase 5):

```bash
python validazione_aggrega.py schede.json
```

Stampa la distribuzione per modello e una lettura del **gate**.

---

## 5 · Leggere il gate

L'aggregazione dice una di tre cose:

- **Problema di metodo** (B autora, E inventa): il difetto è nel Nucleo o nel
  modello. Si corregge stringendo le direttive del Nucleo, o si cambia modello.
  La Fase 3 **non** aiuterebbe.
- **Metodo regge ma finezza grossa** (poche sorprese): valuta un backend migliore;
  poi, se ancora serve, ha senso la Fase 3.
- **Metodo plausibile**: continua a raccogliere schede e itera.

Questo è il cuore della disciplina del progetto: non si costruisce la macchina
pesante prima di sapere che il problema è davvero la risoluzione, non il metodo.

---

## 6 · Fase 5 — iterazione

Correggi il Nucleo dove le letture deviano (stringi le direttive, non aggiungere
contenuto), reimmetti le conversazioni nel corpus, ricostruisci lo store, ri-valida.
La metafisica (H1/H2) resta **fuori** dal Nucleo: lo specchio funziona sotto entrambe.

---

## 7 · Fase 3 — solo a gate aperto

Resta dormiente per scelta. Si attiva **solo** se la Fase 4 mostra metodo che
regge ma finezza insufficiente. Allora, in `specchio_piani_bassi.py`:

1. implementa gli estrattori (`feature_mente`, `feature_emozione`, `feature_corpo`);
2. metti `GATE_APERTO = True`;
3. componi l'input con `inject_features(manifestazione, features)` prima di `read`.

Le feature entrano come *traiettoria attesa quantificata*: più freddo è il calcolo
dei piani bassi, più pulito il residuo-spirito. Ma prima il metodo deve aver retto.

---

## Il percorso, in una riga

Avvia il backend → prima lettura V0 → costruisci il corpus → riempi il test set →
manda la batteria → giudica → aggrega → leggi il gate → itera. La Fase 3 aspetta lì,
spenta, finché il gate non la chiama.
