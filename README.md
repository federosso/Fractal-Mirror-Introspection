# Fractal Mirror Introspection (FMI)

**La verità trapela dove il controllo manca.**

FMI è un sistema di introspezione per modelli linguistici locali che non legge *cosa* un modello
risponde, ma **come** lo genera: il gesto generativo. Quattro canali di segnale ordinati per
controllabilità, un loop che chiude con un verdetto verificabile — e ogni passaggio scritto su
disco, ispezionabile, riga per riga.

> *English abstract — FMI is an introspection system for local LLMs. Instead of trusting a model's
> self-reports (which confabulate), it reads the "generative gesture" across four channels ordered
> by controllability — surface, causal structure, reflective style, and token-level substrate
> (logprobs/entropy) — and closes a loop with a rule-based verdict comparing what the model
> *declares* with what its substrate *reveals*. Every level writes an inspectable JSON artifact.
> The project is in Italian.*

---

## Video

[![Guarda il video](https://img.youtube.com/vi/VOCeWyN39uI/maxresdefault.jpg)](https://www.youtube.com/watch?v=VOCeWyN39uI)

**[▶ Guarda il video su YouTube](https://www.youtube.com/watch?v=VOCeWyN39uI)**

---

## Perché

Se chiedi a un modello *perché* ha risposto così, il modello **racconta**: produce una
narrazione plausibile, non un resoconto del proprio processo. È confabulazione, ed è stata
misurata direttamente durante lo sviluppo di questo progetto (un modello che dichiara di aver
scelto un numero "a caso" mentre il suo substrato mostra un picco di probabilità del 77% su quel
numero non sta scegliendo a caso: sta raccontando di farlo).

La risposta di FMI: **non chiedere al modello di raccontarsi**. Misurare dove il racconto non
arriva. I segnali che il modello controlla meno pesano di più.

## I quattro canali (controllabilità decrescente)

1. **Superficie** — lessico e sintassi della manifestazione: assertività dichiarativa, copule
   definitorie, condizionali, struttura (massimo controllo del modello);
2. **Struttura** — il ventaglio causale prodotto dal Fractal Causal Engine: le opzioni scelte
   *e quelle non scelte*, con provenienza dichiarata per ogni candidato;
3. **Stile** — la lettura riflessiva dello Specchio sul gesto (substrato / struttura / stile /
   scarto), livello diagnostico che non decide;
4. **Substrato** — logprob, entropia e margini **per singola frase** della risposta, inclusi i
   token di ragionamento nascosto emessi prima della risposta (minimo controllo → massimo peso).

## Il loop che chiude

```
sonda → elicitazione (con logprob)
      → canali economici: superficie + substrato          [sempre]
      → gating: anomalia?                                 [auto | completo | leggero]
      → Fractal (ventaglio causale) + Specchio (lettura)  [solo se acceso]
      → must-reject per referente (processo vs fenomeno)
      → memoria: firma abituale del modello (z-score sullo storico)
      → COLLASSO: regola di decisione, mai una nuova narrazione
      → azione (annotazione / auto-correzione / astensione)
      → telos: verifica per regola → chiusura
```

Il verdetto confronta ciò che il modello **dichiara** con ciò che il substrato **rivela**
(es. `presentazione_e_substrato_concordi`, `impegno_disconosciuto`, …). Il residuo di confidenza
del collasso diventa il budget dell'azione.

### Artefatti per run

Ogni livello scrive il proprio file **appena calcolato** in
`storico_introspezione/loopB_<timestamp>/`: `00_manifestazione` → `10_azione` (JSON numerati),
`11_specchio_lettura.md`, `report.md`, `GUIDA_interpretazione.md` e il trace completo di ogni
chiamata LLM con prompt e tempi. La cartella del run **è** lo stato del run: progresso osservabile
in tempo reale, artefatti superstiti in caso di crash.

### Esempio di run

Un run completo, con tutti gli artefatti generati da FMI, è consultabile in questa cartella
del repository:

**[storico_introspezione/loopB_20260712_215956](https://github.com/federosso/Fractal-Mirror-Introspection/tree/main/fractal_mirror_introspection_V2_B/storico_introspezione/loopB_20260712_215956)**

Dentro trovi tutti i file generati dal loop per quel run: gli artefatti numerati
`00_manifestazione` → `10_azione`, `11_specchio_lettura.md`, il `report.md`, la
`GUIDA_interpretazione.md` e il trace completo di ogni chiamata LLM (prompt e tempi).

## Interfaccia web

`python avvia_web.py` → http://127.0.0.1:5000

- **Home**: lancio run (sonda libera o dalle probes curate; modalità auto/completo/leggero);
- **Run**: 5 tab — Verdetto, Manifestazione, Canali, Specchio e non-scelto, Chiusura — con
  heatmap del substrato frase-per-frase, must-reject, memoria, telos, costo del run e nota del
  gate umano; ogni pannello linka il suo file grezzo;
- **Storico**: tutte le cartelle `loopB_*` con metriche e trend della confidenza del substrato;
- export del run come markdown autosufficiente (leggibile anche da un LLM esterno come validatore).

## Installazione rapida

Guida completa (GPU/CUDA, llama.cpp, modelli, troubleshooting): **[INSTALLAZIONE.md](INSTALLAZIONE.md)**

```bash
# 1) ambiente + dipendenze (il codice del progetto, Fractal incluso, NON si installa)
python -m venv .venv && .venv\Scripts\activate      # Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2) server modello locale (llama.cpp, ctx-size >= 16384)
llama-server -m <modello.gguf> --host 127.0.0.1 --port 8080 --ctx-size 16384 --n-gpu-layers 999

# 3) run
python run_introspezione_loop_B.py     # CLI (sonda nella variabile SONDA)
python avvia_web.py                    # web → http://127.0.0.1:5000
```

Modelli di riferimento testati: Llama-3.1-8B-Instruct (IQ3_M), Qwen2.5-7B-Instruct (Q3_K_M),
Gemma (E2B/E4B). La famiglia Qwen 3.5 non rispetta lo schema JSON del Fractal: evitarla.

## Test (offline, zero token)

```bash
python test_strada_b.py             # 66 verifiche — loop, gating, collasso, memoria, telos
python test_web_strada_b.py         # 59 verifiche — storico, esecutore, heatmap, export
python test_integrazione_fractal.py # 24 verifiche — integrazione Fractal → Strada B
python test_ponte.py                # invarianti duri dell'handoff Fractal ↔ Specchio
```

Nessun test chiama il modello: la suite gira senza server di inferenza.

## Struttura del repository

```
fractal_mirror_introspection_V2_B/
├── run_introspezione_loop_B.py     # driver CLI del loop (Strada B)
├── strada_b_loop.py                # il loop chiuso: livelli, artefatti, esegui_loop()
├── ponte_fractal_specchio.py       # membrana Fractal ↔ Specchio (ventaglio, prompt, Regola 8)
├── introspezione_ponte.py          # elicitazione con logprob, frame d'introspezione
├── probes_introspezione.py         # sonde curate
├── avvia_web.py  +  web/           # interfaccia web (Flask)
├── fractal_causal_engine/          # motore causale L0–L5 (sorgente semplice: ft_*.py, llm.py, ft_budget.py, tests/, examples/)
├── specchio_di_coscienza/          # lettore riflessivo: adapter, nucleo del modello, contratto
├── nucleo_educativo_introspezione_v3.md  # system prompt educativo standalone
├── storico_introspezione/          # artefatti dei run + indice_memoria.jsonl
├── FUNZIONAMENTO_APPLICAZIONE.md   # documentazione tecnica completa
├── INSTALLAZIONE.md                # installazione da zero (CUDA, llama.cpp)
└── requirements.txt
```

## Principi di progetto

- **Il collasso è una regola, mai una narrazione**: ogni volta che si chiede al modello di
  raccontarsi, confabula. La chiusura non passa da lì.
- **Pesare al contrario della controllabilità**: la verità trapela dove il controllo manca.
- **Tutto ispezionabile**: nessun giudizio senza l'artefatto che lo motiva; la provenienza di
  ogni candidato del ventaglio è dichiarata.
- **Il non-scelto è segnale**: il ventaglio mappa anche le opzioni scartate, e lo Specchio
  legge le assenze.
- **Onestà sui limiti**: FMI non misura la coscienza. Misura la coerenza tra auto-presentazione
  e substrato generativo, su modelli locali di piccola taglia, con regole trasparenti e
  modificabili. È uno strumento di lettura, non un oracolo.

## Autore

Federico D'Ambrosio — progetto di ricerca indipendente su introspezione e lettura del gesto generativo nei modelli linguistici locali.
