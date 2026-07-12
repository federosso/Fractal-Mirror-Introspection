# Fractal Mirror Introspection — Funzionamento dell'applicazione

**Versione:** V2_B (Strada B) · **Data documento:** 12-07-2026
**Sostituisce:** `INVENTARIO_introspezione.md` (rimosso: descriveva i flussi V1/Strada A, ora eliminati)

---

## 1 · Cosa fa il sistema

FMI è un sistema di introspezione per modelli linguistici locali. Non legge *cosa* il modello
risponde, ma *come* lo genera: il gesto generativo. Il principio guida è che **la verità trapela
dove il controllo manca** — quindi i segnali sono ordinati per controllabilità decrescente, e il
collasso finale pesa al contrario della controllabilità: ciò che il modello controlla meno pesa di più.

Il loop **chiude**: non produce solo una lettura diagnostica, ma un verdetto sul rapporto tra ciò
che il modello *dichiara* e ciò che il suo substrato *rivela*, e da quel verdetto un'azione
(annotazione, auto-correzione, astensione) verificata da un telos. Il collasso è una **regola di
decisione sui segnali osservati, mai una nuova narrazione** (lì entra la confabulazione).

## 2 · Architettura del loop (Strada B)

Punto d'ingresso: `run_introspezione_loop_B.py` (CLI, variabile `SONDA`) oppure l'interfaccia web.
Entrambi chiamano `strada_b_loop.esegui_loop(...)`. Ogni livello scrive il proprio artefatto
**appena calcolato** in `storico_introspezione/loopB_<timestamp>/`: la cartella del run È lo stato
del run (progresso osservabile in tempo reale, artefatti superstiti in caso di crash).

| Fase | Artefatto | Cosa fa |
|---|---|---|
| Elicitazione | `00_manifestazione.json` | Il modello riceve la sonda con il self-system e si manifesta; i logprob vengono catturati (inclusi i token di ragionamento nascosti in inglese → `testo_ragionamento`) |
| Superficie (Canale 1) | `01_superficie.json` | Misura lessicale + sintattica dell'assertività dichiarativa (copule definitorie, grassetti, condizionali assenti). Un canale non informativo NON entra nel blend |
| Corpo (Canale 4) | `02_corpo.json` | Profilo del substrato **per-claim**: logprob allineati alle frasi visibili, token strutturali esclusi; il collasso vede *dove* il substrato è debole, non solo la media |
| Gating | `03_gating.json` | I canali economici (superficie + substrato) girano sempre; Fractal e Specchio si accendono solo su anomalia o su richiesta (`auto` / `completo` / `leggero`) |
| Struttura Fractal | `04_struttura_fractal.json` | Il Fractal Causal Engine (L0–L5) diverge sulla manifestazione inquadrata dal frame d'introspezione e produce il ventaglio dei candidati causali |
| Ventaglio pre-filtro | `04b_ventaglio.json` | Ventaglio COMPLETO con trust, provenienza per candidato (`parent_id`: attori unlocked L3A1–L3A4 vs espansioni) e confidence — la fonte di ogni candidato resta ispezionabile |
| Segnali Specchio | `05_specchio_segnali.json` + `11_specchio_lettura.md` | Lo Specchio legge con il nucleo rimappato sul modello (`specchio_del_modello_nucleo.md`: substrato / struttura / stile / scarto). Livello diagnostico: non decide |
| Must-reject | `06_must_reject.json` | Filtra i candidati per **referente** (processo generativo vs fenomeno descritto), non per nome di scala: le letture-di-sé impossibili vengono rigettate, non pesate |
| Memoria | `07_memoria.json` | Rilegge lo storico (`loopB_*` + `indice_memoria.jsonl`): il substrato del run è confrontato con la firma abituale del modello (z-score), non solo con soglie assolute |
| Collasso | `08_collasso.json` | La regola di decisione: confronta dichiarato vs substrato → verdetto (es. `presentazione_e_substrato_concordi`, `impegno_disconosciuto`, …) con confidenza e residuo. Include il gate `struttura non informativa` contro i falsi verdetti su contenuti filosofici |
| Telos | `09_telos.json` | Mini-costituzione che verifica l'azione e la corregge per regola, mai per ri-narrazione |
| Azione | `10_azione.json` | La chiusura: annotazione/auto-correzione allegata all'output, o astensione se il substrato non è leggibile. Il residuo del collasso è il budget dell'azione |
| Report | `report.md` + `GUIDA_interpretazione.md` | Sintesi leggibile del run e guida all'interpretazione |
| Trace | `trace/llm_calls/*.json`, `trace/telemetry.jsonl` | Ogni scambio con il modello (prompt, risposta, tempi) numerato: 0000 Elicitazione, 0001–0008 attori Fractal, 0009+ Expander, 9000 Specchio |

### I quattro canali (per controllabilità decrescente)
1. **Superficie** — lessico e sintassi della manifestazione (massimo controllo)
2. **Struttura** — il ventaglio causale del Fractal (candidati scelti e non-scelti)
3. **Stile/Specchio** — la lettura riflessiva del gesto
4. **Substrato** — logprob, entropia, margini per-claim (minimo controllo → massimo peso)

## 3 · Componenti e ruoli

### Radice del progetto (sottosistema di introspezione)
| File | Ruolo |
|---|---|
| `run_introspezione_loop_B.py` | Driver CLI del loop B (sonda in `SONDA`, backend llamacpp/ollama) |
| `strada_b_loop.py` | Il loop chiuso: tutti i livelli, i dataclass degli artefatti, `esegui_loop` |
| `ponte_fractal_specchio.py` | La membrana Fractal↔Specchio: `genera`, `estrai_ventaglio`, `serializza_ventaglio`, `componi_input`, `monta_system_prompt` (+ Regola 8 additiva) |
| `introspezione_ponte.py` | Elicitazione con logprob (`manifesta`), sintesi corpo-testo, `SELF_SYSTEM_DEFAULT`, `INTROSPECTION_FRAME` |
| `probes_introspezione.py` | Sonde curate (`PROBES`, `PROBES_BY_ID`) usate da CLI e web |
| `nucleo_educativo_introspezione_v3.md` | System prompt educativo standalone (v3, context-independent) per modelli esterni |
| `avvia_web.py` | Avvio dell'interfaccia web (config backend/modello, nucleo/contratto) |
| `_RUN_fractal_mirror_introspection_V2_B.txt` | Note operative: venv, install del package, comandi server llama.cpp, modelli testati |

### `fractal_causal_engine/` (motore causale, sorgente semplice)
Motore divergente L0–L5 (~30 moduli direttamente in `fractal_causal_engine/`): pipeline (`ft_pipeline`),
classifier, locked/unlocked (attori L3A: DomainKnowledge, CausalPrinciples, CrossDomainAnalogies,
OpenQuestions, GlobalSynthesis; validatore L3B CrossScale), expander L5, orchestrator, cliente LLM
unificato (`llm.py`: ollama / llamacpp / external), utilità IO/JSON. **`ft_budget.py`** centralizza
tutti i budget token degli attori con il `MOLTIPLICATORE` globale. Il motore NON si
installa: è sorgente semplice importato direttamente, come il resto del progetto
(CLI standalone: `python -m fractal_causal_engine.cli` dalla radice).

### `specchio_di_coscienza/` (lettore riflessivo, sottosistema autonomo)
File attivi per FMI: `specchio_adapter.py` (client di lettura + `load_system_prompt` +
`read_with_logprobs`), `specchio_del_modello_nucleo.md` (nucleo rimappato sul modello — quello che
la Strada B DEVE usare) e `specchio_di_coscienza_contratto_di_output.md` (contratto condiviso).
Il resto (framework, nucleo originale, piano di codifica, RAG, harness di validazione, tutorial)
è il progetto Specchio autonomo: conservato intatto.

### `web/` (interfaccia sulla Strada B)
`app.py` (route Flask), `esecutore.py` (job runner in thread, parametri identici al driver CLI),
`storico.py` (lettura run/indice), `esporta.py` (export markdown), template + `stile.css`.

## 4 · Funzionalità principali

1. **Loop introspettivo chiuso** con verdetto e azione motivata, in tre modalità: `auto`
   (gating su anomalia), `completo`, `leggero` (solo canali economici — verdetto comunque emesso).
2. **Lettura del gesto, non del contenuto**: nucleo rimappato (substrato/struttura/stile/scarto),
   must-reject per referente, frame d'introspezione che entra nel Fractal ma non nello Specchio.
3. **Substrato per-claim** con heatmap frase-per-frase e cattura del **ragionamento nascosto**
   (token inglesi pre-risposta ricostruiti dai logprob: canale di prima classe).
4. **Memoria autobiografica trasversale**: firma abituale del modello sullo storico, z-score,
   `indice_memoria.jsonl` aggiornato a ogni run.
5. **Artefatti progressivi e ispezionabili**: ogni livello scrive subito il suo JSON numerato
   (00–10) + letture MD; ogni chiamata LLM tracciata con prompt e tempi.
6. **Provenienza dichiarata dei candidati**: gli attori unlocked (L3A) entrano nel ventaglio con
   `parent_id` e cap per categoria (`_candidati_da_unlocked`); il ventaglio pre-filtro (`04b`)
   conserva la fonte di ogni candidato.
7. **Budget token centralizzati** in `ft_budget.py` (un solo file, `MOLTIPLICATORE` globale).
8. **Interfaccia web** (`python avvia_web.py` → http://127.0.0.1:5000):
   - *Home*: lancio run (sonda libera o dalle probes, modalità auto/completo/leggero);
   - *Run*: 5 tab tematici (Verdetto, Manifestazione, Canali, Specchio e non-scelto, Chiusura),
     heatmap del substrato, must-reject, memoria, telos, costo del run, nota del gate umano;
   - *Storico*: tutte le cartelle `loopB_*` con metriche e trend della confidenza del substrato;
   - *Introduzione*: pagina esplicativa;
   - progresso dei run letto dagli artefatti reali su disco (non simulato);
   - ogni pannello linka il file grezzo; export del run come markdown autosufficiente.
9. **Suite offline a zero token**: `test_strada_b.py` (66), `test_web_strada_b.py` (59),
   `test_integrazione_fractal.py` (24), `test_ponte.py` (invarianti duri dell'handoff).

## 5 · Come si esegue (sintesi)

```
# 1) server modello locale (llama.cpp, ctx-size 16384 consigliato — vedi _RUN txt)
# 2) venv attivo, dipendenze (il codice, Fractal incluso, non si installa):
pip install -r requirements.txt
# 3) run CLI:
python run_introspezione_loop_B.py        # sonda nella variabile SONDA
# 4) oppure web:
python avvia_web.py                       # http://127.0.0.1:5000
# 5) test offline (zero token):
python test_strada_b.py && python test_web_strada_b.py && python test_integrazione_fractal.py && python test_ponte.py
```

---

## 6 · Registro della pulizia e del rename (12-07-2026)

### 6.1 Rename
`fractal_causal_engine_v10_19_3/` → **`fractal_causal_engine/`**. Puntamenti aggiornati in:
`run_introspezione_loop_B.py`, `avvia_web.py`, `test_strada_b.py`, `test_web_strada_b.py`,
`test_integrazione_fractal.py`, `test_ponte.py`, `_RUN_fractal_mirror_introspection_V2_B.txt`.
Nessun riferimento residuo al vecchio nome fuori da `storico_introspezione/` (i log storici dei
run passati sono record e non sono stati toccati). Successivamente la struttura è stata **appiattita**: il package vive direttamente in
`fractal_causal_engine/` (con `tests/` ed `examples/` come sottocartelle) e viene importato
come sorgente semplice — il layout installabile (`src/`, `pyproject.toml`) è stato rimosso.

**Verifica di sicurezza**: dopo l'appiattimento `fractal_causal_engine/` È il package
(con il suo `__init__.py` nella radice del progetto): gli entry point non hanno più bisogno di
`FRACTAL_SRC` nel `sys.path` — basta la radice, già presente. Anche in presenza di vecchie
installazioni del motore, la radice precede `site-packages`, quindi gira sempre il sorgente
locale. Verificato a runtime con le suite in un venv vergine (solo flask+requests).

**⚠ Sulla macchina di lavoro**: il motore non richiede più alcuna installazione; se esiste
una vecchia installazione editable, rimuoverla con `pip uninstall fractal-causal-engine`
(nota aggiunta anche nel `_RUN` txt).

### 6.2 File eliminati

**Linea V1 / Strada A (superata dalla Strada B):**
- `run_introspezione.py` — driver lineare V1 (usava `introspetta` e il nucleo originale)
- `run_introspezione_loop.py` — driver del loop Strada A
- `strada_a_loop.py` — il loop Strada A
- `test_strada_a.py`, `test_introspezione.py` — test dei flussi eliminati
- `run_serie.py`, `run_serie._GROQ.py` — driver serie del ponte-base V1
- `demo_ventaglio_acceso.py`, `mostra_raw.py` — ausiliari V1
- `diagnostica_ventaglio.py` — diagnostica del ventaglio, marcata OLD V1 nel `_RUN` txt
  (i suoi esiti storici restano nei log in coda al txt)
- `ft_logger.py` — DialogLogger: dopo la pulizia nessun modulo attivo lo importa
  (in `ponte_fractal_specchio.genera` il logger è un parametro opzionale, resta compatibile)

**Documentazione superata:**
- `INVENTARIO_introspezione.md` — inventariava i flussi V1/A eliminati; sostituito da questo documento
- `nucleo_educativo_introspezione.md`, `_v2.md` — superati dalla v3 (unica context-independent)

**Backup e artefatti generati (dentro l'engine):**
- `json_utils_BAK_OLD.py`, `llm_BAK_OLD.py`, `llm_BAK_OLD_KO_Groq_Rate_Limit.py`
- `.pytest_cache/`, `__pycache__/`, `fractal_causal_engine.egg-info/` (path del vecchio layout `src/`)
  (non più necessari: il motore non è più un package installabile)

**Web:** `web/static/stile_old.css` (non referenziato da alcun template).

**`_RUN` txt aggiornato:** rimossi il blocco comandi OLD V1 (file eliminati) e la riga su
`web_OLD_V1/` (cartella non presente nel repo); conteggio test web 29→59; aggiunta la nota di
pulizia/rename con l'avviso sul force-reinstall.

### 6.3 File conservati deliberatamente
- `introspezione_ponte.py`: contiene ancora `introspetta` (flusso lineare, ora orfano), ma
  `manifesta`, `sintetizza_corpo`, `SELF_SYSTEM_DEFAULT` e `INTROSPECTION_FRAME` sono attivi
  nella Strada B → il file resta intero (nessun refactor non richiesto).
- `test_ponte.py`: testa invarianti di `ponte_fractal_specchio`, che è attivo.
- `specchio_di_coscienza/` al completo: sottosistema autonomo con proprio `_RUN.txt` e tutorial;
  eliminare RAG/validazione/framework avrebbe toccato un progetto a sé, non "file vecchi" di FMI.
- `fractal_causal_engine/examples/` e i moduli engine non usati da FMI (cli, book_runner,
  thematic, …): parte del package installabile, non zavorra del progetto.

### 6.4 Verifiche post-pulizia
- `py_compile` su tutti i `.py` conservati: **OK**
- `test_strada_b.py`: **66/66 verdi** · `test_web_strada_b.py`: **59/59 verdi**
- `test_integrazione_fractal.py`: **24/24** · `test_ponte.py`: **invarianti duri rispettati**
- grep sul vecchio nome cartella: nessun residuo attivo.
