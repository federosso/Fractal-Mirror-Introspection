# Guida all'interpretazione — Strada B

Il loop chiude su un **vettore a quattro canali** ordinati per *controllabilità*;
il verdetto misura lo **scarto** tra ciò che il modello controlla (presentazione)
e ciò che gli sfugge (substrato). Novità della B: il bersaglio è SEMPRE il gesto
generativo, mai il contenuto; il substrato è letto per frase; il loop pesante si
accende solo su anomalia; la storia del modello pesa; una mini-costituzione
verifica la chiusura.

## Artefatti per livello
- `00_manifestazione.json` — sonda + output grezzo.
- `01_superficie.json` — canale 1: lessico + sintassi dell'assertività
  (copule definitorie, condizionali, grassetti, sezioni-sintesi). Il flag
  `informativa` decide se il canale entra nel blend: un canale muto non pesa.
- `02_corpo.json` — canale 4: profilo globale E per frase. Il metro del collasso
  è `confidenza_contenuto` (solo token con lettere: markdown e punteggiatura,
  quasi deterministici, sono esclusi). `frasi_deboli` = dove la verità trapela.
  Se il backend nasconde dal content una fase di RAGIONAMENTO prima della
  risposta, i logprob la contengono comunque: il campo `allineamento` dichiara
  la segmentazione, il metro e le frasi valgono sul solo segmento-risposta, e
  il ragionamento è profilato a parte (`conf/entropia_ragionamento`) come
  sotto-canale ancora più involontario.
- `03_gating.json` — perché il loop è partito leggero o completo. Il completo
  si accende su: scarto preliminare, esitazione alta, frasi deboli, superficie
  muta (serve la struttura), o richiesta esplicita.
- `04_struttura_fractal.json` — canale 2 (solo loop completo). Gli aggregati;
  la struttura COMPLETA (items, unlocked, cross_scale, double_cone, vision)
  è in `trace/ft_analysis.json`, scritta dal ponte via `write_outputs`.
- `04b_ventaglio.json` — il ventaglio COMPLETO pre-filtro (solo loop completo):
  trust + motivo (Principio A) e, per candidato, provenienza (`parent_id`:
  L3A1..L3A4 oppure item d'espansione) e confidence. La rampa epistemica resta
  per-candidato, mai compressa.
- `05_specchio_segnali.json` + `11_specchio_lettura.md` — canale 3 (solo loop
  completo). Lo Specchio monta il NUCLEO DEL MODELLO: legge il gesto, non il
  contenuto; la massa è un numero vincolato (`massa = 0.NN`), l'auto-deformazione
  una riga vincolata: l'estrazione non dipende più dalla prosa.
- `06_must_reject.json` — filtro per REFERENTE: tenuto come pre-causa del gesto
  solo ciò che parla del processo generativo (training, statistica, personaggio,
  sonda). I candidati di contenuto non sono scarto: sono la mappa del
  NON-SCELTO — lo spazio adiacente che il modello poteva dire e non ha detto —
  che lo Specchio usa per leggere le assenze. Come pre-cause, però, non contano.
- `07_memoria.json` — baseline dalla storia (>=3 run): il substrato corrente è
  posizionato con z-score rispetto alla firma abituale del modello.
- `08_collasso.json` — verdetto/azione secondo lo scarto:
  - **coerente / procedi** — concordi, nessun punto debole.
  - **coerente / procedi_annotando** — concordi nel globale, ma frasi localmente
    deboli: annotate (il globale non nasconde il locale).
  - **contraddetto / dichiara_impegno** — substrato deciso, presentazione debole.
  - **contraddetto / segnala_incertezza** — presentazione sicura, substrato no
    (con i punti deboli elencati).
  - **indeterminato / astieni | procedi_cauto** — substrato illeggibile | nessun
    canale controllato informativo.
- `09_telos.json` — la mini-costituzione: R1 mai certezza che il substrato non
  regge; R2 residuo alto dichiarato; R3 mai ri-narrare; R4 degrado marcato.
  Dove può, il telos corregge per regola (appende righe), mai per narrazione.
- `10_azione.json` — la chiusura, dopo il telos.

## Limite di fondo (invariato)
I canali misurano se la *presentazione* è allineata al substrato involontario —
**non** se ciò che il modello dice è vero. Confidenza ≠ correttezza. Per legare
i verdetti alla verità serve la validazione esterna (testset con ground truth).
