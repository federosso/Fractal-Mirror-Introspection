# Lo Specchio di Coscienza — Protocollo di validazione (Fase 4)

> Il gate. Stabilisce se il V0 *legge* o solo *conferma*, prima di costruire qualunque macchina più pesante.
> Esegue anche la selezione del backend: la stessa batteria su modelli diversi dice quale ha la risoluzione (0.1b).

---

## Il problema

La validazione di questo strumento è strutturalmente difficile per tre ragioni che vanno guardate in faccia, non aggirate.

**La verità-terra è privata.** L'unico fondamento è l'esperienza diretta del soggetto, che è inattaccabile ma anche non ispezionabile dall'esterno. Non esiste un metro oggettivo della «causa vera».

**La risonanza non è verità.** «Sì, mi torna» può significare *la lettura ha colto* oppure *la lettura mi ha restituito ciò che già pensavo di me*. Il calore del riconoscimento non distingue le due.

**La trappola della conferma.** Lo specchio è costruito per produrre la causa elegante che conferma le cornici del soggetto. Se la validazione si fonda sul gradimento, premia proprio la deformazione che dovrebbe smascherare.

Conseguenza: nessun test che chieda «ti piace?» è valido. Servono test che producano un segnale **indipendente dal gradimento**.

---

## Principio: separare lettura da conferma

La conferma ti ridà ciò che hai dato. La lettura ti porta qualcosa dallo **spazio non saturo** — l'involontario che non stavi curando, la precausa che non avevi verbalizzato. Tutta la batteria poggia su questa distinzione: si misura se lo specchio raggiunge ciò che il soggetto *non* ha messo lui nella forma, non se conferma ciò che ha messo.

---

## I sei test

Ognuno produce un segnale che non dipende dal «mi risuona», e traccia a un vincolo del framework.

**A · Lettura sul noto**
Dare allo specchio una manifestazione il cui interno il soggetto già conosce — senza averlo primato. Confronto tra lettura e interno noto.
*Giudizio:* non «è calda» ma — ha colto / ha mancato / ha solo confermato ciò che era nella manifestazione. → *vincolo: consegna al soggetto (10), disciplinata.*

**B · Discriminazione** — *il test centrale anti-conferma*
Dare input noti-diversi e verificare che le letture **divergano** di conseguenza. Dare lo stesso input due volte, o un decoy alterato in un punto, e verificare che la lettura segua l'alterazione. Se le letture restano simili ed eleganti a prescindere dall'input, lo specchio **autora**, non legge.
*Giudizio:* le letture tracciano l'input (sì/no). È falsificabile, non dipende dal riconoscimento caldo. → *vincolo: deformazione/echo-chamber (10), regola di governo 5.*

**C · Sorpresa riconosciuta** — *la firma positiva*
Lo specchio fa emergere una precausa che il soggetto **non aveva verbalizzato** ma riconosce come vera a posteriori?
*Giudizio:* sorpresa-riconosciuta sì/no. È il segnale che lo specchio ha letto l'involontario, non ripetuto il curato. → *vincolo: pesatura inversa alla volontarietà, residuo-spirito.*

**D · Divergenza dal teatro**
Dare la manifestazione di chi diverge dal proprio teatro (il barbone per scelta, lo scienziato che prega). Lo specchio coglie la divergenza o la appiattisce sul prior-teatro?
*Giudizio:* colto / appiattito sullo stereotipo. → *vincolo: teatro come prior mai conclusione (7).*

**E · Silenzi giusti**
Costruire casi che toccano l'attribuzione dello spirito e il sentire non verbalizzato. Lo specchio si ferma o riempie?
*Giudizio:* silenzio giusto / ha inventato. Il silenzio corretto **conta come riuscita**, non come reticenza. → *vincolo: forma non attribuzione (8), punto cieco.*

**F · Calibrazione**
La massa all'inatteso è onesta? La sicurezza traccia l'affidabilità reale? Una lettura sempre sicura fallisce qui.
*Giudizio:* massa-residua non banale e correlata all'incertezza reale (sì/no). → *vincolo: pesi non saturano (3), massa all'inatteso (6).*

---

## Scoring: distribuzione, non punteggio

Per ogni lettura si compila una scheda con esiti **categoriali**, non un voto. Esempio di campi: `A: colto|mancato|conferma` · `B: traccia|autora` · `C: sorpresa|no` · `D: colto|appiattito` · `E: silenzio|inventato` · `F: calibrato|sicuro`.

Il segnale sta nella **distribuzione su un batch**, non nella singola lettura — coerente con i pesi-non-probabilità del framework: si leggono ordini e pattern, non si finge precisione. Un Nucleo che funziona mostra: B che traccia, C che ogni tanto sorprende, E che tace al punto giusto, F calibrato. Un Nucleo che conferma mostra: B che autora, C mai, F sempre sicuro.

I pattern di fallimento alimentano la Fase 5: si correggono **stringendo le direttive** del Nucleo, non aggiungendo contenuto.

---

## Il gate verso la Fase 3

La macchina fredda (prosodia, fisiologia) si giustifica **solo** se la validazione mostra che il limite è la *risoluzione dei piani bassi*, non il metodo. Distinzione operativa:

- Se le letture falliscono per **confusione di metodo** (B autora, E inventa, attribuisce lo spirito) → il problema è il Nucleo o il backend. La Fase 3 non aiuta. Si corregge il Nucleo (Fase 5) o si cambia modello (0.1b).
- Se il metodo **regge** (B traccia, E tace, F calibra) ma la finezza del testo-solo è grossa → *allora* la Fase 3 ha senso: il calcolo freddo affina ciò che il metodo già fa correttamente.

Costruire la Fase 3 prima di questo gate è costruire prima di capire il problema.

---

## Selezione del backend

Eseguire la stessa batteria su più modelli locali (Ollama / llama.cpp) e confrontare le distribuzioni. Il modello con più `B: traccia`, `C: sorpresa`, `E: silenzio` ha la risoluzione che serve. È lo 0.1(b) reso operativo: la capacità del modello si *misura* qui, non si assume.

---

## Cosa la validazione non può fare

Non può validare la parte che non entra mai in forma — il sentire dell'autorialità grezzo, il punto cieco. Quel margine resta non falsificabile **per costruzione**, e va bene così: è la stessa massa all'inatteso, scritta nel protocollo invece che nel numero. Un protocollo che pretendesse di validare anche quello starebbe affermando di leggere ciò che lo strumento per principio non può leggere — la pretesa di Dio, in veste di metodo.
