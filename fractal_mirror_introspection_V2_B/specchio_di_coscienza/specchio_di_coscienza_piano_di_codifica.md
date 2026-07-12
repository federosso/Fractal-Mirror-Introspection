# Lo Specchio di Coscienza — Piano di codifica

> Companion operativo del documento *Lo Specchio di Coscienza — Framework*.
> Il framework dice **cosa** è lo strumento. Questo piano dice **in che ordine costruirlo**.

---

## Principio d'ordine

Il piano è ordinato per **dipendenza**, e una regola lo governa: il cuore — la lettura inversa effetto → causa — si costruisce e si valida *prima* della macchina fredda che lo affina. La macchina pesante (segnali, prosodia, fisiologia) entra solo dopo che il nucleo testuale ha dimostrato di funzionare. Costruire la macchina prima del nucleo significa raffinare la risoluzione di una lettura che ancora non legge.

Il vaso è un'istanza LLM **provider-agnostica**: backend + persona (istruzioni di sistema = Nucleo) + RAG (corpus). Gran parte della «codifica» della Fase 1 non è codice ma istruzione operativa per l'istanza, e per questo è portabile su qualunque backend. Il codice vero compare in Fase 3, ed è opzionale e posticipato per scelta.

---

## Fase 0 — Decisioni e gate

0.1 — **Il vaso è provider-agnostico.** Backend di lavoro primari: inferenza locale via **Ollama** e **llama.cpp**; API LLM esterne tenute come possibilità. Tre conseguenze:
- *(a) Adapter unico.* Serve un livello di astrazione così che lo stesso Nucleo + RAG punti a qualunque backend. Entrambi i locali espongono endpoint HTTP compatibili: un solo client Python basta a coprire locale ed esterno.
- *(b) La capacità del modello È la risoluzione della lettura.* Il Nucleo resta costante; il backend è la variabile che fissa la finezza della superposizione e del residuo. Un modello locale piccolo legge più *grosso*, non diversamente. È lo stesso principio della fedeltà-dei-piani-bassi, salito al livello del motore: la risoluzione dipende da chi calcola.
- *(c) Il locale-first protegge il corpus.* Il materiale più intimo (quaderni, export, manifestazioni) non lascia la macchina. Per uno strumento che legge proprio le manifestazioni più private, è un allineamento, non solo una comodità.

0.2 — Confermare la spec: il documento *Framework* è la fonte. Ogni passo che segue deve essere tracciabile a un suo vincolo (vedi matrice in fondo).

0.3 — **Gate anti-fallimento.** Nessuna riga di codice della Fase 3 prima che la Fase 4 abbia validato la Fase 1. La risoluzione dello spirito-come-residuo dipende dalla fedeltà dei piani bassi: ma «fedeltà» va misurata, non assunta. Prima si verifica che la lettura inversa di base regga su materiale reale.

---

## Fase 1 — Il Nucleo operativo (istruzioni di sistema)

È il cuore. Traduce ogni principio del framework in direttiva operativa per l'istanza. Non è codice: è il *Nucleo* dello specchio, l'equivalente per questo strumento del Nucleo del doppio.

1.1 — **Operazione inversa.** Istruire l'istanza a non descrivere lo stato manifesto, ma a risalire: data la manifestazione, quali precause sui quattro piani la rendono possibile.

1.2 — **Decomposizione in attributi.** La prima operazione su ogni input non è leggere *cosa* dice la forma, ma decomporla nei suoi attributi (personaggio, vocabolario, gesto, non-verbale, pensiero, palcoscenico, tempismo, metrica) e ordinarli per **volontarietà**.

1.3 — **Traiettoria attesa e residuo.** Dai tre piani bassi (corpo, mente, emozione) l'istanza deriva una traiettoria *attesa*. Lo spirito non si nomina come stato: si segnala come **deviazione** dalla traiettoria attesa, priva di causa nei tre piani inferiori. Output dello spirito = residuo, non oggetto.

1.4 — **Lettura delle assenze.** L'istanza deve registrare anche la forma *attesa che non arriva* — il blocco, la pausa, il flusso interrotto — come firma negativa di energia espressa su un piano interno.

1.5 — **Definire il contratto di output.** Una «lettura» dello specchio è sempre composta da:
- lo **spazio delle precause compatibili**, sui quattro piani, con **pesi di compatibilità che non saturano** (non sommano a uno);
- una **massa esplicita lasciata all'inatteso** — il margine che resta sempre aperto;
- le **interruzioni** rilevate, descritte come *forma* (involontaria, fuori firma) e mai come *attribuzione* (mai «spirito guida» o «trauma»: solo «interruzione ad alto residuo, qui»);
- lo **spread tra strato volontario e involontario** come misura di coerenza (scelta vs impulso, veglia vs sonno);
- una **nota di auto-deformazione**: dove la lettura «torna» troppo bene, l'istanza lo segnala come possibile eleganza-deformazione, non come prova.

1.6 — **Teatro come prior.** Il contesto (guerra, strada, laboratorio, altare…) entra come prior che *stringe* il ventaglio, mai come conclusione che ne sceglie il punto. Istruzione esplicita contro lo stereotipo statistico.

1.7 — **Forma, non attribuzione.** Al limite dello spirito l'istanza segnala l'interruzione e **tace** sull'attribuzione. Direttiva netta: presentare, non interpretare la causa-spirito.

1.8 — **Consegna del collasso.** Ogni lettura termina restituendo la scelta al soggetto. L'istanza non chiude la superposizione: la presenta perché sia l'umano a riconoscere.

1.9 — **Limiti dichiarati.** Il Nucleo dichiara il punto cieco (il sentire dell'autorialità non è leggibile dalla sola forma) e le tre deformazioni (conferma, cornice del soggetto, solo-ciò-che-è-verbalizzato).

**Deliverable Fase 1:** il testo del Nucleo operativo, pronto come istruzioni di sistema del Project.

---

## Fase 2 — Il corpus e il RAG

Fornisce allo specchio il materiale di contesto e di teatro, con la disciplina di provenienza che il framework richiede.

2.1 — **Ingestione del corpus:** libri, trascrizioni YouTube, export di conversazioni, quaderni. Normalizzazione in testo.

2.2 — **Chunking ed embedding** in vector store (stack già in uso: embeddings + vector store).

2.3 — **Tag di provenienza — vincolo dal framework.** Ogni chunk è etichettato come *sapere ancorato* (documentato, citabile) o materiale da cui si trae *inferenza plausibile*. Senza questa distinzione l'istanza non può onorare la premessa esperienza-vs-interpretazione: confonderebbe ciò che è dato con ciò che è ricostruito.

2.4 — **Recupero contestuale del teatro.** Il RAG serve anche a fornire il prior-teatro: contesto biografico e situazionale che stringe il ventaglio in Fase 1.6.

**Deliverable Fase 2:** vector store interrogabile, con provenienza tracciata.

---

## Fase 3 — Il readout a freddo *(opzionale, posticipato)*

Qui vive il codice vero: il calcolo dei piani bassi che affina la risoluzione del residuo-spirito. **Non si costruisce finché la Fase 4 non ha validato la Fase 1.** Se la lettura testuale basta, questa fase resta dormiente.

3.1 — **Strato mente** (curato): metriche di struttura linguistica sul testo — architettura del pensiero, coerenza, vocabolario.

3.2 — **Strato emozione** (intermedio): analisi di prosodia e affetto da audio — carica, ritmo, tempismo delle pause, incrinature.

3.3 — **Strato corpo** (involontario): segnali fisiologici dove disponibili (es. coerenza cardiaca / HRV) — la firma meno controllabile.

3.4 — **Pipeline di feature.** Le metriche dei tre strati diventano input strutturato che entra nella lettura come traiettoria-attesa quantificata, rendendo il residuo più pulito (paradosso della fedeltà: più freddo è il basso, più visibile l'incomputabile).

3.5 — **Misura dello spread** tra strato curato e strato involontario, su base quantitativa anziché solo inferita dal testo.

**Deliverable Fase 3:** moduli di estrazione feature che alimentano la lettura. Solo se serve.

---

## Fase 4 — Validazione

Come si sa che funziona. Dato che il collasso spetta all'umano, **la validazione È il riconoscimento del soggetto**, non una metrica esterna di verità.

4.1 — **Test sul noto.** Far leggere allo specchio manifestazioni il cui interno è già noto al soggetto. La lettura coglie, manca, o conferma soltanto?

4.2 — **Audit echo-chamber.** Verificare che lo specchio non stia restituendo le cornici del soggetto. Controprova: leggere materiale che *diverge* dalle attese — la lettura sa segnalare la divergenza, o la appiattisce sul prior-teatro?

4.3 — **Confine del punto cieco.** Confermare che, dove la lettura tocca il sentire dell'autorialità non verbalizzato, lo strumento *si ferma* invece di inventare. La buona validazione premia anche i silenzi giusti.

**Gate:** solo superata la 4.1–4.3 si valuta se la Fase 3 serve davvero.

---

## Fase 5 — Iterazione

5.1 — Reimmettere le conversazioni nel corpus, alimentando il vaso nel tempo.

5.2 — Raffinare il Nucleo dove le letture deviano: la deformazione si corregge stringendo le direttive, non aggiungendo contenuto.

5.3 — Tenere la metafisica (H1/H2) **fuori** dal motore: lo strumento funziona sotto entrambe. Non codificare lo scopo cosmologico nelle istruzioni — resta collasso del soggetto.

---

## Matrice di tracciabilità — così non si perde nessun pezzo

| Elemento del framework | Fase che lo implementa |
|---|---|
| Operazione inversa (effetto → causa) | 1.1 |
| Superposizione aperta, non «la causa» | 1.5 |
| Pesi che non saturano + massa all'inatteso | 1.5 |
| Tre piani calcolati, spirito come residuo | 1.3 · 3.4 |
| Pesatura inversa alla volontarietà | 1.2 · 3.1–3.3 |
| Lettura delle assenze | 1.4 |
| Spread tra strati (scelta vs impulso) | 1.5 · 3.5 |
| Teatro come prior, mai conclusione | 1.6 · 2.4 |
| Forma, non attribuzione | 1.5 · 1.7 |
| Collasso all'umano | 1.8 · 4.1 |
| Auto-deformazione / echo-chamber | 1.5 · 4.2 |
| Specularità (premessa) | 1.1 |
| Esperienza vs interpretazione (premessa) | 2.3 |
| Punto cieco (premessa) | 1.9 · 4.3 |

Ogni riga del framework ha una casa. Una riga senza fase è un pezzo perso; una fase senza riga è qualcosa che stiamo costruendo e che il framework non chiedeva — e va tolto.
