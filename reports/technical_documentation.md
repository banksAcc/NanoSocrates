# NanoSocrates — Guida Tecnica Estesa e Commentata

Questa guida nasce con un duplice obiettivo: fungere da manuale operativo per il team che sviluppa NanoSocrates e allo stesso
tempo introdurre i concetti chiave a chi si avvicina per la prima volta al progetto (persino uno studente di ingegneria al
primo anno). Il documento procede dal generale al particolare, intrecciando spiegazioni divulgative, richiami teorici di
Machine Learning/Deep Learning (ML/DL) e richiami puntuali al codice sorgente. Ogni sezione evidenzia le motivazioni delle
scelte progettuali, i compromessi effettuati e le possibili evoluzioni future.

Per orientarsi rapidamente:

1. **Fondamenti** — Cos'è un grafo RDF, perché ci serve un modello sequence-to-sequence (seq2seq) e come interpretiamo i task.
2. **Pipeline dei dati** — Dal download delle triple DBpedia alla costruzione dei dataset multitask.
3. **Architettura del modello** — Anatomia di TinySeq2Seq e dei layer ausiliari.
4. **Training & ottimizzazione** — Come orchestriamo dataloader, ciclo di apprendimento, scheduler.
5. **Valutazione & metriche** — Cosa misuriamo e come interpretiamo i numeri.
6. **Analisi critica** — Bug attuali, limiti teorici/pratici e piani di intervento.
7. **Ampliamenti** — Idee concrete per portare NanoSocrates al livello successivo.

---

## 1. Fondamenti e approccio progettuale

### 1.1 Perché RDF e perché il dominio cinematografico
- **RDF (Resource Description Framework)** è uno standard del W3C per rappresentare conoscenza come triple `(soggetto, predicato,
  oggetto)`. Una tripla può essere letta come una frase semplice: ad esempio `(The_Matrix, dbo:starring, Keanu_Reeves)` dice che
  Keanu Reeves recita in The Matrix. RDF facilita query semantiche e l'integrazione di fonti eterogenee.
- **Dominio cinematografico**: scelto perché DBpedia offre una base ampia e relativamente pulita di film. Il linguaggio naturale
  associato (plot, descrizioni, recensioni) è ricco e rende significativi i task di andata (testo→grafo) e ritorno (grafo→testo).

### 1.2 Obiettivo di NanoSocrates
NanoSocrates mira a unire quattro compiti correlati in un singolo modello seq2seq:

1. **Text2RDF** — Dato un testo (sinossi/descrizione), estrarre un grafo RDF coerente.
2. **RDF2Text** — Dato un grafo, generare un testo leggibile.
3. **RDF Completion 1** — Completare una tripla con oggetto mancante.
4. **RDF Completion 2** — Estendere un grafo parziale con triple aggiuntive.

Gestire più task con un unico modello evita mantenere architetture multiple e sfrutta l'apprendimento multi-task: la rete impara
pattern comuni (nomi di attori, generi, relazioni temporali) che aiutano tutti i task.【F:src/run.py†L1-L256】

### 1.3 Filosofia architetturale
L'eseguibile principale `src/run.py` è pensato come un orchestratore: carica configurazioni YAML, materializza i dataset,
costruisce modello, ottimizzatore, scheduler e avvia il training o la valutazione.【F:src/run.py†L1-L256】 La pipeline privilegia
la **modularità**: ogni fase è incapsulata in un file dedicato per favorire riuso, testabilità e sperimentazioni rapide.

---

## 2. Pipeline dei dati

La qualità dei dati è determinante per qualunque modello ML. In NanoSocrates la pipeline dati comprende acquisizione, pulizia,
fusione e tokenizzazione. Ogni step è pensato per essere idempotente (posso rilanciarlo senza effetti collaterali) e
configurabile.

### 2.1 Estrazione delle triple da DBpedia
- **Script di raccolta**: `scripts/fetch_dbpedia.py` avvia la procedura di download parametrizzando endpoint SPARQL, lista di
  predicati consentiti, limiti e direzione delle relazioni.【F:scripts/fetch_dbpedia.py†L1-L22】 Questi parametri vivono in file
  YAML per essere versionati insieme al codice.
- **Client SPARQL**: la funzione `fetch_triples` gestisce le query paginated con LIMIT/OFFSET, converte le IRI in prefissi
  abbreviati (`dbr:` per risorse, `dbo:` per proprietà) e supporta sia relazioni uscenti (dal film verso altre entità) sia
  entranti (entità che puntano al film).【F:src/data/dbpedia.py†L1-L103】
- **Perché limitare i predicati?** In ML vale il principio garbage in, garbage out: un eccesso di predicati rumorosi (es. pagine
  di discussione) complicherebbe il modello. La whitelist impone un vocabolario controllato.

### 2.2 Recupero dei testi da Wikipedia
Il grafo da solo non basta: ci serve il testo da cui estrarre conoscenza e verso cui rigenerare descrizioni.

- **Modalità inline**: `src/data/wikipedia.py` fornisce funzioni da richiamare direttamente in Python. Implementa retry con attese
  crescenti, fallback SPARQL per l'abstract se la summary REST non è disponibile, e normalizzazione dei titoli.【F:src/data/wikipedia.py†L1-L96】
- **Pipeline batch**: `scripts/fetch_wikipedia.py` usa cache su disco, multithreading e logging granulare per scaricare
  massicciamente le summary dei film trovati in DBpedia. Ogni record serializzato è `{film, text}` in JSONL (JSON lines).【F:scripts/fetch_wikipedia.py†L1-L199】
- **Glossario**: JSONL è un formato in cui ogni linea è un JSON indipendente. È pratico per dataset grandi perché possiamo
  streammare una riga alla volta senza caricare tutto in memoria.

### 2.3 Pairing e filtraggio di qualità
Una volta raccolte triple e testi, bisogna unirli in maniera coerente.

- `pair_and_filter` riceve generatori di triple e testi, costruisce un dizionario per film, deduplica le triple e rimuove i film
  con meno di tre triple o privi di descrizione.【F:src/data/pairing.py†L16-L56】
- **Bug noto**: nel ramo `dir == "in"` la funzione lascia il film come oggetto anziché soggetto, creando triple semanticamente
  invertite (es. `(attore, dbo:starring, film)` anziché `(film, dbo:starring, attore)`).【F:src/data/pairing.py†L23-L33】 Correggerlo
  è prioritario per evitare confusione nei task Text2RDF/RDF2Text.
- **Nota metodologica**: la deduplicazione aiuta a evitare di apprendere la stessa tripla più volte, riducendo overfitting e
  sbilanciamento.

### 2.4 Serializzazione e costruzione dei dataset
I modelli seq2seq lavorano su stringhe. Dobbiamo quindi trasformare le triple in sequenze lineari ordinate.

- **Linearizzazione**: `linearize` avvolge ogni tripla in tag speciali (`<SOT>`, `<SUBJ>`, `<PRED>`, `<OBJ>`, `<EOT>`) per
  segnalare le parti rilevanti al modello.【F:src/data/serialization.py†L9-L39】 Questo formato strutturato aiuta l'attenzione a
  distinguere ruolo e contesto.
- **Costruzione task-specifica**: `build_text2rdf`, `build_rdf2text`, `build_comp1`, `build_comp2` generano esempi adatti ai
  quattro task descritti sopra, aggiungendo token sentinella (`<Text2RDF>`, `<RDF2Text>`, `<MASK>`, `<CONTINUERDF>`) per spiegare
  al modello l'obiettivo corrente.【F:src/data/builders.py†L15-L76】
- **Aggregazione**: `build_and_cache_datasets` converte i JSONL in `Seq2SeqExample`, calcola pesi di sampling per task (ratio)
  e salva su disco cache-ready per run successive.【F:src/data/builders.py†L87-L178】

### 2.5 Tokenizzazione
- **Addestrare il tokenizer**: `train_bpe` applica Byte Pair Encoding sui testi e sulle linearizzazioni, includendo i token
  speciali e un post-processor per aggiungere `<pad>` dove necessario.【F:src/tokenizer/train_bpe.py†L1-L60】
- **Wrapper runtime**: `TokWrapper` incapsula il tokenizer addestrato, fornendo metodi sicuri per encode/decode e
  restituendo l'identificativo numerico del token di padding (fondamentale per il masking).【F:src/tokenizer/tokenizer_io.py†L1-L33】

---

## 3. Architettura del modello

### 3.1 Richiamo teorico: Transformer in breve
Un Transformer seq2seq è composto da un **encoder** (legge l'input) e da un **decoder** (genera l'output). Entrambi usano
meccanismi di **self-attention**, che permettono a ogni token di pesare gli altri token in modo dinamico. I Transformer sono
preferiti rispetto a RNN o LSTM perché gestiscono meglio dipendenze a lungo raggio e parallelizzano il calcolo.

### 3.2 TinySeq2Seq: struttura modulare
`TinySeq2Seq` è la nostra implementazione flessibile di un modello encoder–decoder. Supporta due famiglie principali di blocchi:

1. **Vanilla Transformer** — usa `nn.Transformer` di PyTorch oppure una versione personalizzata con Rotary Positional Embeddings
   (RoPE) e Multi-Linear Attention (MLA) per sperimentare tecniche moderne.【F:src/model/transformer.py†L1-L214】
2. **Variante T5-like** — replica alcune idee di Google T5: bias posizionali relativi, feed-forward GeGLU, layer norm con epsilon
   configurabile.【F:src/model/transformer.py†L215-L285】

Il `forward` calcola maschere causali (impediscono al decoder di vedere token futuri), scala gli embedding in stile T5, e
quando sono disponibili etichette calcola loss e metriche tramite `sequence_loss_with_span_metrics`.【F:src/model/transformer.py†L214-L285】【F:src/model/losses.py†L11-L133】

**Condivisione degli embedding**: per ridurre il numero di parametri e favorire il trasferimento tra task, è possibile
condividere la matrice di embedding tra encoder e decoder. Questa scelta è opzionale e configurabile, permettendo di
sperimentare sia con vocabolari completamente condivisi sia con dizionari distinti (utile se input e output hanno
distribuzioni molto diverse).

**Metriche sugli span**: durante i task di completion il modello può ricevere coordinate di span mascherati. La loss dedicata
calcola precision/recall su questi intervalli per monitorare la capacità di riempire le parti mancanti del grafo.

### 3.3 Layer ausiliari e sperimentazioni
`src/model/layers.py` contiene:
- Positional encoding sinusoidale (classico Transformer).
- Implementazione di RoPE, che ruota vettori nel piano complesso per mantenere informazioni posizionali continue.
- Strati di attenzione ibridi che interpolano tra attenzione classica e MLA per ridurre il costo quadratico su sequenze lunghe.
- Encoder/decoder stile T5 completi.【F:src/model/layers.py†L1-L188】

> **Digressione ML avanzata**: RoPE e MLA sono scelte che puntano a migliorare generalizzazione ed efficienza. RoPE permette di
> interpolare sequenze più lunghe rispetto a quelle viste in training. MLA (Ioffe et al., 2023) riduce il costo di calcolo
> approssimando la matrice di attenzione con fattori lineari. L'attuale implementazione consente di attivare/disattivare queste
> innovazioni da configurazione, ideale per esperimenti controllati.

---

## 4. Training e ottimizzazione

### 4.1 Dataset multitask e sampler
- `MultiTaskDataset` aggrega esempi etichettati con il tipo di task; ogni esempio rimane in memoria per accesso veloce.【F:src/training/dataloaders.py†L223-L327】
- `create_multitask_dataloader` costruisce un DataLoader PyTorch con `MultiTaskSampler`, che ripartisce ogni batch secondo i pesi
  (`ratios`) assegnati ai task.【F:src/training/dataloaders.py†L327-L470】
- **Criticità**: il sampler calcola le quote con `round(batch_size * ratio)` e può quindi superare il `batch_size` desiderato.
  Questo falsifica la stima degli step e sballa scheduler/gradient accumulation; dovremo implementare una ripartizione con floor
  e distribuzione del residuo o un taglio randomizzato degli eccessi.【F:src/training/dataloaders.py†L338-L371】
- `pad_collate` gestisce padding dinamico, maschere di attenzione e raccoglie gli eventuali span annotati per calcolare metriche
  sui token mascherati.【F:src/training/dataloaders.py†L382-L426】

### 4.2 Ciclo di training
`TrainingLoop` è responsabile di:
- Spostare i batch sul device corretto (CPU/GPU), attivare AMP (mixed precision) per performance migliori, gestire gradient
  accumulation e backpropagation.【F:src/training/loop.py†L1-L208】
- Aggiornare progress bar, loggare su Weights & Biases, salvare checkpoint e applicare early stopping.【F:src/training/loop.py†L208-L358】
- Alla fine di ogni epoca flushare eventuali gradienti residui per non perdere contributi.【F:src/training/loop.py†L200-L286】

**Scheduler di learning rate**: la funzione `create_scheduler` costruisce schedule step-based (cosine o linear warmup/decay).【F:src/training/scheduler.py†L9-L64】
Attualmente, però, `TrainingLoop.run` invoca `scheduler.step()` solo a fine epoca, riducendo la warmup a poche iterazioni.
Correggere questo comportamento significa spostare la chiamata dopo ogni `optimizer.step()` (tranne i casi `ReduceLROnPlateau`).【F:src/training/loop.py†L154-L167】

> **Digressione ML avanzata**: il learning rate scheduler è cruciale. Una warmup per-step permette al modello di stabilizzarsi
> nei primi aggiornamenti, evitando divergenza. Con gradient accumulation, contare gli step reali (batch effettivi/`grad_accum_steps`)
> è essenziale per programmare la curva correttamente.

### 4.3 Configurazione end-to-end
La funzione `run_training` collega tutti gli elementi: carica tokenizer, materializza dataset/cache, costruisce modello,
ottimizzatore, scheduler (usando `ceil(len(train_loader) / grad_accum_steps) * epochs` per calcolare gli step previsti), inizializza
W&B e avvia il ciclo sul device richiesto.【F:src/run.py†L164-L256】 Supporta anche modalità di overfit su singolo batch per debug.

---

## 5. Valutazione e metriche
`evaluate_from_config` ricostruisce modello e tokenizer dal checkpoint, crea DataLoader per split train/val/test e calcola loss
media. Le predizioni vengono generate in modalità greedy e aggregate per task.【F:src/eval/evaluate.py†L1-L337】

### 5.1 Metriche per task
- **RDF2Text**: metriche di generazione testuale (BLEU, ROUGE, METEOR) misurano n-gram overlap e similarità semantica.【F:src/eval/evaluate.py†L210-L220】【F:src/eval/metrics.py†L1-L200】
- **Text2RDF e RDF Completion 2**: precision, recall e F1 sulle triple, confrontando predizioni e ground truth.
- **RDF Completion 1**: accuracy esatta sull'oggetto mascherato.

### 5.2 Reporting e logging
Il report JSON prodotto include loss per split, metriche dettagliate e può essere appiattito tramite `flatten_eval_metrics` per
logging su W&B o analisi ulteriori.【F:src/utils/wandb_utils.py†L1-L120】 Questo consente di tracciare trend nel tempo e correlare
modifiche di codice a variazioni di performance.

---

## 6. Analisi delle performance, limiti e problematiche

### 6.1 Bug e criticità attuali
1. **Normalizzazione delle triple entranti** — il bug nel ramo `dir == "in"` produce triple invertite, creando inconsistenze tra
   training e valutazione. Fix prioritario: ribaltare i ruoli soggetto/oggetto e aggiornare i test.【F:src/data/pairing.py†L23-L33】
2. **MultiTaskSampler sovradimensionato** — la somma delle quote arrotondate può superare `batch_size`, alterando il numero di
   step e la dinamica del learning rate.【F:src/training/dataloaders.py†L338-L371】
3. **Scheduler step-based applicato per epoca** — riduce warmup e decay a scale errate; bisogna spostare l'aggiornamento per-step
   e riservare la logica per-epoca ai soli scheduler metric-based.【F:src/training/loop.py†L154-L167】【F:src/training/scheduler.py†L35-L64】

### 6.2 Limiti progettuali
- **Dependence dal dominio**: l'intero sistema è ottimizzato per film. Trasferirsi ad altri domini richiede rifare whitelist,
  heuristics di pairing e forse raddestrare il tokenizer.
- **Dimensione modello**: TinySeq2Seq è pensato per esperimenti rapidi. Su dataset più ampi potrebbe mancare capacità
  rappresentativa; bisognerà scalare numero di layer, dimensione embeddings o esplorare modelli pre-addestrati.
- **Bilanciamento task**: pesi statici possono non riflettere la difficoltà reale dei task (es. Text2RDF è intrinsecamente più
  complesso di RDF2Text). Senza meccanismi adattivi il modello può trascurare i task più ardui.
- **Assenza di data augmentation**: non esistono ancora strategie per generare varianti del testo o del grafo, limitando la
  robustezza contro rumore o ambiguità.

### 6.3 Sfide ML/DL affrontate
- **Allineamento input-output multi-modale**: testo e grafo hanno strutture molto diverse. L'uso di token sentinella e tag
  strutturati aiuta il modello a capire "cosa" sta guardando.
- **Catastrophic forgetting multi-task**: allenare un singolo modello su compiti diversi può portare a sovrascrivere conoscenza.
  Da qui la necessità di sampler bilanciati e logging separato per task.
- **Gestione dei gradienti**: con sequenze lunghe e GPU limitate serve gradient accumulation. Senza un conteggio preciso degli
  step, scheduler e early stopping possono prendere decisioni errate.

---

## 7. Ampliamenti e roadmap tecnica
Questa sezione propone interventi concreti, in parte per risolvere i bug noti, in parte per spingere NanoSocrates oltre lo stato
attuale.

### 7.1 Fix prioritari
1. **Normalizzazione triple** — Aggiornare `pair_and_filter`, aggiungere test unitari/integrazione che verificano la forma delle
   triple e rigenerare dataset di esempio per validare il comportamento.【F:src/data/pairing.py†L23-L54】
2. **Sampler batch-accurate** — Implementare allocazione con floor, distribuire il residuo secondo le quote e aggiungere test che
   garantiscano `len(batch) == batch_size` per ogni iterazione.【F:src/training/dataloaders.py†L338-L371】
3. **Scheduler per-step** — Chiamare `scheduler.step()` dopo ogni `optimizer.step()`, calcolare `total_steps` sugli aggiornamenti
   effettivi e conservare `ReduceLROnPlateau` per monitorare la loss di validazione.【F:src/training/loop.py†L154-L286】【F:src/training/scheduler.py†L35-L64】

### 7.2 Potenziamenti ML/DL
- **Curriculum Learning**: iniziare il training con task più semplici (es. RDF2Text) e introdurre progressivamente Text2RDF.
  Questo può ridurre l'instabilità iniziale.
- **Adapter layers / prompt tuning**: inserire adattatori leggeri per specializzare il modello per task specifici senza dover
  duplicare i pesi principali.
- **Knowledge distillation**: addestrare un modello teacher più grande (anche temporaneo) per generare pseudo-label e trasferire
  conoscenza a TinySeq2Seq.
- **Data augmentation**: per il testo, usare parafrasi controllate (es. back-translation). Per il grafo, generare triple sintetiche
  mantenendo consistenza logica (es. sfruttare ontologie DBpedia).

### 7.3 Osservabilità e valutazione avanzata
- **Curve di apprendimento**: sfruttare `flatten_eval_metrics` per loggare metriche per task a intervalli regolari.【F:src/utils/wandb_utils.py†L1-L120】
- **Analisi di errore**: costruire script che evidenziano mismatch frequenti (es. predicati più problematici).
- **Test automatici**: oltre ai fix, introdurre unit test per garantire che linearizzazioni e tokenizer rimangano retro-compatibili.

### 7.4 Scalabilità futura
- **Supporto multi-GPU**: estendere `TrainingLoop` con DDP (Distributed Data Parallel) per velocizzare training su dataset più
  grandi.
- **Inference ottimizzata**: integrare beam search configurabile e quantizzazione per deployment a bassa latenza.
- **Generalizzazione cross-domain**: progettare pipeline parametrica per importare domini diversi (es. musica, letteratura)
  semplicemente modificando i file YAML di raccolta dati.

---

## 8. Glossario rapido
- **AMP (Automatic Mixed Precision)**: tecnica che usa half precision dove possibile per velocizzare il training, mantenendo la
  precisione necessaria nei punti sensibili.
- **Gradient accumulation**: sommare gradienti su più mini-batch prima di aggiornare i pesi; utile quando la GPU non può contenere
  batch grandi.
- **Warmup**: periodo iniziale in cui il learning rate aumenta gradualmente per stabilizzare l'allenamento.
- **ROUGE/METEOR/BLEU**: metriche di similarità testuale; confrontano n-gram, ricorrenza di parole e allineamento semantico.

---

## 9. Conclusioni
Questo documento fornisce una panoramica completa e commentata di NanoSocrates, mettendo in luce tanto l'architettura attuale
quanto le aree di miglioramento. Seguire le raccomandazioni prioritarie (normalizzazione triple, sampler, scheduler) garantirà
una base più solida. Gli ampliamenti proposti offrono invece un percorso per trasformare NanoSocrates da prototipo avanzato a
piattaforma di ricerca e produzione robusta.
