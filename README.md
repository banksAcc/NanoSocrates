# NanoSocrates — (Baseline unificata Tex to RDF e viceversa + Completion)

Questo repository ospita una pipeline **end-to-end** per addestrare un **Transformer encoder–decoder** capace di svolgere 4 task nel dominio _film_: **Text2RDF, RDF2Text, RDF Completion 1 (masked), RDF Completion 2 (continuation)**.  

---

## 1) Struttura del repository

```
nanosocrates/
├─ README.md
├─ requirements.txt              # dipendenze Python principali
├─ configs/                 
│  ├─ data/
│  │  ├─ build.yaml              # parametri per costruzione dataset
│  │  ├─ dbpedia.yaml            # endpoint, predicati, limiti SPARQL
│  │  ├─ toy.yaml                # remapping verso data/processed/toy
│  │  └─ wikipedia.yaml          # API REST, lingua, timeout
│  ├─ eval/
│  │  └─ multitask_default.yaml # preset di valutazione multitask (alias-driven)
│  ├─ tokenizer/
│  │  └─ bpe_default.yaml        # addestramento tokenizer + token speciali
│  └─ train/
│     ├─ multitask_default.yaml           # preset T5 più ampio
│     └─ multitask_default.yaml  # preset multitask T5 3:3:2:2
├─ data/                         # directory popolata dagli script (raw/interim/processed/vocab)
├─ scripts/
│  ├─ build_dataset.py           # crea dataset e task JSONL (richiede PYTHONPATH=src)
│  ├─ build_toy_subset.py        # genera il sottoinsieme toy (include setup PYTHONPATH interno)
│  ├─ eval_all.py                # wrapper compatibilità → src.run evaluate
│  ├─ fetch_dbpedia.py           # scarica triple DBpedia (richiede PYTHONPATH=src)
│  ├─ fetch_wikipedia.py         # scarica abstract intro (richiede PYTHONPATH=src)
│  ├─ predict_example.py         # inference minimale da riga di comando
│  ├─ sanity_overfit.py          # scorciatoia per l'overfit di un batch
│  ├─ inspect_mlm_batch.py       # stampa e valida un batch per il masked LM
│  ├─ split_by_film.py           # suddivide pairs JSONL in train/val/test per film
│  └─ train_tokenizer.py         # addestra il tokenizer BPE
└─ src/
   ├─ run.py                     # entrypoint unificato (train/overfit/evaluate/predict)
   ├─ data/                      # fetch DBpedia/Wikipedia, pairing, serializzazione
   ├─ decoding/                  # strategie di decoding e vincoli
   ├─ eval/                      # metriche e orchestratore valutazione
   ├─ model/                     # TinySeq2Seq, layer T5, perdite
   ├─ tokenizer/                 # wrapper IO e libreria per BPE
   ├─ training/                  # dataloader multitask, loop, scheduler
   ├─ utils/                     # config YAML, IO, logging, integrazione W&B
   └─ plots/curves.py            # placeholder per grafici (stub vuoto)
```

---

## 2) Quickstart

### 2.1 Ambiente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Pipeline base (dati → tokenizer → training → valutazione)

1. **Raccogli le sorgenti**

   ```bash
   python -m scripts.fetch_dbpedia --config configs/data/dbpedia.yaml --out data/raw/dbpedia_triples.jsonl

   python -m scripts.fetch_wikipedia --config configs/data/wikipedia.yaml --in data/raw/dbpedia_triples.jsonl --out data/raw/wikipedia_intro.jsonl
   ```

2. **Costruisci il dataset multi-task**

   ```bash
   python -m scripts.build_dataset \
       --config configs/data/build.yaml \
       --dbp data/raw/dbpedia_triples.jsonl \
       --wiki data/raw/wikipedia_intro.jsonl \
       --outdir data/processed \
       --emit_tasks
   ```

3. **Addestra (o aggiorna) il tokenizer**

   ```bash
   python -m scripts.train_tokenizer --config configs/tokenizer/bpe_default.yaml
   ```

4. **Avvia il training**

   ```bash
   python -m src.run train --cfg configs/train/multitask_default.yaml
   ```

   I log mostrano `loss` ed `exact_match` medi su train/validation. `exact_match`
   è pensato come sanity check rapido (sequenza generata identica al target).

5. **Valuta un checkpoint**

   ```bash
   python -m src.run evaluate \
       --cfg configs/train/multitask_default.yaml \
       --checkpoint checkpoints/multitask_default/best.pt \
       --splits validation test
   ```

   Per valutare su un file diverso basta indicare `--splits` (es. `train`).

### 2.3 Sanity check rapido / overfit di un batch

Per verificare rapidamente che il modello e i dati funzionino è disponibile un
comando dedicato (compatibile con `scripts/sanity_overfit.py`):

```bash
python -m src.run overfit --cfg configs/train/multitask_default.yaml --steps 60
```

Il comando applica automaticamente configurazioni adatte all'overfit (batch
singolo, niente dropout/weight decay, shuffle disattivato). L'opzione `--steps`
mappa al numero di epoche ripetute sullo stesso batch.

In alternativa puoi ottenere lo stesso comportamento invocando direttamente il
training con override manuali:

```bash
python -m src.run train \
    --cfg configs/train/multitask_default.yaml \
    --override limit_train_batches=1 shuffle_train=false num_epochs=50 dropout=0.0
```

Se la pipeline è corretta la `loss` scende velocemente (<1 dopo poche decine di
epoche). Per accorciare ulteriormente il test si può creare un sottoinsieme
ridotto copiando poche righe dei file `*.train.jsonl`/`*.val.jsonl` e puntando
il config a tali file tramite `--override datasets[0].train=...` ecc.

### 2.4 Suggerimenti pratici

* Esecuzione su CPU: `--override device=cpu batch_size=2 num_workers=0`.
* Cambiare dimensione batch / epoche: `--override batch_size=8 num_epochs=3`.
* Limitare la valutazione per debug: `--override limit_val_batches=2`.
* Salvare i checkpoint in una directory diversa: `--override save_dir=my_runs/test1`.
tokenizer e il `DataCollatorForLanguageModeling` generino batch coerenti prima di
avviare esperimenti di training. Per usarlo:

1. Assicurati di avere le dipendenze opzionali installate (almeno `transformers`,
   `datasets` e `torch`).
2. Esegui lo script come modulo in modo che il pacchetto `src/` sia risolto
   automaticamente nel `PYTHONPATH`:

   ```bash
   python -m scripts.inspect_mlm_batch --tokenizer data/vocab/bpe.json --text "Ciao mondo" --batch-size 2 --max-length 64
   ```

   Parametri utili:

   - `--tokenizer`: nome Hugging Face o percorso a una directory/file locale del
     tokenizer.
   - `--text`: stringhe raw che verranno tokenizzate (puoi ripetere il flag per
     accumulare più esempi).
   - `--text-file`: file di testo (uno per riga) da cui leggere ulteriori esempi.
   - `--toy`: carica automaticamente frasi di debug da
     `data/processed/toy/rdf2text.train.jsonl` (campo `target`), comodo per
     testare il flusso senza preparare input manuali; usa `--toy-path`,
     `--toy-field` e `--toy-sample` per personalizzare sorgente, colonna e numero
     di esempi.
   - `--batch-size`, `--max-length`, `--mlm-probability`: replicano i campi del
     dataloader e del collator.

3. L'output riporta gli ID dei token speciali e stampa `input_ids`, `labels` e
   `attention_mask`. Lo script solleva un errore se:

   - due token speciali condividono lo stesso ID;
   - le label sul padding non sono a `-100`;
   - la `attention_mask` non è 0/1 o non è allineata al padding.

### 2.7 Strumenti — audit dataset

`scripts/data_audit.py` fornisce una panoramica rapida delle distribuzioni nelle
triple e nelle sequenze dei task JSONL. Esegue automaticamente il parsing dei
`pairs.*.jsonl` (anche compressi) tramite `src.utils.io.read_jsonl` e riporta:

- top-20 predicati, oggetti ed entità soggetto;
- statistiche min/avg/max sulle lunghezze whitespace-token di `input` e `target`
  per ciascun file di task;
- percentuale di esempi che superano la soglia impostata con `--max-len`.

Esempio di utilizzo:

```bash
python -m scripts.data_audit --pairs data/interim --tasks-dir data/processed --max-len 256
```

Argomenti principali:

- `--pairs`: directory, file singolo o glob dei `pairs.*.jsonl` da analizzare;
- `--tasks-dir`: directory che contiene i JSONL dei task (opzionale);
- `--max-len`: soglia di lunghezza (in token separati da spazi) oltre la quale
  viene calcolata la percentuale di esempi fuori limite (default: 512).

---

## 3) Configurazione (YAML)

Vedi esempi in `configs/` per:

- `data/dbpedia.yaml` — endpoint SPARQL, whitelist predicati, direzione (out|both)
- `data/wikipedia.yaml` — lingua, endpoint REST, timeout
- `data/build.yaml` — split, maxlen, filtri qualità
- `data/toy.yaml` — percorsi del sottoinsieme 20-film per debug rapido
- `tokenizer/bpe_default.yaml` — vocab e token speciali
- `train/multitask_default.yaml` — modello, trainer, mixing task
- `decode/constrained.yaml` — vincoli leggeri per RDF
- blocco `wandb:` — parametri di logging (project, entity, run_name, mode, tags, watch)

### Architettura del modello

`TinySeq2Seq` implementa un encoder–decoder ispirato a T5:

- attenzioni multi-head con bias posizionali relativi a bucket;
- LayerNorm in configurazione pre-attention e feed-forward GeGLU;
- embedding condivise fra encoder e decoder scalate di `√d_model`.

I parametri esposti nei config (`d_model`, `nhead`, `enc_layers`, `dec_layers`,
`ff_dim`, `dropout`, `relative_attention_num_buckets`,
`relative_attention_max_distance`, `layer_norm_epsilon`, `max_len`) consentono di
calibrare dimensione e profondità del modello mantenendo fisso il design T5.

---

## 4) Fase Dati (Step 1–3) — Design logico e contratti I/O

### 4.1 DBpedia (SPARQL) — 1-hop filtrato

Raccogli le **triple 1-hop** per ogni `dbo:Film` usando una **whitelist** di predicati (director, starring, writer, musicComposer, releaseDate, runtime, country, language, genre).  
**Output** → `data/raw/dbpedia_triples.jsonl` con campi: `film`, `dir` (out|in), `p`, `o`.

### 4.2 Wikipedia (testo) — intro paragrafo 1

Ottieni **il primo paragrafo** (EN) per ogni film (via REST “page summary”) o, in fallback, `dbo:abstract@en` da DBpedia.  
**Output** → `data/raw/wikipedia_intro.jsonl` con campi: `film`, `text`.

### 4.3 Pairing & Serializzazione

Unisci `{text, triples}` → `pairs.jsonl`, quindi **linearizza** RDF e costruisci i **4 dataset**: `text2rdf.jsonl`, `rdf2text.jsonl`, `rdfcomp1.jsonl`, `rdfcomp2.jsonl`.  
Token speciali: `<SOT> <EOT> <SUBJ> <PRED> <OBJ> <RDF2Text> <Text2RDF> <CONTINUERDF> <MASK>`.

---

## 5) Tokenizer (Step 4)

Addestra **BPE** su (testo + RDF linearizzato) con i token speciali. Artefatti in `data/vocab/`.

---

## 6) Modello & Training (Step 5–6)

Il modello di riferimento è `TinySeq2Seq` con **3 encoder layer + 3 decoder layer**
(`d_model=384`, `nhead=6`, `ff_dim=1536`, dropout `0.1`). Il preset di default
(`configs/train/multitask_default.yaml`) addestra un T5 compatto sui quattro task
Text2RDF/RDF2Text/RDFComp1/RDFComp2 con mixing **3:3:2:2**. Il preset
`multitask_default.yaml` fornisce un'alternativa più ampia mantenendo lo stesso schema di
training. Gli script di sanity (`src.run overfit` o `scripts/sanity_overfit.py`)
permettono di validare rapidamente la pipeline.

---

## 7) Decoding & Post-processing (Step 7)

Il modulo `src/decoding/base.py` implementa il decoding **greedy** autoregressivo
con stop su `<EOT>` (se presente) e filtraggio del token `<pad>`. Il file
`src/decoding/constrained.py` è attualmente uno **stub** pronto per ospitare
vincoli aggiuntivi sul formato RDF. Il post-processing delle triple e la
normalizzazione dei prefissi sono gestiti a livello di dataset (`src/data/serialization.py`).

---

## 8) Valutazione (Step 8)

- **RDF2Text**: ROUGE-L, BLEU, METEOR
- **Text2RDF/Comp-2**: Precision/Recall/**F1** su triple
- **Comp-1**: **Accuracy** sull'entità/predicato mascherato

Le metriche sono calcolate tramite `src/eval/metrics.py` e orchestrate da
`src/eval/evaluate.py`, che carica i checkpoint, costruisce i `DataLoader`
per gli split `val`/`test` e aggrega i risultati per task.

### 8.1 Configurazione & script

Il file `configs/eval/multitask_default.yaml` mostra un esempio completo di
configurazione con percorsi `val`/`test` per ciascun task, parametri di decoding
e destinazione del report JSON. Il campo `checkpoint` è impostato sul
segnaposto `<<override-me>>`: indica agli script di passare il path corretto via
`--override` (o di risolverlo tramite l'alias dichiarato in `model_alias`).
Per eseguire una valutazione completa sul mix multitask T5:

```bash
python -m src.run evaluate --cfg configs/eval/multitask_default.yaml --override checkpoint=checkpoints/multitask_default/best.pt --output reports/eval.json
```

Il comando genera un report strutturato (stampato a terminale e salvato su disco)
ed effettua l'eventuale logging su Weights & Biases se abilitato nel config.
Per retrocompatibilità rimane disponibile anche `python -m scripts.eval_all`,
che reindirizza automaticamente verso il subcomando `evaluate` e sfrutta
`model_alias` per impostare l'override se non già specificato.
Gli alias predefiniti includono anche `mix` (sinonimo di `multitask_default`) e
`baseline` per il vecchio encoder–decoder sinusoidale.

### 8.2 Inference manuale

Per testare rapidamente il modello su un input specifico puoi usare il
subcomando `predict` oppure lo script di esempio `scripts/predict_example.py`:

```bash
python -m src.run predict --checkpoint checkpoints/multitask_default/best.pt --tokenizer data/vocab/bpe.json --task text2rdf --input "Plot ..."

python -m scripts.predict_example --checkpoint checkpoints/multitask_default/best.pt --tokenizer data/vocab/bpe.json --task rdf2text --input "<SOT> ... <RDF2Text>"
```

Il flag `--task` aggiunge automaticamente il marker speciale previsto dal
dataset se non già presente nell'input. L'output del comando è ora un
JSON compatto che riporta input originale, prompt effettivo e statistiche
di lunghezza della predizione; aggiungi `--output reports/pred.json`
per salvare lo stesso payload su disco.

---

## 9) Ablation (Step 9) — breve e mirata

- **Dimensione**: confronta preset compatti (`multitask_default.yaml`).
- **Profondità**: varia `enc_layers`/`dec_layers` mantenendo fisso il mixing multitask **3:3:2:2**.
  Metriche: ROUGE-L, F1 triple, Accuracy Comp-1, costo/epoch.

Esegui i test rapidi sulle varianti con:

```bash
pytest tests/test_transformer_variants.py
```

---

## 10) Riferimento iperparametri & strategie di tuning

Questa tabella raccoglie le chiavi YAML più rilevanti (sezioni `train/` ed
`eval/`) con note pratiche per il tuning.

### 10.1 Architettura del modello (`configs/train/*.yaml`)

| Chiave                      | Significato                       | Range/Note                                            |
| --------------------------- | --------------------------------- | ----------------------------------------------------- |
| `d_model`                   | Dimensione delle embedding/hidden | 384–640 consigliato (multipli di 64)                  |
| `nhead`                     | Teste di attenzione               | 6 o 8 (deve dividere `d_model`)                       |
| `enc_layers` / `dec_layers` | Profondità encoder/decoder        | 3–6; oltre richiede gradient checkpointing            |
| `ff_dim`                    | Dimensione feed-forward           | 1536–2560 (≈4× `d_model`)                             |
| `dropout`                   | Dropout condiviso MHA/FFN         | 0.0–0.2                                               |
| `max_len`                   | Sequenza massima gestita          | 256 (baseline) / 512 (triple lunghe)                  |
| `relative_attention_num_buckets` | Bucket per bias relativo        | 16–64; valori alti gestiscono distanze lunghe         |
| `relative_attention_max_distance` | Distanza massima per i bucket   | 64–256 in base alla lunghezza delle sequenze          |
| `layer_norm_epsilon`        | Epsilon numerico LayerNorm        | Default 1e-6; aumentare solo in caso di instabilità   |

### 10.2 Ottimizzazione & controllo training

| Chiave                          | Descrizione                | Suggerimenti                                                                                    |
| ------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------- |
| `batch_size`                    | Dimensione batch logico    | Con 16 GB VRAM: 16–24; abbassa e aumenta `gradient_accumulation_steps` se memoria insufficiente |
| `gradient_accumulation_steps`   | Accumulo gradiente         | 2–4 per simulare batch grandi                                                                   |
| `num_epochs`                    | Epoche massime             | 8–12 per training da zero; early-stop gestisce uscita anticipata                                |
| `lr`                            | Learning rate iniziale     | 1e-4–5e-4 (training) / 3e-5 (fine-tuning)                                                       |
| `weight_decay`                  | Regolarizzazione L2        | 0.0–0.05                                                                                        |
| `scheduler`                     | Tipo scheduler             | `cosine` o `linear` supportati                                                                  |
| `warmup_ratio` / `warmup_steps` | Warmup iniziale            | Ratio 0.02–0.08 (override `warmup_steps` se specificato)                                        |
| `min_lr_ratio`                  | LR minimo relativo         | 0.01–0.05 per evitare LR→0 nel cosine                                                           |
| `early_stopping.patience`       | Epoche senza miglioramento | 2–4 consigliate                                                                                 |
| `early_stopping.min_delta`      | Miglioramento minimo       | 0.0–0.01                                                                                        |
| `overfit_one_batch`             | Debug di pipeline          | Attivato automaticamente da `src.run overfit`                                                   |

### 10.3 Gestione dati & mixing multitask

| Chiave                    | Descrizione              | Note                                                  |
| ------------------------- | ------------------------ | ----------------------------------------------------- |
| `train_file` / `val_file` | Path singolo task        | Utili per esperimenti single-task o debug mirati      |
| `datasets`                | Lista task con pesi      | `weight` guida il sampler proporzionale (es. 3:3:2:2) |
| `max_len`                 | Troncamento input/target | Deve corrispondere al valore del tokenizer            |
| Flag RUN `--toy`          | Dataset rapido           | Applica automaticamente `configs/data/toy.yaml`       |

### 10.4 Logging & strumentazione

| Chiave                     | Descrizione                                                 |
| -------------------------- | ----------------------------------------------------------- |
| `wandb.mode`               | `online`, `offline`, `disabled`; fallback gestito dalla RUN |
| `wandb.project` / `entity` | Identificativi workspace W&B                                |
| `wandb.run_name`           | Nome leggibile della run                                    |
| `wandb.tags`               | Lista di tag (filtri dashboard)                             |
| `wandb.watch`              | Se `true`, abilita `wandb.watch` sul modello                |

### 10.5 Config valutazione (`configs/eval/*.yaml`)

- `checkpoint`, `tokenizer_file`, `device`: cosa caricare e dove inferire.
- `model_alias`: mapping rapido verso i checkpoint salvati (es. `mix`).
- `batch_size`, `num_workers`: throughput inferenza.
- `decoding.max_new_tokens`: limite della generazione autoregressiva.
- Blocchi `tasks.*.val/test`: percorsi per split; è possibile escludere task
  per valutazioni parziali.

### 10.6 Strategie pratiche di tuning

- **RDF2Text (BLEU/ROUGE)**: aumenta `d_model` a 512, porta
  `enc_layers`/`dec_layers` a 4–5, riduci `dropout` a 0.05, abilita logging W&B
  per monitorare le metriche.
- **Text2RDF (precision/F1)**: prova `scheduler=linear` con
  `warmup_ratio=0.08` e `min_lr_ratio=0.02`; controlla che i marker di task
  siano sempre presenti nell’input.
- **RDF Completion 1 (mask accuracy)**: verifica che gli input contengano il
  marker `<MASK>` e valuta batch più piccoli per ridurre il rumore sulle entità
  lunghe.
- **RDF Completion 2 (triple F1)**: aumenta `ff_dim` e `num_decoder_layers` per
  migliorare la modellazione delle sequenze lunghe.
- **Diagnostica rapida**: `python -m src.run overfit --cfg ... --toy` deve far
  scendere la loss <0.05; in caso contrario ricontrolla tokenizer, dataset e
  sequenza di token speciali.

---
