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
│  │  └─ bpe_default.yaml            # addestramento tokenizer + token speciali
│  └─ train/
│     ├─ baseline.yaml           # encoder-decoder vanilla (sinusoidale)
│     ├─ multitask_default.yaml  # preset multitask T5 3:3:2:2
│     └─ rope_on.yaml            # variante con Rotary Positional Embeddings
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
│  └─ train_tokenizer.py         # addestra il tokenizer BPE
└─ src/
   ├─ run.py                     # entrypoint unificato (train/overfit/evaluate/predict)
   ├─ data/                      # fetch DBpedia/Wikipedia, pairing, serializzazione
   ├─ decoding/                  # strategie di decoding e vincoli
   ├─ eval/                      # metriche e orchestratore valutazione
   ├─ model/                     # TinySeq2Seq, layer MHA/MLA, perdite
   ├─ tokenizer/                 # wrapper IO e libreria per BPE
   ├─ training/                  # dataloader multitask, loop, scheduler
   ├─ utils/                     # config YAML, IO, logging, integrazione W&B
   └─ plots/curves.py            # placeholder per grafici (stub vuoto)
```

---

## 2) Quickstart"""Main training and evaluation loop for the transformer model."""

### 2.1 Ambiente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Pipeline base (dati -> tokenizer → training → valutazione)

1. **Raccogli le sorgenti**

   ```bash
   python -m scripts.fetch_dbpedia --config configs/data/dbpedia.yaml --out data/raw/dbpedia_triples.jsonl

   python -m scripts.fetch_wikipedia --config configs/data/wikipedia.yaml --in data/raw/dbpedia_triples.jsonl --out data/raw/wikipedia_intro.jsonl
   ```

2. **Costruisci il dataset multi-task**
   ```bash
   python -m scripts.build_dataset --config configs/data/build.yaml --dbp data/raw/dbpedia_triples.jsonl --wiki data/raw/wikipedia_intro.jsonl --outdir data/processed --emit_tasks
   ```
3. **Addestra (o aggiorna) il tokenizer**
   ```bash
   python -m scripts.train_tokenizer --config configs/tokenizer/bpe_default.yaml
   ```
4. **Avvia il training**
   ```bash
   python -m src.run train --cfg configs/train/multitask_default.yaml
   ```
5. **Valuta il checkpoint** (report JSON + metriche aggregate)
   ```bash
   python -m src.run evaluate --cfg configs/eval/multitask_default.yaml --override checkpoint=checkpoints/multitask_default/best.pt --output reports/multitask_default_eval.json
   ```

### 2.3 Tutorial — sottoinsieme toy (20 film)

1. Assicurati di avere `data/interim/pairs.all.jsonl` e `data/interim/splits.json`
   generati da `scripts/build_dataset.py`.
2. Rigenera i JSONL ridotti:
   ```bash
   python -m scripts.build_toy_subset --pairs data/interim/pairs.all.jsonl --splits data/interim/splits.json --processed-dir data/processed --outdir data/processed/toy --films 20
   ```
3. Esegui training e valutazione puntando ai nuovi file con il flag `--toy`:
   ```bash
   python -m src.run train --cfg configs/train/multitask_default.yaml --toy
   
   python -m src.run evaluate --cfg configs/eval/multitask_default.yaml --override checkpoint=checkpoints/multitask_default/best.pt --output reports/multitask_default_eval.json --toy
   ```

### 2.4 Tutorial — sanity check (overfit di un batch)

1. Riusa la configurazione standard e forza gli override automatici:
   ```bash
   python -m src.run overfit --cfg configs/train/multitask_default.yaml --toy
   ```
   Il comando forza il dataset a un singolo batch, disattiva l'early stopping e
   per impostazione predefinita esegue 200 aggiornamenti consecutivi sullo stesso batch.
   - Il numero di esempi nel batch coincide con `batch_size` del config (16 nel
     preset `configs/train/multitask_default.yaml`). Se vuoi restringerlo, passa
     `--override batch_size=4` o modifica il valore nel YAML.
   - Usa `--steps N` per cambiare il numero di ottimizzazioni (es. `--steps 400`).
     Qualsiasi ulteriore `--override` passato da RUN viene rispettato.
2. In alternativa esiste lo script dedicato:
   ```bash
   python -m scripts.sanity_overfit --cfg configs/train/multitask_default.yaml --toy
   ```
3. (Opzionale) Per ispezionare il batch che viene ripetuto durante l'overfit usa
   il flag `--print-batch`:

   ```bash
   python -m src.run overfit --cfg configs/train/multitask_default.yaml --toy --print-batch --print-batch-limit 2
   ```

   Il comando stampa sul logger `INFO` un riepilogo del primo batch emesso dal
   `DataLoader`: numero di token non di padding, testo decodificato, eventuali
   campi grezzi (`raw_input`/`raw_target`) e il task associato a ciascun esempio.
   Il limite di esempi mostrati è controllato da `--print-batch-limit` (default: 3).

4. Vedrai i log INFO con le metriche di validazione ad ogni epoca e la perdita nel
   postfix della progress bar. Verifica che la loss scenda rapidamente verso ~0:
   questo conferma che tokenizer, dataloader, loop di training e logging sono collegati.

#### 2.4.1 Interpretazione della loss nel sanity check

- **Prime epoche**: è normale partire da perdite molto alte (anche >150) perché il
  modello sta iniziando da pesi random e il batch toy è estremamente eterogeneo.
- **Andamento atteso**: con `--override dropout=0.0` e AMP attivo, il modello
  memorizza il batch in poche decine di step. In pratica vedrai la loss scendere
  sotto 50 dopo ~5–6 epoche e convergere verso <1 (fino a ~0.05) entro 20–30 epoche.
- **Se la loss resta >10 per decine di epoche**: verifica che `--toy` punti ai file
  ridotti corretti, che il tokenizer sia lo stesso usato per generarli e, in caso di
  dubbi, usa `--print-batch` per stampare gli ID e i testi del mini-dataset (cerca
  valori `-100` nelle labels solo nelle posizioni di padding).
- **Plateau sopra 1**: di solito indica token fuori vocabolario o batch pieni di
  padding; in quel caso ricontrolla il dataset di input e considera di rigenerare il
  toy set.

### 2.5 Tutorial — valutazione con Weights & Biases

1. Modifica il config (o usa gli override) per abilitare W&B.
   ```bash
   python -m src.run train --cfg configs/train/multitask_default.yaml --override wandb.mode=online wandb.project=nanosocrates-demo wandb.run_name=debug
   ```
   I campi supportati sono `mode` (`online`, `offline`, `disabled`), `project`,
   `entity`, `run_name`, `tags` (lista) e `watch` (bool). Se non specifichi
   `run_name`, il RUN genera automaticamente un nome leggibile basato sul file
   di config, sul tipo di esecuzione (`train`/`overfit`) e sul timestamp; in caso
   contrario apparirebbero i nomi casuali di default di W&B. Se la connessione
   fallisce viene eseguito automaticamente il fallback in modalità offline.
2. Per loggare anche la valutazione usa lo stesso approccio:
   ```bash
   python -m src.run evaluate --cfg configs/eval/multitask_default.yaml --override \
       checkpoint=checkpoints/multitask_default/best.pt \
       wandb.mode=online wandb.project=nanosocrates-demo --output reports/multitask_default_eval.json
   ```
   Le metriche vengono appiattite tramite `src.utils.wandb_utils.flatten_eval_metrics`
   e inviate come singolo step alla run già configurata.
3. Per eseguire la valutazione dal RUN unificato mantenendo gli override:
   ```bash
   python -m src.run evaluate --cfg configs/eval/multitask_default.yaml --override \
       checkpoint=checkpoints/multitask_default/best.pt \
       wandb.mode=online wandb.project=nanosocrates-demo --output reports/multitask_default_eval.json
   ```

### 2.6 Debug — ispeziona un batch MLM

Lo script `scripts/inspect_mlm_batch.py` consente di verificare rapidamente che il
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

---

## 3) Configurazione (YAML)

Vedi esempi in `configs/` per:

- `data/dbpedia.yaml` — endpoint SPARQL, whitelist predicati, direzione (out|both)
- `data/wikipedia.yaml` — lingua, endpoint REST, timeout
- `data/build.yaml` — split, maxlen, filtri qualità
- `data/toy.yaml` — percorsi del sottoinsieme 20-film per debug rapido
- `tokenizer/bpe_default.yaml` — vocab e token speciali
- `train/baseline.yaml` — modello, trainer, mixing task
- `decode/constrained.yaml` — vincoli leggeri per RDF
- blocco `wandb:` — parametri di logging (project, entity, run_name, mode, tags, watch)

### Nuove opzioni modello

- `architecture`: scegli `"vanilla"` per mantenere l'encoder–decoder classico
  (nn.Transformer/varianti MLA+RoPE) oppure `"t5"` per attivare blocchi T5 con LayerNorm
  pre-attention, feed-forward GeGLU e bias posizionali relativi a bucket. La variante T5
  applica automaticamente lo scaling `√d_model` sulle embedding e riutilizza un dropout
  condiviso per encoder/decoder.
- `relative_attention_num_buckets` e `relative_attention_max_distance`: controllano la
  discretizzazione delle distanze per il bias relativo T5. Sono ignorati in modalità
  "vanilla" ma diventano obbligatori quando `architecture="t5"`.
- `layer_norm_epsilon`: epsilon numerico per le LayerNorm T5.
- Quando `architecture="t5"` le opzioni `use_rope`, `use_mla` e `interleave_ratio` sono
  disabilitate (sollevano errore in config misti).

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

Addestra **BPE 24k** su (testo + RDF linearizzato) con i token speciali. Artefatti in `data/vocab/`.

---

## 6) Modello & Training (Step 5–6)

Il modello di riferimento è `TinySeq2Seq` con **3 encoder layer + 3 decoder layer**
(`d_model=384`, `nhead=6`, `ff_dim=1536`, dropout `0.1`). Il preset di default
(`configs/train/multitask_default.yaml`) addestra un T5 compatto sui quattro task
Text2RDF/RDF2Text/RDFComp1/RDFComp2 con mixing **3:3:2:2**. Le varianti
`baseline.yaml` e `rope_on.yaml` riusano lo stesso schedule multitask rispettivamente
con posizioni sinusoidali e RoPE per facilitare le ablation. Gli script di sanity
(`src.run overfit` o `scripts/sanity_overfit.py`) permettono di validare rapidamente
la pipeline.

### 6.1 Varianti posizionali/attenzione

I config in `configs/train/*.yaml` espongono tre interruttori per sperimentare
varianti architetturali del `TinySeq2Seq`:

- `use_rope`: abilita le Rotary Positional Embeddings applicate alle
  proiezioni query/key al posto dell'iniezione sinusoidale. Il parametro
  `max_len` del config viene riutilizzato come `max_position_embeddings`.
- `use_mla`: sostituisce l'attenzione classica con un blocco
  **Multi-Linear Attention** leggero; quando combinato con `interleave_ratio`
  consente di fondere MLA e attenzione standard nella stessa testa.
- `interleave_ratio`: coefficiente (0.0–1.0) che controlla quanto del risultato
  dell'attenzione derivi dal ramo MLA (1.0 = solo MLA, 0.5 = mix paritetico).

Gli esempi pronti (`multitask_default.yaml`, `baseline.yaml`, `rope_on.yaml`)
mostrano come attivare/disattivare i flag per le ablation.

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
- **Comp-1**: **Accuracy** sullo span

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
python -m src.run evaluate --cfg configs/eval/multitask_default.yaml \
    --override checkpoint=checkpoints/multitask_default/best.pt --output reports/eval.json
```

Il comando genera un report strutturato (stampato a terminale e salvato su disco)
ed effettua l'eventuale logging su Weights & Biases se abilitato nel config.
Per retrocompatibilità rimane disponibile anche `python -m scripts.eval_all`,
che reindirizza automaticamente verso il subcomando `evaluate` e sfrutta
`model_alias` per impostare l'override se non già specificato. Usa
`--model-alias rope_on` per puntare al checkpoint RoPE (`checkpoints/rope_on/best.pt`).
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

- **Positional**: sinusoidale (`baseline.yaml`) vs **RoPE** (`rope_on.yaml`)
- **Attention**: standard vs **MLA** (abilita `use_mla` e calibra `interleave_ratio`)
- **Architecture**: T5 (`multitask_default.yaml`) vs encoder-decoder vanilla (`baseline.yaml`)
  mantenendo il mixing multitask **3:3:2:2**.
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
| `use_rope`                  | Rotary Position Embeddings        | Alternativa alle sinusoidali classiche                |
| `use_mla`                   | Multihead Latent Attention        | Richiede `CustomTransformer` in `src/model/layers.py` |
| `interleave_ratio`          | Mix MLA ↔ attenzione classica     | 0.0 = disattivato, 0.5 = mix, 1.0 = solo MLA          |
| `enable_entity_spans`       | Propaga span mask nei batch       | Necessario per RDF Completion 1                       |
| `compute_span_metrics`      | Aggiunge metriche sugli span      | Impatta marginalmente sui tempi                       |

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
- `model_alias`: mapping rapido verso i checkpoint salvati (`mix`, `rope_on`, ...).
- `batch_size`, `num_workers`: throughput inferenza.
- `decoding.max_new_tokens`: limite della generazione autoregressiva.
- `enable_entity_spans`: calcola metriche MASK se il checkpoint le supporta.
- Blocchi `tasks.*.val/test`: percorsi per split; è possibile escludere task
  per valutazioni parziali.

### 10.6 Strategie pratiche di tuning

- **RDF2Text (BLEU/ROUGE)**: aumenta `d_model` a 512, porta
  `enc_layers`/`dec_layers` a 4–5, riduci `dropout` a 0.05, abilita logging W&B
  per monitorare le metriche.
- **Text2RDF (precision/F1)**: prova `scheduler=linear` con
  `warmup_ratio=0.08` e `min_lr_ratio=0.02`; controlla che i marker di task
  siano sempre presenti nell’input.
- **RDF Completion 1 (mask accuracy)**: assicurati di avere
  `enable_entity_spans=true` e `compute_span_metrics=true`; valuta batch più
  piccoli per ridurre il rumore sugli span lunghi.
- **RDF Completion 2 (triple F1)**: sperimenta `use_mla=true` con
  `interleave_ratio=0.3` per enfatizzare le dipendenze latenti.
- **Diagnostica rapida**: `python -m src.run overfit --cfg ... --toy` deve far
  scendere la loss <0.05; in caso contrario ricontrolla tokenizer, dataset e
  sequenza di token speciali.

---
