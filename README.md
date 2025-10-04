# NanoSocrates — (Baseline unificata Text↔RDF + Completion)

**Stato**: setup architettura & data layer — GPU target: 16 GB VRAM  
**Data**: 01 ottobre 2025

Questo repository ospita una pipeline **end-to-end** per addestrare un **Transformer encoder–decoder** capace di svolgere 4 task nel dominio *film*: **Text2RDF, RDF2Text, RDF Completion 1 (masked), RDF Completion 2 (continuation)**.  
Il progetto è *compliant* alla traccia: **token speciali**, **tokenizer BPE from scratch**, **multi-task training** e **metriche per task**.

---

## 1) Struttura del repository

```
nanosocrates/
├─ README.md
├─ requirements.txt              # dipendenze Python principali
├─ configs/
│  ├─ base.yaml                  # preset globale con default condivisi
│  ├─ data/
│  │  ├─ build.yaml              # parametri per costruzione dataset
│  │  ├─ dbpedia.yaml            # endpoint, predicati, limiti SPARQL
│  │  ├─ toy.yaml                # remapping verso data/processed/toy
│  │  └─ wikipedia.yaml          # API REST, lingua, timeout
│  ├─ decode/
│  │  └─ constrained.yaml        # vincoli leggeri durante il decoding
│  ├─ eval/
│  │  └─ baseline.yaml           # esempio completo di valutazione
│  ├─ tokenizer/
│  │  └─ bpe_24k.yaml            # addestramento tokenizer + token speciali
│  └─ train/
│     ├─ baseline.yaml           # modello standard 3e+3d
│     ├─ mix_3322.yaml           # mixing alternativo per i task
│     └─ rope_on.yaml            # variante con Rotary Positional Embeddings
├─ data/                         # directory popolata dagli script (raw/interim/processed/vocab)
├─ scripts/
│  ├─ build_dataset.py           # crea dataset e task JSONL (richiede PYTHONPATH=src)
│  ├─ build_toy_subset.py        # genera il sottoinsieme toy (include setup PYTHONPATH interno)
│  ├─ eval_all.py                # valutazione multi-task
│  ├─ fetch_dbpedia.py           # scarica triple DBpedia (richiede PYTHONPATH=src)
│  ├─ fetch_wikipedia.py         # scarica abstract intro (richiede PYTHONPATH=src)
│  ├─ predict_example.py         # inference minimale da riga di comando
│  ├─ sanity_overfit.py          # scorciatoia per l'overfit di un batch
│  └─ train_tokenizer.py         # addestra il tokenizer BPE
├─ src/
│  ├─ cli.py                     # entrypoint unificato (train/overfit/evaluate/predict)
│  ├─ data/                      # fetch DBpedia/Wikipedia, pairing, serializzazione
│  ├─ decoding/                  # strategie di decoding e vincoli
│  ├─ eval/                      # metriche e orchestratore valutazione
│  ├─ model/                     # TinySeq2Seq, layer MHA/MLA, perdite
│  ├─ tokenizer/                 # wrapper IO e libreria per BPE
│  ├─ training/                  # dataloader multitask, loop, scheduler
│  ├─ utils/                     # config YAML, IO, logging, integrazione W&B
│  └─ plots/curves.py            # placeholder per grafici (stub vuoto)
└─ tests/
   ├─ integration/               # scenari end-to-end
   ├─ test_builders.py           # validazione dataset JSONL
   ├─ test_decoding.py           # vincoli e decoding greedy
   ├─ test_dataloaders.py        # collate + span masking
   ├─ test_losses.py             # loss multi-task/spans
   ├─ test_metrics.py            # metriche BLEU/ROUGE/F1
   ├─ test_scheduler.py          # scheduler cosine+warmup
   ├─ test_serialization.py      # linearizzazione RDF ↔ testo
   └─ test_transformer_variants.py # controlli sulle ablation
```

---

## 2) Quickstart

### 2.1 Ambiente
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Suggerimento: i comandi `scripts/*.py` che importano `utils.*` richiedono
> `PYTHONPATH=src`. Imposta una volta `export PYTHONPATH=src` (bash/zsh) oppure
> anteponi `PYTHONPATH=src` al singolo comando.

### 2.2 Pipeline base (dati → tokenizer → training → valutazione)
1. **Raccogli le sorgenti**
   ```bash
   export PYTHONPATH=src                                    # abilita gli import locali
   PYTHONPATH=src python scripts/fetch_dbpedia.py \
       --config configs/data/dbpedia.yaml \
       --out data/raw/dbpedia_triples.jsonl

   PYTHONPATH=src python scripts/fetch_wikipedia.py \
       --config configs/data/wikipedia.yaml \
       --in data/raw/dbpedia_triples.jsonl \
       --out data/raw/wikipedia_intro.jsonl
   ```
2. **Costruisci il dataset multi-task**
   ```bash
   PYTHONPATH=src python scripts/build_dataset.py \
       --config configs/data/build.yaml \
       --dbp data/raw/dbpedia_triples.jsonl \
       --wiki data/raw/wikipedia_intro.jsonl \
       --outdir data/processed \
       --emit_tasks
   ```
3. **Addestra (o aggiorna) il tokenizer**
   ```bash
   python -m scripts.train_tokenizer --config configs/tokenizer/bpe_24k.yaml
   ```
4. **Avvia il training**
   ```bash
   python -m src.cli train --cfg configs/train/baseline.yaml
   ```
5. **Valuta il checkpoint** (report JSON + metriche aggregate)
   ```bash
   python -m scripts.eval_all --cfg configs/eval/baseline.yaml
   # equivalente CLI unificata
   python -m src.cli evaluate --cfg configs/eval/baseline.yaml --output reports/baseline_eval.json
   ```

### 2.3 Tutorial — sottoinsieme toy (20 film)
1. Assicurati di avere `data/interim/pairs.all.jsonl` e `data/interim/splits.json`
   generati da `scripts/build_dataset.py`.
2. Rigenera i JSONL ridotti:
   ```bash
   python -m scripts.build_toy_subset \
       --pairs data/interim/pairs.all.jsonl \
       --splits data/interim/splits.json \
       --processed-dir data/processed \
       --outdir data/processed/toy \
       --films 20
   ```
3. Esegui training e valutazione puntando ai nuovi file con il flag `--toy`:
   ```bash
   python -m src.cli train --cfg configs/train/baseline.yaml --toy
   python -m scripts.eval_all --cfg configs/eval/baseline.yaml --toy
   ```

### 2.4 Tutorial — sanity check (overfit di un batch)
1. Riusa la configurazione standard e forza gli override automatici:
   ```bash
   python -m src.cli overfit --cfg configs/train/baseline.yaml --toy
   ```
   Il comando imposta `num_epochs=1`, `max_steps=1` e `overfit_one_batch=true`,
   mantenendo qualsiasi ulteriore `--override` passato da CLI.
2. In alternativa esiste lo script dedicato:
   ```bash
   python -m scripts.sanity_overfit --cfg configs/train/baseline.yaml --toy
   ```
3. Verifica che la loss scenda rapidamente verso ~0: conferma che tokenizer,
   dataloader, loop di training e logging siano correttamente collegati.

### 2.5 Tutorial — valutazione con Weights & Biases
1. Modifica il config (o usa gli override) per abilitare W&B.
   ```bash
   python -m src.cli train \
       --cfg configs/train/baseline.yaml \
       --override wandb.mode=online wandb.project=nanosocrates-demo wandb.run_name=debug
   ```
   I campi supportati sono `mode` (`online`, `offline`, `disabled`), `project`,
   `entity`, `run_name`, `tags` (lista) e `watch` (bool). Se la connessione fallisce
   viene eseguito automaticamente il fallback in modalità offline.
2. Per loggare anche la valutazione usa lo stesso approccio:
   ```bash
   python -m scripts.eval_all \
       --cfg configs/eval/baseline.yaml \
       --override wandb.mode=online wandb.project=nanosocrates-demo
   ```
   Le metriche vengono appiattite tramite `src.utils.wandb_utils.flatten_eval_metrics`
   e inviate come singolo step alla run già configurata.
3. Per eseguire la valutazione dal CLI unificato mantenendo gli override:
   ```bash
   python -m src.cli evaluate \
       --cfg configs/eval/baseline.yaml \
       --override wandb.mode=online wandb.project=nanosocrates-demo \
       --output reports/baseline_eval.json
   ```

---

## 3) Configurazione (YAML)

Vedi esempi in `configs/` per:  
- `data/dbpedia.yaml` — endpoint SPARQL, whitelist predicati, direzione (out|both)
- `data/wikipedia.yaml` — lingua, endpoint REST, timeout
- `data/build.yaml` — split, maxlen, filtri qualità
- `data/toy.yaml` — percorsi del sottoinsieme 20-film per debug rapido
- `tokenizer/bpe_24k.yaml` — vocab e token speciali
- `train/baseline.yaml` — modello, trainer, mixing task
- `decode/constrained.yaml` — vincoli leggeri per RDF
- blocco `wandb:` — parametri di logging (project, entity, run_name, mode, tags, watch)

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
(`d_model=384`, `nhead=6`, `ff_dim=1536`, dropout `0.1`). Il training baseline
(`configs/train/baseline.yaml`) usa AdamW con scheduler cosine + warmup e opera
su un singolo task (Text2RDF). Per allenare sui quattro task insieme utilizza
`configs/train/mix_3322.yaml`, che imposta il mixing **3:3:2:2** su
Text2RDF/RDF2Text/RDFComp1/RDFComp2. Gli script di sanity (`src.cli overfit` o
`scripts/sanity_overfit.py`) permettono di validare rapidamente la pipeline.

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

Gli esempi pronti (`baseline.yaml`, `rope_on.yaml`, `mix_3322.yaml`) mostrano
come attivare/ disattivare i flag per le ablation.

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

Il file `configs/eval/baseline.yaml` mostra un esempio completo di configurazione
con percorsi `val`/`test` per ciascun task, parametri di decoding e destinazione
del report JSON. Per eseguire una valutazione completa:

```bash
python -m scripts.eval_all --cfg configs/eval/baseline.yaml
```

Lo script genera un report strutturato (stampato a terminale e salvato su disco)
ed effettua l'eventuale logging su Weights & Biases se abilitato nel config.
Lo stesso comportamento è disponibile dal CLI unificato:

```bash
python -m src.cli evaluate --cfg configs/eval/baseline.yaml --output reports/eval.json
```

### 8.2 Inference manuale

Per testare rapidamente il modello su un input specifico puoi usare il
subcomando `predict` oppure lo script di esempio `scripts/predict_example.py`:

```bash
python -m src.cli predict --checkpoint checkpoints/baseline/best.pt \
    --tokenizer data/vocab/bpe.json --task text2rdf --input "Plot ..."

python -m scripts.predict_example --checkpoint checkpoints/baseline/best.pt \
    --tokenizer data/vocab/bpe.json --task rdf2text --input "<SOT> ... <RDF2Text>"
```

Il flag `--task` aggiunge automaticamente il marker speciale previsto dal
dataset se non già presente nell'input.

---

## 9) Ablation (Step 9) — breve e mirata
- **Positional**: sinusoidale (`baseline.yaml`) vs **RoPE** (`rope_on.yaml`)
- **Attention**: standard vs **MLA** (abilita `use_mla` e calibra `interleave_ratio`)
- **Mixing**: single-task (`baseline.yaml`) vs multi-task **3:3:2:2** (`mix_3322.yaml`)
Metriche: ROUGE-L, F1 triple, Accuracy Comp-1, costo/epoch.

Esegui i test rapidi sulle varianti con:

```bash
pytest tests/test_transformer_variants.py
```

---

## 10) Contratti I/O (schemi JSONL)

**pairs.jsonl**
```json
{{"film":"dbr:Inception","text":"...","triples":[["dbr:Inception","dbo:director","dbr:Christopher_Nolan"],["dbr:Inception","dbo:starring","dbr:Leonardo_DiCaprio"]]}}
```

**task jsonl (generico)**
```json
{{"input":"<INPUT TOKENS>","target":"<TARGET TOKENS>","film":"dbr:Inception"}}
```

---

## 11) Playbook CLI completo — end-to-end

Questa sezione raccoglie *tutti* i comandi operativi utili per eseguire l’intera
pipeline: ingestione, preprocessing, training, valutazione e inference manuale.
Si assume un terminale posizionato nella root del repository e, salvo diversa
indicazione, un ambiente virtuale Python attivo.

### 11.1 Preparazione dell’ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
```

Verifica del supporto CUDA (opzionale):

```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only")
PY
```

### 11.2 Raccolta dati e sanity check

1. **Triple DBpedia**
   ```bash
   PYTHONPATH=src python scripts/fetch_dbpedia.py \
       --config configs/data/dbpedia.yaml \
       --out data/raw/dbpedia_triples.jsonl
   ```
2. **Intro Wikipedia**
   ```bash
   PYTHONPATH=src python scripts/fetch_wikipedia.py \
       --config configs/data/wikipedia.yaml \
       --in data/raw/dbpedia_triples.jsonl \
       --out data/raw/wikipedia_intro.jsonl
   ```
3. **Controlli rapidi**
   ```bash
   wc -l data/raw/dbpedia_triples.jsonl data/raw/wikipedia_intro.jsonl
   head -n 2 data/raw/dbpedia_triples.jsonl | jq '.'
   head -n 2 data/raw/wikipedia_intro.jsonl | jq '.'
   ```

### 11.3 Costruzione dataset multitask & subset toy

```bash
PYTHONPATH=src python scripts/build_dataset.py \
    --config configs/data/build.yaml \
    --dbp data/raw/dbpedia_triples.jsonl \
    --wiki data/raw/wikipedia_intro.jsonl \
    --outdir data/processed \
    --emit_tasks
```

Post-controlli consigliati:

```bash
ls data/interim
wc -l data/processed/text2rdf.*.jsonl
python - <<'PY'
from utils.io import read_jsonl
ex = next(read_jsonl('data/processed/text2rdf.train.jsonl'))
print(ex['film'])
print(ex['input'][:120] + '...')
print(ex['target'][:120] + '...')
PY
```

Per generare la versione “toy” (20 film):

```bash
python -m scripts.build_toy_subset \
    --pairs data/interim/pairs.all.jsonl \
    --splits data/interim/splits.json \
    --processed-dir data/processed \
    --outdir data/processed/toy \
    --films 20
```

### 11.4 Tokenizer: addestramento e verifica

1. Aggiorna `configs/tokenizer/bpe_24k.yaml` con pattern (`glob`), file di
   output (`out`), `vocab_size`, `min_freq`, `special_tokens`.
2. Addestra il vocabolario condiviso:
   ```bash
   python -m scripts.train_tokenizer --config configs/tokenizer/bpe_24k.yaml
   ```
3. Verifica dimensioni e token chiave:
   ```bash
   head -n 10 data/vocab/bpe.json
   python - <<'PY'
from src.tokenizer.tokenizer_io import TokWrapper
tok = TokWrapper('data/vocab/bpe.json')
print('vocab:', tok.vocab_size())
print('pad id:', tok.pad_id)
print('SOT id:', tok.token_to_id('<SOT>'))
PY
   ```

### 11.5 Training: modalità supportate

| Scenario | Comando |
|----------|---------|
| **Smoke test (toy)** | `python -m src.cli overfit --cfg configs/train/baseline.yaml --toy` |
| **Baseline Text2RDF** | `python -m src.cli train --cfg configs/train/baseline.yaml` |
| **Multitask 3:3:2:2** | `python -m src.cli train --cfg configs/train/mix_3322.yaml` |
| **Variant RoPE** | `python -m src.cli train --cfg configs/train/rope_on.yaml` |
| **Custom override** | `python -m src.cli train --cfg configs/train/baseline.yaml --override lr=1e-4 num_epochs=5 scheduler=linear` |

Suggerimenti operativi:

- Il flag `--toy` redirige automaticamente ai percorsi definiti in
  `configs/data/toy.yaml`.
- Usa `--override chiave=valore` per modifiche rapide (inclusi parametri
  nidificati, es. `wandb.mode=online`). La CLI provvede al casting numerico.
- `gradient_accumulation_steps` consente batch effettivi più grandi senza
  saturare la VRAM.
- Ogni epoca produce un checkpoint (`epochXXX.pt`) e, in caso di miglioramento,
  aggiorna `best.pt` nella directory `save_dir`.

### 11.6 Monitoraggio, valutazione e logging remoto

Training con W&B online (fallback automatico in offline):

```bash
python -m src.cli train \
    --cfg configs/train/mix_3322.yaml \
    --override wandb.mode=online wandb.project=nanosocrates wandb.run_name=multitask_v1
```

Valutazione completa (val + test) con report JSON e logging opzionale:

```bash
python -m scripts.eval_all --cfg configs/eval/baseline.yaml
python -m src.cli evaluate --cfg configs/eval/baseline.yaml --output reports/baseline_eval.json
python -m src.cli evaluate \
    --cfg configs/eval/baseline.yaml \
    --override wandb.mode=online wandb.project=nanosocrates_eval \
    --output reports/baseline_eval.json
jq '.' reports/baseline_eval.json
```

### 11.7 Inference “online” via CLI

```bash
python -m src.cli predict \
    --checkpoint checkpoints/mix3322/best.pt \
    --tokenizer data/vocab/bpe.json \
    --task rdf2text \
    --input "<SOT> <SUBJ> dbr:Inception <PRED> dbo:director <OBJ> dbr:Christopher_Nolan <EOT>"

python -m src.cli predict \
    --checkpoint checkpoints/mix3322/best.pt \
    --tokenizer data/vocab/bpe.json \
    --task text2rdf \
    --input "Inception is a sci-fi heist film..."
```

Il comando aggiunge il marker di task se assente e rimuove eventuali `<pad>`
prima di stampare l’output. Per batch più ampi usa
`python -m scripts.predict_example.py` con file di input multipli.

### 11.8 Pulizia e reset

- Cancella checkpoint e report: `rm -rf checkpoints/* reports/*`
- Rigenera i dataset eliminando `data/interim` e `data/processed` (preserva
  eventualmente `data/processed/toy`)
- Per riaddestrare il tokenizer da zero elimina `data/vocab/`

---

## 12) Riferimento iperparametri & strategie di tuning

Questa tabella raccoglie le chiavi YAML più rilevanti (sezioni `train/` ed
`eval/`) con note pratiche per il tuning.

### 12.1 Architettura del modello (`configs/train/*.yaml`)

| Chiave | Significato | Range/Note |
|--------|-------------|-----------|
| `d_model` | Dimensione delle embedding/hidden | 384–640 consigliato (multipli di 64) |
| `nhead` | Teste di attenzione | 6 o 8 (deve dividere `d_model`) |
| `enc_layers` / `dec_layers` | Profondità encoder/decoder | 3–6; oltre richiede gradient checkpointing |
| `ff_dim` | Dimensione feed-forward | 1536–2560 (≈4× `d_model`) |
| `dropout` | Dropout condiviso MHA/FFN | 0.0–0.2 |
| `max_len` | Sequenza massima gestita | 256 (baseline) / 512 (triple lunghe) |
| `use_rope` | Rotary Position Embeddings | Alternativa alle sinusoidali classiche |
| `use_mla` | Multihead Latent Attention | Richiede `CustomTransformer` in `src/model/layers.py` |
| `interleave_ratio` | Mix MLA ↔ attenzione classica | 0.0 = disattivato, 0.5 = mix, 1.0 = solo MLA |
| `enable_entity_spans` | Propaga span mask nei batch | Necessario per RDF Completion 1 |
| `compute_span_metrics` | Aggiunge metriche sugli span | Impatta marginalmente sui tempi |

### 12.2 Ottimizzazione & controllo training

| Chiave | Descrizione | Suggerimenti |
|--------|-------------|--------------|
| `batch_size` | Dimensione batch logico | Con 16 GB VRAM: 16–24; abbassa e aumenta `gradient_accumulation_steps` se memoria insufficiente |
| `gradient_accumulation_steps` | Accumulo gradiente | 2–4 per simulare batch grandi |
| `num_epochs` | Epoche massime | 8–12 per training da zero; early-stop gestisce uscita anticipata |
| `lr` | Learning rate iniziale | 1e-4–5e-4 (training) / 3e-5 (fine-tuning) |
| `weight_decay` | Regolarizzazione L2 | 0.0–0.05 |
| `scheduler` | Tipo scheduler | `cosine` o `linear` supportati |
| `warmup_ratio` / `warmup_steps` | Warmup iniziale | Ratio 0.02–0.08 (override `warmup_steps` se specificato) |
| `min_lr_ratio` | LR minimo relativo | 0.01–0.05 per evitare LR→0 nel cosine |
| `early_stopping.patience` | Epoche senza miglioramento | 2–4 consigliate |
| `early_stopping.min_delta` | Miglioramento minimo | 0.0–0.01 |
| `overfit_one_batch` | Debug di pipeline | Attivato automaticamente da `src.cli overfit` |

### 12.3 Gestione dati & mixing multitask

| Chiave | Descrizione | Note |
|--------|-------------|------|
| `train_file` / `val_file` | Path singolo task | Usati nei preset baseline |
| `datasets` | Lista task con pesi | `weight` guida il sampler proporzionale (es. 3:3:2:2) |
| `max_len` | Troncamento input/target | Deve corrispondere al valore del tokenizer |
| Flag CLI `--toy` | Dataset rapido | Applica automaticamente `configs/data/toy.yaml` |

### 12.4 Logging & strumentazione

| Chiave | Descrizione |
|--------|-------------|
| `wandb.mode` | `online`, `offline`, `disabled`; fallback gestito dalla CLI |
| `wandb.project` / `entity` | Identificativi workspace W&B |
| `wandb.run_name` | Nome leggibile della run |
| `wandb.tags` | Lista di tag (filtri dashboard) |
| `wandb.watch` | Se `true`, abilita `wandb.watch` sul modello |

### 12.5 Config valutazione (`configs/eval/*.yaml`)

- `checkpoint`, `tokenizer_file`, `device`: cosa caricare e dove inferire.
- `batch_size`, `num_workers`: throughput inferenza.
- `decoding.max_new_tokens`: limite della generazione autoregressiva.
- `enable_entity_spans`: calcola metriche MASK se il checkpoint le supporta.
- Blocchi `tasks.*.val/test`: percorsi per split; è possibile escludere task
  per valutazioni parziali.

### 12.6 Strategie pratiche di tuning

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
- **Diagnostica rapida**: `python -m src.cli overfit --cfg ... --toy` deve far
  scendere la loss <0.05; in caso contrario ricontrolla tokenizer, dataset e
  sequenza di token speciali.

---

## 13) Linee guida di qualità
Whitelist predicati, split per film (no leakage), max seq 256–512, logging di parsing error/outlier.

---

## 14) Licenze & Dati
DBpedia/Wikipedia: rispettare le licenze; mantenere solo abstract/intro.

---

## 15) Roadmap esecutiva (riassunto)
1. `scripts/fetch_dbpedia.py` + `scripts/fetch_wikipedia.py` → `data/raw/`
2. `scripts/build_dataset.py` → `data/interim/pairs.jsonl` + `data/processed/*.jsonl`
3. `scripts/train_tokenizer.py` → `data/vocab/`
4. `scripts/sanity_overfit.py` → training → eval → ablation
