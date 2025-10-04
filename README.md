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
├─ .gitignore
├─ requirements.txt              # torch, tokenizers, datasets, etc.
├─ configs/
│  ├─ base.yaml                  # iperparametri globali
│  ├─ data/
│  │  ├─ dbpedia.yaml            # endpoint, predicati, limiti
│  │  ├─ wikipedia.yaml          # API, lingua, timeout
│  │  └─ build.yaml              # split, maxlen, filtri qualità
│  ├─ tokenizer/
│  │  └─ bpe_24k.yaml            # vocabolario, special token
│  ├─ train/
│  │  ├─ baseline.yaml           # 4e+4d, d=512, ratio 3:3:2:2
│  │  ├─ rope_on.yaml            # ablation
│  │  └─ mix_2233.yaml           # ablation mixing
│  └─ decode/
│     └─ constrained.yaml        # vincoli leggeri sui tag RDF
├─ data/
│  ├─ raw/                       # dump SPARQL/abstracts per film
│  ├─ interim/                   # intermedio (pairing, pulizia)
│  ├─ processed/                 # jsonl per task (input,target)
│  └─ vocab/                     # tokenizer (vocab.json, merges.txt)
├─ scripts/
│  ├─ fetch_dbpedia.py           # scarica triple
│  ├─ fetch_wikipedia.py         # scarica abstract/intro
│  ├─ build_dataset.py           # crea i 4 task jsonl
│  ├─ train_tokenizer.py         # addestra BPE
│  ├─ sanity_overfit.py          # sanity su 1 batch
│  └─ eval_all.py                # lancia tutte le metriche
├─ src/
│  ├─ __init__.py
│  ├─ cli.py                     # entrypoint a riga di comando
│  ├─ utils/
│  │  ├─ io.py                   # lettura/scrittura jsonl, cache, hashing
│  │  ├─ text.py                 # normalizzazioni, cleaning
│  │  ├─ config.py               # caricamento YAML, override da CLI
│  │  └─ logging.py              # loggers, timers, progress
│  ├─ data/
│  │  ├─ dbpedia.py              # SPARQL query helpers
│  │  ├─ wikipedia.py            # fetch e parsing paragrafi
│  │  ├─ pairing.py              # unione {{text, triples}} per film
│  │  ├─ serialization.py        # linearizzazione RDF + parser inverso
│  │  └─ builders.py             # generatori dataset per i 4 task
│  ├─ tokenizer/
│  │  ├─ train_bpe.py            # libreria per scripts/train_tokenizer.py
│  │  └─ tokenizer_io.py         # load/save tokenizer + special token
│  ├─ model/
│  │  ├─ transformer.py          # encoder–decoder micro
│  │  ├─ layers.py               # MHA/FFN, pos enc (sin, RoPE)
│  │  └─ losses.py               # label smoothing, masking comp-1
│  ├─ training/
│  │  ├─ dataloaders.py          # collate per task, spanned masking
│  │  ├─ loop.py                 # training loop multi-task (AdamW + sched)
│  │  └─ scheduler.py            # warmup + cosine
│  ├─ decoding/
│  │  ├─ base.py                 # greedy/beam/top-k
│  │  └─ constrained.py          # vincoli leggeri su <SUBJ><PRED><OBJ>
│  ├─ eval/
│  │  ├─ metrics.py              # BLEU, ROUGE-L, METEOR; F1 triple; Acc
│  │  └─ evaluate.py             # orchestratore valutazione per-task
│  └─ plots/
│     └─ curves.py               # grafici loss/metriche (opzionale)
└─ tests/
   ├─ test_serialization.py      # round-trip linearize/parse
   ├─ test_metrics.py            # scorer triple e BLEU/ROUGE
   └─ test_builders.py           # dataset 4 task: schemi e lunghezze
```

---

## 2) Quickstart

### 2.1 Ambiente
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2.3 Monitoraggio con Weights & Biases (opzionale)
Il training può loggare automaticamente su [Weights & Biases](https://wandb.ai/). Configura la sezione `wandb` del file YAML
di training (`configs/train/*.yaml`) impostando almeno `project` e `mode` (`online`, `offline` o `disabled`). È possibile
definire anche `entity`, `run_name`, `tags` e abilitare `watch` per tracciare i gradienti del modello. In caso di problemi di
connessione, l'inizializzazione effettua automaticamente il fallback in modalità offline.

### 2.2 Comandi tipici (Makefile)
```bash
make data        # fetch DBpedia/Wikipedia + build dataset (4 task)
make tokenizer   # train BPE 24k + special tokens
make sanity      # overfit su 1 batch (sanity check)
make train       # training multi-task baseline
make eval        # valutazione completa per task
make plots       # grafici di training (opzionale)
```

Oppure usa direttamente gli entrypoint Python:

```bash
python -m scripts.eval_all --cfg configs/eval/baseline.yaml
python -m src.cli evaluate --cfg configs/eval/baseline.yaml
python -m src.cli predict --checkpoint checkpoints/baseline/best.pt \
    --tokenizer data/vocab/bpe.json --task text2rdf --input "Trama..."
```

---

## 3) Configurazione (YAML)

Vedi esempi in `configs/` per:  
- `data/dbpedia.yaml` — endpoint SPARQL, whitelist predicati, direzione (out|both)  
- `data/wikipedia.yaml` — lingua, endpoint REST, timeout  
- `data/build.yaml` — split, maxlen, filtri qualità  
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
Micro Transformer (4e+4d, d=512, 8h, FFN 2048, dropout 0.1).  
Training multi-task mixing **3:3:2:2** (T2RDF:R2Text:Comp1:Comp2), AdamW, warmup+cosine, grad-accum. **Sanity**: toy + overfit 1 batch.

---

## 7) Decoding & Post-processing (Step 7)
Beam=4, length_penalty=0.9, **vincoli leggeri** sui tag RDF (opzionali).  
Parser → triple (S,P,O), normalizzazione prefissi, dedup. **Validity ≥ 95%** su dev.

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

python scripts/predict_example.py --checkpoint checkpoints/baseline/best.pt \
    --tokenizer data/vocab/bpe.json --task rdf2text --input "<SOT> ... <RDF2Text>"
```

Il flag `--task` aggiunge automaticamente il marker speciale previsto dal
dataset se non già presente nell'input.

---

## 9) Ablation (Step 9) — breve e mirata
- **Positional**: sinusoidale vs **RoPE**  
- **Mixing**: 3:3:2:2 vs 2:2:3:3  
- **Decoding**: libero vs **vincoli leggeri**  
Metriche: ROUGE-L, F1 triple, Accuracy Comp-1, validity rate, costo/epoch.

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

## 11) Linee guida di qualità
Whitelist predicati, split per film (no leakage), max seq 256–512, logging di parsing error/outlier.

---

## 12) Licenze & Dati
DBpedia/Wikipedia: rispettare le licenze; mantenere solo abstract/intro.

---

## 13) Roadmap esecutiva (riassunto)
1. `scripts/fetch_dbpedia.py` + `scripts/fetch_wikipedia.py` → `data/raw/`  
2. `scripts/build_dataset.py` → `data/interim/pairs.jsonl` + `data/processed/*.jsonl`  
3. `scripts/train_tokenizer.py` → `data/vocab/`  
4. `scripts/sanity_overfit.py` → training → eval → ablation
