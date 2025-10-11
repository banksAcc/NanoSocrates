# NanoSocrates — Diario di sviluppo personale

## 1. Cosa ho realizzato
Ho impostato un progetto completo per addestrare e valutare un modello Transformer
encoder–decoder sul dominio dei film. Durante questa revisione ho ripulito gli
stub superflui, documentato il codice e lasciato solo gli strumenti realmente
utili per portare a termine la traccia d’esame. In sintesi ho:

* mantenuto gli script essenziali per il download dei dati, la costruzione dei
  dataset multitask e l’addestramento del tokenizer BPE;
* verificato i moduli `src/` uno per uno aggiungendo docstring e spiegazioni;
* eliminato i file vuoti (`configs/base.yaml`, `src/plots/curves.py`, ecc.) e
  l’intera cartella `tests/`, che conteneva soltanto stub non utilizzati;
* rinominato la configurazione del tokenizer in
  `configs/tokenizer/bpe_default.yaml` per renderla coerente con l’uso reale;
* aggiornato questo README in prima persona così da raccontare con chiarezza il
  flusso di lavoro.

## 2. Struttura essenziale del repository
```
nanosocrates/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ data/
│  │  ├─ build.yaml         # istruzioni per costruire i dataset JSONL
│  │  ├─ dbpedia.yaml       # endpoint e parametri SPARQL
│  │  ├─ toy.yaml           # scorciatoia con subset da 20 film
│  │  └─ wikipedia.yaml     # configurazione API Wikipedia
│  ├─ eval/
│  │  └─ baseline.yaml      # esempio completo di valutazione
│  ├─ train/
│  │  ├─ baseline.yaml      # preset encoder-decoder 3e+3d
│  │  ├─ mix_3322.yaml      # mixing multitask 3:3:2:2
│  │  └─ rope_on.yaml       # esempio con Rotary Positional Embeddings
│  └─ tokenizer/
│     └─ bpe_default.yaml   # configurazione BPE condivisa
├─ data/                    # cartella popolata dagli script
├─ scripts/
│  ├─ build_dataset.py      # unisce triple e testo in 4 task
│  ├─ build_toy_subset.py   # genera un sottoinsieme compatto per debug
│  ├─ eval_all.py           # orchestratore della valutazione
│  ├─ fetch_dbpedia.py      # scarica triple da DBpedia
│  ├─ fetch_wikipedia.py    # recupera gli estratti di Wikipedia
│  ├─ predict_example.py    # piccola CLI per testare un checkpoint
│  ├─ sanity_overfit.py     # overfit di un batch per controlli rapidi
│  └─ train_tokenizer.py    # addestra il tokenizer BPE
├─ src/
│  ├─ run.py                # entry point unificato (train/eval/predict)
│  ├─ data/                 # parsing SPARQL, pairing e serialization
│  ├─ decoding/             # funzioni di decoding greedy ben documentate
│  ├─ eval/                 # metriche e valutazione dei checkpoint
│  ├─ model/                # implementazione del TinySeq2Seq
│  ├─ tokenizer/            # wrapper IO per tokenizers
│  ├─ training/             # dataloaders, loop e scheduler
│  └─ utils/                # helper generici (config, IO, logging)
├─ supporto.txt             # appunti rapidi con i comandi principali
└─ .vscode/settings.json    # imposta PYTHONPATH=src per VS Code
```

## 3. Preparazione dell’ambiente
1. Creo (e attivo) l’ambiente virtuale:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Non serve più esportare manualmente `PYTHONPATH=src`: il file
   `.vscode/settings.json` lo imposta automaticamente in ogni terminale di VS
   Code. Se uso un altro editor posso esportare la variabile una tantum.

## 4. Dati e costruzione dei task
1. **Triple DBpedia**
   ```bash
   python scripts/fetch_dbpedia.py \
     --config configs/data/dbpedia.yaml \
     --out data/raw/dbpedia_triples.jsonl
   ```
2. **Intro Wikipedia**
   ```bash
   python scripts/fetch_wikipedia.py \
     --config configs/data/wikipedia.yaml \
     --in data/raw/dbpedia_triples.jsonl \
     --out data/raw/wikipedia_intro.jsonl
   ```
3. **Dataset multitask**
   ```bash
   python scripts/build_dataset.py \
     --config configs/data/build.yaml \
     --dbp data/raw/dbpedia_triples.jsonl \
     --wiki data/raw/wikipedia_intro.jsonl \
     --outdir data/processed \
     --emit_tasks
   ```
   Lo script produce i quattro file `text2rdf`, `rdf2text`, `rdfcomp1`,
   `rdfcomp2` in formato JSONL, più gli split intermedi in `data/interim/`.
4. **Subset “toy”** (20 film) – utile per debug e notebook:
   ```bash
   python -m scripts.build_toy_subset \
     --pairs data/interim/pairs.all.jsonl \
     --splits data/interim/splits.json \
     --processed-dir data/processed \
     --outdir data/processed/toy \
     --films 20
   ```

## 5. Tokenizer, training e valutazione
1. **Tokenizer BPE**
   ```bash
   python -m scripts.train_tokenizer --config configs/tokenizer/bpe_default.yaml
   ```
   Il file YAML indica quali JSONL leggere, la dimensione del vocabolario e i
   token speciali richiesti dalla traccia.
2. **Training**
   ```bash
   python -m src.run train --cfg configs/train/baseline.yaml
   ```
   Posso combinare gli override CLI (es. `--override batch_size=8`) oppure usare
   `configs/train/mix_3322.yaml` per il multitask 3:3:2:2.
3. **Overfit di controllo**
   ```bash
   python -m src.run overfit --cfg configs/train/baseline.yaml --toy
   ```
   Questo comando forza l’overfit di un singolo batch per verificare che
   dataloader, modello e loss siano collegati correttamente.
4. **Valutazione**
   ```bash
   python -m scripts.eval_all --cfg configs/eval/baseline.yaml
   ```
   Il report JSON contiene loss e metriche per ogni task. Ho ripulito il modulo
   `src/eval/evaluate.py` per spiegare chiaramente come vengono calcolati i
   risultati.
5. **Predizione rapida** (facoltativa ma comoda per la relazione):
   ```bash
   python -m scripts.predict_example \
     --checkpoint checkpoints/baseline/best.pt \
     --tokenizer data/vocab/bpe.json \
     --task text2rdf \
     --input "Plot ..."
   ```
   Lo script aggiunge automaticamente il marker del task se manca.

## 6. Note sul codice e sulle semplificazioni
* Ho rivisto tutte le directory di `src/` aggiungendo docstring e commenti
  descrittivi, in particolare per `decoding` ed `eval` che prima risultavano
  criptiche.
* Ho eliminato funzioni e file vuoti per evitare distrazioni durante la
  presentazione del progetto.
* La cartella `tests/` era composta da file vuoti: l’ho rimossa dopo aver
  verificato che non contenesse logica utile.
* I dati di esempio presenti in `data/` mi permettono di eseguire subito gli
  script senza dover attendere il download completo.

## 7. Checklist finale prima della consegna
- [x] Repository pulito da stub e duplicati.
- [x] Documentazione aggiornata in prima persona.
- [x] Configurazioni coerenti con i nuovi nomi (`bpe_default.yaml`).
- [x] Moduli principali corredati di docstring esplicative.
- [x] Script di predizione mantenuto e documentato per uso rapido.

Con questa base posso concentrarmi sulla relazione e sulle prove pratiche senza
perdere tempo a ricostruire il contesto del progetto.
