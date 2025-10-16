# NanoSocrates — pipeline compatta

Questo repository ospita una **pipeline end-to-end** in un unico file Python
(`nanosocrates_pipeline.py`) pensata per addestrare e valutare un modello
Transformer encoder–decoder sui quattro task originari:

1. **Text2RDF**
2. **RDF2Text**
3. **RDF Completion 1** (predizione del token mascherato)
4. **RDF Completion 2** (continuazione delle triple)

Il codice precedente è stato completamente rimpiazzato per massimizzare la
semplicità. Restano invariati i dataset nella cartella `data/`.

---

## Requisiti

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Le dipendenze si appoggiano a PyTorch e `tokenizers` per evitare di
reimplementare componenti di base.

---

## File principale

`nanosocrates_pipeline.py` contiene:

- addestramento del tokenizer BPE (con i token speciali forniti);
- data loader per i quattro task;
- modello Transformer minimalista (encoder–decoder) con Positional Encoding;
- training loop, salvataggio checkpoint e valutazione con exact match;
- comandi CLI per pipeline completa, training, valutazione, overfit/sanity check
  e generazione manuale.

La pipeline salva il tokenizer in `data/vocab/bpe.json` e il modello in
`checkpoints/nanosocrates_transformer.pt`.

---

## Utilizzo rapido

Addestra tokenizer, modello ed esegue valutazione e generazione di esempio:

```bash
python nanosocrates_pipeline.py pipeline
```

Comandi principali:

```bash
# Solo training (riusando il tokenizer se già presente)
python nanosocrates_pipeline.py train --epochs 5 --batch-size 12

# Valutazione di un checkpoint già esistente
python nanosocrates_pipeline.py evaluate --checkpoint checkpoints/nanosocrates_transformer.pt

# Overfit/sanity check su un singolo batch
python nanosocrates_pipeline.py overfit --steps 200

# Generazione libera fornendo input e task token
python nanosocrates_pipeline.py generate "Ca-bau-kan ... <Text2RDF>"
```

Tutti i comandi espongono argomenti per personalizzare dimensioni del modello,
numeri di epoche, learning rate e lunghezza massima delle sequenze. I log della
pipeline riportano la loss per epoca e l'exact match sul validation set.

---

## Struttura residua del repository

```text
nanosocrates/
├─ README.md
├─ nanosocrates_pipeline.py
├─ requirements.txt
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/           # contiene i file JSONL con gli esempi per task/split
│  └─ vocab/               # destinazione del tokenizer addestrato
└─ checkpoints/            # creato automaticamente alla prima esecuzione
```

La directory `data/processed` deve già essere popolata con i JSONL dei quattro
task. La pipeline include controlli per segnalare l'assenza dei file attesi.

---

## Note sull'hardware

Il modello di default è dimensionato per funzionare agevolmente su una GPU con
16GB di VRAM (ad esempio una NVIDIA RTX 3070). È comunque possibile ridurre il
numero di layer, l'ampiezza del modello o le lunghezze massime tramite gli
argomenti CLI.
