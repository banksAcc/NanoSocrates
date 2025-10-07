# NanoSocrates — (Baseline unificata Text↔RDF + Completion)

**Stato**: setup architettura & data layer — GPU target: 16 GB VRAM  
**Data**: 01 ottobre 2025

Questo repository ospita una pipeline **end-to-end** per addestrare un **Transformer encoder–decoder** capace di svolgere 4 task nel dominio _film_: **Text2RDF, RDF2Text, RDF Completion 1 (masked), RDF Completion 2 (continuation)**.  
Il progetto è _compliant_ alla traccia: **token speciali**, **tokenizer BPE from scratch**, **multi-task training** e **metriche per task**.

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
│  ├─ run.py                     # entrypoint unificato (train/overfit/evaluate/predict)
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

## 2) Quickstart"""Main training and evaluation loop for the transformer model."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import wandb
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class TrainingLoop:
    """A class to encapsulate the training and validation loops.

    This class handles the complexities of model training, including:
    - Iterating over epochs and batches.
    - Gradient accumulation to simulate larger batch sizes.
    - Automatic Mixed Precision (AMP) for faster training on compatible GPUs.
    - Checkpointing the best model based on a validation metric.
    - Early stopping to prevent overfitting.
    - Logging metrics to Weights & Biases (wandb).
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: _LRScheduler | None = None,
        device: str | torch.device | None = None,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        log_every_n_steps: int = 100,
        checkpoint_path: str = "best_model.pt",
        early_stopping_patience: int = 5,
        early_stopping_metric: str = "loss",
        early_stopping_mode: Literal["min", "max"] = "min",
        wandb_run: "wandb.sdk.wandb_run.Run" | None = None,
    ):
        """Initializes the TrainingLoop.

        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            scheduler: Optional learning rate scheduler.
            device: The device to train on ('cuda', 'cpu'). If None, it will be
                auto-detected.
            use_amp: Whether to use Automatic Mixed Precision.
            grad_accum_steps: Number of steps to accumulate gradients over.
            log_every_n_steps: How often to log training metrics to wandb.
            checkpoint_path: Path to save the best model checkpoint.
            early_stopping_patience: Number of epochs to wait for improvement
                before stopping.
            early_stopping_metric: The validation metric to monitor for early
                stopping and checkpointing.
            early_stopping_mode: 'min' if a lower metric is better (e.g., loss),
                'max' if a higher metric is better (e.g., accuracy).
            wandb_run: An active wandb run object for logging.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_accum_steps = grad_accum_steps
        self.log_every_n_steps = log_every_n_steps
        self.checkpoint_path = checkpoint_path
        self.wandb_run = wandb_run

        # Early stopping setup
        self.es_patience = early_stopping_patience
        self.es_metric = early_stopping_metric
        self.es_mode = early_stopping_mode
        self.es_counter = 0
        self.best_score = -float("inf") if self.es_mode == "max" else float("inf")

        # Auto-detect device if not provided
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")

        # AMP setup
        self.use_amp = use_amp and self.device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled.")

        self.model.to(self.device)

    def run(self, num_epochs: int) -> dict[str, Any]:
        """Starts and manages the training process for a given number of epochs.

        Args:
            num_epochs: The total number of epochs to train for.

        Returns:
            A dictionary containing the best score achieved and the epoch at which
            it occurred.
        """
        logger.info("Starting training...")
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch()

            if self.scheduler:
                # Some schedulers use metrics (e.g., ReduceLROnPlateau)
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.es_metric])
                else:
                    self.scheduler.step()

            # Log metrics to wandb
            if self.wandb_run:
                metrics_to_log = {
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
                self.wandb_run.log(metrics_to_log)

            logger.info(f"Validation metrics: {val_metrics}")

            # Early stopping and checkpointing
            current_score = val_metrics[self.es_metric]
            if self._check_early_stopping(current_score):
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training finished. Best score: {self.best_score:.4f}")
        return {"best_score": self.best_score, "best_epoch": epoch - self.es_counter}

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Performs one full training pass over the training data.

        Returns:
            A dictionary of average training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        # Use a moving average for smoother loss reporting in tqdm
        smoothing_factor = 0.98
        smoothed_loss = 0.0
        is_first_batch = True

        pbar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )

        for i, batch in enumerate(pbar):
            batch = self._transfer_batch_to_device(batch)
            step = (epoch - 1) * len(self.train_loader) + i

            with autocast(enabled=self.use_amp):
                outputs = self.model(**batch)
                loss = outputs["loss"]
                if loss is None:
                    continue
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # Update progress bar with smoothed loss
            loss_item = loss.item() * self.grad_accum_steps
            total_loss += loss_item
            if is_first_batch:
                smoothed_loss = loss_item
                is_first_batch = False
            else:
                smoothed_loss = (smoothing_factor * smoothed_loss) + (1 - smoothing_factor) * loss_item
            
            pbar.set_postfix({"loss": f"{smoothed_loss:.4f}"})

            # Log to wandb periodically
            if self.wandb_run and step % self.log_every_n_steps == 0:
                log_data = {"train/step_loss": loss_item, "learning_rate": self.optimizer.param_groups[0]["lr"]}
                if "metrics" in outputs and outputs["metrics"] is not None:
                    for k, v in outputs["metrics"].items():
                        log_data[f"train/{k}"] = v
                self.wandb_run.log(log_data, step=step)

        avg_loss = total_loss / len(self.train_loader)
        return {"loss": avg_loss}

    @torch.inference_mode()
    def _validate_epoch(self) -> dict[str, float]:
        """Performs one full validation pass.

        Returns:
            A dictionary of average validation metrics.
        """
        self.model.eval()
        metrics_agg = defaultdict(float)
        total_count = 0

        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            batch = self._transfer_batch_to_device(batch)
            with autocast(enabled=self.use_amp):
                outputs = self.model(**batch)

            if outputs["loss"] is not None:
                metrics_agg["loss"] += outputs["loss"].item() * len(batch["input_ids"])
            
            if "metrics" in outputs and outputs["metrics"] is not None:
                for k, v in outputs["metrics"].items():
                    metrics_agg[k] += v * len(batch["input_ids"])

            total_count += len(batch["input_ids"])

        # Average the metrics over the entire dataset
        avg_metrics = {k: v / total_count for k, v in metrics_agg.items()}
        return avg_metrics

    def _check_early_stopping(self, current_score: float) -> bool:
        """Checks if early stopping criteria are met and saves the best model.

        Args:
            current_score: The validation score from the current epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        is_better = (current_score < self.best_score) if self.es_mode == "min" else (current_score > self.best_score)

        if is_better:
            self.best_score = current_score
            self.es_counter = 0
            logger.info(f"New best score: {self.best_score:.4f}. Saving model...")
            torch.save(self.model.state_dict(), self.checkpoint_path)
        else:
            self.es_counter += 1
            logger.info(f"No improvement. Early stopping counter: {self.es_counter}/{self.es_patience}")

        return self.es_counter >= self.es_patience

    def _transfer_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Moves a batch of data to the configured device."""
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

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
   python -m src.run train --cfg configs/train/baseline.yaml
   ```
5. **Valuta il checkpoint** (report JSON + metriche aggregate)
   ```bash
   python -m scripts.eval_all --cfg configs/eval/baseline.yaml
   # equivalente RUN unificata
   python -m src.run evaluate --cfg configs/eval/baseline.yaml --output reports/baseline_eval.json
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
   python -m src.run train --cfg configs/train/baseline.yaml --toy
   python -m scripts.eval_all --cfg configs/eval/baseline.yaml --toy
   ```

### 2.4 Tutorial — sanity check (overfit di un batch)

1. Riusa la configurazione standard e forza gli override automatici:
   ```bash
   python -m src.run overfit --cfg configs/train/baseline.yaml --toy
   ```
   Il comando abilita `overfit_one_batch=true`, disattiva l'early stopping e,
   per impostazione predefinita, esegue 200 aggiornamenti consecutivi sullo
   stesso batch.
   - Il numero di esempi nel batch coincide con `batch_size` del config (16 nel
     preset `configs/train/mix_3322.yaml`). Se vuoi restringerlo, passa
     `--override batch_size=4` o modifica il valore nel YAML.
   - Usa `--steps N` per cambiare il numero di ottimizzazioni (es. `--steps 400`).
   - In alternativa `--epochs M` forza il numero di epoche (una per aggiornamento
     quando si overfitta un singolo batch).
     Qualsiasi ulteriore `--override` passato da RUN viene rispettato.
2. In alternativa esiste lo script dedicato:
   ```bash
   python -m scripts.sanity_overfit --cfg configs/train/baseline.yaml --toy
   ```
3. Verifica che la loss scenda rapidamente verso ~0: conferma che tokenizer,
   dataloader, loop di training e logging siano correttamente collegati.

### 2.5 Tutorial — valutazione con Weights & Biases

1. Modifica il config (o usa gli override) per abilitare W&B.
   ```bash
   python -m src.run train \
       --cfg configs/train/baseline.yaml \
       --override wandb.mode=online wandb.project=nanosocrates-demo wandb.run_name=debug
   ```
   I campi supportati sono `mode` (`online`, `offline`, `disabled`), `project`,
   `entity`, `run_name`, `tags` (lista) e `watch` (bool). Se non specifichi
   `run_name`, il RUN genera automaticamente un nome leggibile basato sul file
   di config, sul tipo di esecuzione (`train`/`overfit`) e sul timestamp; in caso
   contrario apparirebbero i nomi casuali di default di W&B. Se la connessione
   fallisce viene eseguito automaticamente il fallback in modalità offline.
2. Per loggare anche la valutazione usa lo stesso approccio:
   ```bash
   python -m scripts.eval_all \
       --cfg configs/eval/baseline.yaml \
       --override wandb.mode=online wandb.project=nanosocrates-demo
   ```
   Le metriche vengono appiattite tramite `src.utils.wandb_utils.flatten_eval_metrics`
   e inviate come singolo step alla run già configurata.
3. Per eseguire la valutazione dal RUN unificato mantenendo gli override:
   ```bash
   python -m src.run evaluate \
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
(`d_model=384`, `nhead=6`, `ff_dim=1536`, dropout `0.1`). Il training baseline
(`configs/train/baseline.yaml`) usa AdamW con scheduler cosine + warmup e opera
su un singolo task (Text2RDF). Per allenare sui quattro task insieme utilizza
`configs/train/mix_3322.yaml`, che imposta il mixing **3:3:2:2** su
Text2RDF/RDF2Text/RDFComp1/RDFComp2. Gli script di sanity (`src.run overfit` o
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
Lo stesso comportamento è disponibile dal RUN unificato:

```bash
python -m src.run evaluate --cfg configs/eval/baseline.yaml --output reports/eval.json
```

### 8.2 Inference manuale

Per testare rapidamente il modello su un input specifico puoi usare il
subcomando `predict` oppure lo script di esempio `scripts/predict_example.py`:

```bash
python -m src.run predict --checkpoint checkpoints/baseline/best.pt \
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

## 11) Playbook RUN completo — end-to-end

Questa sezione raccoglie _tutti_ i comandi operativi utili per eseguire l’intera
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

| Scenario              | Comando                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Smoke test (toy)**  | `python -m src.run overfit --cfg configs/train/baseline.yaml --toy [--steps 400]`                            |
| **Baseline Text2RDF** | `python -m src.run train --cfg configs/train/baseline.yaml`                                                  |
| **Multitask 3:3:2:2** | `python -m src.run train --cfg configs/train/mix_3322.yaml`                                                  |
| **Variant RoPE**      | `python -m src.run train --cfg configs/train/rope_on.yaml`                                                   |
| **Custom override**   | `python -m src.run train --cfg configs/train/baseline.yaml --override lr=1e-4 num_epochs=5 scheduler=linear` |

Suggerimenti operativi:

- Il flag `--toy` redirige automaticamente ai percorsi definiti in
  `configs/data/toy.yaml`.
- Usa `--override chiave=valore` per modifiche rapide (inclusi parametri
  nidificati, es. `wandb.mode=online`). La RUN provvede al casting numerico.
- `gradient_accumulation_steps` consente batch effettivi più grandi senza
  saturare la VRAM.
- Ogni epoca produce un checkpoint (`epochXXX.pt`) e, in caso di miglioramento,
  aggiorna `best.pt` nella directory `save_dir`.

### 11.6 Monitoraggio, valutazione e logging remoto

Training con W&B online (fallback automatico in offline):

```bash
python -m src.run train \
    --cfg configs/train/mix_3322.yaml \
    --override wandb.mode=online wandb.project=nanosocrates wandb.run_name=multitask_v1
```

Valutazione completa (val + test) con report JSON e logging opzionale:

```bash
python -m scripts.eval_all --cfg configs/eval/baseline.yaml
python -m src.run evaluate --cfg configs/eval/baseline.yaml --output reports/baseline_eval.json
python -m src.run evaluate \
    --cfg configs/eval/baseline.yaml \
    --override wandb.mode=online wandb.project=nanosocrates_eval \
    --output reports/baseline_eval.json
jq '.' reports/baseline_eval.json
```

### 11.7 Inference “online” via RUN

```bash
python -m src.run predict \
    --checkpoint checkpoints/mix3322/best.pt \
    --tokenizer data/vocab/bpe.json \
    --task rdf2text \
    --input "<SOT> <SUBJ> dbr:Inception <PRED> dbo:director <OBJ> dbr:Christopher_Nolan <EOT>"

python -m src.run predict \
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

### 12.2 Ottimizzazione & controllo training

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

### 12.3 Gestione dati & mixing multitask

| Chiave                    | Descrizione              | Note                                                  |
| ------------------------- | ------------------------ | ----------------------------------------------------- |
| `train_file` / `val_file` | Path singolo task        | Usati nei preset baseline                             |
| `datasets`                | Lista task con pesi      | `weight` guida il sampler proporzionale (es. 3:3:2:2) |
| `max_len`                 | Troncamento input/target | Deve corrispondere al valore del tokenizer            |
| Flag RUN `--toy`          | Dataset rapido           | Applica automaticamente `configs/data/toy.yaml`       |

### 12.4 Logging & strumentazione

| Chiave                     | Descrizione                                                 |
| -------------------------- | ----------------------------------------------------------- |
| `wandb.mode`               | `online`, `offline`, `disabled`; fallback gestito dalla RUN |
| `wandb.project` / `entity` | Identificativi workspace W&B                                |
| `wandb.run_name`           | Nome leggibile della run                                    |
| `wandb.tags`               | Lista di tag (filtri dashboard)                             |
| `wandb.watch`              | Se `true`, abilita `wandb.watch` sul modello                |

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
- **Diagnostica rapida**: `python -m src.run overfit --cfg ... --toy` deve far
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
