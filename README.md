# SchemaSQL — Natural Language to SQL with Retrieval + LoRA

A WebWiz Sophomore Induction 2026–27 AI/ML-track project. SchemaSQL turns a database schema and a natural-language question into a SQL query. It uses the Hugging Face `b-mc2/sql-create-context` dataset, a small seq2seq language model, optional PEFT/LoRA fine-tuning, schema/question embeddings with a FAISS index, and a Streamlit interface.

> **Honest status:** the repository is a complete runnable training/evaluation pipeline, but model training, benchmark scores, and Hugging Face upload are outputs you generate in your own environment. They are not fabricated or pre-filled. Fine-tuning needs a compatible GPU for a practical run; CPU mode is for smoke tests.

## Features

- Downloads the required Hugging Face dataset and selects at least 1,000 examples by default.
- Formats examples as `schema + question -> SQL` and makes train/validation/test splits.
- Builds sentence embeddings and a FAISS retrieval index over training examples.
- Retrieves similar examples and supplies them as context to a small pretrained model.
- Fine-tunes a model with LoRA using Hugging Face Transformers + PEFT.
- Logs training and validation loss; saves trainer logs and a loss plot.
- Compares base and fine-tuned models on the same held-out test set with exact string match and normalized exact match.
- Provides a Streamlit UI for entering a schema and question, retrieving examples, and generating SQL. Model and embedding resources are cached between interactions.
- Shows benchmark metrics and example predictions in the UI after evaluation.
- Includes a model-card template and optional Hugging Face Hub upload script.
- Includes unit tests, configuration, and GitHub Actions checks.

## Architecture

```text
Hugging Face dataset
        |
  clean + format
        |
 train / validation / test split
        |                          train examples
        |                                |
 LoRA fine-tuning                sentence embeddings
        |                                |
 fine-tuned model                FAISS retrieval index
        |                                |
        +---------- Streamlit app -------+
                         |
              schema + question -> SQL
```

## Quick start (Windows PowerShell)

Use Python 3.10 or 3.11 recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, or run `.venv\Scripts\python.exe` directly.

### 1. Prepare data

```powershell
python -m src.prepare_data --max-samples 3000
```

The official task asks for at least 1,000 samples; the default is 3,000 for manageable experiments. Use `--max-samples 1000` for a minimal run. This downloads the public dataset and writes a reproducible split under `data/processed/`.

### 2. Build retrieval index

```powershell
python -m src.build_index
```

This creates sentence-transformer embeddings and a FAISS index from training examples.

### 3. Fine-tune with LoRA

For a GPU with enough memory:

```powershell
python -m src.train --config configs/train.yaml
```

The default model is `google/flan-t5-small` (about 80M parameters). The script uses LoRA and records training/validation losses. Start with 1 epoch and a small subset if compute is limited.

### 4. Evaluate base vs fine-tuned model

```powershell
python -m src.evaluate --max-test-samples 100
```

This writes `reports/evaluation.json` and `reports/predictions.csv`. It evaluates both models on exactly the same test examples. A model checkpoint must exist for fine-tuned evaluation; use `--base-only` to validate the baseline first.

### 5. Run the app

```powershell
streamlit run app.py
```

Enter a schema such as:

```sql
CREATE TABLE employees (
  id INTEGER,
  name TEXT,
  department TEXT,
  salary REAL
);
```

Question: `List the names of employees in Engineering ordered by salary descending.`

The app displays retrieved examples and the generated SQL. Generation can run with the base model before training; choose a local LoRA checkpoint after training.

After running evaluation, the app also displays held-out exact-match accuracy and up to five example outputs. If a fine-tuned adapter exists, the sidebar includes an optional upload form. Authenticate first with `hf auth login` or set `HF_TOKEN`; the app never asks for or stores your token.

## Deploy the app (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository, select the `main` branch, and set the entrypoint to `app.py`.
3. Select Python 3.11 in the app's advanced settings, then deploy.

The first startup downloads the public base model and embedding model from Hugging Face and may take a few minutes. Generated datasets, retrieval indexes, fine-tuned adapters, and evaluation reports are excluded from Git, so a fresh hosted deployment uses the base model and has no local retrieval examples or evaluation metrics. To use those trained outputs in a deployment, provide them through a separate artifact/model host and configure the app to load them there.

## Configuration and compute

- Dataset: `b-mc2/sql-create-context` on Hugging Face Datasets.
- Base model: `google/flan-t5-small` (well under 3B parameters).
- PEFT: LoRA on attention projection layers.
- Retrieval: `sentence-transformers/all-MiniLM-L6-v2` + FAISS cosine similarity.
- Default max input length: 512 tokens; target length: 256 tokens.
- Default epochs: 1; batch size: 4; gradient accumulation: 4; learning rate: 2e-4.
- A CUDA GPU is strongly recommended for fine-tuning. The scripts support a baseline CPU smoke test, but CPU training may be very slow.
- First run downloads model/dataset files from Hugging Face. Internet access is required.

## Upload to Hugging Face (optional)

1. Create a model repository on Hugging Face.
2. Login locally with `hf auth login` or set `HF_TOKEN` in your environment (never commit the token).
3. Update `configs/train.yaml` with your final output directory if needed.
4. Run `python -m src.upload_hub --model-dir artifacts/final_model --repo-id YOUR_USERNAME/schemasql-lora`.
5. Add actual results, intended use, limitations, dataset attribution, and training hardware to the generated model card before publishing.

## Project structure

```text
webwiz_sql_copilot/
├── app.py
├── requirements.txt
├── .env.example
├── configs/train.yaml
├── data/                 # generated datasets (gitignored)
├── artifacts/            # generated models/index (gitignored)
├── reports/              # metrics/plots (generated)
├── src/
│   ├── common.py
│   ├── prepare_data.py
│   ├── build_index.py
│   ├── retrieval.py
│   ├── train.py
│   ├── evaluate.py
│   └── upload_hub.py
└── tests/
```

## Troubleshooting

- **CUDA out of memory:** lower `per_device_train_batch_size`, use `gradient_accumulation_steps`, reduce `max_samples`, and close other GPU applications.
- **No GPU:** run data prep and tests; run `python -m src.evaluate --base-only --max-test-samples 10` for a small baseline. LoRA fine-tuning on CPU may be impractically slow.
- **Model/dataset download fails:** check internet access, Hugging Face status, firewall/proxy settings, and try again.
- **FAISS installation fails on Windows:** install a compatible Python version (3.10/3.11) and use `faiss-cpu`; if unavailable in your environment, use WSL/Linux or the fallback exact-vector retrieval backend in `src/retrieval.py`.
- **Do not execute generated SQL against a production database.** Inspect it first and only run it on a safe, read-only sandbox database.

## Required PI walkthrough

Be ready to explain: (1) how schema/question become model input, (2) why validation and test sets are separate, (3) what LoRA freezes and trains, (4) how retrieval embeddings are indexed, (5) exact match vs normalized exact match, and (6) why generated SQL must be reviewed.
