# SchemaSQL

SchemaSQL turns natural-language questions into SQL using a supplied database schema. It is a WebWiz Sophomore Induction 2026–27 AI/ML-track project built with Streamlit, Hugging Face Transformers, and PEFT.

**Live app:** [sentence-sql-genai.streamlit.app](https://sentence-sql-genai.streamlit.app/)

**Source:** [srishsrujan/SQL-GenAI](https://github.com/srishsrujan/SQL-GenAI)

## Model

The app uses [`juierror/flan-t5-text2sql-with-schema-v2`](https://huggingface.co/juierror/flan-t5-text2sql-with-schema-v2), a pretrained, schema-aware text-to-SQL model. The project also supports LoRA fine-tuning on [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) and optional example retrieval with sentence-transformers and FAISS.

The model is licensed CC-BY-SA-4.0. Preserve attribution and follow the license when redistributing it. Generated SQL can be wrong: inspect every query and test only against a safe database.

## Run locally

Python 3.10 or 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model. To prepare data, build the retrieval index, and train an optional LoRA adapter:

```powershell
python -m src.prepare_data --max-samples 3000
python -m src.build_index
python -m src.train --config configs/train.yaml
python -m src.evaluate --max-test-samples 100
```

Fine-tuning is compute-intensive; a CUDA GPU is recommended. Training outputs and datasets are generated locally and excluded from Git.

## Deployment

The public app is deployed on Streamlit Community Cloud from `app.py` on the `main` branch, using Python 3.11. It downloads the pretrained model on startup. Generated adapters, datasets, retrieval indexes, and evaluation reports are not included in the deployment.

## Project files

- `app.py` — Streamlit interface and SQL generation.
- `src/` — data preparation, retrieval, training, evaluation, and model upload.
- `configs/train.yaml` — dataset, model, and training settings.
- `tests/` — unit tests; run with `python -m pytest -q`.
