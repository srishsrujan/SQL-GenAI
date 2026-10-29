import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi
from src.common import ROOT, load_config

MODEL_CARD = """---
license: cc-by-sa-4.0
library_name: peft
pipeline_tag: text2text-generation
tags:
- text-to-sql
- sql
- lora
- webwiz
---

# SchemaSQL LoRA

A parameter-efficient fine-tune of `{base_model}` for converting a supplied SQL schema and natural-language question into SQL.

This adapter is a separately trained artifact. The current Streamlit app loads the configured base model only and does not apply this adapter.

## Intended use
Educational demonstrations and prototyping on schemas supplied by the user. Review every generated query before execution.

## Training data
`b-mc2/sql-create-context` from Hugging Face Datasets. Please review the dataset card and its license/terms before redistribution.

## Training
LoRA fine-tuning using Hugging Face Transformers and PEFT. Replace this section with the actual dataset size, hyperparameters, hardware, and evaluation scores from your run before public release. Evaluation is performed separately with `python -m src.evaluate`; the Streamlit app does not display benchmark results.

## Limitations
May generate incorrect SQL, assume missing schema details, or fail on dialect-specific syntax. Not intended for unsupervised production database access.
"""


def upload_model(model_dir, repo_id, private=False):
    token = os.getenv("HF_TOKEN")
    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}. Train first.")
    (model_dir / "README.md").write_text(
        MODEL_CARD.format(base_model=load_config()["base_model"]),
        encoding="utf-8",
    )
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id, repo_type="model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/final_model")
    parser.add_argument("--repo-id", required=True, help="YOUR_USERNAME/schemasql-lora")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    upload_model(args.model_dir, args.repo_id, private=args.private)
    print(f"Uploaded https://huggingface.co/{args.repo_id}")
if __name__ == "__main__":
    main()
