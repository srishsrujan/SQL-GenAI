# SchemaSQL LoRA Adapter — Model Card

## Model description
LoRA adapter for schema-conditioned natural-language-to-SQL generation. The project currently configures [`juierror/flan-t5-text2sql-with-schema-v2`](https://huggingface.co/juierror/flan-t5-text2sql-with-schema-v2) as its base model; verify the adapter configuration and update this card if a different base model was used.

This card describes a separately trained adapter, not the model currently used by the Streamlit app. The app loads the configured base model only and does not apply or publish this adapter.

## Training data
Hugging Face dataset: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context). Record the exact number of selected, deduplicated samples after running data preparation, and review the dataset card and license/terms before redistribution.

## Training procedure
- Method: LoRA via PEFT
- Base model: `juierror/flan-t5-text2sql-with-schema-v2` (verify against the trained adapter configuration)
- Dataset split: train / validation / test (seed 42 by default)
- Hyperparameters: see `configs/train.yaml`; confirm and record the actual values used for this adapter.

## Evaluation
Evaluation is run separately with `python -m src.evaluate`. Add results from `reports/evaluation.json` only after evaluation has completed; the Streamlit app does not display evaluation results.

## Intended use and limitations
Educational prototyping. Generated SQL can be incorrect or unsafe. Review before execution; do not use unreviewed output on production systems.

## Bias, risks, and ethical considerations
The model may reproduce errors or biases in training data and may be unreliable for schemas, SQL dialects, or domains not represented in the training data.
