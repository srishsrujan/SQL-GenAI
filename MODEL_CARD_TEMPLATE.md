# SchemaSQL LoRA — Model Card

## Model description
LoRA adapter for `google/flan-t5-small` trained for schema-conditioned natural-language-to-SQL generation.

## Training data
Hugging Face dataset: `b-mc2/sql-create-context`. Record the exact number of selected, deduplicated samples after running data preparation.

## Training procedure
- Method: LoRA via PEFT
- Base model: `google/flan-t5-small`
- Dataset split: train / validation / test (seed 42 by default)
- Hyperparameters: see `configs/train.yaml`; replace with the actual values used.

## Evaluation
Insert actual results from `reports/evaluation.json`. Do not claim a score until evaluation has run.

## Intended use and limitations
Educational prototyping. Generated SQL can be incorrect or unsafe. Review before execution; do not use unreviewed output on production systems.

## Bias, risks, and ethical considerations
The model may reproduce errors or biases in training data and may be unreliable for schemas, SQL dialects, or domains not represented in the training data.
