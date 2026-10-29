import argparse
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from src.common import DATA_DIR, load_config, set_seed

def main():
    parser = argparse.ArgumentParser(description="Download and split SQL-create-context dataset.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    requested = args.max_samples or int(cfg.get("max_samples", 3000))
    if requested < 1000:
        raise SystemExit("The WebWiz task requires at least 1,000 selected samples.")
    ds = load_dataset(cfg["dataset_name"], split="train")
    cols = set(ds.column_names)
    required = {"context", "question", "answer"}
    if not required.issubset(cols):
        raise ValueError(f"Expected columns {required}; found {cols}")
    ds = ds.shuffle(seed=seed).select(range(min(requested, len(ds))))
    df = pd.DataFrame({
        "schema": ds["context"],
        "question": ds["question"],
        "sql": ds["answer"],
    })
    df = df.dropna().drop_duplicates(subset=["schema", "question", "sql"]).reset_index(drop=True)
    if len(df) < 1000:
        raise ValueError(f"Only {len(df)} unique usable examples remain; select a larger subset.")
    test_size = float(cfg.get("test_size", 0.1))
    val_size = float(cfg.get("validation_size", 0.1))
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    val_fraction_of_remaining = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_df, test_size=val_fraction_of_remaining, random_state=seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_json(DATA_DIR / "train.jsonl", orient="records", lines=True, force_ascii=False)
    val_df.to_json(DATA_DIR / "validation.jsonl", orient="records", lines=True, force_ascii=False)
    test_df.to_json(DATA_DIR / "test.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"Dataset: {cfg['dataset_name']}")
    print(f"Unique selected examples: {len(df)}")
    print(f"Train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    print(f"Saved splits to {DATA_DIR}")

if __name__ == "__main__":
    main()
