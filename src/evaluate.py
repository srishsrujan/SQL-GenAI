import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from src.common import DATA_DIR, ROOT, REPORT_DIR, load_config, format_prompt, normalize_sql, write_json
from src.retrieval import ExampleRetriever

def generate(model, tokenizer, prompt, device, max_new_tokens=160):
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**tokens, max_new_tokens=max_new_tokens, num_beams=2)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--max-test-samples", type=int, default=100)
    p.add_argument("--base-only", action="store_true")
    args = p.parse_args()
    if args.max_test_samples < 1:
        p.error("--max-test-samples must be at least 1")
    cfg = load_config(args.config)
    test_path = DATA_DIR / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError("Run `python -m src.prepare_data` first.")
    df = pd.read_json(test_path, lines=True).head(args.max_test_samples)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.to(device).eval()
    tuned = None
    model_dir = ROOT / cfg["final_model_dir"]
    if not args.base_only:
        if not model_dir.exists():
            raise FileNotFoundError("Fine-tuned checkpoint missing. Run training first or use --base-only.")
        tuned = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"]), model_dir)
        tuned.to(device).eval()
    rows = []
    for _, row in df.iterrows():
        prompt = format_prompt(row["schema"], row["question"])
        base_pred = generate(base, tokenizer, prompt, device)
        rec = {"schema": row["schema"], "question": row["question"], "gold_sql": row["sql"], "base_prediction": base_pred,
               "base_exact_match": int(base_pred.strip() == row["sql"].strip()),
               "base_normalized_exact_match": int(normalize_sql(base_pred) == normalize_sql(row["sql"]))}
        if tuned is not None:
            tuned_pred = generate(tuned, tokenizer, prompt, device)
            rec.update({"finetuned_prediction": tuned_pred,
                        "finetuned_exact_match": int(tuned_pred.strip() == row["sql"].strip()),
                        "finetuned_normalized_exact_match": int(normalize_sql(tuned_pred) == normalize_sql(row["sql"]))})
        rows.append(rec)
    out = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_DIR / "predictions.csv", index=False)
    metrics = {"test_samples": len(out),
               "base_exact_match_accuracy": float(out["base_exact_match"].mean()),
               "base_normalized_exact_match_accuracy": float(out["base_normalized_exact_match"].mean())}
    if tuned is not None:
        metrics["finetuned_exact_match_accuracy"] = float(out["finetuned_exact_match"].mean())
        metrics["finetuned_normalized_exact_match_accuracy"] = float(out["finetuned_normalized_exact_match"].mean())
    examples = out.head(5).to_dict(orient="records")
    metrics["examples"] = [
        {
            key: row[key]
            for key in ("question", "gold_sql", "base_prediction", "finetuned_prediction")
            if key in row and pd.notna(row[key])
        }
        for row in examples
    ]
    write_json(REPORT_DIR / "evaluation.json", metrics)
    print(metrics)
    print(f"Predictions: {REPORT_DIR / 'predictions.csv'}")

if __name__ == "__main__":
    main()
