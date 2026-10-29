import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, TaskType, get_peft_model
from src.common import DATA_DIR, ROOT, REPORT_DIR, load_config, set_seed, format_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    if not torch.cuda.is_available():
        print("WARNING: CUDA GPU not detected. Training will run on CPU and may be very slow.")
    for name in ("train", "validation"):
        if not (DATA_DIR / f"{name}.jsonl").exists():
            raise FileNotFoundError("Run `python -m src.prepare_data` first.")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"])
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]), lora_dropout=float(cfg["lora_dropout"]),
        target_modules=["q", "v"], bias="none"
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def load_rows(split):
        df = pd.read_json(DATA_DIR / f"{split}.jsonl", lines=True)
        return Dataset.from_pandas(df, preserve_index=False)

    def tokenize(batch):
        prompts = [format_prompt(s, q) for s, q in zip(batch["schema"], batch["question"])]
        inputs = tokenizer(prompts, max_length=int(cfg["max_input_length"]), truncation=True)
        labels = tokenizer(text_target=batch["sql"], max_length=int(cfg["max_target_length"]), truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = load_rows("train").map(tokenize, batched=True, remove_columns=["schema", "question", "sql"])
    val_ds = load_rows("validation").map(tokenize, batched=True, remove_columns=["schema", "question", "sql"])
    out = ROOT / cfg["output_dir"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        logging_steps=int(cfg["logging_steps"]),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=2,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    final_dir = ROOT / cfg["final_model_dir"]
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logs = pd.DataFrame(trainer.state.log_history)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logs.to_csv(REPORT_DIR / "training_log.csv", index=False)
    if {"loss", "step"}.issubset(logs.columns):
        plt.figure()
        subset = logs.dropna(subset=["loss"])
        plt.plot(subset["step"], subset["loss"], label="Training loss")
        if "eval_loss" in logs:
            evals = logs.dropna(subset=["eval_loss"])
            if len(evals):
                plt.plot(evals["step"], evals["eval_loss"], marker="o", label="Validation loss")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("SchemaSQL training and validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "loss_curve.png", dpi=160)
        plt.close()
    print(f"Saved fine-tuned LoRA model to {final_dir}")
    print(f"Logs saved to {REPORT_DIR / 'training_log.csv'}")

if __name__ == "__main__":
    main()
