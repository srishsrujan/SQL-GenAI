from pathlib import Path
import json
import random
import re
import yaml
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
INDEX_DIR = ARTIFACT_DIR / "retrieval"
for path in (DATA_DIR, ARTIFACT_DIR, REPORT_DIR, INDEX_DIR):
    path.mkdir(parents=True, exist_ok=True)

def load_config(path="configs/train.yaml"):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def normalize_sql(sql):
    sql = (sql or "").strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()

def format_prompt(schema, question, examples=""):
    context = f"Relevant examples:\n{examples}\n" if examples else ""
    return (
        "Convert the question into a SQL query using only the supplied schema. "
        "Return SQL only, with no explanation.\n"
        f"{context}Schema:\n{schema.strip()}\n"
        f"Question: {question.strip()}\nSQL:"
    )
