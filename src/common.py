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

def adapter_matches_base(adapter_dir, base_model):
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.is_file():
        return False
    with config_path.open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    return adapter_config.get("base_model_name_or_path") == base_model

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

def is_sql_output(text):
    return bool(
        re.match(
            r"^\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|PRAGMA|EXPLAIN)\b",
            text,
            re.IGNORECASE,
        )
    )

def _matching_parenthesis(text, open_index):
    depth = 0
    quote = None
    index = open_index
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif char in ("'", '"', "`"):
            quote = char
        elif char == "[":
            quote = "]"
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None

def _split_schema_columns(definition):
    columns = []
    start = 0
    depth = 0
    quote = None
    index = 0
    while index < len(definition):
        char = definition[index]
        if quote:
            if char == quote:
                if index + 1 < len(definition) and definition[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif char in ("'", '"', "`"):
            quote = char
        elif char == "[":
            quote = "]"
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            columns.append(definition[start:index].strip())
            start = index + 1
        index += 1
    columns.append(definition[start:].strip())
    return columns

def _unquote_identifier(identifier):
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier[1:-1].replace('""', '"')
    if identifier.startswith("`") and identifier.endswith("`"):
        return identifier[1:-1].replace("``", "`")
    if identifier.startswith("[") and identifier.endswith("]"):
        return identifier[1:-1]
    return identifier

def schema_to_tables(schema):
    create_table = re.compile(
        r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r'(?P<table>"(?:""|[^"])*"|`(?:``|[^`])*`|\[[^\]]+\]|[^\s(]+)\s*\(',
        re.IGNORECASE,
    )
    column_name = re.compile(
        r'^("(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]+\]|[^\s]+)'
    )
    tables = []
    for match in create_table.finditer(schema):
        open_index = match.end() - 1
        close_index = _matching_parenthesis(schema, open_index)
        if close_index is None:
            continue
        name = _unquote_identifier(match.group("table"))
        columns = []
        for definition in _split_schema_columns(schema[open_index + 1:close_index]):
            if re.match(
                r"^(?:CONSTRAINT|PRIMARY|FOREIGN|UNIQUE|CHECK|INDEX|KEY|EXCLUDE)\b",
                definition,
                re.IGNORECASE,
            ):
                continue
            column = column_name.match(definition)
            if column:
                columns.append(_unquote_identifier(column.group(1)))
        if columns:
            tables.append(f"{name}({','.join(columns)})")
    return ", ".join(tables) if tables else schema.strip()

def format_prompt(schema, question, examples=""):
    tables = schema_to_tables(schema)
    prompt = (
        "convert question and table into SQL query. "
        f"tables: {tables}. question: {question.strip()}"
    )
    if examples:
        prompt = f"{prompt}\nRelevant examples:\n{examples}"
    return prompt
