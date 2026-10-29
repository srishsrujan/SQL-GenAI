import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.common import INDEX_DIR, DATA_DIR, load_config

class ExampleRetriever:
    def __init__(self, model_name=None):
        cfg = load_config()
        self.model_name = model_name or cfg["embedding_model"]
        self.encoder = SentenceTransformer(self.model_name)
        self.rows = []
        self.vectors = None
        self.index = None
        self._load()

    def _load(self):
        rows_path = INDEX_DIR / "rows.json"
        vectors_path = INDEX_DIR / "vectors.npy"
        if rows_path.exists() and vectors_path.exists():
            self.rows = json.loads(rows_path.read_text(encoding="utf-8"))
            self.vectors = np.load(vectors_path)
            try:
                import faiss
                self.index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
            except Exception:
                self.index = None

    def build(self):
        train_path = DATA_DIR / "train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError("Run `python -m src.prepare_data` first.")
        import pandas as pd
        df = pd.read_json(train_path, lines=True)
        self.rows = df.to_dict(orient="records")
        texts = [f"Schema: {r['schema']}\nQuestion: {r['question']}" for r in self.rows]
        self.vectors = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(INDEX_DIR / "vectors.npy", self.vectors)
        (INDEX_DIR / "rows.json").write_text(json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
            faiss.write_index(self.index, str(INDEX_DIR / "index.faiss"))
            print("Saved FAISS cosine-similarity index.")
        except Exception as exc:
            self.index = None
            print(f"FAISS unavailable; saved vector matrix fallback ({exc}).")

    def search(self, schema, question, k=3):
        if self.vectors is None or not self.rows:
            self._load()
        if self.vectors is None:
            return []
        query = self.encoder.encode([f"Schema: {schema}\nQuestion: {question}"], normalize_embeddings=True).astype("float32")
        if self.index is not None:
            scores, ids = self.index.search(query, min(k, len(self.rows)))
            pairs = zip(ids[0].tolist(), scores[0].tolist())
        else:
            scores = (self.vectors @ query[0])
            ids = np.argsort(scores)[::-1][:k]
            pairs = ((int(i), float(scores[i])) for i in ids)
        return [{"score": float(score), **self.rows[int(i)]} for i, score in pairs if int(i) >= 0]
