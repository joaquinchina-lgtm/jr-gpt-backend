# build_index.py
import os, json, csv, re
from pathlib import Path
import numpy as np

from openai import OpenAI
import faiss

# --- CONFIG ---
DATA_DIR = Path("knowledge")
OUT_DIR = Path("rag")
OUT_DIR.mkdir(exist_ok=True)
INDEX_PATH = OUT_DIR / "index.faiss"
TEXTS_PATH = OUT_DIR / "texts.json"

EMBED_MODEL = "text-embedding-3-small"  # barato y preciso
CHUNK_WORDS = 350                       # ~250-500 tokens
CHUNK_OVERLAP = 60

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_csv_file(path: Path) -> str:
    lines = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers:
            lines.append(" | ".join(headers))
        for row in reader:
            lines.append(" | ".join(row))
    return "\n".join(lines)

def load_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return read_txt(path)
    if ext in (".md", ".markdown"):
        return read_md(path)
    if ext == ".csv":
        return read_csv_file(path)
    return ""

def clean_text(t: str) -> str:
    t = t.replace("\r", " ").replace("\t", " ")
    t = re.sub(r" +", " ", t)
    return t.strip()

def chunk_text(text: str, words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + words]
        chunks.append(" ".join(chunk))
        i += (words - overlap)
    return chunks

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    return vectors / norms

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    client = OpenAI()  # usa OPENAI_API_KEY del entorno

    files = []
    for ext in ("*.txt","*.md","*.markdown","*.csv"):
        files.extend(DATA_DIR.glob(ext))

    if not files:
        print("No hay ficheros en knowledge/. Añade .txt/.md/.csv y vuelve a ejecutar.")
        return

    records = []
    for path in files:
        raw = load_file(path)
        if not raw:
            continue
        text = clean_text(raw)
        if not text:
            continue
        chunks = chunk_text(text)
        for ch in chunks:
            if len(ch) < 30:
                continue
            records.append({"source": str(path.name), "text": ch})

    if not records:
        print("No se generaron chunks. Revisa knowledge/.")
        return

    # Embeddings por lotes
    texts = [r["text"] for r in records]
    vectors = []
    BATCH = 64
    print(f"Creando embeddings de {len(texts)} chunks...")
    for batch in batched(texts, BATCH):
        emb = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = [d.embedding for d in emb.data]
        vectors.extend(vecs)

    X = np.array(vectors, dtype="float32")
    X = l2_normalize(X)  # para coseno con producto escalar

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))

    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"OK. Guardado:\n - {INDEX_PATH}\n - {TEXTS_PATH}\nChunks: {len(records)}")

if __name__ == "__main__":
    main()
