"""
Ingesta de documentos (PDF/TXT/MD) desde ./docs -> crea app/data/index.pkl
Uso:
  python ingest.py
"""
import os, re, pickle
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader

BASE = Path(__file__).resolve().parent
DOCS = BASE / "docs"
OUT = BASE / "app" / "data" / "index.pkl"
OUT.parent.mkdir(parents=True, exist_ok=True)

def read_pdf(path: Path) -> str:
    try:
        text = []
        reader = PdfReader(str(path))
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        return ""

def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    tokens = re.split(r"(\s+)", text)
    chunks, cur, cur_len = [], [], 0
    for tok in tokens:
        cur.append(tok)
        cur_len += len(tok)
        if cur_len >= size:
            chunks.append("".join(cur).strip())
            cur = cur[-overlap:]
            cur_len = sum(len(t) for t in cur)
    if cur:
        chunks.append("".join(cur).strip())
    return [c for c in chunks if c]

def main():
    passages: List[Dict[str, Any]] = []
    for path in DOCS.glob("**/*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in [".pdf"]:
            content = read_pdf(path)
        elif path.suffix.lower() in [".txt", ".md"]:
            content = read_text_file(path)
        else:
            continue
        if not content.strip():
            continue
        for i, chunk in enumerate(chunk_text(content)):
            passages.append({"text": chunk, "source": str(path.relative_to(BASE)), "chunk_id": i})

    if not passages:
        print("No se encontraron documentos en ./docs")
        return

    texts = [p["text"] for p in passages]
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    matrix = vec.fit_transform(texts)

    index = {
        "vectorizer": vec,
        "matrix": matrix,
        "passages": passages
    }
    with open(OUT, "wb") as f:
        pickle.dump(index, f)
    print(f"Ingesta completa. Pasajes: {len(passages)} -> {OUT}")

if __name__ == "__main__":
    main()
