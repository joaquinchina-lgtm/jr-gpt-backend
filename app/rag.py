"""
RAG sencillo basado en TF-IDF (sin FAISS) para máxima compatibilidad.
- Ingesta: ejecuta `python ingest.py` para procesar /docs y crear app/data/index.pkl
- Búsqueda: `search_docs(query, k)` devuelve los mejores pasajes
"""
from __future__ import annotations
import os, pickle, re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path(__file__).resolve().parent / "data" / "index.pkl"

@dataclass
class RAGIndex:
    vectorizer: TfidfVectorizer
    matrix: Any  # scipy sparse
    passages: List[Dict[str, Any]]  # {text, source, chunk_id}

def load_index() -> RAGIndex | None:
    if not INDEX_PATH.exists():
        return None
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)

RAG_CACHE = load_index()

def search_docs(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not query or not RAG_CACHE:
        return []
    vec_q = RAG_CACHE.vectorizer.transform([query])
    sims = cosine_similarity(vec_q, RAG_CACHE.matrix).flatten()
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        p = RAG_CACHE.passages[i]
        results.append({
            "score": float(sims[i]),
            "title": p.get("source", "desconocido"),
            "snippet": p.get("text","")[:500],
            "source": p.get("source", ""),
            "chunk_id": p.get("chunk_id", i),
        })
    return results
