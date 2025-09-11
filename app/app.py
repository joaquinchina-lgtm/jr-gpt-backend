# app/app.py
import os
import json
import re
from typing import List, Dict, Any

import numpy as np
import faiss
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from pydantic import BaseModel
from typing import Optional

# =========================
# Configuración y constantes
# =========================
APP_TITLE = "JR GPT Backend"
APP_VERSION = "0.1.0"

# CORS: por defecto tu GitHub Pages. Puedes sobreescribir con env ORIGIN="*"
ORIGIN = os.getenv("ORIGIN", "https://joaquinchina-lgtm.github.io")

# Modelos OpenAI
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

# RAG
RAG_DIR = os.getenv("RAG_DIR", "rag")
INDEX_PATH = os.path.join(RAG_DIR, "index.faiss")
TEXTS_PATH = os.path.join(RAG_DIR, "texts.json")
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.15"))      # umbral similitud coseno
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "9000"))  # límite de contexto a enviar al LLM

# =========================
# App y middlewares
# =========================
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ORIGIN == "*" else [ORIGIN],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)

# =========================
# OpenAI client
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Carga de índice y textos
# =========================
index = None          # tipo: faiss.Index
records: List[Dict[str, Any]] = None  # lista de {source, text}

def _log(msg: str):
    print(f"[RAG] {msg}", flush=True)

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-10
    return v / n

def embed(text: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(e.data[0].embedding, dtype="float32")
    return _normalize(v)

@app.on_event("startup")
def load_rag():
    global index, records
    try:
        if not (os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH)):
            _log("No encuentro el índice/textos. Genera primero rag/index.faiss y rag/texts.json.")
            index = None
            records = []
            return
        index = faiss.read_index(INDEX_PATH)
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        _log(f"Cargado índice y {len(records)} chunks.")
    except Exception as e:
        _log(f"Error cargando índice/textos: {e}")
        index = None
        records = []

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Búsqueda vectorial (coseno vía inner product con vectores normalizados)."""
    if index is None or not records:
        return []
    q = embed(query)
    D, I = index.search(np.array([q]), k)
    out = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        out.append({"score": float(score), **records[idx]})
    return out

# =========================
# Enriquecimiento: entidades/contacto
# =========================
NAME_RE = re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DEPT_WORDS = ["departamento", "unidad", "área", "facultad", "grupo", "laboratorio", "cátedra", "centro"]

def extract_entities(hits: List[Dict[str, Any]]):
    names, emails, depts = set(), set(), set()
    for h in hits:
        t = h["text"]
        for m in EMAIL_RE.findall(t):
            emails.add(m)
        for n in NAME_RE.findall(t):
            if len(n.split()) >= 2:
                names.add(n.strip())
        if any(w in t.lower() for w in DEPT_WORDS):
            depts.add(t)
    names = {n for n in names if len(n) >= 5}
    return list(names)[:12], list(emails)[:12], list(depts)[:12]

def keyword_hits(terms: List[str], limit: int = 12) -> List[Dict[str, Any]]:
    """Segundo salto muy simple sobre los chunks: incluye los que contengan los términos."""
    results: List[Dict[str, Any]] = []
    if not records:
        return results
    total = 0
    lowered_terms = [t.lower() for t in terms if t]
    if not lowered_terms:
        return results
    for r in records:
        txt = r["text"]
        low = txt.lower()
        if any(t in low for t in lowered_terms):
            results.append({"score": 0.99, **r})
            total += 1
            if total >= limit:
                break
    return results

def dedup_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for h in hits:
        key = (h.get("text"), h.get("source"))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out

# =========================
# Endpoints utilitarios
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/status")
def status():
    return {
        "index_loaded": bool(index is not None),
        "num_chunks": (len(records) if records else 0),
        "origin": ORIGIN,
        "min_score": RAG_MIN_SCORE,
        "context_max_chars": CONTEXT_MAX_CHARS,
        "models": {"embed": EMBED_MODEL, "gen": GEN_MODEL},
    }

class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "estricto"
    top_k: Optional[int] = 5


# =========================
# /chat (principal)
# =========================
@app.post("/chat")
async def chat(request: Request):
@app.post("/chat")
async def chat(body: ChatRequest):
    query = (body.message or "").strip()
    mode = (body.mode or "estricto").lower().strip()
    top_k = int(body.top_k or 5)

    try:
        data = await request.json()
    except Exception:
        data = {}

    query = (data.get("message") or "").strip()
    if not query:
        return {"reply": "Escribe una pregunta."}

    mode = (data.get("mode") or "estricto").lower().strip()

    # Parámetros por modo
    if mode == "contextual":
        top_k = int(data.get("top_k", 10))
        min_score = float(os.getenv("RAG_MIN_SCORE_CTX", "0.12"))
        temperature = 0.2
        enable_second_hop = True
    else:
        top_k = int(data.get("top_k", 5))
        min_score = RAG_MIN_SCORE
        temperature = 0.0
        enable_second_hop = False

    # 1) Recuperación inicial
    hits = retrieve(query, k=top_k)
    hits = [h for h in hits if h["score"] >= min_score]
    if not hits:
        return {"reply": "No consta en nuestras fuentes internas."}

    # 2) Segundo salto por entidades (para recoger contacto/organigrama)
    all_hits = hits
    if enable_second_hop:
        names, emails, _depts = extract_entities(hits)
        extra = []
        extra += keyword_hits(names, limit=12)
        extra += keyword_hits(emails, limit=12)
        all_hits = dedup_hits(hits + extra)

    # 3) Construcción del contexto acotado
    context_blocks: List[str] = []
    total = 0
    for i, h in enumerate(all_hits, 1):
        block = f"[{i}] Fuente: {h.get('source','desconocida')}\n{h.get('text','')}\n"
        if total + len(block) > CONTEXT_MAX_CHARS:
            break
        context_blocks.append(block)
        total += len(block)
    context = "\n".join(context_blocks)

    if not context.strip():
        return {"reply": "No hay extractos utilizables en las fuentes internas."}

    # 4) Prompts con guardarraíles
    system_prompt = (
        "Eres el asistente de JR. Trabajas EXCLUSIVAMENTE con los extractos proporcionados.\n"
        "Reglas:\n"
        " - Responde solo con lo que aparece en los extractos.\n"
        " - Si la respuesta no está, di: «No consta en nuestras fuentes internas.»\n"
        " - Cita extractos como [1], [2], ...\n"
        " - No inventes nombres, correos ni teléfonos.\n"
        "Objetivo: dado un tema/petición, devuelve una lista priorizada de personas y/o departamentos relevantes, "
        "con su contacto si aparece, y una breve justificación vinculada a los extractos.\n"
    )
    user_msg = (
        f"Extractos:\n{context}\n\n"
        f"Pregunta del usuario: {query}\n\n"
        "Formato de salida sugerido:\n"
        "1) Nombre — Puesto/Departamento (email/teléfono si aparece)\n"
        "   Motivo de relevancia: … (cita [n])\n"
        "2) …\n"
        "Si no hay datos suficientes, responde exactamente: «No consta en nuestras fuentes internas.»"
    )

    # 5) Llamada al LLM
    try:
        completion = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
        )
        reply = completion.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        _log(f"Error en generación: {e}")
        return {"reply": "No he podido generar una respuesta ahora mismo."}
