# app/app.py
import os, json
import numpy as np
import faiss

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI(title="JR GPT Backend", version="0.1.0")

# --- CORS (ajusta tu dominio si cambias GitHub Pages) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://joaquinchina-lgtm.github.io"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)

# --- OpenAI ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"

# --- RAG config ---
RAG_DIR = os.getenv("RAG_DIR", "rag")
INDEX_PATH = os.path.join(RAG_DIR, "index.faiss")
TEXTS_PATH = os.path.join(RAG_DIR, "texts.json")
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.15"))   # umbral similitud
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "9000"))

index = None
records = None

def embed(text: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(e.data[0].embedding, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-10)
    return v

@app.on_event("startup")
def load_rag():
    global index, records
    if not (os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH)):
        print("[RAG] No encuentro el índice. Genera primero rag/index.faiss y rag/texts.json")
        return
    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"[RAG] Cargado índice y {len(records)} chunks.")

def retrieve(query: str, k: int = 5):
    if index is None or records is None:
        return []
    q = embed(query)
    D, I = index.search(np.array([q]), k)
    out = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        out.append({"score": float(score), **records[idx]})
    return out

@app.get("/health")
def health(): return {"ok": True}

@app.get("/status")
def status():
    return {
        "index_loaded": bool(index is not None),
        "num_chunks": (len(records) if records else 0),
        "min_score": RAG_MIN_SCORE
    }

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = (data.get("message") or "").strip()
    top_k = int(data.get("top_k", 5))

    if not query:
        return {"reply": "Escribe una pregunta."}

    # 1) Recuperar contexto
    hits = retrieve(query, k=top_k)
    if not hits:
        return {"reply": "No consta en nuestras fuentes internas."}

    # 2) Filtrar por umbral
    hits = [h for h in hits if h["score"] >= RAG_MIN_SCORE]
    if not hits:
        return {"reply": "No hay coincidencias suficientemente relevantes en las fuentes internas."}

    # 3) Construir contexto limitado
    context_blocks, total = [], 0
    for i, h in enumerate(hits, 1):
        block = f"[{i}] Fuente: {h['source']}\n{h['text']}\n"
        if total + len(block) > CONTEXT_MAX_CHARS: break
        context_blocks.append(block)
        total += len(block)
    context = "\n".join(context_blocks)

    # 4) Preguntar al modelo con guardarraíles
    system_prompt = (
        "Eres el asistente de JR. Responde ÚNICAMENTE con la información de los extractos proporcionados. "
	"Puedes INFERIR relaciones y sacar conclusiones SIEMPRE que estén respaldadas por los extractos. "
	"Objetivo: dado un tema/petición, devuelve una lista priorizada de líneas/departamentos/personas/ relevantes, "
	"incluyendo contacto (email/teléfono si aparece) y una breve justificación. "
	"Agrupa por persona/departamento y referencia [1], [2]… según el extracto. No inventes nombres ni contactos"
	"Si no hay datos suficientes, di: «No consta en nuestras fuentes internas.» "
	"Prohibido usar conocimiento externo."
    )
    user_msg = f"Extractos:\n{context}\n\nPregunta: {query}"

    completion = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",    "content": user_msg},
        ],
        temperature=0.2,
    )

    reply = completion.choices[0].message.content
    return {"reply": reply}
