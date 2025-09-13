# -*- coding: utf-8 -*-
"""
JR · I+D Finder (CSV-only RAG)
Backend unificado y limpio (FastAPI + FAISS + OpenAI) con:
- /chat            -> respuesta en texto usando contexto RAG
- /report          -> informe de texto (demo)
- /report/pdf      -> informe PDF (demo)
- /_debug/retrieve -> inspección de resultados RAG
- /health, /status -> utilitarios
"""
import os
import io
import re
import json
import traceback
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

# =========================
# Configuración
# =========================
APP_TITLE = "JR · I+D Finder (CSV-only RAG)"
APP_VERSION = "0.1.0"

# Modelos
GEN_MODEL   = os.getenv("GEN_MODEL",   "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# CORS: lista separada por comas
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "https://joaquinchina-lgtm.github.io")

# RAG (rutas por defecto)
RAG_DIR            = os.getenv("RAG_DIR", "rag")
INDEX_PATH         = os.path.join(RAG_DIR, "index.faiss")
TEXTS_PATH         = os.path.join(RAG_DIR, "texts.json")  # lista[dict] con al menos: text, source (opcional)

# Otros
RAG_MIN_SCORE      = float(os.getenv("RAG_MIN_SCORE", "0.0"))  # inicialmente sin filtro
CONTEXT_MAX_CHARS  = int(os.getenv("CONTEXT_MAX_CHARS", "9000"))
ALLOWED_SOURCES_RE = os.getenv("ALLOWED_SOURCES_REGEX", "")  # opcional; ejemplo: "UPNA|La Rioja|UCLM|EHU"
DEBUG              = os.getenv("DEBUG", "0") == "1"

# =========================
# App y CORS
# =========================
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()]
use_wildcard = any(o == "*" for o in _origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if use_wildcard else _origins,
    allow_credentials=not use_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Responder siempre a preflight
@app.options("/{rest_of_path:path}")
def any_options(rest_of_path: str):
    return JSONResponse({"ok": True})

# =========================
# OpenAI
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Utilidades RAG
# =========================
index: Optional[faiss.Index] = None
records: List[Dict[str, Any]] = []  # paralela al índice (misma longitud)

def _log(msg: str) -> None:
    print(f"[JR/RAG] {msg}", flush=True)

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) or 1.0)
    return v / n

def embed_query(text: str) -> np.ndarray:
    """Embedding de consulta con OpenAI; float32 shape (dim,)."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no está configurada.")
    t = (text or "").strip()
    if not t:
        # vector cero de la dim del índice para no romper FAISS
        dim = getattr(index, "d", 1536)
        return np.zeros(dim, dtype="float32")
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[t])
        vec = np.asarray(resp.data[0].embedding, dtype="float32")
        return vec
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar embedding: {e}")

def _fit_to_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    """Ajusta recortando o rellenando con ceros para casar con index.d"""
    arr = np.asarray(vec, dtype="float32").ravel()
    if arr.size > dim:
        return arr[:dim]
    if arr.size < dim:
        pad = np.zeros(dim - arr.size, dtype="float32")
        return np.concatenate([arr, pad], 0)
    return arr

def _meta_to_text(meta: Any) -> str:
    """Convierte metadatos a un texto útil para contexto."""
    if isinstance(meta, dict):
        preferred = (
            "group_name","lineas_investigacion","description","keywords",
            "area","responsable","name","universidad","centro",
            "departamento","email","telefono","title"
        )
        parts = [str(meta.get(k,"")).strip() for k in preferred if meta.get(k)]
        if parts:
            return " ".join(parts)
        # Fallback: aplanar todo
        def _walk(x):
            if isinstance(x, dict):              return " ".join(_walk(v) for v in x.values())
            if isinstance(x, (list,tuple,set)):  return " ".join(_walk(v) for v in x)
            return str(x or "").strip()
        raw = _walk(meta).strip()
        if raw:
            return raw
        return json.dumps(meta, ensure_ascii=False)
    return str(meta or "").strip()

def retrieve(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Busca en FAISS y devuelve [{score,text,source,meta}]"""
    if index is None:
        return []
    q = _fit_to_dim(embed_query(query), index.d)[None, :]  # (1, dim)
    D, I = index.search(q, k)
    out: List[Dict[str, Any]] = []
    src_re = re.compile(ALLOWED_SOURCES_RE) if ALLOWED_SOURCES_RE else None
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        rec: Dict[str, Any] = records[idx] if 0 <= idx < len(records) else {}
        text   = (rec.get("text") or "").strip() if isinstance(rec, dict) else ""
        source = ""
        if isinstance(rec, dict):
            source = (rec.get("source")
                      or rec.get("universidad")
                      or rec.get("centro")
                      or rec.get("departamento") or "").strip()
        if src_re and source and not src_re.search(source):
            continue
        out.append({
            "score": float(score),
            "text": text,
            "source": source or "desconocida",
            "meta": rec if isinstance(rec, dict) else {"record": rec},
        })
    return out

# =========================
# Carga de índice y registros
# =========================
@app.on_event("startup")
def load_rag():
    global index, records
    try:
        if not (os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH)):
            _log("No encuentro rag/index.faiss o rag/texts.json; el RAG seguirá vacío.")
            index = None
            records = []
            return
        index = faiss.read_index(INDEX_PATH)
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        _log(f"Cargado índice FAISS (dim={index.d}) y {len(records)} registros.")
    except Exception as e:
        index = None
        records = []
        _log(f"Error cargando índice/textos: {e}")

# =========================
# Schemas
# =========================
class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "estricto"
    top_k: Optional[int] = 5

class ReportRequest(BaseModel):
    message: str
    mode: Optional[str] = "contextual"
    top_k: Optional[int] = 80

class DebugRequest(BaseModel):
    message: str
    k: int = 25

# =========================
# Endpoints utilitarios
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/status")
def status():
    dim = None
    try:
        dim = index.d  # dimensión del índice FAISS cargado
    except Exception:
        pass
    return {
        "ok": True,
        "gen_model": GEN_MODEL,
        "embed_model": EMBED_MODEL,
        "faiss_dim": dim,
        "records": len(records),
    }

# =========================
# Debug
# =========================
@app.post("/_debug/retrieve")
def _debug_retrieve(body: DebugRequest):
    q = (body.message or "").strip()
    if not q:
        return []
    hits = retrieve(q, k=body.k)
    out = []
    for h in hits[: body.k]:
        meta = h.get("meta") or {}
        out.append({
            "score": float(h.get("score", 0)),
            "source": h.get("source", ""),
            "has_text": bool(h.get("text")),
            "meta_keys": list(meta.keys()) if isinstance(meta, dict) else [],
            "preview": (h.get("text") or _meta_to_text(meta) or "")[:180] + "…"
        })
    return out

# =========================
# /chat
# =========================
@app.post("/chat")
def chat(body: ChatRequest):
    query = (body.message or "").strip()
    if not query:
        return {"reply": "Escribe una pregunta."}

    # estrategia por modo
    mode = (body.mode or "estricto").lower().strip()
    top_k = int(body.top_k or 5)
    if mode == "contextual":
        top_k = max(top_k, 25)
        temperature = 0.2
    else:
        temperature = 0.0

    # Recuperación
    hits = retrieve(query, k=top_k)
    if not hits:
        return {"reply": "No se han encontrado coincidencias en los CSV cargados."}

    # Construcción de contexto robusto
    context_blocks: List[str] = []
    total = 0
    for i, h in enumerate(hits, 1):
        meta = h.get("meta") or {}
        txt  = (h.get("text") or "").strip() or _meta_to_text(meta) or "(sin extracto)"
        src  = (h.get("source") or "desconocida").strip()
        block = f"[{i}] Fuente: {src}\n{txt}\n"
        if total + len(block) > CONTEXT_MAX_CHARS:
            break
        context_blocks.append(block)
        total += len(block)

    if not context_blocks:
        return {"reply": "No consta información relevante en nuestras fuentes."}

    context = "\n".join(context_blocks)

    system_prompt = (
        "Eres el asistente de JR. Trabajas EXCLUSIVAMENTE con los extractos proporcionados. "
        "Ayudas a las empresas a localizar líneas de investigación y oportunidades de I+D compatibles. "
        "Responde solo con lo que aparece en los extractos; si falta, di: «No consta en nuestras fuentes internas.» "
        "Incluye sinónimos y áreas próximas; en caso de duda, inclúyelo. Cita extractos como [1], [2], ..."
    )

    user_msg = (
        f"Extractos:\n{context}\n\n"
        f"Consulta de la empresa: {query}\n\n"
        "Formato de salida:\n"
        "• Línea de investigación\n"
        "  - Descripción breve (si consta)\n"
        "  - Grupo de investigación\n"
        "  - Universidad/centro\n"
        "  - Investigador/a principal (si consta)\n"
        "  - Datos de contacto (si constan)\n"
        "[citas: usa referencias [n] de los extractos]\n\n"
        "Cierra SIEMPRE con: Si deseas asistencia en explorar una colaboración, contacta en **606522663**"
    )

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
        msg = f"{e.__class__.__name__}: {e}"
        if DEBUG:
            msg += " | " + "".join(traceback.format_exc(limit=2))
        _log(f"Error en /chat -> {msg}")
        return {"reply": "No he podido generar una respuesta ahora mismo."}

# =========================
# /report (demo texto) y /report/pdf (demo PDF)
# =========================
@app.post("/report")
def report(body: ReportRequest):
    summary = (
        f"Línea de consulta: {body.message}\n"
        f"Modo: {body.mode}\n\n"
        "- (demo) Sustituye este bloque por tu informe real\n"
        "Si deseas asistencia en explorar una colaboración, contacta en **606522663**"
    )
    return {"report": summary}

@app.post("/report/pdf")
def report_pdf(body: ReportRequest):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.lib.units import mm
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"reportlab no disponible: {e}")

    report_text = (
        f"Consulta: {body.message}\n"
        f"Modo: {body.mode}\n"
        f"Top-K: {body.top_k}\n\n"
        "Informe (demo). Sustituye este bloque por el informe real generado a partir de tus fuentes.\n\n"
        "Si deseas asistencia en explorar una colaboración, contacta en 606522663"
    )

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    left   = 20 * mm; right = 20 * mm; top = 20 * mm; bottom = 20 * mm
    y      = height - top

    title_font_name, title_font_size = "Helvetica-Bold", 16
    body_font_name,  body_font_size  = "Helvetica", 11
    leading = 15
    max_text_width = width - left - right

    c.setFont(title_font_name, title_font_size)
    c.drawString(left, y, "Informe I+D")
    y -= (leading + 6)

    c.setFont(body_font_name, body_font_size)
    for paragraph in report_text.split("\n"):
        line = ""
        words = paragraph.split(" ") if paragraph else [""]
        lines = []
        for w in words:
            test = (line + " " + w).strip()
            if stringWidth(test, body_font_name, body_font_size) <= max_text_width:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        lines.append(line)
        for ln in lines:
            if y <= bottom:
                c.showPage()
                c.setFont(body_font_name, body_font_size)
                y = height - top
            c.drawString(left, y, ln)
            y -= leading

    c.save()
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="informe_id.pdf"'}
    )

