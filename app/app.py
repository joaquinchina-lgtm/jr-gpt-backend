# === imports limpios ===
import os, json, re, io, traceback
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional


# === configuración ===
GEN_MODEL   = os.getenv("GEN_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DEBUG       = os.getenv("DEBUG", "0") == "1"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === FastAPI (solo una vez en todo el archivo) ===
app = FastAPI(title="JR · I+D Finder (CSV-only RAG)")

@app.get("/status")
def status():
    dim = None
    try:
        dim = index.d  # dimensión del índice FAISS
    except Exception:
        pass
    return {
        "ok": True,
        "gen_model": GEN_MODEL,
        "embed_model": EMBED_MODEL,
        "faiss_dim": dim,
    }


class ReportRequest(BaseModel):
    message: str
    mode: Optional[str] = "contextual"
    top_k: Optional[int] = 80

# =========================
# Configuración
# =========================
APP_TITLE = "JR GPT Backend"
APP_VERSION = "0.1.0"

ORIGIN = os.getenv("ORIGIN", "https://joaquinchina-lgtm.github.io")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
RAG_DIR = os.getenv("RAG_DIR", "rag")
INDEX_PATH = os.path.join(RAG_DIR, "index.faiss")
TEXTS_PATH = os.path.join(RAG_DIR, "texts.json")
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.15"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "9000"))



# =========================
# App y middlewares
# =========================
app = FastAPI(title="JR · I+D Finder (CSV-only RAG)")

# --- CORS robusto (pegar justo debajo de app = FastAPI(...)) ---
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Lee la variable de entorno (en Render debe ser exactamente:
# CORS_ALLOW_ORIGINS = https://joaquinchina-lgtm.github.io)
_raw = os.getenv("CORS_ALLOW_ORIGINS", "https://joaquinchina-lgtm.github.io")
origins = [o.strip() for o in _raw.split(",") if o.strip()]

# Si alguien pone '*' en CORS_ALLOW_ORIGINS, los navegadores no permiten credenciales.
use_wildcard = any(o == "*" for o in origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if use_wildcard else origins,
    allow_credentials=not use_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Responder a cualquier preflight OPTIONS, incluso si el handler final falla
@app.options("/{rest_of_path:path}")
def any_options(rest_of_path: str):
    return JSONResponse({"ok": True})

# =========================
# OpenAI client
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/report")
async def report(body: ReportRequest):
    """
    Genera un informe textual sencillo (puedes sustituir 'summary' por tu lógica real).
    """
    summary = (
        f"Línea de consulta: {body.message}\n"
        f"Modo: {body.mode}\n\n"
        "- (demo) Sustituye este bloque por tu informe real\n"
        "Si deseas asistencia en explorar una colaboración, contacta en **606522663**"
    )
    return {"report": summary}


@app.post("/report/pdf")
async def report_pdf(body: ReportRequest):
    """
    Genera y devuelve un PDF con el informe.
    - Lazy import de reportlab (el servicio arranca aunque la lib no esté en memoria al boot).
    - Maquetación básica con ajuste de líneas y salto de página.
    """
    # 1) Importar reportlab SOLO aquí (lazy import)
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.lib.units import mm
    except Exception as e:
        # Si faltara la lib en el build, devolvemos un error claro
        raise HTTPException(status_code=503, detail=f"reportlab no disponible: {e}")

    # 2) Texto del informe (DEMO): sustituye por tu texto real cuando quieras
    report_text = (
        f"Consulta: {body.message}\n"
        f"Modo: {body.mode}\n"
        f"Top-K: {body.top_k}\n\n"
        "Informe (demo). Sustituye este bloque por el informe real generado a partir de tus fuentes.\n\n"
        "Si deseas asistencia en explorar una colaboración, contacta en 606522663"
    )

    # 3) Crear PDF en memoria
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Márgenes y tipografías
    left   = 20 * mm
    right  = 20 * mm
    top    = 20 * mm
    bottom = 20 * mm
    y      = height - top

    title_font_name, title_font_size = "Helvetica-Bold", 16
    body_font_name,  body_font_size  = "Helvetica", 11
    leading = 15  # interlineado
    max_text_width = width - left - right

    # Título
    c.setFont(title_font_name, title_font_size)
    c.drawString(left, y, "Informe I+D")
    y -= (leading + 6)

    # Cuerpo con ajuste de línea
    c.setFont(body_font_name, body_font_size)
    for paragraph in report_text.split("\n"):
        # envolver párrafo al ancho
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

        # escribir líneas; salto de página si hace falta
        for ln in lines:
            if y <= bottom:
                c.showPage()
                c.setFont(body_font_name, body_font_size)
                y = height - top
            c.drawString(left, y, ln)
            y -= leading

    # 4) Finalizar y devolver (sin c.showPage() extra para no añadir página en blanco)
    c.save()
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="informe_id.pdf"'}
    )


# =========================
# Carga de índice y textos
# =========================
index = None
records: List[Dict[str, Any]] = None

def _log(msg: str):
    print(f"[RAG] {msg}", flush=True)

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-10
    return v / n

def embed(text: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(e.data[0].embedding, dtype="float32")
    return _normalize(v)

def embed_query(text: str) -> np.ndarray:
    """
    Devuelve el embedding de la consulta usando OpenAI y el modelo EMBED_MODEL.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no está configurada en el backend")

    txt = (text or "").strip()
    if not txt:
        # vector cero si llega vacío (evita crashear)
        dim = getattr(index, "d", 1536)  # 1536 por defecto
        return np.zeros(dim, dtype="float32")

    try:
        # Llama a OpenAI Embeddings
        resp = client.embeddings.create(model=EMBED_MODEL, input=[txt])
        vec = np.array(resp.data[0].embedding, dtype="float32")
        return vec
    except Exception as e:
        # Mensaje claro en caso de error de API/Modelo
        raise HTTPException(status_code=500, detail=f"Error al generar embedding: {e}")


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

def _fit_to_dim(vec, dim: int):
    arr = np.asarray(vec, dtype="float32").ravel()
    if arr.size > dim:
        return arr[:dim]                     # recorta
    if arr.size < dim:
        pad = np.zeros(dim - arr.size, dtype="float32")
        return np.concatenate([arr, pad], 0) # rellena con ceros
    return arr


def retrieve(query: str, k: int = 5):
    """
    Devuelve una lista de dicts con al menos: score, text, source, meta.
    Requiere que exista: 
      - una función embed_query(query) que genere el embedding de consulta
      - un índice FAISS global 'index'
      - (opcional) una lista/array 'store' con los metadatos por id
    """
    # 1) Embedding de la consulta
    q = embed_query(query)  # <- usa tu función existente

    # 2) Ajuste de dtype/forma
    q = np.asarray(q, dtype="float32").ravel()

    # 3) Ajuste de dimensión (defensivo) según el índice FAISS
    dim = getattr(index, "d", None)
    if dim is None:
        raise HTTPException(status_code=500, detail="Índice FAISS no cargado correctamente")

    if q.size != dim:
        try:
            q = _fit_to_dim(q, dim)
        except NameError:
            # fallback inline si no pegaste _fit_to_dim
            if q.size > dim:
                q = q[:dim]
            else:
                q = np.pad(q, (0, dim - q.size), constant_values=0).astype("float32")

    q = np.ascontiguousarray(q, dtype="float32").reshape(1, -1)

    # 4) Búsqueda FAISS
    D, I = index.search(q, k)

    # 5) Construir resultados
    results = []
    for rank, (d, i) in enumerate(zip(D[0], I[0]), 1):
        if i < 0:
            continue
        # Toma metadatos si tienes una estructura global 'store' (lista de dicts)
        rec = {}
        if "store" in globals():
            try:
                rec = store[i]
            except Exception:
                rec = {}
        results.append({
            "score": float(d),
            "text":  rec.get("text", ""),
            "source": rec.get("source", ""),
            "meta":  rec
        })
    return results

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
    }

# =========================
# /chat (principal)
# =========================
class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "estricto"
    top_k: Optional[int] = 5

class DebugRequest(BaseModel):
    message: str
    k: int = 25

@app.post("/_debug/retrieve")
def _debug_retrieve(body: DebugRequest):
    qs = (body.message or "").strip()
    if not qs:
        return []
    hits = retrieve(qs, k=body.k)
    preview = []
    for h in hits[: body.k]:
        preview.append({
            "score": float(h.get("score", 0)),
            "source": h.get("source", ""),
            "text": (h.get("text","")[:220] + "…") if h.get("text") else ""
        })
    return preview

@app.post("/chat")
async def chat(body: ChatRequest):
    query = (body.message or "").strip()
    if not query:
        return {"reply": "Escribe una pregunta."}

    mode = (body.mode or "estricto").lower().strip()
    top_k = int(body.top_k or 5)

    # Configuración por modo (dejamos min_score pero NO filtramos por score ahora)
    if mode == "contextual":
        top_k = max(top_k, 25)
        min_score = float(os.getenv("RAG_MIN_SCORE_CTX", "0.0"))
        temperature = 0.2
    else:
        min_score = float(os.getenv("RAG_MIN_SCORE", "0.0"))
        temperature = 0.0

    # Recuperación (sin filtrar por score)
    hits = retrieve(query, k=top_k)
    if not hits:
        return {"reply": "No se han encontrado coincidencias en los CSV cargados."}

    # Construcción de contexto (si falta 'text', montamos con meta: group_name, description, etc.)
    context_blocks = []
    total = 0
    for i, h in enumerate(hits, 1):
        meta = h.get("meta") or {}
        txt = (h.get("text") or "").strip()
        if not txt:
            fields = (
                "group_name", "lineas_investigacion", "description", "keywords",
                "area", "responsable", "name"
            )
            txt = " ".join(str(meta.get(k, "")) for k in fields if meta.get(k)).strip()

        if not txt:
            continue

        src = h.get("source", "desconocida")
        block = f"[{i}] Fuente: {src}\n{txt}\n"
        if total + len(block) > CONTEXT_MAX_CHARS:
            break
        context_blocks.append(block)
        total += len(block)

    if not context_blocks:
        return {"reply": "No constan extractos útiles en los CSV cargados."}

    context = "\n".join(context_blocks)

    # Prompt del asistente (texto seguro, sin comillas “raras”)
    system_prompt = (
        "Eres el asistente de JR. Trabajas EXCLUSIVAMENTE con los extractos proporcionados. "
        "Ayudas a las empresas a localizar líneas de investigación y oportunidades de I+D lo más "
        "compatibles con su actividad. Responde solo con lo que aparece en los extractos; si falta, "
        "di: «No consta en nuestras fuentes internas.» "
        "No te limites al texto literal: incluye sinónimos y áreas próximas. Devuelve todos los grupos "
        "cuya actividad pueda estar vinculada directa o indirectamente con el tema de la consulta. "
        "Incluye coincidencias aunque la información esté fragmentada en nombre del grupo, keywords, "
        "descripción o cualquier campo. En caso de duda, inclúyelo. Cita extractos como [1], [2], ..."
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
        _log(f"Error en generación: {e}")
        return {"reply": "No he podido generar una respuesta ahora mismo."}


    except HTTPException:
        # Re-lanza tal cual validaciones explícitas
        raise
    except Exception as e:
        # Detalle útil para depurar
        msg = f"{e.__class__.__name__}: {e}"
        if DEBUG:
            msg += " | " + "".join(traceback.format_exc(limit=3))
        # Log (si tienes _log; si no, usa print)
        try:
            _log(f"Error en /chat: {msg}")
        except NameError:
            print(f"[JR] Error en /chat: {msg}")
        raise HTTPException(status_code=500, detail=msg)

# -------------------------
# INFORME (texto simple DEMO)
# -------------------------
@app.post("/report")
async def report(body: ReportRequest):
    """
    Genera un informe textual sencillo (DEMO).
    Sustituye 'summary' por tu lógica real cuando quieras.
    """
    summary = (
        f"Línea de consulta: {body.message}\n"
        f"Modo: {body.mode}\n\n"
        "- (demo) Sustituye este bloque por tu informe real\n"
        "Si deseas asistencia en explorar una colaboración, contacta en **606522663**"
    )
    return {"report": summary}


# -------------------------
# INFORME en PDF (DEMO)
# -------------------------
@app.post("/report/pdf")
async def report_pdf(body: ReportRequest):
    """
    Genera un PDF descargable con el informe (DEMO).
    Importa reportlab SOLO aquí (lazy import) para que el servicio arranque
    aunque la librería no esté instalada aún.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
    except Exception as e:
        # Mensaje claro si falta la dependencia en el build
        raise HTTPException(status_code=503, detail=f"reportlab no disponible: {e}")

    # Texto de ejemplo (pon aquí tu informe real cuando esté listo)
    text = (
        f"Consulta: {body.message}\n"
        f"Modo: {body.mode}\n\n"
        "Informe (demo)\n"
        "Si deseas asistencia en explorar una colaboración, contacta en 606522663"
    )

    # Crear PDF en memoria
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "Informe I+D")
    y -= 24

    c.setFont("Helvetica", 11)
    for line in text.split("\n"):
        c.drawString(72, y, line[:110])
        y -= 16
        if y < 72:
            c.showPage()
            y = height - 72
            c.setFont("Helvetica", 11)

    c.showPage()
    c.save()
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="informe_id.pdf"'}
    )

