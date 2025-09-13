# backend.py — RAG SOLO desde CSV en ./knowledge (sin Tavily) — FIXED QUOTES
import os, re, csv, logging, unicodedata
from typing import List, Dict, Any
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI

# =========================
# Configuración por entorno
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
KNOWLEDGE_DIR  = os.getenv("KNOWLEDGE_DIR", "knowledge")
MAX_K          = int(os.getenv("MAX_K", "12"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "800"))
ALLOW_CHAT     = os.getenv("ALLOW_CHAT", "true").lower() in ("1","true","yes")

CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS")
if CORS_ALLOW_ORIGINS:
    ALLOWED_ORIGINS = set(o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip())
else:
    ALLOWED_ORIGINS = {
        "https://joaquinchina-lgtm.github.io",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"
    }

DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# ============
# App & CORS
# ============
app = Flask(__name__)
cors_origins = "*" if "*" in ALLOWED_ORIGINS else list(ALLOWED_ORIGINS)
CORS(app, resources={r"/*": {"origins": cors_origins}},
     supports_credentials=False,
     methods=["GET","POST","OPTIONS"],
     allow_headers=["Content-Type","Authorization"],
     expose_headers=["Content-Type"],
     max_age=86400)

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
log = logging.getLogger("backend")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin", "")
    if "*" in ALLOWED_ORIGINS:
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    else:
        resp.headers.setdefault("Access-Control-Allow-Origin", origin or "")
    resp.headers.setdefault("Vary", "Origin")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return resp

# =============================
# Utilidades OpenAI (Responses)
# =============================
def extract_text(resp) -> str:
    # 1) atajo del SDK
    try:
        txt = getattr(resp, "output_text", None)
        if txt:
            return str(txt).strip()
    except Exception:
        pass
    # 2) recorrer output
    text = ""
    try:
        out = getattr(resp, "output", None) or []
        for item in out:
            parts = getattr(item, "content", None)
            if parts is None and hasattr(item, "message"):
                parts = getattr(item.message, "content", [])
            for c in (parts or []):
                if getattr(c, "type", None) == "text":
                    text += getattr(c, "text", "")
    except Exception:
        return str(resp)
    return text.strip() or str(resp)

# ======================
# Carga de conocimiento
# ======================
def _norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

STOPWORDS = set("""de la los las el un una unas unos y o a ante bajo cabe con contra desde durante en entre hacia hasta mediante para por según sin so sobre tras que como donde cuando cual cuales quien quienes cuyo cuya cuyos cuyas del al""".split())

def tokenize(s: str) -> List[str]:
    return [w for w in re.findall(r"\w+", _norm(s)) if w not in STOPWORDS and len(w) > 2]

def load_knowledge(dirpath: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not os.path.isdir(dirpath):
        log.warning("No existe la carpeta de conocimiento: %s", dirpath)
        return docs
    for root, _, files in os.walk(dirpath):
        for fname in files:
            if not fname.lower().endswith(".csv"):
                continue
            path = os.path.join(root, fname)
            rel  = os.path.relpath(path, dirpath)
            try:
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.reader(f))
            except Exception as e:
                log.warning("No pude leer %s: %s", path, e)
                continue
            headers = rows[0] if rows else []
            data_rows = rows[1:] if headers else rows
            for i, row in enumerate(data_rows):
                idx = i + 1 if headers else i
                pieces = []
                for j, cell in enumerate(row):
                    colname = headers[j] if headers and j < len(headers) else f"col{j+1}"
                    cell = (cell or "").strip()
                    if cell:
                        pieces.append(f"{colname}: {cell}")
                text = " | ".join(pieces)
                if not text:
                    continue
                docs.append({
                    "id": f"{rel}#row{idx}",
                    "file": rel,
                    "row": idx,
                    "text": text,
                    "tokens": tokenize(text)
                })
    log.info("Conocimiento cargado: %d fragmentos desde %s", len(docs), dirpath)
    return docs

KNOWLEDGE: List[Dict[str, Any]] = load_knowledge(KNOWLEDGE_DIR)

# ======================
# Búsqueda en conocimiento
# ======================
def score_doc(query: str, doc: Dict[str, Any]) -> float:
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    overlap = sum(doc["tokens"].count(t) for t in q_tokens)
    phrase = 1.0 if _norm(query) in _norm(doc["text"]) else 0.0
    sim = SequenceMatcher(None, _norm(query)[:2000], _norm(doc["text"])[:2000]).quick_ratio()  # 0..1
    return overlap + 1.5*phrase + 2.0*sim  # pesos sencillos

def top_k(query: str, k: int, neighbors: int = 1) -> List[Dict[str, Any]]:
    scored = [(score_doc(query, d), d) for d in KNOWLEDGE]
    scored.sort(key=lambda x: x[0], reverse=True)
    results: List[Dict[str, Any]] = []
    seen = set()
    for score, d in scored:
        if score <= 0:
            break
        if d["id"] in seen:
            continue
        results.append(d); seen.add(d["id"])
        # añade vecinos +-1 fila del mismo fichero para contexto
        if neighbors:
            for dd in KNOWLEDGE:
                if dd["file"] == d["file"] and abs(dd["row"] - d["row"]) <= neighbors and dd["id"] not in seen:
                    results.append(dd); seen.add(dd["id"])
        if len(results) >= k:
            break
    return results[:k]

def build_prompt(query: str, frags: List[Dict[str, Any]], language: str = "es") -> str:
    lines = []
    lines.append(f"Idioma de respuesta: {language}")
    lines.append("Contesta EXCLUSIVAMENTE usando los fragmentos listados como evidencias.")
    lines.append("Si la información no está en las evidencias, responde literalmente: 'No aparece en las fuentes'.")
    lines.append("Cita como [n] y lista al final 'Fuentes: [n]...' con fichero y fila.")
    lines.append("\nEVIDENCIAS:")
    for i, d in enumerate(frags, 1):
        content = d['text'][:CONTEXT_MAX_CHARS]
        lines.append(f"""### [{i}] {d['file']} (fila {d['row']})
{content}
""".strip())
    lines.append(f"\nPREGUNTA: {query}")
    lines.append("- Estructura: resumen breve y, si procede, lista de puntos clave.")
    lines.append("- No inventes. Sólo usa lo que está en las evidencias.")
    return "\n".join(lines)

# =======
# Rutas
# =======
@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify(status="ok", knowledge=len(KNOWLEDGE)), 200

@app.route("/", methods=["GET"])
def root():
    return "ok", 200

@app.route("/api/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return make_response(("", 204))
    if not ALLOW_CHAT:
        return jsonify(error={"message":"/api/ask deshabilitado. Usa /api/rag"}), 403

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(error={"message": "Missing 'message'"}), 400

    temperature = float(data.get("temperature", 0.2))
    top_p = float(data.get("top_p", 1.0))
    system = data.get("system") or "Eres un asistente útil y conciso."

    if not client:
        return jsonify(reply=f"[ECO] {message}"), 200

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            top_p=top_p,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": message}],
        )
        return jsonify(reply=extract_text(resp)), 200
    except Exception as e:
        return jsonify(error={"message": str(e)}), 500

@app.route("/api/rag", methods=["POST", "OPTIONS"])
def rag():
    if request.method == "OPTIONS":
        return make_response(("", 204))

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or body.get("message") or "").strip()
    if not query:
        return jsonify(error={"message": "Missing 'query'"}), 400

    k = min(int(body.get("k", 8)), MAX_K)
    language = body.get("language", "es")
    temperature = float(body.get("temperature", 0.2))
    top_p = float(body.get("top_p", 1.0))
    system = body.get("system") or "Eres un analista que sólo responde con las evidencias proporcionadas."

    frags = top_k(query, k=k, neighbors=1)
    if not frags:
        return jsonify(reply="No aparece en las fuentes.", citations=[], query=query), 200

    prompt = build_prompt(query, frags, language=language)

    if not client:
        return jsonify(
            reply="[ECO] Sin OpenAI, devuelvo evidencias.",
            citations=[{"file": d["file"], "row": d["row"]} for d in frags],
            raw=[{"id": d["id"], "text": d["text"][:CONTEXT_MAX_CHARS]} for d in frags]
        ), 200

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            top_p=top_p,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": prompt}],
        )
        text = extract_text(resp)
        citations = [{"file": d["file"], "row": d["row"]} for d in frags]
        return jsonify(reply=text, citations=citations, query=query), 200
    except Exception as e:
        return jsonify(error={"message": str(e)}), 500

@app.errorhandler(404)
def notfound(e):
    return jsonify(error="route_not_found", hint="usa /api/rag o /health"), 404

# Alias antiguo de compatibilidad
@app.route("/_debug/retrieve", methods=["POST", "OPTIONS"])
def old_alias():
    return rag()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
