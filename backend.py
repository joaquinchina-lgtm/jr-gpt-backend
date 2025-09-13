# backend.py (Flask) — listo para Render
import os
import logging
import re
from typing import Optional, List, Dict, Any

import requests
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI

# =========================
# Configuración por entorno
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # opcional

# Permitir configurar orígenes por env (CSV o "*")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS")
if CORS_ALLOW_ORIGINS:
    ALLOWED_ORIGINS = set(o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip())
else:
    ALLOWED_ORIGINS = {
        "https://joaquinchina-lgtm.github.io",  # tu GitHub Pages
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"  # útil en pruebas si no usas credenciales
    }

MAX_K = int(os.getenv("MAX_K", "10"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "1200"))
ALLOWED_SOURCES_REGEX = os.getenv("ALLOWED_SOURCES_REGEX")  # p.ej. r"(navarra\.es|miteco\.gob\.es)$"
SOURCE_RE = re.compile(ALLOWED_SOURCES_REGEX) if ALLOWED_SOURCES_REGEX else None

DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# ============
# Inicializar
# ============
app = Flask(__name__)

# CORS (si hay "*" en la lista, usa wildcard global)
cors_origins = "*" if "*" in ALLOWED_ORIGINS else list(ALLOWED_ORIGINS)
CORS(
    app,
    resources={r"/*": {"origins": cors_origins}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
    max_age=86400,
)

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
log = logging.getLogger("backend")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Añadimos cabeceras CORS por si un proxy las pierde
@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin", "")
    # Si configuraste "*" permitimos a todos los orígenes,
    # en otro caso reflejamos el origen recibido.
    if "*" in ALLOWED_ORIGINS:
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    else:
        resp.headers.setdefault("Access-Control-Allow-Origin", origin or "")
    resp.headers.setdefault("Vary", "Origin")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return resp

# ========
# Utilidades
# ========
def _extract_text(resp) -> str:
    text = ""
    out = getattr(resp, "output", None) or []
    for item in out:
        if getattr(item, "type", None) == "message":
            for c in item.message.content:
                if getattr(c, "type", None) == "text":
                    text += c.text
    return text.strip() or "No he recibido contenido útil del modelo."

def tavily_search(
    query: str,
    k: int = 6,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    time_range: Optional[str] = None,   # 'd','w','m','y'
    search_depth: str = "advanced",      # 'basic' | 'advanced'
    include_answer: bool = True
) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        return {"results": [], "answer": None, "note": "TAVILY_API_KEY not set"}
    payload = {
        "query": query,
        "search_depth": search_depth,
        "max_results": int(k),
        "include_answer": include_answer,
    }
    if include_domains: payload["include_domains"] = include_domains
    if exclude_domains: payload["exclude_domains"] = exclude_domains
    if time_range: payload["time_range"] = time_range

    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    r = requests.post("https://api.tavily.com/search", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def build_retrieval_prompt(query: str, results: List[Dict[str, Any]], language: str = "es") -> str:
    lines = []
    lines.append(f"Idioma de respuesta: {language}")
    lines.append("Responde conciso y con rigor. Usa SOLO las evidencias listadas y cita como [n].\n")
    lines.append("EVIDENCIAS:")
    for i, it in enumerate(results, 1):
        title = it.get("title") or it.get("url")
        url = it.get("url", "")
        content = (it.get("content") or "")[:CONTEXT_MAX_CHARS]
        lines.append(f"### [{i}] {title}\n{url}\n{content}\n")
    lines.append(f"\nPREGUNTA DEL USUARIO: {query}")
    lines.append("- Resumen claro en 1-2 párrafos.")
    lines.append("- Lista de hallazgos clave si procede.")
    lines.append("- Cierra con 'Fuentes: [1], [2]...' solo con los índices usados.")
    return "\n".join(lines)

# =======
# Rutas
# =======
@app.route("/health", methods=["GET", "HEAD"])
def health():
    # HEAD devolverá mismo status sin cuerpo
    return jsonify(status="ok"), 200

@app.route("/", methods=["GET"])
def root():
    # Para evitar "Not Found" en el health checker si mira "/"
    return "ok", 200

@app.route("/api/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return make_response(("", 204))

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    temperature = float(data.get("temperature", 0.3))
    top_p = float(data.get("top_p", 1.0))
    system = data.get("system") or "Eres un asistente útil y conciso."

    if not message:
        return jsonify(error={"message": "Missing 'message'"}), 400

    if client is None:
        log.warning("OPENAI_API_KEY no configurada. Devolviendo eco.")
        return jsonify(reply=f"[ECO backend sin OpenAI] {message}"), 200

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            top_p=top_p,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
        )
        text = _extract_text(resp)
        return jsonify(reply=text), 200
    except Exception as e:
        log.exception("Error en /api/ask")
        return jsonify(error={"message": str(e)}), 500

@app.route("/api/retrieve", methods=["POST", "OPTIONS"])
def retrieve():
    if request.method == "OPTIONS":
        return make_response(("", 204))

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or body.get("message") or "").strip()
    if not query:
        return jsonify(error={"message": "Missing 'query'"}), 400

    k = min(int(body.get("k", 6)), MAX_K)
    include_domains = body.get("include_domains") or body.get("domains") or None
    exclude_domains = body.get("exclude_domains") or None
    time_range = body.get("time_range") or None
    search_depth = body.get("search_depth", "advanced")
    language = body.get("language", "es")
    temperature = float(body.get("temperature", 0.2))
    top_p = float(body.get("top_p", 1.0))
    system = body.get("system") or "Eres un analista que sintetiza evidencias y cita fuentes."

    tavily = tavily_search(
        query=query,
        k=k,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        time_range=time_range,
        search_depth=search_depth,
        include_answer=True,
    )
    results = tavily.get("results", [])

    # Filtro por regex si procede
    if SOURCE_RE:
        results = [r for r in results if SOURCE_RE.search((r.get("url") or ""))]

    prompt = build_retrieval_prompt(query, results, language=language)

    if client is None:
        return jsonify(
            reply="[ECO backend sin OpenAI] Devuelvo evidencias tal cual.",
            citations=[{"title": r.get("title"), "url": r.get("url")} for r in results],
            raw=tavily,
        ), 200

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            top_p=top_p,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        text = _extract_text(resp)
        citations = [{"title": r.get("title"), "url": r.get("url")} for r in results[:k]]
        return jsonify(
            reply=text,
            citations=citations,
            query=query,
            params={
                "k": k,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "time_range": time_range,
                "search_depth": search_depth,
                "temperature": temperature,
                "top_p": top_p,
                "language": language,
            },
        ), 200
    except Exception as e:
        log.exception("Error en /api/retrieve")
        return jsonify(error={"message": str(e), "tavily_note": tavily.get("note")}), 500

# 404 informativo (una sola vez)
@app.errorhandler(404)
def notfound(e):
    return jsonify(error="route_not_found",
                   hint="usa /api/ask, /api/retrieve o /health"), 404

# Alias de compatibilidad con tu endpoint antiguo
@app.route("/_debug/retrieve", methods=["POST", "OPTIONS"])
def old_alias():
    return retrieve()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
