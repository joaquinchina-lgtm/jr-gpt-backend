# backend.py — Retrieve SIN Tavily (DuckDuckGo + extracción simple)
import os
import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urlencode

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# CORS
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS")
if CORS_ALLOW_ORIGINS:
    ALLOWED_ORIGINS = set(o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip())
else:
    ALLOWED_ORIGINS = {"https://joaquinchina-lgtm.github.io", "*", "http://localhost:5173", "http://127.0.0.1:5500"}

MAX_K = int(os.getenv("MAX_K", "6"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "1200"))
ALLOWED_SOURCES_REGEX = os.getenv("ALLOWED_SOURCES_REGEX")  # p.ej. r"(navarra\.es|miteco\.gob\.es)$"
SOURCE_RE = re.compile(ALLOWED_SOURCES_REGEX) if ALLOWED_SOURCES_REGEX else None

DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# ---------- App ----------
app = Flask(__name__)
cors_origins = "*" if "*" in ALLOWED_ORIGINS else list(ALLOWED_ORIGINS)
CORS(app, resources={r"/*": {"origins": cors_origins}}, supports_credentials=False)

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

# ---------- Utilidades OpenAI ----------
def extract_text(resp) -> str:
    try:
        txt = getattr(resp, "output_text", None)
        if txt:
            return str(txt).strip()
    except Exception:
        pass
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

# ---------- Búsqueda web SIN Tavily ----------
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

def ddg_search(query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    Usa DuckDuckGo HTML (sin JS) para obtener resultados sin API key.
    Devuelve: [{title, url, snippet?}, ...]
    """
    params = {"q": query}
    url = "https://duckduckgo.com/html/?" + urlencode(params)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    # CSS selectores robustos para distintas plantillas
    for res in soup.select(".result")[: k * 2]:  # parseamos un poco más y filtramos después
        a = res.select_one("a.result__a") or res.select_one("a[href]")
        if not a: 
            continue
        href = a.get("href")
        title = a.get_text(strip=True)
        # snippet
        snippet_el = res.select_one(".result__snippet") or res.select_one(".result__extras") or res
        snippet = snippet_el.get_text(" ", strip=True)[:220] if snippet_el else ""
        if href and href.startswith("http"):
            results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= k:
            break
    return results

def same_domain(u: str) -> str:
    try:
        return urlparse(u).netloc or ""
    except Exception:
        return ""

def fetch_page_text(url: str, limit: int = CONTEXT_MAX_CHARS) -> str:
    """
    Descarga la página y extrae texto básico con BeautifulSoup.
    Retorna texto truncado a 'limit' chars.
    """
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "es-ES,es;q=0.9"}, timeout=20)
        r.raise_for_status()
    except Exception as e:
        return f"[No se pudo abrir la página: {e}]"
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text[:limit]

def build_prompt(query: str, results: List[Dict[str, Any]], language: str = "es") -> str:
    lines = []
    lines.append(f"Idioma de respuesta: {language}")
    lines.append("Analiza y responde con contexto. Usa SOLO las evidencias listadas y cita como [n].")
    lines.append("EVIDENCIAS:")
    for i, it in enumerate(results, 1):
        title = it.get("title") or it.get("url")
        url = it.get("url", "")
        content = it.get("content", "")[:CONTEXT_MAX_CHARS]
        lines.append(f"### [{i}] {title}\n{url}\n{content}\n")
    lines.append(f"\nPREGUNTA: {query}")
    lines.append("- Respuesta clara en 1-2 párrafos.")
    lines.append("- Lista de hallazgos clave si procede.")
    lines.append("- Cierra con 'Fuentes: [1], [2]...' solo si las usaste.")
    return "\n".join(lines)

# ---------- Rutas ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/", methods=["GET"])
def root():
    return "ok", 200

@app.route("/api/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return make_response(("", 204))
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
    language = body.get("language", "es")
    temperature = float(body.get("temperature", 0.2))
    top_p = float(body.get("top_p", 1.0))
    system = body.get("system") or "Eres un analista que sintetiza evidencias y cita fuentes."

    # 1) Buscar
    results = ddg_search(query, k=k+4)  # obtenemos extra por si filtramos

    # 2) Filtrado por dominios
    if include_domains:
        incl = set(include_domains)
        results = [r for r in results if any(d in same_domain(r['url']) for d in incl)]
    if exclude_domains:
        excl = set(exclude_domains)
        results = [r for r in results if all(d not in same_domain(r['url']) for d in excl)]
    if SOURCE_RE:
        results = [r for r in results if SOURCE_RE.search(r.get("url",""))]

    # cortar a k
    results = results[:k]

    # 3) Extraer texto de cada URL (contexto)
    for r in results:
        r["content"] = fetch_page_text(r["url"], limit=CONTEXT_MAX_CHARS)

    prompt = build_prompt(query, results, language=language)

    if not client:
        return jsonify(
            reply="[ECO] Sin OpenAI, devuelvo evidencias.",
            citations=[{"title": r.get("title"), "url": r.get("url")} for r in results],
            raw=results,
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
        citations = [{"title": r.get("title"), "url": r.get("url")} for r in results]
        return jsonify(reply=text, citations=citations, query=query), 200
    except Exception as e:
        return jsonify(error={"message": str(e)}), 500

@app.errorhandler(404)
def notfound(e):
    return jsonify(error="route_not_found", hint="usa /api/ask, /api/retrieve o /health"), 404

@app.route("/_debug/retrieve", methods=["POST", "OPTIONS"])
def old_alias():
    return retrieve()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
