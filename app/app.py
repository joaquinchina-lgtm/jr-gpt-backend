import os, json, time
from typing import Dict, Any, List, Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security

from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI

from .rag import search_docs

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY. Crea un archivo .env (copia .env.example)")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="JR GPT Backend", version="0.1.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://joaquinchina-lgtm.github.io"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

class ChatInput(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str] = None

class ChatOutput(BaseModel):
    reply: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    latency_ms: int

security = HTTPBearer(auto_error=False)

def require_auth(credentials: HTTPAuthorizationCredentials = Security(security)):
    expected = os.getenv("JR_API_KEY")
    if expected:
        if not credentials or credentials.scheme.lower() != "bearer":
            raise HTTPException(401, "Falta cabecera Authorization: Bearer <JR_API_KEY>")
        if credentials.credentials != expected:
            raise HTTPException(401, "API key incorrecta")
    return True

SESSIONS: Dict[str, List[Dict[str, str]]] = {}

SYSTEM_PROMPT = (
    "Eres un asistente para una web de innovación y educación en Navarra. "
    "Objetivo: ayudar con dudas técnicas, RAG sobre documentos cargados y dar respuestas claras. "
    "Si te piden fuentes, cita resultados de 'search_docs'. "
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Busca pasajes relevantes en el corpus interno",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }
]

def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    if name == "search_docs":
        q = arguments.get("query","")
        return search_docs(q, k=5)
    return {"error":"Tool not found"}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/chat", response_model=ChatOutput)
def chat(body: ChatInput, _: bool = Depends(require_auth)):
    start = time.time()

    short_memory = ""
    if SESSIONS.get(body.session_id):
        last = SESSIONS[body.session_id][-3:]
        for turn in last:
            short_memory += f"User: {turn['user']}\\nAssistant: {turn['assistant']}\\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f" Memoria breve:\\n{short_memory}"},
        {"role": "user", "content": body.message},
    ]

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.3
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    tool_results = []

    if tool_calls:
        messages.append(resp.choices[0].message)
        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = call_tool(tc.function.name, args)
            tool_results.append({"id": tc.id, "name": tc.function.name, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": json.dumps(result, ensure_ascii=False)
            })

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3
        )

    answer = resp.choices[0].message.content

    SESSIONS.setdefault(body.session_id, []).append({"user": body.message, "assistant": answer})
    latency_ms = int((time.time() - start) * 1000)

    return ChatOutput(reply=answer, tool_calls=tool_results or None, latency_ms=latency_ms)
