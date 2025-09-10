# JR GPT Backend · paquete mínimo con RAG sencillo (TF-IDF)

Este proyecto te da **tu propio backend** tipo GPT, con:
- **/chat**: endpoint de conversación con memoria breve y **tools** (incluye `search_docs`).
- **RAG sencillo** (sin instalaciones complicadas): ingesta de PDF/TXT/MD y búsqueda por similitud con TF‑IDF.
- **Frontend** (caja de texto discreta) en `/frontend/index.html` para incrustar en tu web.
- **Docker opcional**.

## 0) Requisitos
- Python 3.10+ (Windows, macOS o Linux).
- Una clave de OpenAI (cópiala en `.env`).

## 1) Instalar
```bash
# 1) Entra a la carpeta
cd jr_gpt_backend

# 2) Crea tu entorno (opcional pero recomendado)
python -m venv .venv
# En Windows
.venv\Scripts\activate
# En macOS/Linux
source .venv/bin/activate

# 3) Instala dependencias
pip install -r requirements.txt

# 4) Copia la plantilla de variables
copy .env.example .env   # Windows
# o
cp .env.example .env     # macOS/Linux

# 5) Edita .env y pega tu OPENAI_API_KEY
```

## 2) Añade documentos (opcional, para RAG)
Coloca **PDFs, .txt o .md** dentro de la carpeta `docs/` (puedes crear subcarpetas).

## 3) Construye el índice (RAG)
```bash
python ingest.py
```
- Esto genera `app/data/index.pkl` con pasajes troceados y el índice TF‑IDF.

## 4) Arranca el backend
```bash
uvicorn main:app --reload
```
- Se abrirá en `http://127.0.0.1:8000`
- Documentación interactiva: `http://127.0.0.1:8000/docs`
- **/chat** es el endpoint principal (método POST).

> Seguridad: si pones `JR_API_KEY` en `.env`, el backend exigirá `Authorization: Bearer <JR_API_KEY>` en cada llamada.

## 5) Probar desde la terminal
```bash
curl -X POST http://127.0.0.1:8000/chat   -H "Content-Type: application/json"   -H "Authorization: Bearer cambia-esta-clave"   -d "{\"message\":\"Hola, ¿qué sabes de mis documentos?\", \"session_id\":\"test-1\"}"
```

## 6) Usar la “caja de texto discreta” (frontend)
Abre `frontend/index.html` en tu navegador y escribe.  
En producción, sirve este archivo desde tu web y configura el **fetch** a tu dominio del backend.

## 7) Estructura de carpetas
```
jr_gpt_backend/
├─ app/
│  ├─ app.py          # Lógica de /chat y tools
│  ├─ rag.py          # Búsqueda TF-IDF
│  └─ data/           # Aquí se guarda el índice (index.pkl)
├─ docs/              # Pon tus PDFs/TXT/MD aquí
├─ frontend/
│  └─ index.html      # Caja de texto discreta para incrustar
├─ ingest.py          # Script de ingesta y creación del índice
├─ main.py            # Punto de entrada uvicorn
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
└─ README.md
```

## 8) Cómo funciona (explicado llano)
1. **Tú escribes** en la caja discreta (o llamas a `/chat`).  
2. El backend monta un **prompt** con: reglas del sistema + tu mensaje + una memoria muy corta de los últimos turnos.  
3. El modelo puede pedir usar la **tool** `search_docs`.  
4. Si la pide, el backend **busca** en tu índice y devuelve **pasajes**.  
5. El modelo **recibe** esos pasajes y redacta una respuesta citando (si procede).  
6. El backend te devuelve el **texto final** y guarda una memoria mínima de la conversación (en RAM en este ejemplo).

## 9) Personalizar
- Cambia el tono y políticas en `SYSTEM_PROMPT` (app/app.py).  
- Añade **nuevas tools** siguiendo el patrón de `tools` + `call_tool`.  
- Sustituye TF‑IDF por **pgvector** o **FAISS** si más adelante quieres RAG avanzado.  
- Guarda memoria larga en una base de datos (Postgres/SQLite).

## 10) Docker (opcional)
```bash
docker compose up --build
```
- Expone el backend en `http://localhost:8000`
- Usa tus variables desde `.env`

---

### Preguntas rápidas
- **¿Necesito RAG?** No. Si no pones documentos, /chat funciona igual (solo que sin búsquedas).  
- **¿Puedo usarlo en tu web ya?** Sí: sube el backend a tu servidor y apunta el `fetch('/chat', ...)` de `frontend/index.html` al dominio de tu API.  
- **¿Es seguro?** Activa `JR_API_KEY` y usa HTTPS. No envíes secretos en prompts.
