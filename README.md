# Nepal Travel Bot API

Production-style RAG chatbot for Nepal travel questions.
**FastAPI + LangChain + Gemini + ChromaDB.**

## Project structure

```
nepal-travel-api/
├── app/
│   ├── main.py              # FastAPI app factory, middleware, startup
│   ├── core/
│   │   ├── config.py        # settings from .env (pydantic-settings)
│   │   └── logging.py
│   ├── api/
│   │   ├── router.py        # combines all routes under /api/v1
│   │   ├── deps.py          # shared dependencies (RAG service, admin auth)
│   │   └── routes/
│   │       ├── chat.py      # POST /chat, POST /chat/stream
│   │       ├── ingest.py    # add data: text, url, reload folder
│   │       └── health.py    # /health, /ready
│   ├── schemas/             # request / response models
│   └── rag/
│       ├── rag.py           # RAGService: ChromaDB + Gemini + LangChain
│       ├── prompts.py       # the prompt template
│       └── loaders.py       # read files, fetch web pages
├── data/nepal_travel.txt    # starter knowledge
├── tests/                   # pytest tests
├── manage.py                # terminal tool (ingest, add, add-url, chat)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Setup (Windows PowerShell)

```powershell
cd nepal-travel-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
 uvicorn app.main:app --host 0.0.0.0 --port 8000
Open `.env` in Notepad and set `GOOGLE_API_KEY=` to your Gemini key
(get one at https://aistudio.google.com/app/apikey).

Mac / Linux: use `source venv/bin/activate` and `cp .env.example .env`.

## Run the API

```powershell
uvicorn app.main:app --reload
```

On the first start it loads `data/nepal_travel.txt` into ChromaDB automatically
(`AUTO_INGEST=true`). Then open **http://127.0.0.1:8000/docs** to try every
endpoint in your browser.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | server is alive |
| GET  | `/ready` | database has data |
| POST | `/api/v1/chat` | ask a question, get answer + sources |
| POST | `/api/v1/chat/stream` | same, streamed word by word (Server-Sent Events) |
| POST | `/api/v1/ingest/text` | add text to the database |
| POST | `/api/v1/ingest/url` | fetch a web page and add it |
| POST | `/api/v1/ingest/reload` | re-scan the `data/` folder |

Ingest endpoints need the `X-API-Key` header when `ADMIN_API_KEY` is set
(required when `ENVIRONMENT=production`).

### Example: ask a question (PowerShell)

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/v1/chat `
  -ContentType "application/json" `
  -Body '{"question": "How many days is Everest Base Camp trek?"}'
```

### Example: with conversation history

```json
{
  "question": "And what permits do I need?",
  "history": [
    {"role": "user", "content": "How many days is Everest Base Camp trek?"},
    {"role": "assistant", "content": "About 12 to 14 days round trip."}
  ]
}
```
The API is stateless: your frontend sends the recent history with each request.

### Example: add data (PowerShell)

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/v1/ingest/text `
  -ContentType "application/json" `
  -Headers @{ "X-API-Key" = "your-admin-key" } `
  -Body '{"text": "Phewa Lake boat rental is about NPR 800 per hour.", "source": "my-notes"}'
```

## Add data from the terminal

```powershell
python manage.py ingest                     # load everything in data/
python manage.py add "Some travel fact"     # add a sentence
python manage.py add-url https://en.wikipedia.org/wiki/Annapurna_Circuit
python manage.py chat                       # chat in the terminal
```
You can also drop any `.txt` or `.md` file into `data/` and run `ingest`.
Chunks already stored are skipped, so re-running is safe and saves API quota.

## Configuration (.env)

| Variable | Default | Meaning |
|----------|---------|---------|
| `GOOGLE_API_KEY` | required | Gemini API key |
| `CHAT_MODEL` | `gemini-3.6-flash` | change this if Google retires the model |
| `EMBED_MODEL` | `models/gemini-embedding-001` | embedding model |
| `TOP_K` | `4` | chunks retrieved per question |
| `AUTO_INGEST` | `true` | load `data/` on startup if DB is empty |
| `CORS_ORIGINS` | `*` | allowed website origins, comma separated |
| `ADMIN_API_KEY` | empty | protects ingest endpoints |
| `SUVYATRA_API_URL` | empty | optional SuvYatra API base URL for current hotels, flights, transport, restaurants and destinations |
| `ENVIRONMENT` | `development` | set to `production` when deployed |

When the SuvYatra API is running on the host and the chatbot runs in Docker
Desktop, set `SUVYATRA_API_URL=http://host.docker.internal:5000/api`. Without
this setting, the bot explicitly says it cannot verify live inventory.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest
```

## Docker

```bash
docker compose up --build
```
The database is stored in a Docker volume, so it survives restarts.
Set `ADMIN_API_KEY` and `CORS_ORIGINS` in `.env` before deploying.

## Production notes

- Run **one worker** (the default). Local ChromaDB is not designed for many
  writer processes. To scale out, move to Chroma server mode or a hosted
  vector database.
- Put the API behind HTTPS (Nginx, Caddy, or your cloud provider) and add rate
  limiting there, because each chat request uses your Gemini quota.
- Never commit `.env` or share your API key.
- The URL ingest endpoint blocks private and local addresses, but keep it
  behind the admin key.
- Starter data is general information. Verify prices, permits and rules
  with official sources.

## Troubleshooting

- **`GOOGLE_API_KEY` validation error on start**: `.env` is missing or the key is empty.
- **404 model not found**: set `CHAT_MODEL` in `.env` to a current Gemini model name.
- **429 / quota errors**: free-tier limit reached; wait and retry.
- **503 "Knowledge base is empty"**: run `python manage.py ingest`.
- **Reset the database**: stop the server, delete the `chroma_db/` folder, start again.
