# 🤖 BlueBug Self-RAG Chatbot

<div align="center">

**An intelligent company knowledge-base chatbot powered by Self-RAG — built for accuracy, efficiency, and real-time streaming.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Self--RAG-ff6b35?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PGVector-336791?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📺 Demo Walkthrough

> Click the thumbnail below to watch the full project walkthrough.

<div align="center">
  <a href="https://drive.google.com/file/d/1Ows-lOiZOlAnno1KS4QOQ10RIqX_GFNB/view?usp=sharing" target="_blank">
    <img src="assets/screenshot.png" alt="BlueBug Self-RAG Demo Walkthrough" width="780" style="border-radius: 10px; border: 2px solid #2c7be5;" />
  </a>
  <br/>
  <sub>▶ Click to watch the walkthrough video</sub>
</div>

---

## 📌 What is This?

**BlueBug Self-RAG Chatbot** is a production-ready, API-efficient conversational AI backend that answers questions about BlueBug's internal company knowledge — policies, products, pricing, culture, financials, and more.

Unlike a naive RAG system that retrieves documents on every single query, this system uses **Self-RAG** (Self-Reflective Retrieval-Augmented Generation) — a graph-based reasoning pipeline that only retrieves when it actually needs to, verifies what it generates, and corrects itself before responding.

---

## ✨ Why Self-RAG?

Most RAG systems retrieve on every query. Self-RAG is smarter:

| Feature | Naive RAG | Self-RAG (This Project) |
|---|---|---|
| Retrieval | Always, even for greetings | Only when needed |
| Answer verification | None | Checks if answer is grounded in context |
| Self-correction | None | Revises ungrounded answers automatically |
| Hallucination control | Weak | Strong — refuses to answer from fabrication |
| API cost | High (unnecessary calls) | Optimised — skips retrieval for ~30–40% of queries |
| Memory | Stateless | Persistent per session with summarisation |

### 🎯 API Efficiency by Design

This system uses **two separate LLM instances** with different cost profiles, assigned by task weight:

| LLM Role | Used For | Why |
|---|---|---|
| `chat_llm` — higher capability | Answer generation only (streaming) | Final answer quality matters most |
| `inspect_llm` — lower cost | All routing, grounding checks, revision, summarisation | High-frequency internal calls — cost adds up fast |

The inspection LLM runs 4–6 times per turn (retrieval decision, relevance checks, grounding verification, usefulness check). Using a cheaper model here reduces cost significantly without affecting answer quality, since these are binary/structured decisions — not open-ended generation.

> **Example:** Set `CHAT_MODEL=gpt-4o` and `INSPECT_MODEL=gpt-4o-mini`. You get high-quality answers at a fraction of the cost of running `gpt-4o` for everything.

### 🧠 Hallucination Prevention

Self-RAG has a built-in **grounding verification loop**:

```
Generate answer → Is it supported by the retrieved context?
    ├── fully_supported     → proceed to user
    ├── partially_supported → revise using only direct context quotes → re-verify
    └── not_supported       → revise or fall back gracefully
```

The system **never fabricates answers**. If the knowledge base doesn't contain the information, it says so cleanly and directs the user to the right contact — rather than making something up.

### 💾 Memory-Driven Retrieval Savings

Because every session has persistent memory, the system avoids redundant retrieval:

- A user asks *"What are the pricing plans?"* → retrieval runs, answer stored in session memory
- User follows up *"Which one is cheapest?"* → answered **directly from memory**, zero retrieval, zero embedding cost
- After every N turns, history is **compressed into a running summary** — context is preserved without growing the prompt indefinitely

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                  Self-RAG Graph (LangGraph)              │
│                                                          │
│  decide_retrieval                                        │
│       │                                                  │
│       ├─── No retrieval needed ──► generate_direct       │
│       │    (greetings, history)                          │
│       │                                                  │
│       └─── Retrieval needed ──► retrieve                 │
│                                     │                    │
│                                is_relevant               │
│                                     │                    │
│                     ┌───────────────┴──────────────┐     │
│                     │ No relevant docs              │     │
│                     ▼                              ▼     │
│              no_answer_found        generate_from_context│
│                     │                              │     │
│                     │                        check_is_sup│
│                     │                        ┌─────┴───┐ │
│                     │                   revise_answer   │ │
│                     │                        │  check   │ │
│                     │                        │ _is_use  │ │
│                     │                              │     │
│                     └──────────────────────────────┘     │
│                                     │                    │
│                                save_memory               │
│                                     │                    │
│                     ┌───────────────┴───────────────┐    │
│              summarize_conversation             (end)    │
│              (if history > threshold)                    │
└─────────────────────────────────────────────────────────┘
     │
     ▼
SSE Stream → Frontend (token by token)
```

### 📁 Project Structure

```
self_rag_chatbot/
│
├── backend/                        # FastAPI production backend
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat.py             # POST /api/chat/stream  (SSE endpoint)
│   │   │   ├── sessions.py         # Session lifecycle endpoints
│   │   │   └── health.py           # GET /health
│   │   ├── core/
│   │   │   ├── graph.py            # LangGraph graph compiler
│   │   │   ├── nodes.py            # All async graph nodes + routing
│   │   │   ├── prompts.py          # All LLM prompts (centralised)
│   │   │   └── state.py            # TypedDict state schema
│   │   ├── db/
│   │   │   ├── pool.py             # Shared async PostgreSQL connection pool
│   │   │   └── sessions.py         # Session table + TTL cleanup
│   │   ├── streaming/
│   │   │   └── sse.py              # LangGraph → SSE event translator
│   │   ├── config.py               # All settings via pydantic-settings
│   │   └── main.py                 # FastAPI app + lifespan startup
│   ├── documents/                  # Place your company PDF files here
│   ├── scripts/
│   │   └── ingest.py               # One-time PDF ingestion into PGVector
│   ├── .env.example                # Environment variable template
│   └── run.py                      # Entry point (cross-platform)
│
├── assets/                         # Screenshots and media
├── docker-compose.yml              # PostgreSQL + pgvector container
└── requirements.txt                # All Python dependencies
```

---

## 📡 SSE Streaming Protocol

The chat endpoint streams responses in real time using **Server-Sent Events (SSE)** — giving users a live, token-by-token experience rather than waiting for the full answer.

### Event Types

```
POST /api/chat/stream
Content-Type: text/event-stream
```

| Event | Payload | When |
|---|---|---|
| `progress` | `{ type, node, message }` | Each graph node starts — shows thinking steps live |
| `token` | `{ type, content }` | Each word chunk from the answer LLM |
| `done` | `{ type, answer, session_id, need_retrieval }` | Stream complete — full answer + metadata |
| `error` | `{ type, content }` | Something went wrong |

### Example Stream

```
data: {"type": "progress", "node": "decide_retrieval", "message": "Deciding whether to search the knowledge base…"}

data: {"type": "progress", "node": "retrieve", "message": "Searching the knowledge base…"}

data: {"type": "progress", "node": "generate_from_context", "message": "Generating answer from context…"}

data: {"type": "token", "content": "BlueBug"}
data: {"type": "token", "content": " offers"}
data: {"type": "token", "content": " three"}
...

data: {"type": "done", "answer": "BlueBug offers three plans...", "session_id": "uuid", "need_retrieval": true}
```

The `need_retrieval` flag in the `done` event lets the frontend show a *"📚 From knowledge base"* or *"💬 From general knowledge"* badge on each message.

---

## 🧠 Persistent Memory

Every conversation session is backed by **PostgreSQL-persisted memory** via LangGraph's checkpointing system:

- **Short-term:** Recent Q&A turns stored per `thread_id` in checkpoint tables
- **Summarisation:** After every N turns, older history is compressed into a running summary — preserving context without growing the prompt indefinitely
- **Cross-turn awareness:** Follow-up questions are answered from memory — no re-retrieval, no extra embedding cost

### Session Lifecycle

```
POST /api/sessions               → create session, get thread_id
POST /api/chat/stream            → send messages (include thread_id each time)
DELETE /api/sessions/{thread_id} → end session, wipe all checkpoint data
```

Sessions idle longer than 30 minutes are automatically purged by a background cleanup task — no orphaned data accumulates in the database.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Graph Framework | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM | OpenAI GPT (dual-model — configurable) |
| Embeddings | OpenAI `text-embedding-3-large` |
| Vector Store | [PGVector](https://github.com/pgvector/pgvector) (PostgreSQL extension) |
| Checkpointing | LangGraph `AsyncPostgresSaver` |
| Backend API | FastAPI + Uvicorn |
| Streaming | Server-Sent Events (SSE) |
| Data Validation | Pydantic v2 |
| Database | PostgreSQL 16 (via Docker) |
| Frontend | [Lovable](https://lovable.dev) |

---

## 🛠️ Local Setup

### Prerequisites

- Python 3.10 or higher
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — for the PostgreSQL database
- OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/FahimS45/advRag.git
cd self_rag_chatbot
```

### 2. Start the database

The `docker-compose.yml` spins up a PostgreSQL 16 instance with the pgvector extension pre-installed. **This must be running before you start the API.**

```bash
docker compose up -d
```

The database will be available at `localhost:5442`. To stop it later: `docker compose down`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cd backend
cp .env.example .env
```

Open `.env` and fill in your values:

```env
OPENAI_API_KEY=sk-...

# PostgreSQL (must match docker-compose.yml)
DB_URI=postgresql://postgres:postgres@localhost:5442/postgres
VECTOR_DB_URI=postgresql+psycopg://postgres:postgres@localhost:5442/postgres
```

> ⚠️ Never commit `.env` to version control. Add it to your `.gitignore`.

### 5. Ingest your documents

Place your company PDF files in `backend/documents/`, then run from the `backend/` directory:

```bash
# First-time ingestion
python -m scripts.ingest

# Wipe existing data and re-ingest from scratch
python -m scripts.ingest --reset
```

### 6. Start the API server

```bash
# From the backend/ directory
python run.py
```

The API will be live at `http://localhost:8000`  
Interactive docs available at `http://localhost:8000/docs`

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Readiness check — DB + graph status |
| `POST` | `/api/sessions` | Create a new chat session → returns `thread_id` |
| `GET` | `/api/sessions/{thread_id}` | Check if a session is still alive |
| `DELETE` | `/api/sessions/{thread_id}` | Terminate session + erase all checkpoint data |
| `POST` | `/api/chat/stream` | Send a question → SSE token stream |

### Typical frontend flow

```
1. POST /api/sessions              →  { thread_id: "uuid" }
2. POST /api/chat/stream           →  SSE stream (progress → tokens → done)
   { "question": "...", "thread_id": "uuid" }
3. DELETE /api/sessions/{thread_id}  →  call on conversation end
```

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key |
| `DB_URI` | ✅ | `postgresql://postgres:postgres@localhost:5442/postgres` | PostgreSQL URI for LangGraph checkpointer |
| `VECTOR_DB_URI` | ✅ | `postgresql+psycopg://...` | PostgreSQL URI for PGVector (psycopg3 dialect) |
| `VECTOR_TABLE_NAME` | | `company_docs` | PGVector collection name |
| `CHAT_MODEL` | | `gpt-4o-mini` | LLM for answer generation (streaming) |
| `INSPECT_MODEL` | | `gpt-4o-mini` | LLM for routing & verification (non-streaming) |
| `CHAT_TEMPERATURE` | | `0.7` | Generation temperature |
| `RETRIEVAL_K` | | `6` | Number of documents to retrieve |
| `SESSION_TTL_MINUTES` | | `30` | Idle session expiry time |
| `CLEANUP_INTERVAL_SECONDS` | | `300` | How often the cleanup task runs |
| `SUMMARIZE_AFTER` | | `3` | Summarise chat history after N turns |
| `MAX_RETRIES` | | `3` | Max answer revision attempts |
| `ALLOWED_ORIGINS` | | `["http://localhost:3000"]` | CORS allowed origins |

> **Cost tip:** Set `CHAT_MODEL=gpt-4o` and `INSPECT_MODEL=gpt-4o-mini` to get the best answer quality while keeping inspection costs low.

---

## 🔒 Data Privacy

- All session data lives in your **own PostgreSQL instance** — nothing is sent to third-party storage
- `DELETE /api/sessions/{thread_id}` immediately wipes the session and all associated checkpoint data
- Idle sessions are auto-purged after 30 minutes — no long-term conversation retention by default

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ using LangGraph · FastAPI · PostgreSQL · OpenAI</sub>
</div>
