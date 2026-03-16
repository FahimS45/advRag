# BlueBug Self-RAG — Frontend Integration Guide
### For Lovable (React + TypeScript)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [API Reference](#2-api-reference)
3. [SSE Event Protocol](#3-sse-event-protocol)
4. [Session Lifecycle](#4-session-lifecycle)
5. [React Hooks](#5-react-hooks)
6. [Full Chat Component](#6-full-chat-component)
7. [Progress Messages](#7-progress-messages)
8. [Error Handling](#8-error-handling)
9. [Deployment Checklist](#9-deployment-checklist)

---

## 1. Architecture Overview

```
User types question
      │
      ▼
[POST /api/chat/stream]  ──► FastAPI backend
                                    │
                          LangGraph Self-RAG graph
                          (decides → retrieves → generates)
                                    │
                          token-by-token SSE stream
                                    │
                                    ▼
                     React reads EventSource → renders live
```

The backend **never buffers the full answer** — it streams each word as it is
generated, giving a ChatGPT-like typing feel.

---

## 2. API Reference

Base URL: `http://localhost:8000`  (replace with your deployed URL)

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/sessions` | Create a new chat session → returns `thread_id` |
| `GET` | `/api/sessions/{thread_id}` | Check if a session is alive |
| `DELETE` | `/api/sessions/{thread_id}` | End session and erase all data |

### Chat

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat/stream` | Send a question → SSE stream |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness check |

---

## 3. SSE Event Protocol

Every event arrives as a `data:` line with a JSON payload.

```
data: {"type": "progress", "node": "retrieve", "message": "Searching the knowledge base…"}\n\n
data: {"type": "token",    "content": "The "}\n\n
data: {"type": "token",    "content": "answer "}\n\n
data: {"type": "token",    "content": "is…"}\n\n
data: {"type": "done",     "answer": "The answer is…", "session_id": "uuid", "need_retrieval": true}\n\n
```

### Event shapes

```typescript
type ProgressEvent = {
  type: "progress";
  node: string;       // graph node name, e.g. "retrieve"
  message: string;    // human-readable label
};

type TokenEvent = {
  type: "token";
  content: string;    // one text chunk — append to your answer buffer
};

type DoneEvent = {
  type: "done";
  answer: string;        // complete assembled answer
  session_id: string;
  need_retrieval: boolean; // true = answer came from knowledge base
};

type ErrorEvent = {
  type: "error";
  content: string;
};

type SseEvent = ProgressEvent | TokenEvent | DoneEvent | ErrorEvent;
```

---

## 4. Session Lifecycle

```
App mounts / user opens chat
        │
        ▼
POST /api/sessions  ──► store thread_id in React state (or localStorage)
        │
        ▼
User sends message
        │
        ▼
POST /api/chat/stream  { question, thread_id }
        │
        ▼
Stream events until `done`
        │
        ▼
User closes chat / navigates away
        │
        ▼
DELETE /api/sessions/{thread_id}  ──► all data wiped from DB
```

> **Important:** Always call `DELETE /api/sessions/{thread_id}` when the user
> is done. Idle sessions are automatically purged after 30 minutes, but
> explicit deletion is instant and is the right UX pattern.

---

## 5. React Hooks

### `useSession.ts`

```typescript
import { useState, useEffect, useCallback } from "react";

const API_BASE = process.env.REACT_APP_API_URL ?? "http://localhost:8000";

export function useSession() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);

  // Create a new session on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const res  = await fetch(`${API_BASE}/api/sessions`, { method: "POST" });
        const data = await res.json();
        if (!cancelled) setThreadId(data.thread_id);
      } catch (e) {
        if (!cancelled) setError("Failed to start session.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Destroy session on unmount or explicit call
  const endSession = useCallback(async () => {
    if (!threadId) return;
    try {
      await fetch(`${API_BASE}/api/sessions/${threadId}`, { method: "DELETE" });
    } finally {
      setThreadId(null);
    }
  }, [threadId]);

  useEffect(() => {
    // Clean up when the component unmounts (user closes tab / navigates away)
    const handleUnload = () => {
      if (threadId) {
        // navigator.sendBeacon is fire-and-forget, perfect for unload
        navigator.sendBeacon(`${API_BASE}/api/sessions/${threadId}`);
      }
    };
    window.addEventListener("beforeunload", handleUnload);
    return () => {
      window.removeEventListener("beforeunload", handleUnload);
      endSession();
    };
  }, [threadId, endSession]);

  return { threadId, loading, error, endSession };
}
```

---

### `useChat.ts`

```typescript
import { useState, useRef, useCallback } from "react";

const API_BASE = process.env.REACT_APP_API_URL ?? "http://localhost:8000";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  usedRetrieval?: boolean;
};

export type ChatStatus =
  | "idle"
  | "thinking"          // graph is processing (progress events)
  | "streaming"         // tokens are arriving
  | "done"
  | "error";

export function useChat(threadId: string | null) {
  const [messages, setMessages]     = useState<Message[]>([]);
  const [status, setStatus]         = useState<ChatStatus>("idle");
  const [progress, setProgress]     = useState<string>("");
  const [error, setError]           = useState<string | null>(null);
  const abortRef                    = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (question: string) => {
      if (!threadId || status === "thinking" || status === "streaming") return;

      // Add user message immediately
      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: question,
      };
      setMessages((prev) => [...prev, userMsg]);
      setStatus("thinking");
      setProgress("");
      setError(null);

      // Placeholder for streaming assistant reply
      const assistantId = crypto.randomUUID();
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "" },
      ]);

      abortRef.current = new AbortController();

      try {
        const res = await fetch(`${API_BASE}/api/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, thread_id: threadId }),
          signal: abortRef.current.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`HTTP ${res.status}`);
        }

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let   buffer  = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";          // keep incomplete line

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (!raw) continue;

            let event: any;
            try { event = JSON.parse(raw); } catch { continue; }

            switch (event.type) {
              case "progress":
                setStatus("thinking");
                setProgress(event.message);
                break;

              case "token":
                setStatus("streaming");
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: m.content + event.content }
                      : m
                  )
                );
                break;

              case "done":
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: event.answer, usedRetrieval: event.need_retrieval }
                      : m
                  )
                );
                setStatus("done");
                setProgress("");
                break;

              case "error":
                setError(event.content);
                setStatus("error");
                break;
            }
          }
        }
      } catch (e: any) {
        if (e?.name !== "AbortError") {
          setError(e?.message ?? "Unknown error");
          setStatus("error");
        }
      } finally {
        if (status !== "done") setStatus("idle");
        abortRef.current = null;
      }
    },
    [threadId, status]
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    setStatus("idle");
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setStatus("idle");
    setProgress("");
    setError(null);
  }, []);

  return {
    messages,
    status,
    progress,
    error,
    sendMessage,
    stopStreaming,
    clearMessages,
  };
}
```

---

## 6. Full Chat Component

```tsx
// ChatWindow.tsx
import React, { useState, useRef, useEffect } from "react";
import { useSession }    from "./hooks/useSession";
import { useChat }       from "./hooks/useChat";
import type { Message }  from "./hooks/useChat";

export default function ChatWindow() {
  const { threadId, loading: sessionLoading, error: sessionError } = useSession();
  const { messages, status, progress, error, sendMessage, stopStreaming } =
    useChat(threadId);

  const [input, setInput]   = useState("");
  const bottomRef           = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const q = input.trim();
    if (!q || !threadId) return;
    setInput("");
    sendMessage(q);
  };

  if (sessionLoading) return <div className="p-4">Starting session…</div>;
  if (sessionError)   return <div className="p-4 text-red-500">{sessionError}</div>;

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">

      {/* Message list */}
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Progress indicator */}
        {(status === "thinking") && progress && (
          <div className="flex items-center gap-2 text-sm text-gray-400 italic">
            <span className="animate-pulse">⚙</span>
            {progress}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Error banner */}
      {error && (
        <div className="mb-2 px-3 py-2 bg-red-50 border border-red-200 text-red-700 rounded text-sm">
          {error}
        </div>
      )}

      {/* Input bar */}
      <div className="flex gap-2">
        <input
          className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
          placeholder="Ask anything…"
          disabled={!threadId || status === "thinking" || status === "streaming"}
        />
        {(status === "thinking" || status === "streaming") ? (
          <button
            onClick={stopStreaming}
            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!input.trim() || !threadId}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-prose px-4 py-2 rounded-2xl text-sm whitespace-pre-wrap
          ${isUser
            ? "bg-blue-600 text-white rounded-br-sm"
            : "bg-gray-100 text-gray-800 rounded-bl-sm"}`}
      >
        {message.content || (
          <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse rounded" />
        )}
        {!isUser && message.usedRetrieval !== undefined && message.content && (
          <span className="block mt-1 text-xs text-gray-400">
            {message.usedRetrieval ? "📚 From knowledge base" : "💬 From general knowledge"}
          </span>
        )}
      </div>
    </div>
  );
}
```

---

## 7. Progress Messages

The backend emits these `progress` events in order (RAG path):

| `node`                  | `message`                                      |
|-------------------------|------------------------------------------------|
| `decide_retrieval`      | Deciding whether to search the knowledge base… |
| `retrieve`              | Searching the knowledge base…                  |
| `is_relevant`           | Checking document relevance…                   |
| `generate_from_context` | Generating answer from context…                |
| `check_is_sup`          | Verifying answer is grounded in sources…       |
| `check_is_use`          | Checking if answer is useful…                  |
| `revise_answer`         | Revising answer for accuracy…                  |
| `rewrite_question`      | Rewriting query for better retrieval…          |
| `no_answer_found`       | No relevant answer found in knowledge base.    |
| `generate_direct`       | Generating answer… (no-RAG path)               |

Render these as a subtle status line below the last message.

---

## 8. Error Handling

### Network errors (fetch fails before stream opens)
```typescript
if (!res.ok) {
  const body = await res.json().catch(() => ({}));
  // body.detail contains the FastAPI error message
  setError(body.detail ?? `HTTP ${res.status}`);
}
```

### Common HTTP errors

| Status | Meaning | Fix |
|--------|---------|-----|
| `404` | Session not found | Call `POST /api/sessions` again |
| `503` | Graph not ready | Retry after 2–3 s (server still booting) |
| `422` | Validation error | Check request body matches schema |

### SSE `error` event
```typescript
case "error":
  setError(event.content);
  setStatus("error");
  // Show retry button — call sendMessage(lastQuestion) to retry
  break;
```

### User closes tab
`useSession` attaches a `beforeunload` listener that calls
`navigator.sendBeacon` to fire a best-effort DELETE. No guaranteed delivery,
but the 30-minute TTL auto-cleanup handles the rest.

---

## 9. Deployment Checklist

### Backend (e.g. Railway / Render / EC2)

- [ ] Set `OPENAI_API_KEY` environment variable
- [ ] Set `DB_URI` and `VECTOR_DB_URI` to your hosted PostgreSQL instance
- [ ] Add your Lovable app URL to `ALLOWED_ORIGINS`
  ```
  ALLOWED_ORIGINS=["https://your-app.lovable.app"]
  ```
- [ ] Run document ingestion once after deploying:
  ```bash
  python -m scripts.ingest
  ```
- [ ] Start the server:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```

### Frontend (Lovable)

- [ ] Set `REACT_APP_API_URL` (or `VITE_API_URL`) to your backend URL
- [ ] Copy `useSession.ts`, `useChat.ts` into `src/hooks/`
- [ ] Copy `ChatWindow.tsx` into `src/components/`
- [ ] Import and render `<ChatWindow />` in your page

### Nginx / proxy (if applicable)

SSE requires disabled proxy buffering. Add to your location block:

```nginx
location /api/chat/stream {
    proxy_pass         http://backend:8000;
    proxy_buffering    off;
    proxy_cache        off;
    proxy_set_header   Connection '';
    proxy_http_version 1.1;
    chunked_transfer_encoding on;
}
```

---

*Happy building! 🚀*
