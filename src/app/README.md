# FastAPI FAQ chat

The `src/app` module sits alongside `src/citation_grounding_agent`. It uses
Jinja2 templates, Tailwind CDN, and a small browser script. `main.py` calls your
existing `HybridRetriever.from_paths(...)` and `ask_grounded_question(...)`
functions directly. There is no AG-UI dependency or separate frontend build.

## Run with uv

From the repository root in PowerShell:

```powershell
uv sync
Copy-Item src/app/.env.example src/app/.env
```

Set `OPENAI_API_KEY` in `src/app/.env`, then start the app:

```powershell
uv run uvicorn app.main:app --app-dir src --reload
```

Open <http://127.0.0.1:8000>. API documentation is at
<http://127.0.0.1:8000/docs>. Restart the server after changing `.env`.
The `--app-dir src` argument makes the new module available alongside the
existing package without changing existing Python files.

## Choose the search mode

**Hybrid is the default.** Choose `hybrid`, `semantic`, or `bm25` in the chat
selector, or pass `mode` in the API request body:

```json
{
  "question": "How much annual leave does a full-time employee receive?",
  "mode": "hybrid",
  "k": 5
}
```

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/chat -Method Post -ContentType 'application/json' -Body '{"question":"How much annual leave does a full-time employee receive?","mode":"hybrid","k":5}'
```

Only `question` is required. `mode` defaults to `hybrid`; `k` defaults to `5`
and must be an integer from `1` to `20`. Questions have a 4,000-character limit.
The parameters are forwarded directly to your agent. Its existing retry logic
may retrieve extra candidates. The response preserves the existing
`GroundedAnswer` fields: `status`, `evidence_status`, `answer`, `claims`,
`citations`, and `reason`.

Hybrid and semantic search use your existing populated Chroma `cedarbridge`
collection and the existing `google/embeddinggemma-300m` model. The first request
may load or download the embedding model; configure Hugging Face access as
needed. BM25 skips embeddings. Paths and the embedding device can be configured
in `.env`. The app does not explicitly rebuild or reindex your data.

## Endpoints

- `GET /`: rendered chat page.
- `POST /api/chat`: the existing grounded agent, with `question`, `mode`, and `k`.
- `GET /health`: basic syntax, API-key presence, FAQ-file, and default hybrid-index
  checks; `503` with a setup message when something is missing. This does not
  load models, verify credentials, or check index contents.
- `GET /docs`: interactive API documentation.

Retrievers are cached. Calls are serialized within each server process because
the existing BM25 retriever changes `k` per request. The interface shows final,
validated claims with citation buttons and quotation cards. Insufficient
evidence is displayed as such.

The conversation exists only in the current page. Each question is independent,
matching your existing function. Refreshing or choosing New conversation clears
the display. Source paths are metadata; arbitrary local files are not served.

## Existing source issue

At implementation time, `src/citation_grounding_agent/grounded_agent.py` was
missing a comma after `tools=[search_internal_faq]` in its `create_agent` call.
Correct that existing syntax error separately before calling the agent. The
chat page still opens and shows a setup message. No original `.py` file was
changed by this app.

## Check the endpoints

```powershell
uv run pytest src/app/tests -q -p no:cacheprovider
```

These tests substitute the agent at the HTTP boundary, so they do not make API
calls or load the embedding model. Live answers also require the existing
syntax fix, valid credentials, and a usable index.

Tailwind uses the requested [Play CDN](https://tailwindcss.com/docs/installation/play-cdn).
It requires internet access and is intended for development.
