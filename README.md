# Job Search Agent — LangChain Project

An AI agent that searches the web for real job postings and returns structured results — runnable as a CLI script or through a web UI.

Built while following the **LangChain & LangGraph AI Engineering course**

## Features

- 🔍 **Real web search** — the agent calls `TavilySearch` and answers only from real results (no fabricated postings)
- 🧩 **Structured output** — results are parsed into a Pydantic schema (`JobPosting` + `Source`) via `create_agent(response_format=...)`
- 🌐 **Two interfaces** — a CLI (`main.py`) and a web app (`app.py` + `index.html`)
- 🎯 **Per-request configuration** — choose the job site (LinkedIn, Jobstreet, Glints, Kalibrr, Indeed, or any site) and the number of postings
- � **Runs on free models** — uses OpenRouter free-tier models with automatic fallback routing

## Tech Stack

| Component | Tech |
|---|---|
| Agent framework | LangChain 1.x (`create_agent`) |
| Search tool | `langchain-tavily` (Tavily API) |
| LLM | OpenRouter (OpenAI-compatible) — `nvidia/nemotron-3-ultra-550b-a55b:free` with fallbacks |
| Structured output | Pydantic schema via `ToolStrategy` |
| Web backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Package manager | `uv` |

## Project Structure

```
├── main.py        # CLI entrypoint — runs one hardcoded search and prints the result
├── app.py         # FastAPI backend — serves the UI and POST /api/search
├── index.html     # Web UI — query input, site dropdown, result count, job cards
├── static/        # Static assets (logos)
├── .env           # API keys (not committed)
└── pyproject.toml # Dependencies (managed by uv)
```

## Prerequisites

- Python 3.14+ and [`uv`](https://docs.astral.sh/uv/)
- API keys:
  - **`TAVILY_API_KEY`** — get one at [tavily.com](https://tavily.com) (required — powers the search tool)
  - **`OPENROUTER_API_KEY`** — get one at [openrouter.ai](https://openrouter.ai) (required — powers the LLM)
  - Optional: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `OLLAMA_API_KEY`, `LANGSMITH_*` — only needed if you switch providers or enable LangSmith tracing

## Setup

```bash
git clone https://github.com/denisantana/langchain-course.git
cd langchain-course
uv sync
```

Create a `.env` file:

```bash
TAVILY_API_KEY=tvly-...
OPENROUTER_API_KEY=sk-or-v1-...
```

## How to Run

### CLI

```bash
uv run python main.py
```

Runs a fixed search ("AI engineer jobs that mention LangChain in Jakarta on LinkedIn") and prints the full agent state — messages, tool calls, and the final `structured_response`.

### Web App

```bash
uv run uvicorn app:app --port 8001 --reload
```

Open http://127.0.0.1:8001 in your browser.

## How to Use the Web UI

1. **What are you looking for?** — free text, e.g. `AI engineer jobs that mention LangChain, located in Jakarta Indonesia`
2. **Job site** — restrict results to `linkedin.com`, `jobstreet.co.id`, `glints.com`, `kalibrr.com`, `indeed.com`, or select *Any site*
3. **Number of postings** — 1–10
4. Click **Search** — results take ~30–60s (the agent calls Tavily, then structures the output). Each job renders as its own card with title, company, location, requirements, and a link to the posting. Raw sources are in a collapsible section below.

### API

```bash
curl -X POST http://127.0.0.1:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI engineer jobs that mention LangChain in Jakarta Indonesia",
       "site": "linkedin.com",
       "count": 3}'
```

Response shape:

```json
{
  "answer": "Found 3 AI engineer job postings in Jakarta ...",
  "jobs": [
    {
      "title": "AI Engineer",
      "company": "ATI Business Group",
      "location": "Central Jakarta, Indonesia",
      "requirements": "Practical experience with LangGraph, LangChain, ...",
      "url": "https://www.linkedin.com/jobs/view/..."
    }
  ],
  "sources": [{ "title": "...", "url": "..." }]
}
```

## How It Works

```
User query ──► create_agent(llm, tools=[TavilySearch], response_format=AgentResponse)
                  │
                  ├─► LLM decides to call tavily_search(query, include_domains=[site])
                  ├─► Tavily returns real search results
                  └─► LLM emits an AgentResponse tool call (answer + jobs + sources)
```

- `tools=[TavilySearch(max_results=count*2, include_domains=[site])]` — built per request so the site/count options apply
- `response_format=AgentResponse` uses the default `ToolStrategy`: the model "calls" the schema as a tool to produce validated structured output
- LLM requests go through `ChatOpenAI` pointed at `https://openrouter.ai/api/v1`, with a `models` fallback list so free-tier rate limits auto-failover
- `temperature=0`, `max_tokens=4096` for deterministic, non-truncated output

### Switching models / providers

- **Other OpenRouter model**: change the `model=` slug in `app.py` / `main.py`. Find free tool-capable models with:
  ```bash
  curl -s https://openrouter.ai/api/v1/models | jq -r \
    '.data[] | select(.id | endswith(":free")) | select(.supported_parameters | index("tools")) | .id'
  ```
- **Local Ollama**: replace `ChatOpenAI` with `ChatOllama(model="qwen3:4b")` (remove `base_url`/`api_key`). Note: small local models may skip tool calls or produce malformed structured output.

## Credits

- **LangChain / LangGraph** — agent framework and docs
- **Tavily** — web search API
- **OpenRouter** — LLM gateway providing the free models used here
- **FastAPI / Uvicorn / Pydantic** — web backend and schemas

