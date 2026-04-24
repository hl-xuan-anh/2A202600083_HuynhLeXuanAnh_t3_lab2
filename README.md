# Lab #17 — Multi‑Memory Agent with LangGraph (Complete Project)

Implements a runnable **Multi‑Memory Agent** using **LangGraph** with:

- Short‑term memory: in‑memory ConversationBuffer
- Long‑term memory: **Redis** (user facts/preferences)
- Episodic memory: **JSON log file**
- Semantic memory: **Chroma** vector DB (embedding-based retrieval)

Also includes a full benchmark pipeline comparing:
- **WITH memory**
- **WITHOUT memory**

## 1) Setup

### Prereqs
- Python 3.11+

### Install deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure env

```powershell
Copy-Item .env.example .env
```

Optional (recommended): set `OPENAI_API_KEY` in `.env` to use a real LLM and OpenAI embeddings.

Notes:
- **Chroma does not require Docker**: this project uses embedded persistent Chroma at `./data/chroma` by default.
- **Redis does not require Docker**: by default, this project does **not** require Redis (`USE_REDIS=false`) and stores long‑term facts in `./data/profile_facts.json`.
  - If you want to use Redis anyway, install/run `redis-server` locally and set `USE_REDIS=true` + `REDIS_URL=...` in `.env`.
- If `USE_OPENAI_EMBEDDINGS=false`, the project uses a deterministic local embedding function (still embedding-based) so it runs without external APIs.
- The agent uses an OpenAI-compatible interface; you can set `OPENAI_BASE_URL` for a compatible provider.

## 2) Run chat (CLI)

```powershell
python -m agent.cli --user-id demo
```

Commands:
- `/reset` clears short-term buffer
- `/clear-all` clears all memories for the user (Redis + episodic + Chroma additions + short-term)
- `/exit`

## 3) Run benchmark

```powershell
python -m benchmark.run_benchmark
```

Outputs:
- `benchmark/out/results.json`
- `benchmark/out/results.csv`
- `benchmark/out/summary.md`

## 4) Repo structure

```
.
├── agent/
│   ├── graph.py
│   ├── router.py
│   ├── context_manager.py
│   ├── llm.py
│   └── cli.py
├── memory/
│   ├── base.py
│   ├── short_term.py
│   ├── long_term_redis.py
│   ├── episodic_json.py
│   └── semantic_chroma.py
├── benchmark/
│   ├── test_conversations.py
│   ├── evaluator.py
│   └── run_benchmark.py
├── config.py
├── requirements.txt
└── .env.example
```
