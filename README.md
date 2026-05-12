# HEART

**HEART: Harness Engineering via Agent-native Reusable Tool primitives**

A multi-agent tool-calling system that decomposes user queries into a plan, retrieves the relevant tools from a centralized **ToolFace** registry, executes each step through a per-tool Tool Primitive, and recovers from failures via feedback-driven re-planning.

The four agents are:

- **Planner** — intent analysis, information-sufficiency check, plan generation
- **Router** — per-step tool retrieval and parameter mapping
- **Tool Primitive** — schema resolution and isolated execution of one tool
- **Verifier** — four-criteria evaluation, feedback for re-planning

## Install

```bash
pip install requests flask
```

Python 3.9+. Flask is only needed for the web frontend.

## Run the demos (no API key)

```bash
python examples/demo_offline.py
python examples/demo_clarify_and_replan.py
python tests/test_all.py
```

## Run with a real LLM

```bash
export OPENAI_API_KEY=...
python examples/demo_live_llm.py --backend openai --model gpt-4o-mini "your query"

export ANTHROPIC_API_KEY=...
python examples/demo_live_llm.py --backend anthropic --model claude-sonnet-4-6 "your query"
```

## Web frontend

```bash
python web/server.py
# open http://127.0.0.1:8000
```

`web/index.html` is a browser UI for exploring the full ToolFace registry — schema view, parameter table, and formatted call examples.

Endpoints, all returning `{"ok": true, "result": ...}` on success:

- `GET /` — serve index.html
- `GET /tools` — list executable tools
- `POST /execute/<tool_id>` — dispatch by id, body `{"arguments": {...}}`
- `POST /tools/<domain>/<tool>` — TAU2/NESTFUL alias for the same dispatch
- `POST /ace/<tool>` — ACE alias

