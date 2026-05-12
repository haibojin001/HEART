# HEART

HEART: Harness Engineering via Agent-native Reusable Tool primitives

A multi-agent tool-calling system (Planner / Router / Tool Primitive / Verifier) with a centralized **ToolFace** registry.

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


# HEART

**HEART: Harness Engineering via Agent-native Reusable Tool primitives**

A multi-agent tool-calling system that decomposes user queries into a plan, retrieves the relevant tools from a centralized **ToolFace** registry, executes each step through a per-tool Tool Primitive, and recovers from failures via feedback-driven re-planning.

The four agents are:

- **Planner** — intent analysis, information-sufficiency check, plan generation
- **Router** — per-step tool retrieval and parameter mapping
- **Tool Primitive** — schema resolution and isolated execution of one tool
- **Verifier** — four-criteria evaluation, feedback for re-planning

The orchestrator in `orchestrator.py` implements the paper's Algorithm 1 end-to-end. Default re-planning budget is `B = 3` (paper Table 7).

## Install

```bash
pip install requests flask
```

Python 3.9+. `requests` is needed for the live-LLM backends; `flask` is needed only for the web frontend. The offline demos and the test suite run on stdlib alone.

Optional for dense retrieval instead of the default TF-IDF:

```bash
pip install sentence-transformers
```

## Run the demos (no API key)

All three commands use a `MockBackend` that returns deterministic JSON, so the full pipeline runs without any network access:

```bash
python examples/demo_offline.py
python examples/demo_clarify_and_replan.py
python tests/test_all.py
```

- `demo_offline.py` walks two scenarios: the stolen-wallet flow from paper Figure 1 (`freeze_debit_card` → `report_card_stolen`), then a nested-math flow (`list_recent_transactions` → `math_sum_list` → `math_percentage`) showing inter-tool communication via the upstream-context channel.
- `demo_clarify_and_replan.py` shows two failure-recovery paths: a clarification loop when the Planner returns `INSUFFICIENT`, and a Verifier-triggered re-plan when the initial plan picks a bogus argument.
- `test_all.py` runs 13 unit tests covering ToolFace, Tool Primitive validation, Verifier fast-path, Planner JSON parsing, Router fallback, and full orchestrator runs.

Expected: `demo_offline` prints `success: True, plan_revisions: 0` for both scenarios; `demo_clarify_and_replan` prints `success=True, plan_revisions=1` for the recovery scenario; tests print `13 passed, 0 failed.`

## Run with a real LLM

`examples/demo_live_llm.py` is an argparse CLI that wires HEART to a real backend:

```bash
# OpenAI
export OPENAI_API_KEY=...
python examples/demo_live_llm.py --backend openai --model gpt-4o-mini \
       "I'm user u_001 and my wallet was stolen. Freeze my cards."

# Anthropic
export ANTHROPIC_API_KEY=...
python examples/demo_live_llm.py --backend anthropic --model claude-sonnet-4-6 \
       "Find flights from ORD to SFO on 2026-06-15 and book the cheapest."

# Self-hosted Qwen3-8B via vLLM (paper's default backbone)
vllm serve Qwen/Qwen3-8B-Instruct --port 8000
python examples/demo_live_llm.py --backend openai \
       --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B-Instruct \
       "your query"
```

The `HEART` constructor in `orchestrator.py` also accepts per-role backend overrides (`planner_backend=`, `router_backend=`, `primitive_backend=`, `verifier_backend=`), which reproduces the hybrid configuration in paper Table 12 (frontier model for Planner/Router/Verifier, small local model for Tool Primitives).

## Web frontend

```bash
python web/server.py
# open http://127.0.0.1:8000
```

`web/index.html` is a browser UI for exploring the full ToolFace registry — schema view, parameter table, formatted call examples, and a "Try It" tab. `web/server.py` is a Flask shim that serves the page and routes "Try It" calls into the 19 bundled tools (other tools in the registry are browse-only).

Endpoints, all returning `{"ok": true, "result": ...}` on success:

- `GET /` — serve index.html
- `GET /tools` — list executable tools
- `POST /execute/<tool_id>` — dispatch by id, body `{"arguments": {...}}`
- `POST /tools/<domain>/<tool>` — TAU2/NESTFUL alias for the same dispatch
- `POST /ace/<tool>` — ACE alias

