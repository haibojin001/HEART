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
