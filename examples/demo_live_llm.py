from __future__ import annotations

import argparse
import json
import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PKG_DIR))

from heart.llm_backend import AnthropicBackend, OpenAIBackend
from heart.orchestrator import HEART
from heart.toolface import ToolFace
from heart.tools import register_all


def cli_clarifier(question: str) -> str:
    print(f"\n[HEART] Planner needs more info:\n  {question}")
    return input("Your answer: ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Natural-language user query.")
    ap.add_argument("--backend", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", default=None,
                    help="Model name. Defaults: openai=gpt-4o-mini, anthropic=claude-sonnet-4-6")
    ap.add_argument("--base-url", default=None,
                    help="Override OPENAI_BASE_URL (use to point at vLLM/local servers).")
    ap.add_argument("--replan-budget", type=int, default=3)
    ap.add_argument("--retrieval-k", type=int, default=8,
                    help="Top-K retrieved Tool Primitives per step.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.backend == "openai":
        backend = OpenAIBackend(
            model=args.model or "gpt-4o-mini",
            base_url=args.base_url,
        )
    else:
        backend = AnthropicBackend(model=args.model or "claude-sonnet-4-6")

    tf = ToolFace()
    register_all(tf)
    print(f"ToolFace: {len(tf)} tools across "
          f"{len({s.category for s in tf.list_schemas()})} categories.\n")

    heart = HEART(
        toolface=tf, backend=backend,
        replan_budget=args.replan_budget,
        retrieval_k=args.retrieval_k,
        clarifier=cli_clarifier,
        verbose=args.verbose,
    )

    report = heart.run(args.query)
    print("\n" + "═" * 70)
    print(f" success        : {report.success}")
    print(f" plan_revisions : {report.plan_revisions}")
    if report.failure_reason:
        print(f" failure        : {report.failure_reason}")
    if report.final_result:
        print(f" final_summary  : {report.final_result.summary}")
        print(f" final_result   :\n{json.dumps(report.final_result.result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
