from __future__ import annotations

import json
import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PKG_DIR))

from heart.llm_backend import MockBackend
from heart.orchestrator import HEART
from heart.toolface import ToolFace
from heart.tools import register_all


def _wire_freeze_scenario(mock: MockBackend) -> None:

    mock.register(
        "Planner",
        r"freeze.*debit cards.*u_001|wallet.*stolen",
        json.dumps({
            "status": "SUFFICIENT",
            "intent": {
                "task": "Handle stolen wallet for user u_001",
                "tool_categories": ["finance"],
                "constraints": ["priority=high"],
            },
            "plan": [
                {"step": 1, "tool_category": "finance",
                 "tool_hint": "freeze_debit_card",
                 "objective": "Freeze every debit card on u_001's account",
                 "dependencies": []},
                {"step": 2, "tool_category": "finance",
                 "tool_hint": "report_card_stolen",
                 "objective": "File a stolen-card report for c_4321",
                 "dependencies": [1]},
            ],
        }),
    )

    mock.register(
        "Router",
        r"freeze_debit_card.*report_card_stolen",
        json.dumps({
            "status": "READY",
            "dispatch": [
                {"step": 1, "target_tool": "freeze_debit_card",
                 "invocation_request":
                     "Freeze every debit card on user u_001 by passing card_id='auto' and op='lock'.",
                 "resolved_bindings": {"card_id": "auto", "op": "lock"},
                 "h_params": {"retry": 5, "priority": "high", "timeout_s": 30}},
                {"step": 2, "target_tool": "report_card_stolen",
                 "invocation_request":
                     "File a stolen-card report for c_4321; follow up by email.",
                 "resolved_bindings": {"card_id": "c_4321", "contact_method": "email"},
                 "h_params": {"retry": 3, "priority": "high", "timeout_s": 30}},
            ],
        }),
    )

    mock.register(
        "Tool Primitive",
        r"id:\s*freeze_debit_card",
        json.dumps({
            "status": "RESOLVED",
            "tool": "freeze_debit_card",
            "bound_arguments": {"card_id": "auto", "op": "lock"},
        }),
    )

    mock.register(
        "Tool Primitive",
        r"id:\s*report_card_stolen",
        json.dumps({
            "status": "RESOLVED",
            "tool": "report_card_stolen",
            "bound_arguments": {"card_id": "c_4321", "contact_method": "email"},
        }),
    )

    mock.register(
        "Verifier",
        r"freeze_debit_card|report_card_stolen",
        lambda payload: json.dumps({
            "status": "PASS",
            "step": 1 if "freeze_debit_card" in payload else 2,
            "criteria": {
                "task_completion": "pass",
                "argument_consistency": "pass",
                "execution_validity": "pass",
                "constraint_satisfaction": "pass",
            },
        }),
    )


def _wire_nested_math_scenario(mock: MockBackend) -> None:

    mock.register(
        "Planner",
        r"percentage.*amazon|amazon.*percentage|amazon charge",
        json.dumps({
            "status": "SUFFICIENT",
            "intent": {
                "task": "Compute what fraction of u_001's last 3 transactions was the Amazon charge",
                "tool_categories": ["finance", "math"],
                "constraints": [],
            },
            "plan": [
                {"step": 1, "tool_category": "finance",
                 "tool_hint": "list_recent_transactions",
                 "objective": "Fetch last 3 transactions for u_001",
                 "dependencies": []},
                {"step": 2, "tool_category": "math",
                 "tool_hint": "math_sum_list",
                 "objective": "Sum the amounts from step 1's transactions",
                 "dependencies": [1]},
                {"step": 3, "tool_category": "math",
                 "tool_hint": "math_percentage",
                 "objective": "Compute Amazon-amount / total * 100",
                 "dependencies": [1, 2]},
            ],
        }),
    )

    mock.register(
        "Router",
        r"list_recent_transactions.*math_sum_list|math_sum_list.*math_percentage",
        json.dumps({
            "status": "READY",
            "dispatch": [
                {"step": 1, "target_tool": "list_recent_transactions",
                 "invocation_request":
                     "List the 3 most recent transactions for user u_001.",
                 "resolved_bindings": {"user_id": "u_001", "limit": 3},
                 "h_params": {"retry": 2, "priority": "normal", "timeout_s": 15}},
                {"step": 2, "target_tool": "math_sum_list",
                 "invocation_request":
                     "Sum the `amount` field of every transaction returned by step 1.",
                 "resolved_bindings": {"values": "$step_1.result[*].amount"},
                 "h_params": {"retry": 1, "priority": "normal", "timeout_s": 5}},
                {"step": 3, "target_tool": "math_percentage",
                 "invocation_request":
                     "Compute 199.99 (Amazon) as a percentage of the total from step 2.",
                 "resolved_bindings": {"part": 199.99, "whole": "$step_2.result"},
                 "h_params": {"retry": 1, "priority": "normal", "timeout_s": 5}},
            ],
        }),
    )

    mock.register(
        "Tool Primitive",
        r"id:\s*list_recent_transactions",
        json.dumps({
            "status": "RESOLVED",
            "tool": "list_recent_transactions",
            "bound_arguments": {"user_id": "u_001", "limit": 3},
        }),
    )

    def _resolve_sum(payload: str) -> str:
        import re
        amounts = [float(m) for m in re.findall(r'"amount":\s*([\d.]+)', payload)]
        return json.dumps({
            "status": "RESOLVED",
            "tool": "math_sum_list",
            "bound_arguments": {"values": amounts},
        })
    mock.register("Tool Primitive", r"id:\s*math_sum_list", _resolve_sum)

    def _resolve_pct(payload: str) -> str:
        import re
        m = re.search(r'"result":\s*([\d.]+)', payload)
        whole = float(m.group(1)) if m else 0.0
        return json.dumps({
            "status": "RESOLVED",
            "tool": "math_percentage",
            "bound_arguments": {"part": 199.99, "whole": whole},
        })
    mock.register("Tool Primitive", r"id:\s*math_percentage", _resolve_pct)

    mock.register(
        "Verifier",
        r"list_recent_transactions|math_sum_list|math_percentage",
        lambda payload: json.dumps({
            "status": "PASS",
            "step": (1 if "list_recent_transactions" in payload
                     else 2 if "math_sum_list" in payload else 3),
            "criteria": {
                "task_completion": "pass",
                "argument_consistency": "pass",
                "execution_validity": "pass",
                "constraint_satisfaction": "pass",
            },
        }),
    )


def _print_report(label: str, report) -> None:
    print("\n" + "═" * 70)
    print(f"  {label}")
    print("═" * 70)
    print(f"  success       : {report.success}")
    print(f"  plan_revisions: {report.plan_revisions}")
    if report.failure_reason:
        print(f"  failure       : {report.failure_reason}")
    if report.final_result:
        print(f"  final_summary : {report.final_result.summary}")
        print(f"  final_result  : {json.dumps(report.final_result.result, default=str)[:300]}")
    print("\n  Trace:")
    for i, t in enumerate(report.trace, 1):
        if t.kind == "plan":
            steps = t.payload.get("plan", [])
            print(f"   {i:2}. PLAN     -> {len(steps)} steps: "
                  f"{[s['tool_hint'] for s in steps]}")
        elif t.kind == "dispatch":
            ds = t.payload.get("dispatches", [])
            print(f"   {i:2}. DISPATCH -> {[d['target_tool'] for d in ds]}")
        elif t.kind == "execute":
            print(f"   {i:2}. EXECUTE  -> step={t.payload.get('step')} "
                  f"tool={t.payload.get('tool')} status={t.payload.get('status')} "
                  f":: {t.payload.get('summary','')[:80]}")
        elif t.kind == "verify":
            print(f"   {i:2}. VERIFY   -> step={t.payload.get('step')} "
                  f"status={t.payload.get('status') or t.payload.get('status').value if hasattr(t.payload.get('status'),'value') else t.payload.get('status')}")
        elif t.kind == "replan":
            print(f"   {i:2}. REPLAN")
        elif t.kind == "clarify":
            print(f"   {i:2}. CLARIFY  -> {t.payload}")


def main() -> None:
    mock = MockBackend()
    _wire_freeze_scenario(mock)
    _wire_nested_math_scenario(mock)

    tf = ToolFace()
    register_all(tf)
    print(f"ToolFace loaded: {len(tf)} tools across "
          f"{len({s.category for s in tf.list_schemas()})} categories.")

    heart = HEART(toolface=tf, backend=mock, replan_budget=3, verbose=False)

    r1 = heart.run(
        "My wallet recently got stolen. Please freeze every debit card on "
        "user u_001 and report card c_4321 as stolen."
    )
    _print_report("Scenario 1 — stolen wallet (Figure-1 canonical)", r1)

    r2 = heart.run(
        "On user u_001's last three transactions, what percentage did the "
        "Amazon charge make up? Compute it precisely."
    )
    _print_report("Scenario 2 — nested math (NESTFUL-style)", r2)


if __name__ == "__main__":
    main()
