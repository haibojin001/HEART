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


def scenario_clarification():
    print("\n" + "═" * 70)
    print("  Scenario A — Planner clarification loop")
    print("═" * 70)

    mock = MockBackend()

    planner_calls = {"n": 0}

    def planner_response(payload: str) -> str:
        planner_calls["n"] += 1
        if planner_calls["n"] == 1:
            return json.dumps({
                "status": "INSUFFICIENT",
                "clarification": "Which city are you flying to?",
            })
        return json.dumps({
            "status": "SUFFICIENT",
            "intent": {"task": "Find flights", "tool_categories": ["travel"], "constraints": []},
            "plan": [{
                "step": 1, "tool_category": "travel",
                "tool_hint": "search_flights",
                "objective": "Find ORD -> SFO flights on 2026-06-12",
                "dependencies": [],
            }],
        })

    mock.register("Planner", r".*", planner_response)
    mock.register("Router", r".*", json.dumps({
        "status": "READY",
        "dispatch": [{
            "step": 1, "target_tool": "search_flights",
            "invocation_request": "Find flights from ORD to SFO on 2026-06-12.",
            "resolved_bindings": {"origin": "ORD", "destination": "SFO", "date": "2026-06-12"},
            "h_params": {"retry": 2, "priority": "normal", "timeout_s": 15},
        }],
    }))
    mock.register("Tool Primitive", r"id:\s*search_flights", json.dumps({
        "status": "RESOLVED", "tool": "search_flights",
        "bound_arguments": {"origin": "ORD", "destination": "SFO", "date": "2026-06-12"},
    }))
    mock.register("Verifier", r".*", json.dumps({
        "status": "PASS", "step": 1,
        "criteria": {"task_completion": "pass", "argument_consistency": "pass",
                     "execution_validity": "pass", "constraint_satisfaction": "pass"},
    }))

    tf = ToolFace(); register_all(tf)
    heart = HEART(
        toolface=tf, backend=mock, replan_budget=0,
        clarifier=lambda q: print(f"  Planner asked: {q}") or "San Francisco, departing 2026-06-12",
    )

    r = heart.run("I'd like to book a flight from Chicago.")
    print(f"  success={r.success}  final={r.final_result.summary if r.final_result else 'N/A'}")
    print(f"  clarifications recorded: {len(r.context.clarifications)}")
    for c in r.context.clarifications:
        print(f"    Q: {c['q']!r}\n    A: {c['a']!r}")


def scenario_replanning():
    print("\n" + "═" * 70)
    print("  Scenario B — Verifier-driven re-planning (feedback recovery)")
    print("═" * 70)

    mock = MockBackend()

    planner_calls = {"n": 0}

    def planner_response(payload: str) -> str:
        planner_calls["n"] += 1
        if planner_calls["n"] == 1:
            return json.dumps({
                "status": "SUFFICIENT",
                "intent": {"task": "Book ORD->SFO flight",
                           "tool_categories": ["travel"], "constraints": []},
                "plan": [{
                    "step": 1, "tool_category": "travel",
                    "tool_hint": "book_flight",
                    "objective": "Book a flight from ORD to SFO under name Alex Chen",
                    "dependencies": [],
                }],
            })
        return json.dumps({
            "status": "SUFFICIENT",
            "intent": {"task": "Search then book", "tool_categories": ["travel"], "constraints": []},
            "plan": [
                {"step": 1, "tool_category": "travel",
                 "tool_hint": "search_flights",
                 "objective": "Find a real ORD->SFO flight first",
                 "dependencies": [], "replanned": True},
                {"step": 2, "tool_category": "travel",
                 "tool_hint": "book_flight",
                 "objective": "Book the first flight returned by step 1",
                 "dependencies": [1], "replanned": True},
            ],
        })
    mock.register("Planner", r".*", planner_response)

    router_calls = {"n": 0}

    def router_response(payload: str) -> str:
        router_calls["n"] += 1
        if router_calls["n"] == 1:
            return json.dumps({
                "status": "READY",
                "dispatch": [{
                    "step": 1, "target_tool": "book_flight",
                    "invocation_request": "Book flight FL-DOES-NOT-EXIST for Alex Chen.",
                    "resolved_bindings": {"flight_id": "FL-DOES-NOT-EXIST",
                                          "passenger_name": "Alex Chen"},
                    "h_params": {"retry": 2, "priority": "high", "timeout_s": 20},
                }],
            })
        return json.dumps({
            "status": "READY",
            "dispatch": [
                {"step": 1, "target_tool": "search_flights",
                 "invocation_request": "List ORD->SFO flights on 2026-06-12.",
                 "resolved_bindings": {"origin": "ORD", "destination": "SFO",
                                       "date": "2026-06-12"},
                 "h_params": {"retry": 2, "priority": "normal", "timeout_s": 15}},
                {"step": 2, "target_tool": "book_flight",
                 "invocation_request": "Book the first flight from step 1 under Alex Chen.",
                 "resolved_bindings": {"flight_id": "$step_1.result[0].id",
                                       "passenger_name": "Alex Chen"},
                 "h_params": {"retry": 3, "priority": "high", "timeout_s": 30}},
            ],
        })
    mock.register("Router", r".*", router_response)

    mock.register("Tool Primitive", r"id:\s*book_flight", lambda payload: (
        json.dumps({
            "status": "RESOLVED", "tool": "book_flight",
            "bound_arguments": {"flight_id": "FL-AA101", "passenger_name": "Alex Chen"},
        }) if "FL-AA101" in payload or "$step_1" in payload else json.dumps({
            "status": "RESOLVED", "tool": "book_flight",
            "bound_arguments": {"flight_id": "FL-DOES-NOT-EXIST", "passenger_name": "Alex Chen"},
        })
    ))
    mock.register("Tool Primitive", r"id:\s*search_flights", json.dumps({
        "status": "RESOLVED", "tool": "search_flights",
        "bound_arguments": {"origin": "ORD", "destination": "SFO", "date": "2026-06-12"},
    }))

    mock.register("Verifier", r".*", json.dumps({
        "status": "PASS", "step": 1,
        "criteria": {"task_completion": "pass", "argument_consistency": "pass",
                     "execution_validity": "pass", "constraint_satisfaction": "pass"},
    }))

    tf = ToolFace(); register_all(tf)
    heart = HEART(toolface=tf, backend=mock, replan_budget=3)

    r = heart.run("Book me a flight from Chicago to SF on 2026-06-12 under Alex Chen.")
    print(f"  success={r.success}  plan_revisions={r.plan_revisions}")
    print(f"  final_summary: {r.final_result.summary if r.final_result else 'N/A'}")
    print("  Trace (high-level):")
    for t in r.trace:
        if t.kind in ("plan", "replan"):
            steps = t.payload.get("plan", [])
            print(f"    - {t.kind.upper():8s} {[s['tool_hint'] for s in steps]}")
        elif t.kind == "execute":
            print(f"    - EXECUTE  step={t.payload.get('step')} "
                  f"tool={t.payload.get('tool')} "
                  f"status={t.payload.get('status')}")
        elif t.kind == "verify":
            print(f"    - VERIFY   step={t.payload.get('step')} "
                  f"status={t.payload.get('status')}")


if __name__ == "__main__":
    scenario_clarification()
    scenario_replanning()
