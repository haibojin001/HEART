from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Callable, List, Tuple

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PKG_DIR))

from heart.agents import Planner, Router, ToolPrimitive, Verifier
from heart.llm_backend import MockBackend
from heart.orchestrator import HEART
from heart.schemas import (
    Context, Dispatch, ExecStatus, ExecutionResult, HParams,
    InvocationStep, PlannerStatus, ToolParameter, ToolSchema,
    VerifyStatus,
)
from heart.toolface import ToolFace
from heart.tools import register_all


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def test(fn: Callable[[], None]) -> Callable[[], None]:
    _TESTS.append((fn.__name__, fn))
    return fn


def expect(cond: bool, msg: str = "") -> None:
    if not cond:
        raise AssertionError(msg or "expectation failed")


def expect_eq(a, b, msg: str = "") -> None:
    if a != b:
        raise AssertionError(msg or f"expected {a!r} == {b!r}")


@test
def toolface_register_and_lookup():
    tf = ToolFace()
    tf.register(
        ToolSchema(id="echo", name="echo", description="returns its input",
                   category="util",
                   parameters=[ToolParameter("msg", "string", required=True,
                                             description="text to echo")]),
        lambda msg: msg,
    )
    expect_eq(len(tf), 1)
    expect_eq(tf.get_schema("echo").id, "echo")
    expect_eq(tf.get_function("echo").fn("hi"), "hi")


@test
def toolface_id_collision_raises():
    tf = ToolFace()
    s = ToolSchema(id="dup", name="dup", description="d", category="x")
    tf.register(s, lambda: None)
    try:
        tf.register(s, lambda: None)
    except ValueError:
        return
    raise AssertionError("expected ValueError on duplicate registration")


@test
def toolface_search_lexical_ranks_relevant_first():
    tf = ToolFace()
    register_all(tf)
    hits = tf.search("freeze a stolen debit card", top_k=3)
    ids = [s.id for s, _ in hits]
    expect("freeze_debit_card" in ids, f"freeze_debit_card not in top-3: {ids}")


@test
def toolface_search_filters_by_category():
    tf = ToolFace()
    register_all(tf)
    hits = tf.search("add", top_k=5, category="math")
    ids = [s.id for s, _ in hits]
    expect(all(tf.get_schema(i).category == "math" for i in ids),
           f"non-math results leaked: {ids}")


@test
def primitive_validates_and_executes_correctly():
    tf = ToolFace()
    tf.register(
        ToolSchema(id="add2", name="add2",
                   description="a+b",
                   category="math",
                   parameters=[
                       ToolParameter("a", "integer", required=True, description="lhs"),
                       ToolParameter("b", "integer", required=True, description="rhs"),
                   ]),
        lambda a, b: a + b,
    )
    mock = MockBackend()
    mock.register("Tool Primitive", r"id:\s*add2", json.dumps({
        "status": "RESOLVED", "tool": "add2",
        "bound_arguments": {"a": "3", "b": "4"},
    }))
    prim = ToolPrimitive(mock, tf, "add2")
    disp = Dispatch(step=1, target_tool="add2",
                    invocation_request="add 3 and 4",
                    resolved_bindings={}, h_params=HParams())
    r = prim.invoke(disp, Context("test"))
    expect_eq(r.status, ExecStatus.SUCCESS)
    expect_eq(r.result, 7)
    expect_eq(r.bound_arguments, {"a": 3, "b": 4})


@test
def primitive_surfaces_missing_required_param():
    tf = ToolFace()
    tf.register(
        ToolSchema(id="needs_x", name="needs_x", description="d",
                   category="x",
                   parameters=[ToolParameter("x", "string", required=True,
                                             description="required")]),
        lambda x: x,
    )
    mock = MockBackend()
    mock.register("Tool Primitive", r"id:\s*needs_x", json.dumps({
        "status": "RESOLVED", "tool": "needs_x", "bound_arguments": {},
    }))
    prim = ToolPrimitive(mock, tf, "needs_x")
    r = prim.invoke(
        Dispatch(step=1, target_tool="needs_x",
                 invocation_request="please call it",
                 resolved_bindings={}, h_params=HParams()),
        Context("test"),
    )
    expect_eq(r.status, ExecStatus.FAILURE)
    expect("missing required parameter" in (r.details or ""),
           f"unexpected details: {r.details!r}")


@test
def primitive_rejects_enum_violation():
    tf = ToolFace()
    tf.register(
        ToolSchema(id="picker", name="picker", description="d",
                   category="x",
                   parameters=[ToolParameter("color", "string", required=True,
                                             enum=["red", "blue"],
                                             description="color")]),
        lambda color: color,
    )
    mock = MockBackend()
    mock.register("Tool Primitive", r"id:\s*picker", json.dumps({
        "status": "RESOLVED", "tool": "picker", "bound_arguments": {"color": "green"},
    }))
    prim = ToolPrimitive(mock, tf, "picker")
    r = prim.invoke(
        Dispatch(step=1, target_tool="picker",
                 invocation_request="pick green",
                 resolved_bindings={}, h_params=HParams()),
        Context("test"),
    )
    expect_eq(r.status, ExecStatus.FAILURE)
    expect("enum" in (r.details or "").lower(),
           f"expected enum error, got {r.details!r}")


@test
def verifier_fast_path_on_failure_result():
    mock = MockBackend()
    v = Verifier(mock)
    step = InvocationStep(step=1, tool_category="x", tool_hint="t", objective="o")
    schema = ToolSchema(id="t", name="t", description="d", category="x")
    failed = ExecutionResult(status=ExecStatus.FAILURE, tool="t",
                             details="schema explosion", step=1)
    verdict = v.verify(failed, step, schema, "do thing")
    expect_eq(verdict.status, VerifyStatus.FAIL)
    expect("schema explosion" in (verdict.feedback or ""))


@test
def planner_parses_insufficient():
    mock = MockBackend()
    mock.register("Planner", r".*", json.dumps({
        "status": "INSUFFICIENT", "clarification": "What city?",
    }))
    p = Planner(mock)
    out = p.step(Context("book flight"))
    expect_eq(out.status, PlannerStatus.INSUFFICIENT)
    expect_eq(out.clarification, "What city?")


@test
def planner_parses_sufficient_plan():
    mock = MockBackend()
    mock.register("Planner", r".*", json.dumps({
        "status": "SUFFICIENT",
        "plan": [
            {"step": 1, "tool_category": "math", "tool_hint": "add",
             "objective": "add a and b", "dependencies": []},
            {"step": 2, "tool_category": "math", "tool_hint": "sub",
             "objective": "sub c", "dependencies": [1]},
        ],
    }))
    p = Planner(mock)
    out = p.step(Context("compute"))
    expect_eq(out.status, PlannerStatus.SUFFICIENT)
    expect_eq(len(out.plan), 2)
    expect_eq(out.plan[1].dependencies, [1])


@test
def router_falls_back_when_target_not_in_candidates():
    tf = ToolFace(); register_all(tf)
    mock = MockBackend()
    mock.register("Router", r".*", json.dumps({
        "status": "READY",
        "dispatch": [{
            "step": 1, "target_tool": "TOTALLY_MADE_UP_TOOL",
            "invocation_request": "do thing",
            "resolved_bindings": {},
            "h_params": {"retry": 1, "priority": "low", "timeout_s": 5},
        }],
    }))
    r = Router(mock, tf, retrieval_k=5)
    plan = [InvocationStep(step=1, tool_category="finance",
                           tool_hint="freeze a debit card",
                           objective="freeze")]
    out = r.route(plan, Context("freeze it"))
    expect_eq(len(out), 1)
    expect(out[0].target_tool != "TOTALLY_MADE_UP_TOOL",
           "Router did not fall back to a real candidate")


@test
def orchestrator_full_happy_path():
    mock = MockBackend()
    mock.register("Planner", r".*", json.dumps({
        "status": "SUFFICIENT",
        "plan": [{"step": 1, "tool_category": "math", "tool_hint": "add",
                  "objective": "1+2", "dependencies": []}],
    }))
    mock.register("Router", r".*", json.dumps({
        "status": "READY",
        "dispatch": [{
            "step": 1, "target_tool": "math_add",
            "invocation_request": "Add 1 and 2.",
            "resolved_bindings": {"a": 1, "b": 2},
            "h_params": {"retry": 1, "priority": "low", "timeout_s": 5},
        }],
    }))
    mock.register("Tool Primitive", r"id:\s*math_add", json.dumps({
        "status": "RESOLVED", "tool": "math_add",
        "bound_arguments": {"a": 1, "b": 2},
    }))
    mock.register("Verifier", r".*", json.dumps({
        "status": "PASS", "step": 1,
        "criteria": {"task_completion": "pass",
                     "argument_consistency": "pass",
                     "execution_validity": "pass",
                     "constraint_satisfaction": "pass"},
    }))

    tf = ToolFace(); register_all(tf)
    h = HEART(toolface=tf, backend=mock, replan_budget=0)
    r = h.run("Add 1 and 2.")
    expect(r.success, f"expected success, got {r.failure_reason}")
    expect_eq(r.final_result.result, 3)
    expect_eq(r.plan_revisions, 0)


@test
def orchestrator_recovers_after_one_replan():
    mock = MockBackend()
    planner_n = {"i": 0}

    def planner(payload: str) -> str:
        planner_n["i"] += 1
        if planner_n["i"] == 1:
            return json.dumps({
                "status": "SUFFICIENT",
                "plan": [{"step": 1, "tool_category": "math",
                          "tool_hint": "divide",
                          "objective": "divide a by b",
                          "dependencies": []}],
            })
        return json.dumps({
            "status": "SUFFICIENT",
            "plan": [{"step": 1, "tool_category": "math",
                      "tool_hint": "add",
                      "objective": "switched to add after divide failure",
                      "dependencies": [], "replanned": True}],
        })

    mock.register("Planner", r".*", planner)

    router_n = {"i": 0}

    def router(payload: str) -> str:
        router_n["i"] += 1
        if router_n["i"] == 1:
            return json.dumps({
                "status": "READY",
                "dispatch": [{
                    "step": 1, "target_tool": "math_divide",
                    "invocation_request": "Divide 10 by 0.",
                    "resolved_bindings": {"a": 10, "b": 0},
                    "h_params": {"retry": 1, "priority": "low", "timeout_s": 5},
                }],
            })
        return json.dumps({
            "status": "READY",
            "dispatch": [{
                "step": 1, "target_tool": "math_add",
                "invocation_request": "Add 10 and 0.",
                "resolved_bindings": {"a": 10, "b": 0},
                "h_params": {"retry": 1, "priority": "low", "timeout_s": 5},
            }],
        })
    mock.register("Router", r".*", router)

    mock.register("Tool Primitive", r"id:\s*math_divide", json.dumps({
        "status": "RESOLVED", "tool": "math_divide",
        "bound_arguments": {"a": 10, "b": 0},
    }))
    mock.register("Tool Primitive", r"id:\s*math_add", json.dumps({
        "status": "RESOLVED", "tool": "math_add",
        "bound_arguments": {"a": 10, "b": 0},
    }))
    mock.register("Verifier", r".*", json.dumps({
        "status": "PASS", "step": 1,
        "criteria": {"task_completion": "pass",
                     "argument_consistency": "pass",
                     "execution_validity": "pass",
                     "constraint_satisfaction": "pass"},
    }))

    tf = ToolFace(); register_all(tf)
    h = HEART(toolface=tf, backend=mock, replan_budget=3)
    r = h.run("Compute 10 / 0, please.")
    expect(r.success, f"expected success after replan, got {r.failure_reason}")
    expect_eq(r.plan_revisions, 1)
    expect_eq(r.final_result.tool, "math_add")
    expect_eq(r.final_result.result, 10)


def main():
    passed = failed = 0
    for name, fn in _TESTS:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
