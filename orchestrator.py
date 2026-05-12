from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .agents import Planner, Router, ToolPrimitive, Verifier
from .llm_backend import LLMBackend
from .schemas import (
    Context,
    Dispatch,
    ExecStatus,
    ExecutionResult,
    InvocationStep,
    PlannerOutput,
    PlannerStatus,
    Verification,
    VerifyStatus,
)
from .toolface import ToolFace


log = logging.getLogger("heart")


@dataclass
class TraceEntry:
    kind: str
    payload: dict = field(default_factory=dict)


@dataclass
class RunReport:
    success: bool
    final_result: Optional[ExecutionResult]
    context: Context
    plan_revisions: int
    trace: List[TraceEntry] = field(default_factory=list)
    failure_reason: Optional[str] = None


class HEART:

    def __init__(
        self,
        toolface: ToolFace,
        backend: LLMBackend,
        planner_backend: Optional[LLMBackend] = None,
        router_backend: Optional[LLMBackend] = None,
        primitive_backend: Optional[LLMBackend] = None,
        verifier_backend: Optional[LLMBackend] = None,
        replan_budget: int = 3,
        retrieval_k: int = 8,
        clarifier: Optional[Callable[[str], str]] = None,
        max_clarify_rounds: int = 3,
        verbose: bool = False,
    ):
        self.toolface = toolface
        self.backend = backend
        self.replan_budget = replan_budget
        self.retrieval_k = retrieval_k
        self.max_clarify_rounds = max_clarify_rounds
        self.verbose = verbose
        self.clarifier = clarifier

        self.planner = Planner(planner_backend or backend, clarifier=clarifier)
        self.router = Router(router_backend or backend, toolface, retrieval_k=retrieval_k)
        self.verifier = Verifier(verifier_backend or backend)
        self._primitive_backend = primitive_backend or backend

    def run(self, query: str) -> RunReport:
        ctx = Context(user_query=query)
        trace: List[TraceEntry] = []

        seed_candidates = [s for s, _ in self.toolface.search(query, top_k=self.retrieval_k * 2)]
        planner_out = self.planner.plan_until_sufficient(
            ctx, candidate_tools=seed_candidates,
            max_clarify=self.max_clarify_rounds,
        )

        if planner_out.status != PlannerStatus.SUFFICIENT:
            trace.append(TraceEntry("clarify",
                                    {"unresolved": planner_out.clarification}))
            return RunReport(
                success=False, final_result=None, context=ctx,
                plan_revisions=0, trace=trace,
                failure_reason=f"Could not gather sufficient information: "
                               f"{planner_out.clarification}",
            )

        plan = planner_out.plan
        trace.append(TraceEntry("plan", {
            "intent": planner_out.raw_intent,
            "plan": [s.__dict__ for s in plan],
        }))
        self._log(f"Plan ({len(plan)} steps): "
                  f"{[s.tool_hint for s in plan]}")

        budget_used = 0
        while budget_used <= self.replan_budget:
            outcome, last_result = self._run_plan_once(plan, ctx, trace)
            if outcome == "ok":
                return RunReport(
                    success=True, final_result=last_result, context=ctx,
                    plan_revisions=budget_used, trace=trace,
                )
            if outcome == "fatal":
                return RunReport(
                    success=False, final_result=last_result, context=ctx,
                    plan_revisions=budget_used, trace=trace,
                    failure_reason="execution failed and re-planning could not recover",
                )
            if budget_used >= self.replan_budget:
                break
            budget_used += 1
            self._log(f"Re-planning (round {budget_used}/{self.replan_budget})")
            seed_candidates = [s for s, _ in self.toolface.search(query, top_k=self.retrieval_k * 2)]
            planner_out = self.planner.step(ctx, candidate_tools=seed_candidates)
            if planner_out.status != PlannerStatus.SUFFICIENT or not planner_out.plan:
                return RunReport(
                    success=False, final_result=last_result, context=ctx,
                    plan_revisions=budget_used, trace=trace,
                    failure_reason="re-planning did not produce a usable plan",
                )
            plan = planner_out.plan
            trace.append(TraceEntry("replan", {"plan": [s.__dict__ for s in plan]}))

        return RunReport(
            success=False, final_result=last_result, context=ctx,
            plan_revisions=budget_used, trace=trace,
            failure_reason=f"exhausted re-planning budget ({self.replan_budget})",
        )

    def _run_plan_once(
        self,
        plan: List[InvocationStep],
        ctx: Context,
        trace: List[TraceEntry],
    ):
        try:
            dispatches: List[Dispatch] = self.router.route(plan, ctx)
        except Exception as e:
            log.exception("Router failure")
            trace.append(TraceEntry("dispatch", {"error": str(e)}))
            return "fatal", None
        trace.append(TraceEntry("dispatch", {
            "dispatches": [
                {"step": d.step, "target_tool": d.target_tool,
                 "request": d.invocation_request,
                 "bindings": d.resolved_bindings,
                 "h_params": d.h_params.__dict__}
                for d in dispatches
            ]
        }))

        disp_by_step = {d.step: d for d in dispatches}
        last_result: Optional[ExecutionResult] = None

        for step in plan:
            disp = disp_by_step.get(step.step)
            if disp is None:
                self._log(f"  step {step.step}: no dispatch produced; abort")
                return "replan", last_result

            try:
                primitive = ToolPrimitive(self._primitive_backend, self.toolface, disp.target_tool)
            except KeyError as e:
                self._log(f"  step {step.step}: unknown tool {disp.target_tool}: {e}")
                trace.append(TraceEntry("execute", {
                    "step": step.step, "tool": disp.target_tool, "error": str(e),
                }))
                last_result = ExecutionResult(
                    status=ExecStatus.FAILURE, tool=disp.target_tool,
                    failure_type=None, details=str(e),
                    summary=f"unknown tool: {disp.target_tool}",
                    step=step.step,
                )
                ctx.add_result(last_result)
                v = Verification(
                    status=VerifyStatus.FAIL, step=step.step,
                    criteria={"task_completion": "fail",
                              "argument_consistency": "fail",
                              "execution_validity": "fail",
                              "constraint_satisfaction": "fail"},
                    feedback=f"target_tool {disp.target_tool!r} is not in ToolFace; "
                             f"Planner should pick a different tool.",
                )
                ctx.add_feedback(v)
                trace.append(TraceEntry("verify", v.__dict__))
                return "replan", last_result

            upstream = None
            if step.dependencies:
                for dep in sorted(step.dependencies, reverse=True):
                    upstream = ctx.result_for_step(dep)
                    if upstream is not None:
                        break

            result = primitive.invoke(disp, ctx, upstream=upstream)
            ctx.add_result(result)
            last_result = result
            trace.append(TraceEntry("execute", {
                "step": step.step, "tool": result.tool,
                "status": result.status.value,
                "args": result.bound_arguments,
                "result": result.result,
                "summary": result.summary,
                "latency_ms": result.latency_ms,
            }))
            self._log(f"  step {step.step} [{result.tool}] -> "
                      f"{result.status.value}: {result.summary}")

            verdict = self.verifier.verify(
                result, step, primitive.schema, ctx.user_query
            )
            trace.append(TraceEntry("verify", verdict.__dict__))
            if verdict.status == VerifyStatus.FAIL:
                ctx.add_feedback(verdict)
                self._log(f"  step {step.step} VERIFY FAIL: {verdict.feedback}")
                return "replan", last_result

        return "ok", last_result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[HEART] {msg}")
