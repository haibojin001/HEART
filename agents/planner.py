from __future__ import annotations

import json
from typing import Callable, List, Optional

from ..llm_backend import LLMBackend
from ..prompts import PLANNER_SYS
from ..schemas import (
    Context,
    InvocationStep,
    PlannerOutput,
    PlannerStatus,
    ToolSchema,
)


class Planner:
    def __init__(
        self,
        backend: LLMBackend,
        clarifier: Optional[Callable[[str], str]] = None,
    ):
        self.backend = backend
        self.clarifier = clarifier

    def step(self, ctx: Context, candidate_tools: Optional[List[ToolSchema]] = None) -> PlannerOutput:
        user_payload = self._build_user_prompt(ctx, candidate_tools)
        raw = self.backend.chat_json(
            [
                {"role": "system", "content": PLANNER_SYS},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        return self._parse(raw)

    def plan_until_sufficient(
        self,
        ctx: Context,
        candidate_tools: Optional[List[ToolSchema]] = None,
        max_clarify: int = 3,
    ) -> PlannerOutput:
        for _ in range(max_clarify + 1):
            out = self.step(ctx, candidate_tools)
            if out.status == PlannerStatus.SUFFICIENT:
                return out
            if not self.clarifier:
                return out
            answer = self.clarifier(out.clarification or "Please clarify.")
            ctx.add_clarification(out.clarification or "", answer)
        return out

    def _build_user_prompt(
        self,
        ctx: Context,
        candidate_tools: Optional[List[ToolSchema]],
    ) -> str:
        parts = [
            "Current context (C_t):",
            ctx.to_prompt_view(),
            "",
        ]
        if candidate_tools:
            parts.append(
                "Candidate Tool Primitives retrieved from ToolFace "
                "(use these names in tool_hint when relevant):"
            )
            for s in candidate_tools[:12]:
                parts.append(f"  - {s.id} [{s.category}]: {s.description}")
            parts.append("")
        if ctx.verifier_feedback:
            last_fail = next((v for v in reversed(ctx.verifier_feedback)
                              if v.status.value == "FAIL"), None)
            if last_fail:
                parts.append(
                    f"Most recent Verifier feedback for step {last_fail.step}: "
                    f"{last_fail.feedback}"
                )
                parts.append("Re-plan to address this failure.")
                parts.append("")
        parts.append(
            "Produce the JSON object now (either SUFFICIENT with a plan, "
            "or INSUFFICIENT with one clarification)."
        )
        return "\n".join(parts)

    @staticmethod
    def _parse(raw: dict) -> PlannerOutput:
        status_str = (raw.get("status") or "").upper()
        if status_str == PlannerStatus.INSUFFICIENT.value:
            return PlannerOutput(
                status=PlannerStatus.INSUFFICIENT,
                clarification=raw.get("clarification", "Could you clarify?"),
                raw_intent=raw.get("intent"),
            )
        if status_str != PlannerStatus.SUFFICIENT.value:
            return PlannerOutput(
                status=PlannerStatus.INSUFFICIENT,
                clarification=f"Planner returned unexpected status: {status_str!r}.",
            )
        plan_raw = raw.get("plan") or []
        steps: List[InvocationStep] = []
        for entry in plan_raw:
            steps.append(
                InvocationStep(
                    step=int(entry.get("step", len(steps) + 1)),
                    tool_category=entry.get("tool_category", ""),
                    tool_hint=entry.get("tool_hint", ""),
                    objective=entry.get("objective", ""),
                    dependencies=list(entry.get("dependencies") or []),
                    replanned=bool(entry.get("replanned", False)),
                )
            )
        return PlannerOutput(
            status=PlannerStatus.SUFFICIENT,
            plan=steps,
            raw_intent=raw.get("intent"),
        )
