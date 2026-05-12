from __future__ import annotations

import json
from typing import Dict, List

from ..llm_backend import LLMBackend
from ..prompts import ROUTER_SYS
from ..schemas import Context, Dispatch, HParams, InvocationStep, ToolSchema
from ..toolface import ToolFace


class Router:
    def __init__(self, backend: LLMBackend, toolface: ToolFace, retrieval_k: int = 8):
        self.backend = backend
        self.toolface = toolface
        self.retrieval_k = retrieval_k

    def route(self, plan: List[InvocationStep], ctx: Context) -> List[Dispatch]:
        step_candidates: Dict[int, List[ToolSchema]] = {}
        for step in plan:
            q = f"{step.tool_hint} {step.objective} {step.tool_category}".strip()
            results = self.toolface.search(q, top_k=self.retrieval_k)
            step_candidates[step.step] = [s for s, _ in results]

        user_payload = self._build_user_prompt(plan, ctx, step_candidates)
        raw = self.backend.chat_json(
            [
                {"role": "system", "content": ROUTER_SYS},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.1,
            max_tokens=1600,
        )
        return self._parse(raw, plan, step_candidates)

    @staticmethod
    def _format_schema(s: ToolSchema) -> str:
        params = [
            f"{p.name}({p.type}{', required' if p.required else ''})"
            f"{(' enum=' + str(p.enum)) if p.enum else ''}"
            f": {p.description}"
            for p in s.parameters
        ]
        return (
            f"  - id: {s.id}\n"
            f"    category: {s.category}\n"
            f"    description: {s.description}\n"
            f"    parameters: {params}\n"
            f"    returns: {s.returns}"
        )

    def _build_user_prompt(
        self,
        plan: List[InvocationStep],
        ctx: Context,
        step_candidates: Dict[int, List[ToolSchema]],
    ) -> str:
        parts = [
            "Current context (C_t):",
            ctx.to_prompt_view(),
            "",
            "Invocation plan to route:",
        ]
        for step in plan:
            parts.append(
                f"  step {step.step}: category={step.tool_category} "
                f"hint={step.tool_hint!r} objective={step.objective!r} "
                f"depends_on={step.dependencies}"
            )
        parts.append("")
        parts.append("Candidate Tool Primitives retrieved for each step:")
        seen = set()
        for step in plan:
            parts.append(f"  [step {step.step}]")
            for s in step_candidates.get(step.step, []):
                if s.id in seen:
                    parts.append(f"    - {s.id}  (see above)")
                    continue
                seen.add(s.id)
                parts.append(self._format_schema(s))
        parts.append("")
        parts.append(
            "Produce the dispatch JSON now. Emit one entry per plan step, "
            "in the original step order, choosing target_tool from the "
            "candidates retrieved for that step."
        )
        return "\n".join(parts)

    def _parse(
        self,
        raw: dict,
        plan: List[InvocationStep],
        step_candidates: Dict[int, List[ToolSchema]],
    ) -> List[Dispatch]:
        dispatches: List[Dispatch] = []
        for entry in raw.get("dispatch") or []:
            step_no = int(entry.get("step", 0))
            target = entry.get("target_tool", "")
            cand_ids = {s.id for s in step_candidates.get(step_no, [])}
            if cand_ids and target not in cand_ids:
                top = step_candidates[step_no][0] if step_candidates[step_no] else None
                target = top.id if top else target
            hp = entry.get("h_params") or {}
            dispatches.append(
                Dispatch(
                    step=step_no,
                    target_tool=target,
                    invocation_request=entry.get("invocation_request", ""),
                    resolved_bindings=dict(entry.get("resolved_bindings") or {}),
                    h_params=HParams(
                        retry=int(hp.get("retry", 3)),
                        priority=str(hp.get("priority", "normal")),
                        timeout_s=int(hp.get("timeout_s", 30)),
                    ),
                )
            )
        dispatches.sort(key=lambda d: d.step)
        return dispatches
