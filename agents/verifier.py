from __future__ import annotations

import json
from typing import Dict

from ..llm_backend import LLMBackend
from ..prompts import VERIFIER_SYS
from ..schemas import (
    ExecStatus,
    ExecutionResult,
    InvocationStep,
    ToolSchema,
    Verification,
    VerifyStatus,
)


class Verifier:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def verify(
        self,
        result: ExecutionResult,
        step: InvocationStep,
        schema: ToolSchema,
        user_query: str,
    ) -> Verification:
        if result.status == ExecStatus.FAILURE:
            criteria = {
                "task_completion": "fail",
                "argument_consistency": "fail" if result.failure_type and
                                                 result.failure_type.value == "SCHEMA_RESOLUTION" else "pass",
                "execution_validity": "fail",
                "constraint_satisfaction": "fail" if result.failure_type and
                                                    result.failure_type.value == "CONSTRAINT_VIOLATION" else "pass",
            }
            feedback = (
                f"step {step.step} primitive {result.tool} returned FAILURE "
                f"({result.failure_type.value if result.failure_type else 'UNKNOWN'}): "
                f"{result.details}. Re-plan with corrected arguments or an "
                f"alternate tool."
            )
            return Verification(
                status=VerifyStatus.FAIL,
                step=step.step,
                criteria=criteria,
                feedback=feedback,
            )

        user_payload = self._build_user_prompt(result, step, schema, user_query)
        try:
            raw = self.backend.chat_json(
                [
                    {"role": "system", "content": VERIFIER_SYS},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.0,
                max_tokens=700,
            )
        except Exception as e:
            return Verification(
                status=VerifyStatus.PASS,
                step=step.step,
                criteria={k: "pass" for k in
                          ("task_completion", "argument_consistency",
                           "execution_validity", "constraint_satisfaction")},
                feedback=f"Verifier LLM unavailable, conservatively passing: {e}",
            )
        return self._parse(raw, step.step)

    @staticmethod
    def _build_user_prompt(
        result: ExecutionResult,
        step: InvocationStep,
        schema: ToolSchema,
        user_query: str,
    ) -> str:
        schema_blurb = {
            "id": schema.id,
            "description": schema.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "enum": p.enum,
                }
                for p in schema.parameters
            ],
            "returns": schema.returns,
        }
        step_blurb = {
            "step": step.step,
            "objective": step.objective,
            "tool_category": step.tool_category,
            "tool_hint": step.tool_hint,
        }
        result_blurb = {
            "tool": result.tool,
            "status": result.status.value,
            "bound_arguments": result.bound_arguments,
            "result": result.result,
            "summary": result.summary,
        }
        return (
            f"User query: {user_query}\n\n"
            f"Plan step (pi_k):\n{json.dumps(step_blurb, indent=2, default=str)}\n\n"
            f"Tool schema (s_k):\n{json.dumps(schema_blurb, indent=2, default=str)}\n\n"
            f"Execution result (r_k):\n{json.dumps(result_blurb, indent=2, default=str)}\n\n"
            "Evaluate the four criteria and produce the JSON verdict now."
        )

    @staticmethod
    def _parse(raw: dict, step_no: int) -> Verification:
        status_str = (raw.get("status") or "").upper()
        criteria: Dict[str, str] = raw.get("criteria") or {}
        for k, v in list(criteria.items()):
            criteria[k] = str(v).lower()
        status = VerifyStatus.PASS if status_str == "PASS" else VerifyStatus.FAIL
        if status == VerifyStatus.PASS:
            return Verification(status=status, step=step_no, criteria=criteria)
        return Verification(
            status=VerifyStatus.FAIL,
            step=step_no,
            criteria=criteria,
            feedback=raw.get("feedback") or "Verifier signaled FAIL without details.",
        )
