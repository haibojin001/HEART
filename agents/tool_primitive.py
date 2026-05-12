from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from ..llm_backend import LLMBackend
from ..prompts import TOOL_PRIMITIVE_SYS
from ..schemas import (
    Context,
    Dispatch,
    ExecStatus,
    ExecutionResult,
    FailureType,
    ToolFunction,
    ToolSchema,
)
from ..toolface import ToolFace


class ToolPrimitive:

    def __init__(self, backend: LLMBackend, toolface: ToolFace, schema_id: str):
        self.backend = backend
        self.toolface = toolface
        self.schema: ToolSchema = toolface.get_schema(schema_id)
        self.function: ToolFunction = toolface.get_function(schema_id)

    def invoke(
        self,
        dispatch: Dispatch,
        ctx: Context,
        upstream: Optional[ExecutionResult] = None,
    ) -> ExecutionResult:
        t0 = time.time()

        resolution = self._resolve_arguments(dispatch, ctx, upstream)
        if resolution["status"] == "FAILURE":
            return ExecutionResult(
                status=ExecStatus.FAILURE,
                tool=self.schema.id,
                bound_arguments=resolution.get("bound_arguments", {}),
                failure_type=FailureType(resolution.get("failure_type", FailureType.SCHEMA_RESOLUTION.value)),
                details=resolution.get("details"),
                summary=f"failed to resolve arguments: {resolution.get('details', '')}",
                latency_ms=(time.time() - t0) * 1000,
                step=dispatch.step,
            )

        bound = resolution.get("bound_arguments", {}) or {}
        for k, v in dispatch.resolved_bindings.items():
            bound.setdefault(k, v)

        ok, msg, cleaned = self._validate_against_schema(bound)
        if not ok:
            return ExecutionResult(
                status=ExecStatus.FAILURE,
                tool=self.schema.id,
                bound_arguments=bound,
                failure_type=FailureType.CONSTRAINT_VIOLATION,
                details=msg,
                summary=f"schema validation failed: {msg}",
                latency_ms=(time.time() - t0) * 1000,
                step=dispatch.step,
            )

        last_err = None
        for attempt in range(max(1, dispatch.h_params.retry)):
            try:
                raw_result = self.function.fn(**cleaned)
                summary = self._summarize_result(raw_result)
                return ExecutionResult(
                    status=ExecStatus.SUCCESS,
                    tool=self.schema.id,
                    bound_arguments=cleaned,
                    result=raw_result,
                    summary=summary,
                    latency_ms=(time.time() - t0) * 1000,
                    step=dispatch.step,
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue
        return ExecutionResult(
            status=ExecStatus.FAILURE,
            tool=self.schema.id,
            bound_arguments=cleaned,
            failure_type=FailureType.RUNTIME_ERROR,
            details=last_err,
            summary=f"runtime error: {last_err}",
            latency_ms=(time.time() - t0) * 1000,
            step=dispatch.step,
        )

    def _resolve_arguments(
        self,
        dispatch: Dispatch,
        ctx: Context,
        upstream: Optional[ExecutionResult],
    ) -> Dict[str, Any]:
        user_payload = self._build_user_prompt(dispatch, ctx, upstream)
        try:
            return self.backend.chat_json(
                [
                    {"role": "system", "content": TOOL_PRIMITIVE_SYS},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.0,
                max_tokens=800,
            )
        except Exception as e:
            return {
                "status": "FAILURE",
                "tool": self.schema.id,
                "failure_type": FailureType.SCHEMA_RESOLUTION.value,
                "details": f"LLM resolution call failed: {e}",
                "bound_arguments": dispatch.resolved_bindings,
            }

    def _build_user_prompt(
        self,
        dispatch: Dispatch,
        ctx: Context,
        upstream: Optional[ExecutionResult],
    ) -> str:
        params_doc = []
        for p in self.schema.parameters:
            extra = []
            if p.enum is not None:
                extra.append(f"enum={p.enum}")
            if p.default is not None:
                extra.append(f"default={p.default}")
            extra_s = (" " + " ".join(extra)) if extra else ""
            params_doc.append(
                f"    - {p.name}: {p.type}{' (required)' if p.required else ''}{extra_s}\n"
                f"      desc: {p.description}"
            )
        schema_block = (
            f"Tool schema (s_i):\n"
            f"  id: {self.schema.id}\n"
            f"  description: {self.schema.description}\n"
            f"  parameters:\n" + "\n".join(params_doc) + "\n"
            f"  returns: {self.schema.returns}"
        )
        upstream_block = (
            "Upstream context (c):\n  " + json.dumps(upstream.to_context_blob(), default=str)
            if upstream else "Upstream context (c): NONE"
        )
        request_block = (
            "Router-provided invocation request (x):\n  "
            f"{dispatch.invocation_request}\n"
            f"Router-resolved bindings (use as fallback): "
            f"{json.dumps(dispatch.resolved_bindings, default=str)}"
        )
        return "\n\n".join([schema_block, upstream_block, request_block])

    def _validate_against_schema(self, bound: Dict[str, Any]):
        cleaned: Dict[str, Any] = {}
        for p in self.schema.parameters:
            if p.name not in bound:
                if p.required:
                    return False, f"missing required parameter '{p.name}'", {}
                if p.default is not None:
                    cleaned[p.name] = p.default
                continue
            v = bound[p.name]
            try:
                v = _coerce(v, p.type)
            except Exception as e:
                return False, f"type coercion failed for {p.name!r} -> {p.type}: {e}", {}
            if p.enum is not None and v not in p.enum:
                return False, f"{p.name}={v!r} not in enum {p.enum}", {}
            cleaned[p.name] = v
        return True, "ok", cleaned

    def _summarize_result(self, result: Any) -> str:
        try:
            s = json.dumps(result, default=str)
        except Exception:
            s = str(result)
        if len(s) > 240:
            s = s[:237] + "..."
        return f"{self.schema.id} returned {s}"


def _coerce(v: Any, type_: str) -> Any:
    type_ = (type_ or "string").lower()
    if v is None:
        return None
    if type_ == "string":
        return str(v)
    if type_ == "integer":
        if isinstance(v, bool):
            return int(v)
        return int(v) if not isinstance(v, int) else v
    if type_ == "number":
        return float(v)
    if type_ == "boolean":
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0"}:
            return False
        raise ValueError(f"cannot coerce {v!r} to boolean")
    if type_ == "array":
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [s.strip() for s in v.split(",") if s.strip()]
        raise ValueError(f"cannot coerce {v!r} to array")
    if type_ == "object":
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        raise ValueError(f"cannot coerce {v!r} to object")
    return v
