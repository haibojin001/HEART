from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import json
import time
import uuid


@dataclass
class ToolParameter:
    name: str
    type: str
    required: bool = False
    description: str = ""
    enum: Optional[List[Any]] = None
    default: Any = None


@dataclass
class ToolSchema:
    id: str
    name: str
    description: str
    category: str
    source: str = "manual"
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: str = ""
    return_schema: Dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> Dict[str, Any]:
        props, required = {}, []
        for p in self.parameters:
            entry = {"type": p.type, "description": p.description}
            if p.enum is not None:
                entry["enum"] = p.enum
            if p.default is not None:
                entry["default"] = p.default
            props[p.name] = entry
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


@dataclass
class ToolFunction:
    schema_id: str
    fn: Callable[..., Any]


class PlannerStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class InvocationStep:
    step: int
    tool_category: str
    tool_hint: str
    objective: str
    dependencies: List[int] = field(default_factory=list)
    replanned: bool = False


@dataclass
class PlannerOutput:
    status: PlannerStatus
    plan: List[InvocationStep] = field(default_factory=list)
    clarification: Optional[str] = None
    raw_intent: Optional[Dict[str, Any]] = None


@dataclass
class HParams:
    retry: int = 3
    priority: str = "normal"
    timeout_s: int = 30


@dataclass
class Dispatch:
    step: int
    target_tool: str
    invocation_request: str
    resolved_bindings: Dict[str, Any] = field(default_factory=dict)
    h_params: HParams = field(default_factory=HParams)


class ExecStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class FailureType(str, Enum):
    SCHEMA_RESOLUTION = "SCHEMA_RESOLUTION"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    RUNTIME_ERROR = "RUNTIME_ERROR"


@dataclass
class ExecutionResult:
    status: ExecStatus
    tool: str
    bound_arguments: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    summary: str = ""
    failure_type: Optional[FailureType] = None
    details: Optional[str] = None
    latency_ms: float = 0.0
    step: int = 0

    def to_context_blob(self) -> Dict[str, Any]:
        return {
            "from_step": self.step,
            "tool": self.tool,
            "status": self.status.value,
            "summary": self.summary,
            "result": self.result,
        }


class VerifyStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class Verification:
    status: VerifyStatus
    step: int
    criteria: Dict[str, str] = field(default_factory=dict)
    feedback: Optional[str] = None


@dataclass
class Context:
    user_query: str
    clarifications: List[Dict[str, str]] = field(default_factory=list)
    execution_history: List[ExecutionResult] = field(default_factory=list)
    verifier_feedback: List[Verification] = field(default_factory=list)
    conversation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)

    def add_clarification(self, question: str, answer: str) -> None:
        self.clarifications.append({"q": question, "a": answer})

    def add_result(self, r: ExecutionResult) -> None:
        self.execution_history.append(r)

    def add_feedback(self, v: Verification) -> None:
        self.verifier_feedback.append(v)

    def last_result(self) -> Optional[ExecutionResult]:
        return self.execution_history[-1] if self.execution_history else None

    def result_for_step(self, step: int) -> Optional[ExecutionResult]:
        for r in reversed(self.execution_history):
            if r.step == step:
                return r
        return None

    def to_prompt_view(self) -> str:
        lines = [f"USER_QUERY: {self.user_query}"]
        for i, c in enumerate(self.clarifications, 1):
            lines.append(f"CLARIFICATION_{i}: Q={c['q']}  A={c['a']}")
        for r in self.execution_history:
            lines.append(
                f"STEP_{r.step}_RESULT[{r.tool}]: status={r.status.value} "
                f"summary={r.summary} result={json.dumps(r.result, default=str)[:400]}"
            )
        for v in self.verifier_feedback:
            if v.status == VerifyStatus.FAIL:
                lines.append(f"FEEDBACK_STEP_{v.step}: {v.feedback}")
        return "\n".join(lines)
