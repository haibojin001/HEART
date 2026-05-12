from __future__ import annotations

from typing import List

from ..schemas import ToolParameter, ToolSchema
from ..toolface import ToolFace


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("division by zero")
    return a / b


def sum_list(values: list) -> float:
    return float(sum(float(v) for v in values))


def mean_list(values: list) -> float:
    vs = [float(v) for v in values]
    if not vs:
        raise ValueError("empty list")
    return sum(vs) / len(vs)


def percentage(part: float, whole: float) -> float:
    if whole == 0:
        raise ValueError("whole must be nonzero")
    return 100.0 * part / whole


def register(tf: ToolFace) -> None:
    def numeric(name: str):
        return ToolParameter(name, "number", required=True, description=f"Numeric operand {name}.")

    tf.register(
        ToolSchema(
            id="math_add",
            name="Add two numbers",
            description="Return the sum a + b.",
            category="math",
            source="nestful",
            parameters=[numeric("a"), numeric("b")],
            returns="The numeric sum.",
        ),
        add,
    )
    tf.register(
        ToolSchema(
            id="math_subtract",
            name="Subtract two numbers",
            description="Return a - b.",
            category="math",
            source="nestful",
            parameters=[numeric("a"), numeric("b")],
            returns="The numeric difference.",
        ),
        subtract,
    )
    tf.register(
        ToolSchema(
            id="math_multiply",
            name="Multiply two numbers",
            description="Return a * b.",
            category="math",
            source="nestful",
            parameters=[numeric("a"), numeric("b")],
            returns="The numeric product.",
        ),
        multiply,
    )
    tf.register(
        ToolSchema(
            id="math_divide",
            name="Divide two numbers",
            description="Return a / b. Raises if b == 0.",
            category="math",
            source="nestful",
            parameters=[numeric("a"), numeric("b")],
            returns="The numeric quotient.",
        ),
        divide,
    )
    tf.register(
        ToolSchema(
            id="math_sum_list",
            name="Sum a list of numbers",
            description="Return the sum of every value in `values`.",
            category="math",
            source="nestful",
            parameters=[
                ToolParameter("values", "array", required=True,
                              description="Array of numeric values to sum."),
            ],
            returns="Numeric total.",
        ),
        sum_list,
    )
    tf.register(
        ToolSchema(
            id="math_mean_list",
            name="Mean of a list",
            description="Return the arithmetic mean of `values`.",
            category="math",
            source="nestful",
            parameters=[
                ToolParameter("values", "array", required=True,
                              description="Array of numeric values."),
            ],
            returns="Arithmetic mean.",
        ),
        mean_list,
    )
    tf.register(
        ToolSchema(
            id="math_percentage",
            name="Compute percentage",
            description="Return 100 * part / whole.",
            category="math",
            source="nestful",
            parameters=[
                ToolParameter("part", "number", required=True,
                              description="Numerator value."),
                ToolParameter("whole", "number", required=True,
                              description="Denominator value (must be nonzero)."),
            ],
            returns="The percentage as a float (e.g., 25.0 means 25%).",
        ),
        percentage,
    )
