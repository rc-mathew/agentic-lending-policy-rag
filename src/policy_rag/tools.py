"""Deterministic finance tools the agents call instead of doing arithmetic in the LLM.

Every function is pure, typed and validated, so the ReAct loop and the LangGraph agent can expose
them as tools with the same JSON schema. Amounts are in rupees, rates in percent per year.
"""

from __future__ import annotations

import ast
import math
import operator
import os


def _check_positive(**values: float) -> None:
    for name, v in values.items():
        if v is None or v <= 0 or math.isnan(v):
            raise ValueError(f"{name} must be a positive number, got {v!r}")


def emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Equated monthly instalment for a reducing-balance loan, rounded to the rupee."""
    _check_positive(principal=principal, tenure_months=tenure_months)
    if annual_rate_pct < 0:
        raise ValueError("annual_rate_pct cannot be negative")
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return round(principal / tenure_months)
    f = (1 + r) ** tenure_months
    return round(principal * r * f / (f - 1))


def tenure_for_emi(principal: float, annual_rate_pct: float, monthly_emi: float) -> float:
    """Months needed to repay `principal` at a fixed EMI (e.g. keep the EMI after a rate rise).

    Raises if the EMI does not even cover the first month's interest.
    """
    _check_positive(principal=principal, annual_rate_pct=annual_rate_pct, monthly_emi=monthly_emi)
    r = annual_rate_pct / 12 / 100
    if monthly_emi <= principal * r:
        raise ValueError("EMI does not cover the monthly interest; the loan never amortises")
    return round(-math.log(1 - principal * r / monthly_emi) / math.log(1 + r), 1)


def foir(monthly_income: float, existing_emis: float, new_emi: float = 0.0) -> float:
    """Fixed Obligation to Income Ratio in percent: (existing EMIs + new EMI) / gross income.

    No limit is applied: FOIR caps are each lender's credit policy, not an RBI rule in the corpus.
    """
    _check_positive(monthly_income=monthly_income)
    if existing_emis < 0 or new_emi < 0:
        raise ValueError("EMIs cannot be negative")
    return round((existing_emis + new_emi) / monthly_income * 100, 2)


def max_loan_for_ltv(collateral_value: float, ltv_pct: float) -> float:
    """Largest loan a collateral value supports at a given LTV cap."""
    _check_positive(collateral_value=collateral_value, ltv_pct=ltv_pct)
    if ltv_pct > 100:
        raise ValueError("ltv_pct cannot exceed 100")
    return round(collateral_value * ltv_pct / 100)


def max_loan_by_slabs(collateral_value: float, slabs: list[dict]) -> dict:
    """Largest loan under slab-wise LTV caps, where the slab depends on the loan amount itself.

    `slabs` come from the passages, e.g. RBI housing loans:
    [{"up_to": 3000000, "ltv_pct": 90}, {"up_to": 7500000, "ltv_pct": 80}, {"up_to": null,
    "ltv_pct": 75}]. A loan L sits in the first slab whose `up_to` is >= L. For each slab the
    best feasible loan is min(up_to, value * ltv); it counts only if it lands inside that slab.
    This is the step models get wrong by hand: 85% of a ₹3 lakh gold item is ₹2.55 lakh, which
    is above the ₹2.5 lakh line, where the cap is 80% (₹2.4 lakh), back below the line, so the
    answer is the line itself, ₹2.5 lakh.
    """
    _check_positive(collateral_value=collateral_value)
    if not slabs:
        raise ValueError("slabs must not be empty")
    for s in slabs:
        # Strict keys: a slab written as {"max": ...} or {"limit": ...} used to read as having
        # no upper bound, so every slab looked open-ended and the highest LTV won (holdout f01:
        # 85% of ₹5 lakh = ₹4.25 lakh instead of ₹4 lakh). A wrong key must be an error.
        if not isinstance(s, dict) or set(s) != {"up_to", "ltv_pct"}:
            raise ValueError('each slab must have exactly the keys "up_to" (number, or null '
                             f'for the top slab) and "ltv_pct", got {s!r}')
    if sum(s["up_to"] is None for s in slabs) > 1:
        raise ValueError('only the top slab may have "up_to": null')
    try:
        ordered = sorted(slabs, key=lambda s: math.inf if s["up_to"] is None else s["up_to"])
        best, best_slab, lower = 0.0, None, 0.0
        for s in ordered:
            up_to = math.inf if s["up_to"] is None else float(s["up_to"])
            ltv = float(s["ltv_pct"])
            if not 0 < ltv <= 100:
                raise ValueError(f"ltv_pct must be in (0, 100], got {ltv}")
            loan = min(up_to, collateral_value * ltv / 100)
            if loan > lower and loan > best:
                best, best_slab = loan, s
            lower = up_to
    except (KeyError, TypeError) as e:
        raise ValueError(f"each slab needs up_to (number or null) and ltv_pct: {e}") from e
    if best_slab is None:
        raise ValueError("no slab admits a positive loan")
    return {"max_loan": round(best), "ltv_pct_applied": best_slab["ltv_pct"],
            "ltv_actual_pct": round(best / collateral_value * 100, 2)}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos}


def arithmetic(expression: str) -> float:
    """Evaluate plain arithmetic (+ - * / ** and parentheses on numbers), nothing else.

    For the sums the finance tools do not cover: 15 days x ₹100, 0.40% of ₹50 crore.
    Parsed with `ast`, never `eval`, so a model cannot run code through it.
    """
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("expression must be a string of at most 200 characters")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers, + - * / ** and parentheses are allowed")

    try:
        value = ev(ast.parse(expression.replace(",", ""), mode="eval"))
    except (SyntaxError, ZeroDivisionError) as e:
        raise ValueError(f"cannot evaluate {expression!r}: {e}") from e
    return round(value, 4)


# JSON schemas for tool calling (shared by the hand-written ReAct loop and LangGraph).
TOOL_SPECS = [
    {
        "name": "emi",
        "description": "Monthly EMI for a loan. Use for any instalment calculation.",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Loan amount in rupees"},
                "annual_rate_pct": {"type": "number", "description": "Interest rate, % per year"},
                "tenure_months": {"type": "integer", "description": "Loan tenure in months"},
            },
            "required": ["principal", "annual_rate_pct", "tenure_months"],
        },
    },
    {
        "name": "tenure_for_emi",
        "description": "Months needed to repay a loan at a fixed EMI, e.g. after a rate reset.",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number"},
                "annual_rate_pct": {"type": "number"},
                "monthly_emi": {"type": "number"},
            },
            "required": ["principal", "annual_rate_pct", "monthly_emi"],
        },
    },
    {
        "name": "foir",
        "description": "Fixed Obligation to Income Ratio (%) = all EMIs / gross monthly income.",
        "parameters": {
            "type": "object",
            "properties": {
                "monthly_income": {"type": "number"},
                "existing_emis": {"type": "number"},
                "new_emi": {"type": "number"},
            },
            "required": ["monthly_income", "existing_emis"],
        },
    },
    {
        "name": "max_loan_for_ltv",
        "description": "Largest loan allowed for a collateral value at an LTV cap (%).",
        "parameters": {
            "type": "object",
            "properties": {
                "collateral_value": {"type": "number"},
                "ltv_pct": {"type": "number"},
            },
            "required": ["collateral_value", "ltv_pct"],
        },
    },
]

TOOL_SPECS += [
    {
        "name": "max_loan_by_slabs",
        "description": "Largest loan when the LTV cap depends on the loan size (slabs, e.g. "
                       "housing or gold loans). Pass every slab from the passage as "
                       '[{"up_to": <rupees, or null for the top slab>, "ltv_pct": <percent>}]; '
                       "use exactly those two keys.",
        "parameters": {
            "type": "object",
            "properties": {
                "collateral_value": {"type": "number"},
                "slabs": {
                    "type": "array",
                    "description": 'Exactly these keys, e.g. [{"up_to": 3000000, "ltv_pct": 90}, '
                                   '{"up_to": null, "ltv_pct": 75}]; null only for the top slab.',
                    "items": {
                        "type": "object",
                        "properties": {"up_to": {"type": ["number", "null"]},
                                       "ltv_pct": {"type": "number"}},
                        "required": ["up_to", "ltv_pct"],
                    },
                },
            },
            "required": ["collateral_value", "slabs"],
        },
    },
    {
        "name": "arithmetic",
        "description": "Plain arithmetic for anything else: amounts x days, a percentage of an "
                       "amount, differences. Numbers and + - * / ** ( ) only.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. (45 - 30) * 100"}},
            "required": ["expression"],
        },
    },
]

TOOLS = {"emi": emi, "tenure_for_emi": tenure_for_emi, "foir": foir,
         "max_loan_for_ltv": max_loan_for_ltv, "max_loan_by_slabs": max_loan_by_slabs,
         "arithmetic": arithmetic}


# CALC_MODE=v1 reproduces the scored runs' calculator setup for an ablation: the original four
# tools, results passed to the answer step as unnumbered notes. Default v2: all tools, results
# as citable sources. Read when an agent module is imported, so set it before importing them.
V1_TOOLS = ("emi", "tenure_for_emi", "foir", "max_loan_for_ltv")


def calc_mode() -> str:
    mode = os.getenv("CALC_MODE", "v2").strip().lower()
    if mode not in ("v1", "v2"):
        raise ValueError(f"CALC_MODE must be v1 or v2, got {mode!r}")
    return mode


def active_tool_names() -> tuple[str, ...]:
    return V1_TOOLS if calc_mode() == "v1" else tuple(TOOLS)


def active_specs() -> list[dict]:
    names = active_tool_names()
    return [t for t in TOOL_SPECS if t["name"] in names]


def call_tool(name: str, args: dict) -> dict:
    """Run a tool by name; errors come back as data so the agent can recover."""
    fn = TOOLS.get(name) if name in active_tool_names() else None
    if fn is None:
        return {"error": f"unknown tool {name!r}; available: {sorted(active_tool_names())}"}
    try:
        return {"result": fn(**args)}
    except (TypeError, ValueError) as e:
        return {"error": str(e)}
