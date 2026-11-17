"""Deterministic finance tools the agents call instead of doing arithmetic in the LLM.

Every function is pure, typed and validated, so the ReAct loop and the LangGraph agent can expose
them as tools with the same JSON schema. Amounts are in rupees, rates in percent per year.
"""

from __future__ import annotations

import math


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

TOOLS = {"emi": emi, "tenure_for_emi": tenure_for_emi, "foir": foir,
         "max_loan_for_ltv": max_loan_for_ltv}


def call_tool(name: str, args: dict) -> dict:
    """Run a tool by name; errors come back as data so the agent can recover."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}; available: {sorted(TOOLS)}"}
    try:
        return {"result": fn(**args)}
    except (TypeError, ValueError) as e:
        return {"error": str(e)}
