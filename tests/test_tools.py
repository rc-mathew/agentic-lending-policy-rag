import pytest

from policy_rag import tools


def test_emi_matches_gold_g50():
    assert tools.emi(2_000_000, 9, 240) == 17_995
    assert tools.emi(2_000_000, 9.5, 240) == 18_643


def test_tenure_extends_when_emi_kept_after_rate_rise():
    assert tools.tenure_for_emi(2_000_000, 9.5, 17_995) == pytest.approx(268.8, abs=0.2)


def test_emi_zero_rate():
    assert tools.emi(120_000, 0, 12) == 10_000


def test_foir():
    assert tools.foir(80_000, 15_000, 17_995) == pytest.approx(41.24, abs=0.01)


def test_max_loan_for_ltv():
    assert tools.max_loan_for_ltv(4_000_000, 80) == 3_200_000


@pytest.mark.parametrize("call", [
    lambda: tools.emi(-1, 9, 12),
    lambda: tools.emi(100, 9, 0),
    lambda: tools.tenure_for_emi(2_000_000, 9.5, 10_000),  # EMI below monthly interest
    lambda: tools.max_loan_for_ltv(100, 120),
])
def test_bad_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def test_call_tool_returns_errors_as_data():
    assert tools.call_tool("emi", {"principal": 120_000, "annual_rate_pct": 0, "tenure_months": 12}) == {"result": 10_000}
    assert "error" in tools.call_tool("emi", {"principal": -5, "annual_rate_pct": 9, "tenure_months": 12})
    assert "unknown tool" in tools.call_tool("sip", {})["error"]
    assert "error" in tools.call_tool("emi", {"principal": 1})  # missing args


def test_specs_match_functions():
    assert {s["name"] for s in tools.TOOL_SPECS} == set(tools.TOOLS)


# ---- v2 tools: slab-wise LTV and plain arithmetic ----

GOLD_SLABS = [{"up_to": 250_000, "ltv_pct": 85}, {"up_to": 500_000, "ltv_pct": 80},
              {"up_to": None, "ltv_pct": 75}]
HOUSING_SLABS = [{"up_to": 3_000_000, "ltv_pct": 90}, {"up_to": 7_500_000, "ltv_pct": 80},
                 {"up_to": None, "ltv_pct": 75}]


def test_slabs_boundary_case_binds_at_the_line():
    # 85% of 3 lakh = 2.55 lakh (above the 2.5 lakh line); 80% = 2.4 lakh (back below it).
    out = tools.max_loan_by_slabs(300_000, GOLD_SLABS)
    assert out["max_loan"] == 250_000 and out["ltv_pct_applied"] == 85


@pytest.mark.parametrize("value, loan, ltv", [
    (4_000_000, 3_200_000, 80),    # 90% would be 36 lakh, above 30 lakh: 80% slab
    (2_500_000, 2_250_000, 90),    # stays in the first slab
    (20_000_000, 15_000_000, 75),  # top slab
])
def test_slabs_housing(value, loan, ltv):
    out = tools.max_loan_by_slabs(value, HOUSING_SLABS)
    assert (out["max_loan"], out["ltv_pct_applied"]) == (loan, ltv)


def test_slabs_order_does_not_matter_and_bad_input_is_an_error():
    assert tools.max_loan_by_slabs(4_000_000, list(reversed(HOUSING_SLABS)))["max_loan"] == 3_200_000
    assert "error" in tools.call_tool("max_loan_by_slabs", {"collateral_value": 1, "slabs": [{}]})
    assert "error" in tools.call_tool("max_loan_by_slabs",
                                      {"collateral_value": 1, "slabs": [{"up_to": 1, "ltv_pct": 150}]})


@pytest.mark.parametrize("expr, value", [
    ("(45 - 30) * 100", 1500), ("50 * 10000000 * 0.40 / 100", 2_000_000),
    ("1,50,000 * 1.1", 165_000), ("-2 + 3", 1), ("2 ** 10", 1024),
])
def test_arithmetic(expr, value):
    assert tools.arithmetic(expr) == pytest.approx(value)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "open('x')", "a + 1", "1 / 0", "9 ** 999", "x" * 201,
])
def test_arithmetic_refuses_anything_but_arithmetic(expr):
    assert "error" in tools.call_tool("arithmetic", {"expression": expr})


def test_calc_mode_v1_reproduces_the_scored_runs(monkeypatch):
    from policy_rag.rag import prompt_with_calculations
    from policy_rag.retrieve import Hit

    hit = Hit(1.0, "c1", "d", "Doc", "RBI", "in_force", 3, 3, "Ch 1", "LTV 80%")
    calcs = ['max_loan_for_ltv({"collateral_value": 4000000, "ltv_pct": 80}) = 3200000']

    monkeypatch.setenv("CALC_MODE", "v2")
    prompt, hits = prompt_with_calculations("q", [hit], calcs)
    assert len(hits) == 2 and "[2] Calculator: max_loan_for_ltv" in prompt
    assert "result" in tools.call_tool("arithmetic", {"expression": "2 * 3"})

    monkeypatch.setenv("CALC_MODE", "v1")
    prompt, hits = prompt_with_calculations("q", [hit], calcs)
    assert len(hits) == 1 and "Calculator results (exact" in prompt and "[2]" not in prompt
    assert "error" in tools.call_tool("arithmetic", {"expression": "2 * 3"})
    assert tools.active_tool_names() == tools.V1_TOOLS


def test_agents_can_choose_the_new_tools():
    # Regression: both agents' action schemas listed only the original four tools.
    import typing

    from policy_rag import graph_agent, react_agent

    assert "arithmetic" in typing.get_args(react_agent.Action)
    assert "max_loan_by_slabs" in typing.get_args(graph_agent.ToolCall.model_fields["tool"].annotation)
