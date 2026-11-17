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
