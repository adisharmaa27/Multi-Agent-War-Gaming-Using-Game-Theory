import numpy as np
import pytest

import engine as E


def test_prisoners_dilemma_pure_equilibrium():
    # Years in prison as negative payoffs (from the Week 1 report).
    # Row 0 = silent, row 1 = betray; same layout works for the solver.
    R = np.array([[-1, -10], [0, -5]], float)
    B = np.array([[-1, 0], [-10, -5]], float)
    assert E.solve(R, B)["pure"] == [(1, 1)]


def test_matching_pennies_mixed_is_half_half():
    R = np.array([[1, -1], [-1, 1]], float)
    res = E.solve(R, -R)
    assert res["pure"] == []
    assert res["mixed"]["x"] == pytest.approx(0.5)
    assert res["mixed"]["y"] == pytest.approx(0.5)


def test_mixed_equilibrium_makes_opponent_indifferent():
    R, B = E.matrices(E.DEFAULT_CONFIG["scenarios"]["Red Low vs Blue Low"])
    m = E.solve(R, B)["mixed"]
    xv, yv = np.array([m["x"], 1 - m["x"]]), np.array([m["y"], 1 - m["y"]])
    red_rows, blue_cols = R @ yv, xv @ B
    assert red_rows[0] == pytest.approx(red_rows[1])
    assert blue_cols[0] == pytest.approx(blue_cols[1])
    assert m["x"] == pytest.approx(1 / 3) and m["y"] == pytest.approx(1 / 3)


def test_threshold_matches_indifference_probability():
    R, B = E.matrices(E.DEFAULT_CONFIG["scenarios"]["Red Low vs Blue Low"])
    res = E.solve(R, B)
    assert res["red_threshold"]["q_star"] == pytest.approx(res["mixed"]["y"])
    assert res["blue_threshold"]["q_star"] == pytest.approx(res["mixed"]["x"])


def test_every_default_scenario_has_an_equilibrium():
    for name, s in E.DEFAULT_CONFIG["scenarios"].items():
        res = E.solve(*E.matrices(s))
        assert res["pure"] or res["mixed"], name


def test_pure_equilibria_are_mutual_best_responses():
    for s in E.DEFAULT_CONFIG["scenarios"].values():
        R, B = E.matrices(s)
        for i, j in E.solve(R, B)["pure"]:
            assert R[i, j] >= R[1 - i, j] and B[i, j] >= B[i, 1 - j]


def test_weights_sum_to_one_and_match_example():
    w = E.weights(0.3, 0.4)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["Red High vs Blue High"] == pytest.approx(0.12)
    assert w["Red Low vs Blue Low"] == pytest.approx(0.42)


def test_analyze_aggregate_in_bounds():
    a = E.analyze(E.default_config())["aggregate"]
    assert 0 <= a["attack_prob"] <= 1 and 0 <= a["defend_prob"] <= 1


def test_sweep_and_grid_shapes():
    cfg = E.default_config()
    assert len(E.sweep_capability(cfg, "Red", 11)) == 11
    assert E.grid(cfg, n=5).shape == (5, 5)


def test_payoff_table_marks_equilibrium():
    R, B = E.matrices(E.DEFAULT_CONFIG["scenarios"]["Red High vs Blue Low"])
    t = E.payoff_table(R, B, E.solve(R, B)["pure"])
    assert "*" in t.loc["Red Attack", "Blue Withdraw"]
