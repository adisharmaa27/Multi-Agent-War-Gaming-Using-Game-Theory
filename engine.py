"""Stage 1 engine: Red (attacker) vs Blue (defender), general-sum 2x2 games.

Red: Attack / Hold.  Blue: Defend / Withdraw.
Each side is High or Low capability, giving four scenarios, each with its own
payoff matrix.  In Stage 1 both sides know each other's type, so every scenario
is a complete-information game.  The type probabilities say how likely each
scenario is.  Stage 2 will make the types hidden (Bayesian).

Conventions: R[i][j] / B[i][j] = Red / Blue payoff when Red plays row i
(0=Attack, 1=Hold) and Blue plays column j (0=Defend, 1=Withdraw).
x = P(Red attacks), y = P(Blue defends).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

RED_ACTIONS = ["Attack", "Hold"]
BLUE_ACTIONS = ["Defend", "Withdraw"]
OUTCOMES = ["Attack/Defend", "Attack/Withdraw", "Hold/Defend", "Hold/Withdraw"]
LEVELS = ["High", "Low"]
TOL = 1e-9


def scenario_name(red: str, blue: str) -> str:
    return f"Red {red} vs Blue {blue}"


SCENARIOS = [scenario_name(r, b) for r in LEVELS for b in LEVELS]

# Each outcome maps to [Red payoff, Blue payoff]. Scale is roughly -10..+10.
DEFAULT_CONFIG = {
    "p_red_high": 0.3,
    "p_blue_high": 0.4,
    "scenarios": {
        "Red High vs Blue High": {
            "Attack/Defend": [-3, -3], "Attack/Withdraw": [5, -6],
            "Hold/Defend": [0, -1], "Hold/Withdraw": [1, -2]},
        "Red High vs Blue Low": {
            "Attack/Defend": [4, -8], "Attack/Withdraw": [6, -5],
            "Hold/Defend": [0, -1], "Hold/Withdraw": [1, -2]},
        "Red Low vs Blue High": {
            "Attack/Defend": [-8, 2], "Attack/Withdraw": [2, -4],
            "Hold/Defend": [0, -1], "Hold/Withdraw": [1, -2]},
        "Red Low vs Blue Low": {
            "Attack/Defend": [-4, -4], "Attack/Withdraw": [3, -6],
            "Hold/Defend": [0, -2], "Hold/Withdraw": [1, -1]},
    },
}


def default_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def matrices(s: dict) -> tuple[np.ndarray, np.ndarray]:
    """Turn one scenario dict into Red and Blue 2x2 payoff matrices."""
    R = np.array([[s["Attack/Defend"][0], s["Attack/Withdraw"][0]],
                  [s["Hold/Defend"][0], s["Hold/Withdraw"][0]]], dtype=float)
    B = np.array([[s["Attack/Defend"][1], s["Attack/Withdraw"][1]],
                  [s["Hold/Defend"][1], s["Hold/Withdraw"][1]]], dtype=float)
    return R, B


def _threshold(a: float, b: float) -> dict:
    """A player prefers its first action iff f(q) = b + (a-b)*q > 0, where q is
    the belief that the opponent plays ITS first action.
    Returns the indifference belief q* and the rule on either side of it."""
    if abs(a - b) < TOL:
        rule = "always" if b > TOL else "never" if b < -TOL else "indifferent"
        return {"q_star": None, "rule": rule}
    q = -b / (a - b)
    if q <= 0 or q >= 1:
        return {"q_star": None, "rule": "always" if (b + (a - b) * 0.5) > 0 else "never"}
    return {"q_star": q, "rule": "below" if a - b < 0 else "above"}


def solve(R: np.ndarray, B: np.ndarray) -> dict:
    """Pure and mixed Nash equilibria, belief thresholds and a primary equilibrium."""
    pure = [(i, j) for i in range(2) for j in range(2)
            if R[i, j] >= R[1 - i, j] - TOL and B[i, j] >= B[i, 1 - j] - TOL]

    mixed = None
    dx = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
    dy = R[0, 0] - R[0, 1] - R[1, 0] + R[1, 1]
    if abs(dx) > TOL and abs(dy) > TOL:
        x = (B[1, 1] - B[1, 0]) / dx          # makes Blue indifferent
        y = (R[1, 1] - R[0, 1]) / dy          # makes Red indifferent
        if 0 < x < 1 and 0 < y < 1:
            mixed = {"x": float(x), "y": float(y)}

    # Red: prefers Attack iff a*q + b*(1-q) > 0, q = P(Blue defends)
    red_thr = _threshold(R[0, 0] - R[1, 0], R[0, 1] - R[1, 1])
    # Blue: prefers Defend iff a*p + b*(1-p) > 0, p = P(Red attacks)
    blue_thr = _threshold(B[0, 0] - B[0, 1], B[1, 0] - B[1, 1])

    pure.sort(key=lambda c: -(R[c] + B[c]))   # best joint payoff first
    if pure:
        i, j = pure[0]
        x, y, kind = (1.0 if i == 0 else 0.0), (1.0 if j == 0 else 0.0), "pure"
        note = "multiple pure equilibria; the one with the best joint payoff is shown" if len(pure) > 1 else ""
    else:
        x, y, kind, note = mixed["x"], mixed["y"], "mixed", ""
    xv, yv = np.array([x, 1 - x]), np.array([y, 1 - y])
    primary = {"kind": kind, "attack_prob": x, "defend_prob": y, "note": note,
               "red_payoff": float(xv @ R @ yv), "blue_payoff": float(xv @ B @ yv)}
    if mixed:
        mx, my = np.array([mixed["x"], 1 - mixed["x"]]), np.array([mixed["y"], 1 - mixed["y"]])
        mixed["red_payoff"], mixed["blue_payoff"] = float(mx @ R @ my), float(mx @ B @ my)
    return {"pure": pure, "mixed": mixed, "red_threshold": red_thr,
            "blue_threshold": blue_thr, "primary": primary}


def weights(p_red_high: float, p_blue_high: float) -> dict:
    """Probability of each scenario (Red and Blue types assumed independent)."""
    pr = {"High": p_red_high, "Low": 1 - p_red_high}
    pb = {"High": p_blue_high, "Low": 1 - p_blue_high}
    return {scenario_name(r, b): pr[r] * pb[b] for r in LEVELS for b in LEVELS}


def analyze(cfg: dict) -> dict:
    """Solve all four scenarios, then aggregate with the type probabilities."""
    w = weights(cfg["p_red_high"], cfg["p_blue_high"])
    per, EM_R, EM_B = {}, np.zeros((2, 2)), np.zeros((2, 2))
    for name in SCENARIOS:
        R, B = matrices(cfg["scenarios"][name])
        per[name] = {"R": R, "B": B, **solve(R, B)}
        EM_R += w[name] * R
        EM_B += w[name] * B
    agg = {k: sum(w[n] * per[n]["primary"][k] for n in SCENARIOS)
           for k in ("attack_prob", "defend_prob", "red_payoff", "blue_payoff")}
    return {"weights": w, "scenarios": per, "aggregate": agg,
            "expected_game": {"R": EM_R, "B": EM_B, **solve(EM_R, EM_B)}}


def payoff_table(R: np.ndarray, B: np.ndarray, pure: list | None = None) -> pd.DataFrame:
    """Decision (payoff) matrix as '(Red, Blue)' strings; pure equilibria marked with *."""
    pure = pure or []
    cells = [[f"({R[i, j]:g}, {B[i, j]:g})" + (" *" if (i, j) in pure else "")
              for j in range(2)] for i in range(2)]
    return pd.DataFrame(cells, index=[f"Red {a}" for a in RED_ACTIONS],
                        columns=[f"Blue {a}" for a in BLUE_ACTIONS])


def describe_threshold(player: str, thr: dict) -> str:
    """Plain-English belief-threshold rule."""
    if player == "Red":
        act, other, belief = "attacks", "holds", "P(Blue defends)"
    else:
        act, other, belief = "defends", "withdraws", "P(Red attacks)"
    q, rule = thr["q_star"], thr["rule"]
    if q is None:
        return {"always": f"{player} {act} whatever it believes (dominant action).",
                "never": f"{player} never {act}; it {other} whatever it believes.",
                "indifferent": f"{player} is indifferent between its actions."}[rule]
    return f"{player} {act} only if {belief} is {rule} {q:.3f}."


def sweep_capability(cfg: dict, who: str, n: int = 21) -> pd.DataFrame:
    """Vary P(Red High) or P(Blue High) from 0 to 1; track scenario-weighted outcomes."""
    key = "p_red_high" if who == "Red" else "p_blue_high"
    rows = []
    for p in np.linspace(0, 1, n):
        c = copy.deepcopy(cfg)
        c[key] = float(p)
        a = analyze(c)["aggregate"]
        rows.append({f"P({who} High)": p, "P(Attack)": a["attack_prob"],
                     "P(Defend)": a["defend_prob"], "Red payoff": a["red_payoff"],
                     "Blue payoff": a["blue_payoff"]})
    return pd.DataFrame(rows)


def grid(cfg: dict, metric: str = "attack_prob", n: int = 5) -> pd.DataFrame:
    """Metric over a grid of P(Red High) (rows) x P(Blue High) (columns)."""
    ps = np.linspace(0, 1, n)
    out = {}
    for pb in ps:
        col = []
        for pr in ps:
            c = copy.deepcopy(cfg)
            c["p_red_high"], c["p_blue_high"] = float(pr), float(pb)
            col.append(analyze(c)["aggregate"][metric])
        out[f"Blue High={pb:.2f}"] = col
    return pd.DataFrame(out, index=[f"Red High={p:.2f}" for p in ps])


def belief_sweep(R: np.ndarray, B: np.ndarray, n: int = 101) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expected payoff of each action as a function of the belief about the opponent."""
    q = np.linspace(0, 1, n)
    red = pd.DataFrame({"P(Blue defends)": q,
                        "Red: Attack": q * R[0, 0] + (1 - q) * R[0, 1],
                        "Red: Hold": q * R[1, 0] + (1 - q) * R[1, 1]}).set_index("P(Blue defends)")
    blue = pd.DataFrame({"P(Red attacks)": q,
                         "Blue: Defend": q * B[0, 0] + (1 - q) * B[1, 0],
                         "Blue: Withdraw": q * B[0, 1] + (1 - q) * B[1, 1]}).set_index("P(Red attacks)")
    return red, blue
