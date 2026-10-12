import copy
import json

import pandas as pd
import streamlit as st

import engine as E

st.set_page_config(page_title="Red vs Blue - Stage 1", layout="wide")


def load_cfg(cfg: dict) -> None:
    st.session_state.cfg = copy.deepcopy(cfg)
    st.session_state.base = {
        n: pd.DataFrame({o: cfg["scenarios"][n][o] for o in E.OUTCOMES}, index=["Red", "Blue"]).T
        for n in E.SCENARIOS}
    st.session_state.version = st.session_state.get("version", 0) + 1


if "cfg" not in st.session_state:
    load_cfg(E.default_config())
cfg = st.session_state.cfg

#                               sidebar 
st.sidebar.header("Capability probabilities")
cfg["p_red_high"] = st.sidebar.slider("P(Red is High capability)", 0.0, 1.0, float(cfg["p_red_high"]), 0.05)
cfg["p_blue_high"] = st.sidebar.slider("P(Blue is High capability)", 0.0, 1.0, float(cfg["p_blue_high"]), 0.05)
st.sidebar.caption("Stage 1: both sides know each other's type. These probabilities "
                   "say how likely each of the four scenarios is.")
st.sidebar.divider()
if st.sidebar.button("Reset to default payoffs"):
    load_cfg(E.default_config())
    st.rerun()
st.sidebar.download_button("Download config (JSON)", json.dumps(cfg, indent=2), "redblue_config.json")
up = st.sidebar.file_uploader("Load config (JSON)", type="json")
if up is not None and st.sidebar.button("Apply uploaded config"):
    try:
        new = json.load(up)
        assert set(new["scenarios"]) == set(E.SCENARIOS)
        load_cfg(new)
        st.rerun()
    except Exception:
        st.sidebar.error("Invalid config file.")

st.title("Multi-Agent War Gaming Using Game Theory: Stage 1")
st.caption("Red (attacker: Attack / Hold) vs Blue (defender: Defend / Withdraw). "
           "General-sum payoffs, four capability scenarios.")

t_pay, t_scn, t_all, t_sens, t_about = st.tabs(
    ["1. Payoffs (editable)", "2. Scenario results", "3. Combined view", "4. Sensitivity", "About"])

# ---------------- 1. payoffs ----------------
with t_pay:
    st.markdown("Edit any number. A **payoff** is a score for how good an outcome is for that "
               "side (higher is better; negative means a loss). Default scale is about -10 to +10.")
    cols = st.columns(2)
    for k, name in enumerate(E.SCENARIOS):
        with cols[k % 2]:
            st.subheader(name)
            ed = st.data_editor(st.session_state.base[name], key=f"ed_{name}_{st.session_state.version}",
                                use_container_width=True)
            for o in E.OUTCOMES:
                cfg["scenarios"][name][o] = [float(ed.loc[o, "Red"]), float(ed.loc[o, "Blue"])]
            st.caption("Rows are outcomes written as RedAction/BlueAction.")

res = E.analyze(cfg)

# ---------------- 2. per-scenario ----------------
with t_scn:
    cols = st.columns(2)
    for k, name in enumerate(E.SCENARIOS):
        s = res["scenarios"][name]
        with cols[k % 2]:
            st.subheader(f"{name}  (weight {res['weights'][name]:.2f})")
            st.table(E.payoff_table(s["R"], s["B"], s["pure"]))
            st.caption("Cells are (Red payoff, Blue payoff). * = pure-strategy Nash equilibrium.")
            if s["pure"]:
                for i, j in s["pure"]:
                    st.write(f"**Pure equilibrium:** Red {E.RED_ACTIONS[i]}, Blue {E.BLUE_ACTIONS[j]}  "
                             f"-> payoffs ({s['R'][i, j]:g}, {s['B'][i, j]:g})")
            if s["mixed"]:
                m = s["mixed"]
                st.write(f"**Mixed equilibrium:** Red attacks with prob {m['x']:.3f}, Blue defends with "
                         f"prob {m['y']:.3f} -> expected payoffs ({m['red_payoff']:.2f}, {m['blue_payoff']:.2f})")
            st.write("**Belief thresholds:**")
            st.write("- " + E.describe_threshold("Red", s["red_threshold"]))
            st.write("- " + E.describe_threshold("Blue", s["blue_threshold"]))
            if s["primary"]["note"]:
                st.info(s["primary"]["note"])

# ---------------- 3. combined ----------------
with t_all:
    rows = []
    for name in E.SCENARIOS:
        p = res["scenarios"][name]["primary"]
        rows.append({"Scenario": name, "Weight": res["weights"][name], "Equilibrium": p["kind"],
                     "P(Attack)": p["attack_prob"], "P(Defend)": p["defend_prob"],
                     "Red payoff": p["red_payoff"], "Blue payoff": p["blue_payoff"]})
    st.subheader("Equilibrium in each scenario")
    st.dataframe(pd.DataFrame(rows).set_index("Scenario").round(3), use_container_width=True)
    a = res["aggregate"]
    st.subheader("Scenario-weighted outcome")
    c = st.columns(4)
    c[0].metric("P(Attack)", f"{a['attack_prob']:.2f}")
    c[1].metric("P(Defend)", f"{a['defend_prob']:.2f}")
    c[2].metric("Red expected payoff", f"{a['red_payoff']:.2f}")
    c[3].metric("Blue expected payoff", f"{a['blue_payoff']:.2f}")
    eg = res["expected_game"]
    st.subheader("Expected payoff matrix (all scenarios averaged by weight)")
    st.table(E.payoff_table(eg["R"], eg["B"], eg["pure"]))
    if eg["pure"]:
        i, j = eg["pure"][0]
        st.write(f"Equilibrium of the averaged game: Red {E.RED_ACTIONS[i]}, Blue {E.BLUE_ACTIONS[j]}.")
    elif eg["mixed"]:
        st.write(f"Mixed equilibrium of the averaged game: Red attacks {eg['mixed']['x']:.3f}, "
                 f"Blue defends {eg['mixed']['y']:.3f}.")
    st.caption("The averaged matrix is a quick summary. In Stage 1 types are known, so the "
               "scenario-by-scenario results above are the real analysis.")

# ---------------- 4. sensitivity ----------------
with t_sens:
    st.subheader("How outcomes change with capability probabilities")
    who = st.radio("Vary", ["Red", "Blue"], horizontal=True, format_func=lambda w: f"P({w} High)")
    sw = E.sweep_capability(cfg, who).set_index(f"P({who} High)")
    c1, c2 = st.columns(2)
    c1.line_chart(sw[["P(Attack)", "P(Defend)"]])
    c2.line_chart(sw[["Red payoff", "Blue payoff"]])
    st.subheader("Grid: P(Attack) over both probabilities")
    st.dataframe(E.grid(cfg).round(2), use_container_width=True)

    st.subheader("Belief thresholds")
    name = st.selectbox("Scenario", E.SCENARIOS)
    s = res["scenarios"][name]
    red_df, blue_df = E.belief_sweep(s["R"], s["B"])
    c1, c2 = st.columns(2)
    c1.line_chart(red_df)
    c1.caption(E.describe_threshold("Red", s["red_threshold"]))
    c2.line_chart(blue_df)
    c2.caption(E.describe_threshold("Blue", s["blue_threshold"]))

# ---------------- about ----------------
with t_about:
    st.markdown("""
**Stage 1 scope.** Four 2x2 general-sum games (Red High/Low x Blue High/Low). Both sides know the
types. Output: payoff matrices, Nash equilibria (pure and mixed), belief thresholds and sensitivity.

**Conventions.** x = P(Red attacks), y = P(Blue defends). A *belief threshold* is the opponent-action
probability at which a player is indifferent; a player switches action when its belief crosses it.

**Coming next.** Stage 2 hidden types (Bayesian), Stage 3 leadership (Stackelberg),
Stage 4 repeated interaction, Stage 5 AI-enabled decisions.
""")
