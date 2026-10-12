# Red vs Blue - Stage 1 (Multi-Agent War Gaming Using Game Theory)

Run:   pip install -r requirements.txt && streamlit run app.py
Test:  pytest

- engine.py  game logic (2x2 general-sum solver, scenarios, thresholds, sensitivity)
- app.py     Streamlit interface (editable payoffs, results, sensitivity)
- test_*.py  tests

Payoff = a score for how good an outcome is for that side (higher is better).
Defaults are illustrative and editable in the app or via the JSON config.
