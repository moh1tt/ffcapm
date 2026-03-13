# examples/aapl_analysis.py
# ─────────────────────────────────────────────
# Run this after `pip install factorlib`
# or `pip install -e .` from the repo root
# ─────────────────────────────────────────────

from factorlib import FactorModel

# ── 1. Create model ────────────────────────────────────────────────────
model = FactorModel('AAPL', '2018-01-01', '2024-01-01')

# ── 2. Fit all three models ────────────────────────────────────────────
model.fit_all()

# ── 3. Print summaries ─────────────────────────────────────────────────
model.summary()

# ── 4. Risk-adjusted metrics table ────────────────────────────────────
print(model.metrics())

# ── 5. Visualisations ─────────────────────────────────────────────────
model.plot_loadings('FF5')
model.plot_residuals('FF5')
model.plot_rolling_alpha('FF5', window=126)
model.compare()
