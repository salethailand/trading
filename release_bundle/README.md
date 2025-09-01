# p39 Release
- Basis: equity-mode monthlies (E0=$100k)  [or: proxy-mode]
- Winner: $(cat runs/winner_dir_equity_fixed.txt)
- Score knobs: dd_weight=20, strict on, gates: min_trades=5, min_months=3, monthly_target=10%
- See manifest_*.json for exact CLI + filters.
- Risk limits: see risk_limits.json (derived from backtest).
