# p39 Release Notes

- Primary: **equity-mode** (E0=$100k) → winner `p39_seed131_dd04_broadhours_strict_open_short_mc_dd06_fee2x_slip0`
- Alternative: proxy-mode → winner `p39_seed131_fee0004_slip0008`
- Scoring: `dd_weight=20`, `--strict`, gates: `min_trades=5`, `min_months=3`, `monthly_target=10%`
- Monthlies computed per `--monthly-mode`; equity mode uses E0 fixed at $100k.
- Risk guardrails derived from loss-only daily quantiles (see `risk_limits.json`).

## Operational Guidance
- Start paper-trade with **$100k** equity, **same fees/slippage** as backtest.
- Enforce risk limits:
  - throttle at `max_daily_loss_soft`
  - flatten at `max_daily_loss_hard`
  - disable at `max_drawdown_kill_pct`
- Monitor: realized slippage vs backtest, reject rate, fill latency, trades/day.
