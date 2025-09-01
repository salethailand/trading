# RUNBOOK – p39 paper trade

## Inputs
- Winners: see `release_bundle/VERIFY.md`
- Risk: `release_bundle/risk_limits.json` (auto-derived from backtest)

## Start
1) Fund paper account with **$100k** notional.
2) Fees/slippage same as backtest.
3) Enable guardrails (soft daily stop -> throttle; hard stop -> flatten+disable; DD kill).

## Monitor (every hour)
- Realized slippage vs backtest (bps)
- Reject rate (%), fill latency (ms), trades/day
- Daily PnL vs soft/hard stops

## Stop criteria
- Hit `max_daily_loss_hard` → flatten + disable
- Drawdown ≥ `max_drawdown_kill_pct` → flatten + disable
- Sustained live slippage > backtest by 2× for 3 sessions → pause + review

## Reproduce
- Commands in `release_bundle/VERIFY.md`
- Assets in GitHub Release `release-YYYYMMDD-p39`
