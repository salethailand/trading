# Verify p39 Release

## Winner(s)
- Equity (E0=$100k): `p39_seed131_dd04_broadhours_strict_open_short_mc_dd06_fee2x_slip0`
- Proxy: `p39_seed131_fee0004_slip0008`

## Exact CLI (equity fixed E0)
python3 score_runs_min.py --input 'runs/p39_*'   --strict --min-trades 5 --min-months 3 --monthly-target 0.10   --monthly-mode equity --equity0 100000 --dd-weight 20 --topk 25   --report --report-out reports/equity_fixed100k   --winner-file runs/winner_dir_equity_fixed.txt --dump-winner-monthlies

## Exact CLI (proxy)
python3 score_runs_min.py --input 'runs/p39_*'   --strict --min-trades 5 --min-months 3 --monthly-target 0.10   --monthly-mode proxy --dd-weight 20 --topk 25   --report --report-out reports/proxy   --winner-file runs/winner_dir_proxy.txt --dump-winner-monthlies

## Artifacts
- manifests: release_bundle/manifest_equity_fixed100k.json, release_bundle/manifest_proxy.json
- scoreboards: release_bundle/scoreboard_equity_fixed100k.csv, release_bundle/scoreboard_proxy.csv
- trades/metrics: release_bundle/*_test_trades.csv, release_bundle/*_trials_metrics.csv
- risk limits: release_bundle/risk_limits.json
- checksums: p39_release_*.tgz + p39_release_checksums.txt
