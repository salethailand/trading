#!/usr/bin/env bash
set -euo pipefail
DATE=$(date +%Y%m%d)

E_WIN=$(<runs/winner_dir_equity_fixed.txt)
P_WIN=$(<runs/winner_dir_proxy.txt)

mkdir -p release_bundle
cp runs/"$E_WIN"/trials_metrics.csv         "release_bundle/${E_WIN}_trials_metrics.csv"
cp runs/"$E_WIN"/test_trades.csv            "release_bundle/${E_WIN}_test_trades.csv"
cp reports/equity_fixed100k/scoreboard.csv  "release_bundle/scoreboard_equity_fixed100k.csv"
cp reports/equity_fixed100k/release_manifest.json "release_bundle/manifest_equity_fixed100k.json"

cp runs/"$P_WIN"/trials_metrics.csv         "release_bundle/${P_WIN}_trials_metrics.csv"
cp runs/"$P_WIN"/test_trades.csv            "release_bundle/${P_WIN}_test_trades.csv"
cp reports/proxy/scoreboard.csv             "release_bundle/scoreboard_proxy.csv"
cp reports/proxy/release_manifest.json      "release_bundle/manifest_proxy.json"

tar -czf "p39_release_${DATE}.tgz" -C release_bundle .
shasum -a 256 p39_release_*.tgz > p39_release_checksums.txt

echo "Bundle done:"
tar -tzf "p39_release_${DATE}.tgz" | egrep "manifest_(equity_fixed100k|proxy)\.json|scoreboard_(equity_fixed100k|proxy)\.csv|risk_limits\.json|_test_trades\.csv|_trials_metrics\.csv"
