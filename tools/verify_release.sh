#!/usr/bin/env bash
set -euo pipefail
TAG=${1:?usage: $0 release-tag}
work=$(mktemp -d)
gh release download "$TAG" -D "$work" >/dev/null
( cd "$work" && sha256sum -c p39_release_checksums.txt )
found=0
for a in "$work"/p39_release_*.tgz; do
  tar -tzf "$a" | egrep \
'manifest_(equity_fixed100k|proxy)\.json|scoreboard_(equity_fixed100k|proxy)\.csv|risk_limits\.json|_test_trades\.csv|_trials_metrics\.csv' >/dev/null && found=1
done
[ $found -eq 1 ] || { echo "no matching archives found" >&2; exit 1; }
echo "OK: checksums and expected files present"
