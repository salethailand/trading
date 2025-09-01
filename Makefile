SHELL := /bin/bash
DATE := $(shell date +%Y%m%d)

.PHONY: equity proxy equity_fixed bundle risk tag

equity:
	python3 score_runs_min.py --input 'runs/p39_*' \
	  --strict --min-trades 5 --min-months 3 --monthly-target 0.10 \
	  --monthly-mode equity --dd-weight 20 --topk 25 \
	  --report --report-out reports/equity \
	  --winner-file runs/winner_dir_equity.txt --dump-winner-monthlies

proxy:
	python3 score_runs_min.py --input 'runs/p39_*' \
	  --strict --min-trades 5 --min-months 3 --monthly-target 0.10 \
	  --monthly-mode proxy --dd-weight 20 --topk 25 \
	  --report --report-out reports/proxy \
	  --winner-file runs/winner_dir_proxy.txt --dump-winner-monthlies

equity_fixed:
	python3 score_runs_min.py --input 'runs/p39_*' \
	  --strict --min-trades 5 --min-months 3 --monthly-target 0.10 \
	  --monthly-mode equity --equity0 100000 --dd-weight 20 --topk 25 \
	  --report --report-out reports/equity_fixed100k \
	  --winner-file runs/winner_dir_equity_fixed.txt --dump-winner-monthlies

bundle:
	mkdir -p release_bundle
	# equity fixed
	EQF_WIN=$$(cat runs/winner_dir_equity_fixed.txt); \
	cp runs/$$EQF_WIN/trials_metrics.csv release_bundle/$${EQF_WIN}_trials_metrics.csv; \
	cp runs/$$EQF_WIN/test_trades.csv    release_bundle/$${EQF_WIN}_test_trades.csv; \
	cp reports/equity_fixed100k/scoreboard.csv        release_bundle/scoreboard_equity_fixed100k.csv; \
	cp reports/equity_fixed100k/release_manifest.json release_bundle/manifest_equity_fixed100k.json; \
	# proxy
	PX_WIN=$$(cat runs/winner_dir_proxy.txt); \
	cp runs/$$PX_WIN/trials_metrics.csv release_bundle/$${PX_WIN}_trials_metrics.csv; \
	cp runs/$$PX_WIN/test_trades.csv    release_bundle/$${PX_WIN}_test_trades.csv; \
	cp reports/proxy/scoreboard.csv     release_bundle/scoreboard_proxy.csv; \
	cp reports/proxy/release_manifest.json release_bundle/manifest_proxy.json; \
	# docs
	cp reports/*/release_manifest.json release_bundle/manifest_equity.json || true; \
	cp reports/*/scoreboard.csv release_bundle/scoreboard_equity.csv || true; \
	tar -czf p39_release_$(DATE).tgz -C release_bundle .

risk:
	python3 - <<'PY'
import pandas as pd, numpy as np, pathlib, json
equity0=100000.0
runs=pathlib.Path("runs")
wfile=runs/"winner_dir_equity_fixed.txt" if (runs/"winner_dir_equity_fixed.txt").exists() else runs/"winner_dir_equity.txt"
w=wfile.read_text().strip()
wdir=(pathlib.Path(w) if pathlib.Path(w).exists() else runs/w)
t=pd.read_csv(wdir/"test_trades.csv")
h=["timestamp","ts","time","date","datetime","exit","open","close","entry"]
ts=max(t.columns,key=lambda c: pd.to_datetime(t[c],errors="coerce",utc=True).notna().mean() if any(x in c.lower() for x in h) else -1)
t[ts]=pd.to_datetime(t[ts],errors="coerce",utc=True).dt.tz_convert(None); t=t.dropna(subset=[ts]).sort_values(ts)
pc=next(c for c in t.columns if c.lower() in {"pnl","realized_pnl","net_pnl","profit","loss"} or "pnl" in c.lower())
t["pnl"]=pd.to_numeric(t[pc],errors="coerce").fillna(0.0); t["equity"]=equity0+t["pnl"].cumsum()
t["date"]=t[ts].dt.date; daily=t.groupby("date")["pnl"].sum().astype(float)
neg=daily[daily<0]; p10=float(neg.quantile(0.10)) if len(neg) else 0.0; p01=float(neg.quantile(0.01)) if len(neg) else 0.0; worst=float(neg.min()) if len(neg) else 0.0
peak=-np.inf; mdd=0.0
for eq in t["equity"]:
    peak=max(peak,eq); mdd=min(mdd,eq/peak-1.0)
mdd_pct=abs(mdd)
limits={"equity0":equity0,"stats_loss_only":{"p10_loss":round(p10,2),"p01_loss":round(p01,2),"worst_loss":round(worst,2),"backtest_mdd_pct":round(mdd_pct*100,2)},
        "limits":{"max_daily_loss_soft":float(round(abs(p10)*1.25,2)),"max_daily_loss_hard":float(round(max(abs(p01),abs(worst))*1.05,2)),"max_drawdown_kill_pct":float(round(mdd_pct*0.5*100,2)),
                  "throttle_on_soft_stop":True,"flatten_on_hard_stop":True,"disable_on_dd_kill":True}}
path=pathlib.Path("release_bundle/risk_limits.json"); path.write_text(json.dumps(limits,indent=2)); print("wrote",path)
PY

tag:
	git init
	git config user.name "Your Name"
	git config user.email "you@example.com"
	printf ".venv/\n__pycache__/\n*.pyc\nrelease_bundle/*.tgz\n" > .gitignore
	git add -A
	git commit -m "Release: p39 equity (E0=100k) + proxy reports; bundle prepared"
	git tag -a "release-$(DATE)-p39" -m "p39 release (equity fixed E0)"

.PHONY: bundle verify release
bundle:
	./bundle.sh

verify:
	@:[ -n "$$TAG" ] || (echo "usage: make verify TAG=release-YYYYMMDD-p39" && exit 1)
	./tools/verify_release.sh "$$TAG"

release:
	@:[ -n "$$TAG" ] || (echo "usage: make release TAG=release-YYYYMMDD-p39" && exit 1)
	@DATE=$$(echo "$$TAG" | sed -E 's/^release-([0-9]{8})-p39$$/\1/') && \
	 gh release create "$$TAG" "p39_release_$${DATE}.tgz" "p39_release_checksums.txt" \
	   -t "p39 Release" -F release_bundle/RELEASE_NOTES.md
