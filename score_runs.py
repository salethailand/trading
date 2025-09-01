#!/usr/bin/env python3
# score_runs.py
import json, sys, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Matplotlib is optional; plotting will be skipped if not available
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _maybe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def _session_from_params(params: dict):
    sess = params.get('include_hours', None)
    if not sess:
        return "[0, 24]"
    if isinstance(sess, str):
        # normalize "19,20,22" -> [19, 20, 22]
        try:
            hours = [int(s.strip()) for s in sess.split(',') if s.strip() != ""]
            return "[" + ", ".join(str(h) for h in hours) + "]"
        except Exception:
            return f"[{sess}]"
    if isinstance(sess, (list, tuple)):
        try:
            return "[" + ", ".join(str(int(h)) for h in sess) + "]"
        except Exception:
            return str(sess)
    return str(sess)

def _equity_from_trades(df: pd.DataFrame, start_equity: float):
    """Return equity series aligned to trade exits (sorted by exit_time)."""
    if df is None or df.empty:
        return None
    dfx = df.copy()
    if 'exit_time' in dfx.columns:
        dfx = dfx.sort_values('exit_time')
    # robust to column name variants
    pnl_col = 'pnl' if 'pnl' in dfx.columns else None
    if pnl_col is None:
        return None
    eq = start_equity + dfx[pnl_col].cumsum()
    eq.index = dfx['exit_time'] if 'exit_time' in dfx.columns else np.arange(len(eq))
    return eq

def _score_row(ret_frac, maxdd_frac, trades):
    """
    Simple scoring:
      score = return - 0.7*DD_penalty + tiny penalty for extremely low trade count
    Where ret_frac and maxdd_frac are already fractions (e.g., 0.0005 == 0.05%).
    """
    dd_pen = abs(min(0.0, float(maxdd_frac)))  # maxDD comes as <= 0
    base = float(ret_frac) - 0.7 * dd_pen

    # small penalty for tiny sample sizes so we don't 'win' with 0-2 trades
    penalty = 0.0
    try:
        t = int(trades)
        if t == 0:
            penalty = -3.0e-5
        elif t < 3:
            penalty = -2.9e-5
        elif t < 5:
            penalty = -2.7e-5
    except Exception:
        pass

    return base + penalty

def summarize_run(run_dir: Path):
    prof = run_dir / "strategy_profile.json"
    trades_csv = run_dir / "test_trades.csv"

    js = _read_json(prof)
    tm = js.get("test_metrics", {}) or {}
    params = js.get("params", {}) or {}

    # Prefer test_metrics if present; fall back to trades file if needed
    ret_frac = _maybe_float(tm.get("return_pct"), 0.0)          # already a fraction
    maxdd_frac = _maybe_float(tm.get("max_drawdown_pct"), 0.0)  # usually <= 0
    wr = _maybe_float(tm.get("win_rate"), np.nan)
    trades = int(tm.get("trades", 0) or 0)

    start_equity = _maybe_float(params.get("start_equity"), 20000.0)
    fee = _maybe_float(params.get("fee"), _maybe_float(js.get("exchange", {}).get("fee"), np.nan))
    slip = _maybe_float(params.get("fixed_slippage"), np.nan)
    fill = params.get("fill_timing", "next_open")
    prio = params.get("bar_priority", "stop_first")
    sess = _session_from_params(params)

    df = None
    pnl_abs = np.nan
    if trades_csv.exists():
        try:
            df = pd.read_csv(trades_csv, parse_dates=["entry_time","exit_time"])
            pnl_abs = float(df["pnl"].sum())
            if trades == 0:
                trades = len(df)
            if math.isnan(wr) and "pnl" in df.columns and len(df) > 0:
                wr = (df["pnl"] > 0).mean()
            # If return not in JSON, infer from trades
            if ret_frac == 0.0 and start_equity > 0:
                ret_frac = pnl_abs / start_equity
            # If maxDD not in JSON, estimate from equity path
            if (maxdd_frac == 0.0 or math.isnan(maxdd_frac)) and start_equity > 0:
                eq = _equity_from_trades(df, start_equity)
                if eq is not None and len(eq) > 0:
                    peak = eq.cummax()
                    dd = (eq / peak - 1.0).min()
                    maxdd_frac = float(dd)
        except Exception as e:
            warnings.warn(f"Failed reading {trades_csv}: {e}")

    score = _score_row(ret_frac, maxdd_frac, trades)

    row = dict(
        dir=str(run_dir),
        **{
            "ret%": ret_frac,
            "maxDD%": maxdd_frac,
            "pnl": pnl_abs,
            "trades": trades,
            "win%": wr,
            "fee": fee,
            "slip": slip,
            "fill": fill,
            "prio": prio,
            "sess": sess,
            "score": score,
        },
    )
    return row, df, start_equity

def maybe_plot_equity(run_dir: Path, df: pd.DataFrame, start_equity: float):
    if df is None or df.empty or not HAVE_MPL:
        return False
    try:
        eq = _equity_from_trades(df, start_equity)
        if eq is None or eq.empty:
            return False
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 3))
        plt.plot(eq.index, eq.values)
        plt.title(run_dir.name + " — equity (test trades)")
        plt.xlabel("exit_time")
        plt.ylabel("equity")
        plt.tight_layout()
        out = run_dir / "equity.png"
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")
        return True
    except Exception as e:
        warnings.warn(f"Plot failed for {run_dir}: {e}")
        return False

def main(argv):
    if len(argv) < 2:
        print("Usage: python score_runs.py <run_dir1> <run_dir2> ...")
        sys.exit(0)

    rows = []
    for p in argv[1:]:
        run_dir = Path(p)
        if not run_dir.exists():
            warnings.warn(f"Missing: {run_dir}")
            continue
        row, df, start_eq = summarize_run(run_dir)
        rows.append(row)
        maybe_plot_equity(run_dir, df, start_eq)

    if not rows:
        print("No valid run directories.")
        sys.exit(0)

    dfres = pd.DataFrame(rows)

    # Pretty print in the same column order you showed
    col_order = [
        "dir","ret%","maxDD%","pnl","trades","win%","fee","slip","fill","prio","sess","score"
    ]
    for c in col_order:
        if c not in dfres.columns:
            dfres[c] = np.nan
    dfres = dfres[col_order]

    # Sort best first by score
    dfres = dfres.sort_values("score", ascending=False)

    # Nicely formatted numeric columns
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dfres.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))

    # Winner file (under the common 'runs' root if present)
    winner_dir = Path(dfres.iloc[0]["dir"])
    runs_root = Path("runs")
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
        (runs_root / "winner_dir.txt").write_text(str(winner_dir))
        print(f"\nWinner written to runs/winner_dir.txt -> {winner_dir}")
    except Exception:
        # fallback next to winner
        (winner_dir / "winner_dir.txt").write_text(str(winner_dir))
        print(f"\nWinner written to {winner_dir / 'winner_dir.txt'} -> {winner_dir}")

if __name__ == "__main__":
    main(sys.argv)
