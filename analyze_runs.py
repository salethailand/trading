import json, sys, math
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 120)


def fmt_pct(x):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "nan"
        return f"{x:.2%}"
    except Exception:
        return str(x)


def _profit_factor(s: pd.Series) -> float:
    s = pd.Series(s, dtype="float64")
    wins = s[s > 0].sum()
    losses = -s[s < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def _streaks(win_bool: pd.Series | np.ndarray):
    """Return (max_consec_wins, max_consec_losses)."""
    arr = pd.Series(win_bool).astype(bool).to_numpy()
    max_w = max_l = cur_w = cur_l = 0
    for w in arr:
        if w:
            cur_w += 1
            max_w = max(max_w, cur_w)
            cur_l = 0
        else:
            cur_l += 1
            max_l = max(max_l, cur_l)
            cur_w = 0
    return max_w, max_l


def _equity_stats_from_trades(trades_df: pd.DataFrame, start_equity: float | None):
    if trades_df.empty:
        return None

    df = trades_df.copy()

    # Robust time sorting: coerce to datetime regardless of CSV weirdness
    for col in ("exit_time",):
        if col in df.columns:
            df[col] = pd.to_datetime(pd.Series(df[col]), errors="coerce", utc=True)
    if "exit_time" in df.columns:
        df = df.sort_values("exit_time", kind="stable")

    eq0 = float(start_equity) if (start_equity is not None and start_equity > 0) else 0.0
    pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    eq = eq0 + pnl.cumsum()

    # Peak, drawdown (absolute) and drawdown (%) done correctly
    peak = eq.cummax()
    dd = eq - peak
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = dd / peak.replace(0, np.nan)

    max_dd = float(dd.min()) if len(dd) else 0.0
    max_dd_pct = float(dd_pct.min()) if len(dd_pct) else float("nan")
    ret_pct = (float(eq.iloc[-1]) / eq0 - 1.0) if eq0 > 0 else float("nan")

    # MAR-like ratio
    mar = float(ret_pct / abs(max_dd_pct)) if (np.isfinite(ret_pct) and np.isfinite(max_dd_pct) and abs(max_dd_pct) > 1e-12) else float("nan")

    # CAGR: use first->last exit_time span (days>=1). Only when final equity positive.
    if "exit_time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["exit_time"]):
        t0, t1 = df["exit_time"].dropna().iloc[[0, -1]]
        days = max((t1 - t0).days, 1)
        if eq0 > 0 and eq.iloc[-1] > 0:
            cagr = float((eq.iloc[-1] / eq0) ** (365.0 / days) - 1.0)
        else:
            cagr = float("nan")
    else:
        cagr = float("nan")

    return {
        "final_equity": float(eq.iloc[-1]),
        "return_pct": ret_pct,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "mar_ratio": mar,
        "cagr": cagr,
    }


def _expectancy_stats(pnl):
    pnl = pd.to_numeric(pd.Series(pnl), errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    wr = float((pnl > 0).mean()) if len(pnl) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative or 0
    payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else (float("inf") if avg_win > 0 else 0.0)
    pf = _profit_factor(pnl)
    exp_per_trade = float(pnl.mean()) if len(pnl) else 0.0

    q = pnl.quantile([0.1, 0.25, 0.5, 0.75, 0.9]) if len(pnl) else pd.Series(dtype=float)
    def _q(p):
        try:
            v = float(q.loc[p])
            return v if np.isfinite(v) else float("nan")
        except Exception:
            return float("nan")

    return {
        "wr": wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff,
        "profit_factor": pf,
        "expectancy": exp_per_trade,
        "q10": _q(0.1),
        "q25": _q(0.25),
        "q50": _q(0.5),
        "q75": _q(0.75),
        "q90": _q(0.9),
    }


def _group_table(df: pd.DataFrame, by_col: str) -> pd.DataFrame:
    grp = df.groupby(by_col, dropna=False)["pnl"]
    # count via .size() (robust across pandas versions)
    count = grp.size().rename("size")
    wr = grp.apply(lambda s: (s > 0).mean()).rename("wr")
    avg = grp.mean().rename("avg")
    pf = grp.apply(_profit_factor).rename("pf")
    out = pd.concat([count, wr, avg, pf], axis=1)
    out = out.sort_values("size", ascending=False)
    return out


def summarize_run(run: Path):
    prof = run / "strategy_profile.json"
    trades_csv = run / "test_trades.csv"
    trials_csv = run / "trials_metrics.csv"
    name = run.name
    out = [f"\n=== {name} ==="]

    start_equity = None

    if prof.exists():
        with open(prof, "r") as f:
            js = json.load(f)
        tm = js.get("test_metrics", {}) or {}
        trm = js.get("train_metrics", {}) or {}
        sel = js.get("selection", {}) or {}
        params = js.get("params", {}) or {}
        start_equity = params.get("start_equity", None)

        out += [
            f"TEST: ret%={fmt_pct(tm.get('return_pct'))}, maxDD%={fmt_pct(tm.get('max_drawdown_pct'))}, "
            f"WR={fmt_pct(tm.get('win_rate'))}, trades={tm.get('trades')}",
            f"TRAIN: ret%={fmt_pct(trm.get('return_pct'))}, maxDD%={fmt_pct(trm.get('max_drawdown_pct'))}, "
            f"WR={fmt_pct(trm.get('win_rate'))}, trades={trm.get('trades')}",
            f"Selection: top_k={sel.get('top_k')} best_trial={sel.get('best_trial')} DD_cap={fmt_pct(sel.get('dd_cap_pct', 0))}",
        ]
        suspicious = (js.get("metrics_consistent", {}) or {}).get("suspicious_run", False)
        if suspicious:
            out += ["[!] Suspicious run flags were set — double-check data splits and slippage."]

        p = params
        out += [
            f"Entry strategy={p.get('strategy')}  tp_mult={p.get('tp_mult')}  tsl_mult={p.get('tsl_mult')}  time_stop={p.get('time_stop')}",
            f"Global bb_std_mult={p.get('bb_std_mult')}  rsi_required={p.get('rsi_required')}  use_adx={p.get('use_adx')}  use_atr={p.get('use_atr')}",
        ]
        reg = (js.get("regime", {}) or {}).get("overrides")
        if reg:
            for k, v in reg.items():
                ts = v.get("time_stop")
                tp = v.get("tp_mult")
                bb = v.get("bb_std_mult")
                out += [
                    f"  {k}: {v.get('strategy')} bb={bb} tp={tp} tsl={v.get('tsl_mult')} ts={ts} "
                    f"rsi_req={v.get('rsi_required')} adx={v.get('adx_thresh')} atr=[{v.get('atr_min_pct')},{v.get('atr_max_pct')}]"
                ]
    else:
        out += ["(no strategy_profile.json)"]

    # Trials quick glance (best objective vs. DD)
    if trials_csv.exists():
        try:
            tdf = pd.read_csv(trials_csv)
            ret_col = "ret_pct" if "ret_pct" in tdf.columns else ("ret_pct_true" if "ret_pct_true" in tdf.columns else None)
            dd_col = "max_dd_pct" if "max_dd_pct" in tdf.columns else ("max_dd_pct_true" if "max_dd_pct_true" in tdf.columns else None)
            cols = [c for c in (ret_col, dd_col, "trades") if c]
            if ret_col and dd_col and all(c in tdf.columns for c in cols):
                tt = tdf[cols].astype(float)
                best_idx = tt[ret_col].idxmax()
                out += [
                    f"Trials: best_ret%={fmt_pct(tt.loc[best_idx, ret_col])} "
                    f"at drawdown={fmt_pct(tt.loc[best_idx, dd_col])} "
                    f"trades={int(tt.loc[best_idx, 'trades'])}"
                ]
        except Exception:
            pass

    # Trade-level analyses
    if trades_csv.exists():
        df = pd.read_csv(trades_csv)

        # Coerce/standardize time columns (prevents: "cannot subtract DatetimeArray from ndarray")
        for col in ("entry_time", "exit_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(pd.Series(df[col]), errors="coerce", utc=True)

        if not df.empty:
            df = df.dropna(subset=["entry_time", "exit_time"]) if {"entry_time", "exit_time"}.issubset(df.columns) else df

            # Holding time
            if {"entry_time", "exit_time"}.issubset(df.columns) and \
               pd.api.types.is_datetime64_any_dtype(df["entry_time"]) and \
               pd.api.types.is_datetime64_any_dtype(df["exit_time"]):
                df["hold_min"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
            else:
                df["hold_min"] = np.nan

            # Ensure numeric pnl
            df["pnl"] = pd.to_numeric(df.get("pnl", 0.0), errors="coerce").fillna(0.0)

            # Headline derived stats
            wr = float((df["pnl"] > 0).mean()) if len(df) else 0.0
            exp_stats = _expectancy_stats(df["pnl"])
            max_w, max_l = _streaks(df["pnl"] > 0)
            eq_stats = _equity_stats_from_trades(df, start_equity)

            out += [
                f"Trades: {len(df)}  WR={fmt_pct(wr)}  Median hold (min)={np.nanmedian(df['hold_min']):.1f}",
                f"Expectancy per trade={exp_stats['expectancy']:.4f} | PF={exp_stats['profit_factor']:.3f} "
                f"| Payoff={exp_stats['payoff']:.3f} | AvgWin={exp_stats['avg_win']:.4f} | AvgLoss={exp_stats['avg_loss']:.4f}",
                f"Streaks: max consec wins={max_w} | max consec losses={max_l}",
                f"PnL quantiles: q10={exp_stats['q10']:.4f}, q25={exp_stats['q25']:.4f}, q50={exp_stats['q50']:.4f}, "
                f"q75={exp_stats['q75']:.4f}, q90={exp_stats['q90']:.4f}",
            ]
            if eq_stats is not None:
                mar_s = f"{eq_stats['mar_ratio']:.3f}" if (isinstance(eq_stats['mar_ratio'], (int, float)) and np.isfinite(eq_stats['mar_ratio'])) else "nan"
                cagr_s = fmt_pct(eq_stats["cagr"]) if isinstance(eq_stats["cagr"], (int, float)) else "nan"
                out += [
                    f"Equity (from trades): ret%={fmt_pct(eq_stats['return_pct'])} "
                    f"maxDD%={fmt_pct(eq_stats['max_drawdown_pct'])} "
                    f"MAR={mar_s} CAGR={cagr_s}"
                ]

            # Reason table
            if "reason" in df.columns:
                tab = _group_table(df, "reason")
                if "hold_min" in df.columns:
                    med_hold = df.groupby("reason")["hold_min"].median().rename("med_hold_min")
                    tab = tab.join(med_hold)
                out += ["By reason:", tab.to_string(float_format=lambda x: f"{x:.4f}")]

            # Side table
            if "side" in df.columns:
                tab = _group_table(df, "side")
                if "hold_min" in df.columns:
                    med_hold = df.groupby("side")["hold_min"].median().rename("med_hold_min")
                    tab = tab.join(med_hold)
                out += ["By side:", tab.to_string(float_format=lambda x: f"{x:.4f}")]

            # Hour-of-day (by entry)
            if "entry_time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["entry_time"]):
                df["entry_hour"] = df["entry_time"].dt.hour.astype("Int64")
                hod = _group_table(df, "entry_hour")
                out += ["By entry hour:", hod.to_string(float_format=lambda x: f"{x:.4f}")]

        else:
            out += ["(test_trades.csv is empty)"]
    else:
        out += ["(no test_trades.csv; rerun with --dump-trades test_trades.csv)"]

    print("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python analyze_runs.py <run_dir1> <run_dir2> ...")
        sys.exit(0)
    for p in sys.argv[1:]:
        summarize_run(Path(p))
