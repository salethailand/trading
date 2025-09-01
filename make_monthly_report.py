# make_monthly_report.py  (robust timestamp parsing)
import sys, glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_col(cols, cands):
    for c in cands:
        if c in cols: return c
    return None

def parse_datetime_any(series):
    """Return tz-naive pandas datetime (UTC dropped) from many formats."""
    s = series.copy()
    # Try numeric epoch first
    if np.issubdtype(s.dropna().infer_objects().dtype, np.number):
        m = s.dropna().astype("float").abs().max()
        # Heuristic for epoch unit
        # ~1e9=s, ~1e12=ms, ~1e15=us, ~1e18=ns
        if m >= 1e17:
            unit = "ns"
        elif m >= 1e14:
            unit = "us"
        elif m >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        dt = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        # drop timezone to get tz-naive (works fine for grouping/plots)
        return dt.dt.tz_convert("UTC").dt.tz_localize(None)
    # Otherwise treat as string/date-like
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    # If tz-aware, drop tz; if tz-naive, this is a no-op
    try:
        return dt.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        return dt  # already tz-naive

def load_trades(csv_path):
    df = pd.read_csv(csv_path)

    # Prefer exit time; fall back to entry/time/ts
    tcol = find_col(
        df.columns,
        ["exit_ts","exit_time","exit_at","exit",
         "entry_ts","entry_time",
         "ts","timestamp","time"]
    )
    if tcol is None:
        raise ValueError(f"No timestamp-like column found. Columns={list(df.columns)}")

    # PnL column (amount). If missing, fallback to pct (not ideal, but useful for visualization)
    pnlcol = find_col(df.columns, ["pnl","pnl_quote","pnl_realized","pnl_usd","pnl_$"])
    if pnlcol is None:
        pnlcol = find_col(df.columns, ["pnl_pct","ret","return","ret_pct"])
        if pnlcol is None:
            raise ValueError(f"No PnL column found. Columns={list(df.columns)}")

    df["dt"] = parse_datetime_any(df[tcol])
    # Drop rows we couldn't parse
    df = df[~df["dt"].isna()].copy()
    if df.empty:
        raise ValueError("All timestamps failed to parse")

    df["is_win"] = df[pnlcol] > 0
    df["is_loss"] = df[pnlcol] < 0
    return df, pnlcol

def monthly_summary(df, pnlcol):
    # Group by calendar month on the 'dt' column directly
    g = df.groupby(pd.Grouper(key="dt", freq="M"))
    out = pd.DataFrame({
        "trades": g.size(),
        "wins": g["is_win"].sum(),
        "losses": g["is_loss"].sum(),
        "win_rate": (g["is_win"].mean() * 100.0),
        "pnl_sum": g[pnlcol].sum(),
        "pnl_mean": g[pnlcol].mean(),
        "pnl_median": g[pnlcol].median(),
    }).fillna(0.0)
    out["win_rate"] = out["win_rate"].round(2)
    out["cum_pnl"] = out["pnl_sum"].cumsum()
    out.index.name = "month"
    return out

def main(patterns):
    run_rows = []
    combined = []
    outdir = Path("runs_summary")
    outdir.mkdir(exist_ok=True, parents=True)

    run_dirs = []
    for pat in patterns:
        run_dirs += [Path(p) for p in glob.glob(pat)]
    run_dirs = sorted([d for d in run_dirs if d.is_dir()])

    for rd in run_dirs:
        csv_path = rd / "test_trades.csv"
        if not csv_path.exists():
            continue
        try:
            df, pnlcol = load_trades(csv_path)
            m = monthly_summary(df, pnlcol)
        except Exception as e:
            print(f"[WARN] {rd}: {e}")
            continue

        # per-run CSV + charts
        m.to_csv(rd / "monthly_summary.csv", index=True)

        for col, title, fname in [
            ("pnl_sum", "Monthly Net PnL", "month_pnl.png"),
            ("win_rate", "Monthly Win Rate (%)", "month_winrate.png"),
            ("cum_pnl", "Cumulative PnL by Month", "month_cum_pnl.png"),
        ]:
            plt.figure(figsize=(9,4.5))
            (m[col].plot(kind="bar") if col!="win_rate" else m[col].plot(kind="line"))
            plt.title(f"{rd.name} — {title}")
            plt.xlabel("Month")
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(rd / fname, dpi=150)
            plt.close()

        tmp = m[["pnl_sum"]].rename(columns={"pnl_sum": rd.name})
        combined.append(tmp)

        run_rows.append({
            "run": rd.name,
            "months": len(m),
            "trades": int(m["trades"].sum()),
            "total_pnl": float(m["pnl_sum"].sum()),
            "avg_month_pnl": float(m["pnl_sum"].mean()),
            "avg_month_winrate_%": float(m["win_rate"].mean()),
        })

    if combined:
        panel = pd.concat(combined, axis=1).fillna(0.0).sort_index()
        panel.to_csv(outdir / "monthly_pnl_panel.csv")

        # monthly PnL comparison
        plt.figure(figsize=(10.5,5))
        panel.plot(ax=plt.gca())
        plt.title("Monthly Net PnL by Run")
        plt.xlabel("Month")
        plt.ylabel("PnL")
        plt.tight_layout()
        plt.savefig(outdir / "monthly_pnl_by_run.png", dpi=160)
        plt.close()

        # cumulative comparison
        cpanel = panel.cumsum()
        cpanel.to_csv(outdir / "monthly_cum_pnl_panel.csv")
        plt.figure(figsize=(10.5,5))
        cpanel.plot(ax=plt.gca())
        plt.title("Cumulative Monthly PnL by Run")
        plt.xlabel("Month")
        plt.ylabel("Cumulative PnL")
        plt.tight_layout()
        plt.savefig(outdir / "monthly_cum_pnl_by_run.png", dpi=160)
        plt.close()

        summary = pd.DataFrame(run_rows).sort_values("total_pnl", ascending=False)
        summary.to_csv(outdir / "monthly_summary_overview.csv", index=False)
        print(summary)

if __name__ == "__main__":
    pats = sys.argv[1:] or ["runs/p27_*", "runs/p28_*"]
    main(pats)
