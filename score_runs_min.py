#!/usr/bin/env python3
import argparse, glob, sys, warnings, json, hashlib, platform
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Silence the "Period drops timezone" warning
warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information."
)

try:
    from tabulate import tabulate
    HAVE_TABULATE = True
except Exception:
    HAVE_TABULATE = False


# ----------------------------- helpers ----------------------------- #

def discover_run_dirs(inputs):
    found, seen = [], set()
    for pat in inputs:
        for hit in glob.glob(pat, recursive=True):
            p = Path(hit)
            if p.is_file() and p.name == "trials_metrics.csv":
                d = p.parent
                if d not in seen:
                    found.append(d); seen.add(d)
            elif p.is_dir():
                for csv in p.rglob("trials_metrics.csv"):
                    d = csv.parent
                    if d not in seen:
                        found.append(d); seen.add(d)
    return found


def _read_metrics_csv(run_dir: Path):
    f = run_dir / "trials_metrics.csv"
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f)
        df["__dir"] = str(run_dir)
        return df
    except Exception:
        return None


def _pick_cols(df):
    ret = "ret_pct_true" if "ret_pct_true" in df.columns else ("ret_pct" if "ret_pct" in df.columns else None)
    dd  = "max_dd_pct_true" if "max_dd_pct_true" in df.columns else ("max_dd_pct" if "max_dd_pct" in df.columns else None)
    return ret, dd


def _as_percent(x):
    if pd.isna(x): return np.nan
    x = float(x)
    # If value looks like a fraction (<=1.5 in magnitude), treat as fraction and convert to %
    return x*100.0 if abs(x) <= 1.5 else x


def _best_row(df: pd.DataFrame):
    ret_col, dd_col = _pick_cols(df)
    if ret_col is None or not df[ret_col].notna().any():
        return None
    idx = df[ret_col].astype(float).idxmax()
    row = df.loc[idx].to_dict()
    row["_ret"] = float(row[ret_col])
    row["_dd"]  = float(row.get(dd_col, np.nan)) if dd_col in df.columns else np.nan
    row["_trades"] = int(row.get("trades", 0)) if pd.notna(row.get("trades", np.nan)) else 0
    row["_dir"] = row.get("__dir", "")
    return row


def _find_ts_col(df: pd.DataFrame):
    hints = ["timestamp","ts","time","date","datetime","exit","close","open","dt","entry"]
    best, best_score = None, -1.0
    for c in df.columns:
        if any(h in c.lower() for h in hints):
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            score = s.notna().mean()
            if score > best_score:
                best, best_score = c, score
    return best


def _find_pnl_col(df: pd.DataFrame):
    cand = [c for c in df.columns if any(k in c.lower() for k in ["pnl","p&l","realized","net_pnl","profit","loss"])]
    return cand[0] if cand else None


def _geomean_from_returns(r: pd.Series):
    """Return exp(mean(log(1+r)))-1 with protection against r <= -1."""
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) == 0:
        return np.nan
    r = r.clip(lower=-0.999999)
    return float(np.expm1(np.log1p(r).mean()))


def _risk_from_monthlies(mon: pd.Series):
    """Compute risk stats from monthly returns series (fractions)."""
    mon = pd.to_numeric(mon, errors="coerce").dropna()
    if len(mon) == 0:
        return {
            "months": 0, "geo_monthly": np.nan, "cagr": np.nan,
            "sharpe_ann": np.nan, "sortino_ann": np.nan,
            "maxdd": np.nan, "mar": np.nan, "ulcer_index": np.nan
        }

    geo_m = _geomean_from_returns(mon)
    cagr = (1.0 + geo_m)**12 - 1.0

    # Sharpe (annualized, rf=0)
    sd = float(mon.std(ddof=1)) if len(mon) > 1 else float("nan")
    sharpe_ann = (mon.mean() / sd * np.sqrt(12.0)) if (sd and sd > 0) else np.nan

    # Sortino (annualized, rf=0)
    downside = mon.clip(upper=0.0)
    ddv = float(np.sqrt((downside**2).mean()))
    sortino_ann = (mon.mean() / ddv * np.sqrt(12.0)) if ddv > 0 else np.nan

    # MaxDD & Ulcer from NAV path (start at 1)
    nav = (1.0 + mon).cumprod()
    peak = nav.cummax()
    dd_series = nav / peak - 1.0
    maxdd = float(abs(dd_series.min()))
    ulcer_index = float(np.sqrt(np.mean((dd_series * 100.0)**2))) / 100.0  # back to fraction

    mar = (cagr / maxdd) if (maxdd and maxdd > 0) else np.nan

    return {
        "months": int(len(mon)), "geo_monthly": float(geo_m), "cagr": float(cagr),
        "sharpe_ann": float(sharpe_ann) if not pd.isna(sharpe_ann) else np.nan,
        "sortino_ann": float(sortino_ann) if not pd.isna(sortino_ann) else np.nan,
        "maxdd": float(maxdd) if not pd.isna(maxdd) else np.nan,
        "mar": float(mar) if not pd.isna(mar) else np.nan,
        "ulcer_index": float(ulcer_index) if not pd.isna(ulcer_index) else np.nan,
    }


def _compute_monthlies_from_trades(run_dir: Path, mode: str = "equity", equity0: float | None = None):
    tfile = run_dir / "test_trades.csv"
    if not tfile.exists():
        return None
    try:
        trades = pd.read_csv(tfile)
    except Exception:
        return None

    ts_col = _find_ts_col(trades)
    if ts_col is None:
        return None
    pnl_col = _find_pnl_col(trades)
    if pnl_col is None:
        return None

    # Parse timestamps and PnL
    trades[ts_col] = pd.to_datetime(trades[ts_col], errors="coerce", utc=True).dt.tz_convert(None)
    trades = trades.dropna(subset=[ts_col]).sort_values(ts_col).copy()
    trades["pnl"] = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0)
    trades["month"] = trades[ts_col].dt.to_period("M")

    if mode == "proxy":
        # Scale-free monthly fractions from raw PnL
        m = trades.groupby("month")["pnl"].sum()
        mon = (m / m.abs().sum()) if m.abs().sum() != 0 else m * 0
        risk = _risk_from_monthlies(mon)
        # Rough MDD proxy from normalized cumulative PnL
        cum = trades["pnl"].cumsum()
        if cum.abs().max() > 0:
            norm = (cum - cum.min()) / (max(cum.max() - cum.min(), 1e-12)) + 1.0  # make positive
            peak = -np.inf; mdd_curve = 0.0
            for v in norm:
                peak = max(peak, v)
                dd = (v / peak) - 1.0
                mdd_curve = min(mdd_curve, dd)
            mdd_curve = abs(mdd_curve)
        else:
            mdd_curve = np.nan

        return {
            "months": risk["months"],
            "geo_monthly": risk["geo_monthly"],
            "mdd": float(mdd_curve) if pd.notna(mdd_curve) else np.nan,
            "series": mon.to_timestamp(),
            "E0": None,
            "E0_origin": None,
            "mode": "proxy",
            "risk": risk,
        }

    # --- equity mode ---
    E0 = 100_000.0 if equity0 is None else float(equity0)

    # Only infer E0 from metrics if user did not supply --equity0
    if equity0 is None:
        mfile = run_dir / "trials_metrics.csv"
        if mfile.exists():
            try:
                mdf = pd.read_csv(mfile)
                for cand in ["ret_pct_true", "ret_pct"]:
                    if cand in mdf.columns and mdf[cand].notna().any():
                        rp = float(mdf[cand].max())
                        rp = rp if abs(rp) > 1.5 else rp * 100.0  # fraction -> %
                        if abs(rp) > 1e-12:
                            total_pnl = float(trades["pnl"].sum())
                            E0 = abs(total_pnl / (rp / 100.0))
                        break
            except Exception:
                pass

    trades["equity"] = E0 + trades["pnl"].cumsum()
    eom = trades.groupby("month")["equity"].last().astype(float)
    mon = eom.pct_change().dropna()
    risk = _risk_from_monthlies(mon)

    # MDD from equity path
    peak = -np.inf; mdd = 0.0
    for eq in trades["equity"]:
        peak = max(peak, eq)
        dd = (eq / peak) - 1.0
        mdd = min(mdd, dd)

    return {
        "months": risk["months"],
        "geo_monthly": risk["geo_monthly"],
        "mdd": float(abs(mdd)),
        "series": mon.to_timestamp(),
        "E0": float(E0),
        "E0_origin": "fixed" if equity0 is not None else "inferred",
        "mode": "equity",
        "risk": risk,
    }


def _sha256(fp: Path):
    try:
        h = hashlib.sha256()
        with fp.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# ----------------------------- main ----------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--dd-weight", type=float, default=10.0, help="Score = ret%% - dd_weight * |maxDD%%|")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--monthly-target", type=float, default=None, help="e.g. 0.10 for 10%%")
    ap.add_argument("--min-months", type=int, default=2)
    ap.add_argument("--winner-file", default="runs/winner_dir.txt")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--why", action="store_true", help="Print filter diagnostics")

    # New flags
    ap.add_argument("--monthly-mode", choices=["equity", "proxy"], default="equity",
                    help="How to compute monthly returns/mdd from trades")
    ap.add_argument("--equity0", type=float, default=None,
                    help="Override starting equity in equity-mode monthlies (disables E0 inference)")
    ap.add_argument("--dump-winner-monthlies", action="store_true",
                    help="Print the winner's month-by-month returns (respecting --monthly-mode)")

    # Report
    ap.add_argument("--report", action="store_true",
                    help="Write scoreboard.csv, winner_monthlies.csv and a release manifest JSON")
    ap.add_argument("--report-out", default=None,
                    help="Directory for report artifacts (default: reports/<timestamp>)")

    args = ap.parse_args()

    run_dirs = discover_run_dirs(args.input)
    if not run_dirs:
        print("No readable data found in inputs."); sys.exit(1)

    rows = []
    with_trade_logs = 0

    for rd in sorted(run_dirs):
        mdf = _read_metrics_csv(rd)
        if mdf is None or mdf.empty:
            continue
        best = _best_row(mdf)
        if not best:
            continue

        ret_pct = _as_percent(best["_ret"])
        dd_pct  = _as_percent(best["_dd"]) if pd.notna(best["_dd"]) else np.nan
        trades  = best["_trades"]

        monthly = _compute_monthlies_from_trades(rd, mode=args.monthly_mode, equity0=args.equity0)
        if monthly:
            with_trade_logs += 1
            geo_m = monthly["geo_monthly"] * 100.0 if pd.notna(monthly["geo_monthly"]) else np.nan
            mdd_m = monthly["mdd"] * 100.0 if pd.notna(monthly["mdd"]) else np.nan
            months= monthly["months"]
        else:
            geo_m = np.nan; mdd_m = np.nan; months = 0

        # drawdown to use for scoring: prefer maxDD% if present & nonzero else mdd_from_trades%
        dd_use = dd_pct if (pd.notna(dd_pct) and dd_pct != 0) else mdd_m

        rows.append({
            "dir": rd.name,
            "ret%": ret_pct,
            "maxDD%": dd_pct,
            "trades": trades,
            "geo_monthly%": geo_m,
            "months": months,
            "mdd_from_trades%": mdd_m,
            "dd_used%": abs(dd_use) if pd.notna(dd_use) else np.nan,
            "score": ret_pct - args.dd_weight * (abs(dd_use) if pd.notna(dd_use) else 0.0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No readable data found in inputs."); sys.exit(1)

    print(f"Found {len(run_dirs)} run dirs | scored rows: {len(df)} | with trade logs: {with_trade_logs}")

    # STRICT FILTERS
    if args.strict:
        if args.why:
            before = len(df)
            m1 = df["trades"] >= args.min_trades
            m2 = df["ret%"].between(-20000, 20000, inclusive="both")
            m3 = df["maxDD%"].abs().between(0, 200, inclusive="both") | df["maxDD%"].isna()
            print(
                f"[STRICT] start={before}  "
                f"drop_trades<{args.min_trades}: {np.count_nonzero(~m1)}  "
                f"drop_ret_bounds: {np.count_nonzero(~m2)}  "
                f"drop_dd_bounds: {np.count_nonzero(~m3)}"
            )
        df = df[
            (df["trades"] >= args.min_trades) &
            (df["ret%"].between(-20000, 20000, inclusive="both")) &
            (df["maxDD%"].abs().between(0, 200, inclusive="both") | df["maxDD%"].isna())
        ]

    # MONTHLY TARGET FILTER
    if args.monthly_target is not None:
        target_pct = args.monthly_target * 100.0
        if args.why:
            before = len(df)
            m_has = df["geo_monthly%"].notna()
            m_months = df["months"] >= args.min_months
            m_target = df["geo_monthly%"] >= target_pct
            print(
                f"[MONTHLY] start={before}  "
                f"no_tradelog: {np.count_nonzero(~m_has)}  "
                f"short_history<{args.min_months}m: {np.count_nonzero(m_has & ~m_months)}  "
                f"below_target<{target_pct:.2f}%: {np.count_nonzero(m_has & m_months & ~m_target)}"
            )
        df = df[(df["geo_monthly%"].notna()) & (df["months"] >= args.min_months) & (df["geo_monthly%"] >= target_pct)]

    if df.empty:
        print("All rows filtered out. Try removing --strict or --monthly-target.")
        sys.exit(2)

    df = df.sort_values("score", ascending=False)
    show_cols = ["dir","ret%","maxDD%","trades","geo_monthly%","months","mdd_from_trades%","dd_used%","score"]

    print("\nAll runs (sorted by score, desc)")
    if HAVE_TABULATE:
        print(tabulate(df[show_cols].head(args.topk), headers="keys", tablefmt="github", floatfmt=".4f"))
    else:
        print(df[show_cols].head(args.topk).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    winner = df.iloc[0]["dir"]
    if not args.no_write:
        Path(args.winner_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.winner_file).write_text(str(winner))
        print(f"\nWinner written to {args.winner_file} -> {winner}")
    else:
        print(f"\nWinner (not written): {winner}")

    # Optional: dump winner monthlies & E0 context
    if args.dump_winner_monthlies:
        try:
            wdir = next(p for p in run_dirs if p.name == winner)
        except StopIteration:
            wdir = None
        if wdir is not None:
            m = _compute_monthlies_from_trades(wdir, mode=args.monthly_mode, equity0=args.equity0)
            if m:
                print("\nWinner monthly series:")
                extra = ""
                if args.monthly_mode == "equity":
                    extra = f"  E0={m.get('E0'):.2f} ({m.get('E0_origin')})"
                print(f"months={m['months']}  geo_monthly={m['geo_monthly']*100:.2f}%  mode={m.get('mode')}{extra}")
                print(m["series"].to_frame("monthly_ret"))
        else:
            print("\n[WARN] Could not locate winner directory to dump monthlies.")

    # -------- report artifacts --------
    if args.report:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        outdir = Path(args.report_out) if args.report_out else Path("reports")/ts
        outdir.mkdir(parents=True, exist_ok=True)

        # 1) scoreboard
        scoreboard_path = outdir/"scoreboard.csv"
        df.to_csv(scoreboard_path, index=False)

        # 2) winner monthlies + risk
        try:
            wdir = next(p for p in run_dirs if p.name == winner)
        except StopIteration:
            wdir = None

        winner_monthlies_path = outdir/"winner_monthlies.csv"
        manifest = {
            "generated_utc": ts,
            "env": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "args": {k:v for k,v in vars(args).items()},
            "inputs": [str(p) for p in args.input],
            "counts": {
                "run_dirs_found": len(run_dirs),
                "rows_scored": int(len(df)),
                "with_trade_logs": int(df["geo_monthly%"].notna().sum()),
            },
            "winner": winner,
            "scoreboard_top": df[show_cols].head(args.topk).to_dict(orient="records"),
            "artifacts": {
                "scoreboard_csv": str(scoreboard_path),
                "winner_monthlies_csv": str(winner_monthlies_path),
                "winner_file": str(args.winner_file),
            },
        }

        if wdir is not None:
            m = _compute_monthlies_from_trades(wdir, mode=args.monthly_mode, equity0=args.equity0)
            if m:
                # write monthlies
                m["series"].rename("monthly_ret").to_csv(winner_monthlies_path, header=True)

                # file hashes
                tm = wdir/"trials_metrics.csv"
                tt = wdir/"test_trades.csv"
                manifest["winner_detail"] = {
                    "dir": str(wdir),
                    "E0": m.get("E0"),
                    "E0_origin": m.get("E0_origin"),
                    "monthly_mode": args.monthly_mode,
                    "months": m["risk"]["months"],
                    "geo_monthly": m["risk"]["geo_monthly"],
                    "cagr": m["risk"]["cagr"],
                    "sharpe_ann": m["risk"]["sharpe_ann"],
                    "sortino_ann": m["risk"]["sortino_ann"],
                    "maxdd_from_monthlies": m["risk"]["maxdd"],
                    "ulcer_index": m["risk"]["ulcer_index"],
                    "mdd_from_curve": m["mdd"],
                    "files": {
                        "trials_metrics.csv": {
                            "exists": tm.exists(),
                            "sha256": _sha256(tm) if tm.exists() else None,
                            "size": tm.stat().st_size if tm.exists() else None,
                        },
                        "test_trades.csv": {
                            "exists": tt.exists(),
                            "sha256": _sha256(tt) if tt.exists() else None,
                            "size": tt.stat().st_size if tt.exists() else None,
                        }
                    }
                }

        # 3) manifest
        manifest_path = outdir/"release_manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\nReport written to: {outdir}")
        print(f" - scoreboard: {scoreboard_path}")
        print(f" - winner monthlies: {winner_monthlies_path}")
        print(f" - manifest: {manifest_path}")


if __name__ == "__main__":
    main()
