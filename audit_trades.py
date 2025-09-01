#!/usr/bin/env python3
import argparse, json, os, sys
import pandas as pd
import numpy as np

def dd_stats(eq):
    eq = np.asarray(eq, dtype=float)
    peaks = np.maximum.accumulate(eq)
    dd = eq / peaks - 1.0
    return dd.min()

def load_trades(csv_path):
    df = pd.read_csv(csv_path)
    # Normalize common column names
    cols = {c.lower(): c for c in df.columns}
    # Heuristics for side sign
    if 'side' in cols:
        s = df[cols['side']].astype(str).str.lower()
        sign = np.where(s.str.startswith('s'), -1.0, 1.0)
    else:
        sign = np.ones(len(df))
    # Try to get absolute PnL if present
    pnl_abs = None
    for k in ['pnl', 'pnl_quote', 'pnl_usd', 'pnl_usdc', 'gross_pnl', 'net_pnl']:
        if k in cols:
            pnl_abs = df[cols[k]].astype(float).values
            break
    # Try to get pct return if present
    ret_pct = None
    for k in ['ret', 'return', 'ret_pct', 'pct_return']:
        if k in cols:
            ret_pct = df[cols[k]].astype(float).values
            # If looks like percent (e.g., 2.5 for 2.5%), convert to fraction
            if np.nanmedian(np.abs(ret_pct)) > 0.5:  # >50% median absolute? assume in %
                ret_pct = ret_pct / 100.0
            break

    # Fallback: compute pct from price/qty if we can
    if ret_pct is None and pnl_abs is None:
        price_in = None; price_out = None; qty = None
        for k in ['entry_price','price_in','in_price','open_price','entry']:
            if k in cols: price_in = df[cols[k]].astype(float).values; break
        for k in ['exit_price','price_out','out_price','close_price','exit']:
            if k in cols: price_out = df[cols[k]].astype(float).values; break
        for k in ['qty','quantity','size','base_qty','base_size']:
            if k in cols: qty = df[cols[k]].astype(float).values; break
        if price_in is not None and price_out is not None and qty is not None:
            pnl_abs = (price_out - price_in) * qty * sign

    return df, pnl_abs, ret_pct

def recompute_equity(csv_path, start_equity=20000.0, assume_abs_if_both=False):
    df, pnl_abs, ret_pct = load_trades(csv_path)

    results = []
    # Mode A: use absolute PnL if available
    if pnl_abs is not None:
        eq = [start_equity]
        for p in pnl_abs:
            eq.append(eq[-1] + float(p))
        eq = np.array(eq[1:], dtype=float)
        results.append(('ABS', eq))

    # Mode B: use percentage return if available (compound multiplicatively)
    if ret_pct is not None:
        eq = [start_equity]
        for r in ret_pct:
            eq.append(eq[-1] * (1.0 + float(r)))
        eq = np.array(eq[1:], dtype=float)
        results.append(('PCT', eq))

    if not results:
        raise RuntimeError("Could not infer PnL or returns from the CSV headers.")

    # If both available, prefer ABS unless user asks otherwise
    if len(results) == 2 and not assume_abs_if_both:
        # Pick the one with larger drawdown magnitude (more conservative)
        ddA = dd_stats(results[0][1])
        ddB = dd_stats(results[1][1])
        mode, eq = (results[0] if ddA < ddB else results[1])
    else:
        mode, eq = results[0]

    total_ret = (eq[-1] / start_equity - 1.0)
    max_dd = dd_stats(eq)
    print(f"[AUDIT] Mode={mode} | Trades={len(eq)} | TotalRet={total_ret*100:.2f}% | MaxDD={max_dd*100:.2f}%")
    return mode, eq, (total_ret, max_dd)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to test_trades.csv")
    ap.add_argument("--start-equity", type=float, default=20000.0)
    ap.add_argument("--prefer-abs", action="store_true",
                    help="If both absolute PnL and percent returns exist, prefer ABS.")
    args = ap.parse_args()
    recompute_equity(args.csv, start_equity=args.start_equity, assume_abs_if_both=args.prefer_abs)
