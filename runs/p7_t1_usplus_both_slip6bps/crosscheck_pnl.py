#!/usr/bin/env python3
import pandas as pd, numpy as np, sys

def pick(df, names): 
    for n in names:
        if n in df.columns: return n
    return None

path = sys.argv[1]
df = pd.read_csv(path)

c_side = pick(df, ["side","direction"])
c_qty  = pick(df, ["qty","size","quantity"])
c_ep   = pick(df, ["entry_price","entry","open_price_in"])
c_xp   = pick(df, ["exit_price","exit","close_price_out"])
c_fee  = pick(df, ["fee_paid","fees","total_fee"])
c_pnl  = pick(df, ["net_pnl","pnl_usdc","pnl_usd","pnl_quote","pnl","gross_pnl"])

need = [c_side,c_qty,c_ep,c_xp,c_pnl]
assert all(need), f"Missing cols: {need}"

side_sign = np.where(df[c_side].astype(str).str.lower().str.startswith("s"), -1.0, 1.0)
qty = df[c_qty].astype(float).values
ep  = df[c_ep].astype(float).values
xp  = df[c_xp].astype(float).values
fee = df[c_fee].astype(float).values if c_fee else np.zeros(len(df))
pnl_engine = df[c_pnl].astype(float).values

pnl_calc = side_sign * (xp - ep) * qty - fee
diff = pnl_engine - pnl_calc
abs_err = np.abs(diff)
rel_err = abs_err / (np.maximum(1e-9, np.abs(pnl_calc)))

print(f"[CROSSCHECK] {path}")
print(f"  corr(sign(engine), sign(calc)) = {np.corrcoef(np.sign(pnl_engine), np.sign(pnl_calc))[0,1]:.4f}")
print(f"  median abs diff = {np.median(abs_err):.6g}")
print(f"  95% abs diff    = {np.quantile(abs_err,0.95):.6g}")
print(f"  median rel diff = {np.median(rel_err):.6g}")
print(f"  95% rel diff    = {np.quantile(rel_err,0.95):.6g}")

neg_frac_engine = np.mean(pnl_engine < 0)
neg_frac_calc   = np.mean(pnl_calc   < 0)
print(f"  negative pnl fraction: engine={neg_frac_engine:.3f} | calc={neg_frac_calc:.3f}")
