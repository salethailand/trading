#!/usr/bin/env python3
"""
True Walk-Forward Optimization with Ensemble and Parallel Execution
- Re-optimize hyperparameters on each train slice
- Pick top-N combos per window, ensemble on test slice
- Aggregate out-of-sample performance across windows
- Parallelized backtesting
- Window date logging
"""
import pandas as pd
import numpy as np
import itertools
import joblib
import json
import logging
from datetime import datetime
from joblib import Parallel, delayed

# --- configuration ---
FEE_DEFAULT = 0.001  # 0.1% per side
TOP_N = 3            # number of top combos to ensemble per window
N_JOBS = -1          # parallel jobs (-1 = all cores)

# --- logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# --- load and prepare data ---
raw = pd.read_csv('10k.csv', sep=r'[,\t]+', engine='python')
raw.columns = [c.strip().lower() for c in raw.columns]
raw['timestamp'] = pd.to_datetime(raw['timestamp'])
df = raw.set_index('timestamp')[['open','high','low','close','volume']].dropna()

# --- prepare indicators ---
def prepare_indicators(df):
    df = df.copy()
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    plus = df['high'].diff().where(df['high'].diff() > df['low'].diff().abs(), 0.0)
    minus = df['low'].diff().abs().where(df['low'].diff().abs() > df['high'].diff(), 0.0)
    dm_plus = plus.rolling(14).sum()
    dm_minus= minus.rolling(14).sum()
    df['adx14'] = ((dm_plus-dm_minus).abs()/(dm_plus+dm_minus)*100).rolling(14).mean()
    for p in [14, 28, 56]:
        d = df['close'].diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        rs= g / l
        df[f'rsi{p}'] = 100 - 100/(1+rs)
    for span in [9, 21]:
        df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    return df.dropna()

logging.info("Preparing indicators...")
df = prepare_indicators(df)

# --- performance metrics ---
def sharpe_ratio(returns, freq=252):
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    return (mu / sigma) * np.sqrt(freq) if sigma else np.nan

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return ((equity - peak) / peak).min()

# --- load config & model ---
with open('config.json') as f:
    cfg = json.load(f)
strat = cfg['strategy']
trade = cfg['trading']
try:
    ml_model = joblib.load(cfg['model']['path'])
except EOFError:
    logging.error(f"Failed to load ML model from {cfg['model']['path']}: file may be corrupted or incomplete.")
    raise SystemExit("Exiting due to model load error. Please check your model file and path.")
nfeat = ml_model.n_features_in_

# --- MACD cache ---
macd_cache = {}
def get_macd(series, fast, slow, signal):
    key = (fast, slow, signal)
    if key not in macd_cache:
        e_f = series.ewm(span=fast, adjust=False).mean()
        e_s = series.ewm(span=slow, adjust=False).mean()
        m   = e_f - e_s
        sig = m.ewm(span=signal, adjust=False).mean()
        macd_cache[key] = m - sig
    return macd_cache[key]

# --- slice backtest ---
def backtest_slice(df_slice, params):
    fee = params.get('fee', FEE_DEFAULT)
    bbm = params['bb_std_mult']
    rsi_p = params['rsi_period']; os_l = params['rsi_os']; ob_l = params['rsi_ob']
    use_rsi = params['use_rsi']; use_ema = params['use_ema']; use_macd = params['use_macd']
    vol_s = params['vol_spike']; imb = params['imb_thresh']
    tp = params['tp_mult']; tsl = params['tsl_mult']
    ml_w = params['ml_weight']; ml_t = params['ml_thresh']
    fast, slow, sig = params['macd_fast'], params['macd_slow'], params['macd_signal']
    macd_diff = get_macd(df_slice['close'], fast, slow, sig) if use_macd else None

    equity = [0.0]
    rets = []
    in_pos = False
    entry = 0.0
    maxp = 0.0

    for prev, cur in zip(df_slice.itertuples(), df_slice.iloc[1:].itertuples()):
        bb_h = prev.sma20 + bbm * prev.std20
        bb_l = prev.sma20 - bbm * prev.std20
        feats = []
        if use_rsi:
            feats.append(getattr(prev, f'rsi{rsi_p}'))
        if use_ema:
            feats.extend([prev.ema9, prev.ema21])
        if use_macd:
            feats.append(macd_diff.loc[prev.Index])
        feats.extend([
            prev.atr14,
            prev.adx14,
            prev.std20,
            prev.close - bb_l,
            prev.adx14 / prev.atr14
        ])
        vec = np.pad(np.array(feats), (0, nfeat - len(feats)), 'constant').reshape(1, -1)
        ml_conf = ml_model.predict_proba(vec)[0][1]

        cond_entry = (
            not in_pos and
            prev.close > bb_h and
            (not use_rsi or getattr(prev, f'rsi{rsi_p}') < os_l) and
            (prev.close - bb_l) >= vol_s * prev.std20 and
            (prev.adx14 / prev.atr14) >= imb and
            (ml_conf * ml_w) >= ml_t
        )
        if cond_entry:
            in_pos = True
            entry = cur.open
            maxp = entry
        elif in_pos:
            maxp = max(maxp, cur.high)
            cond_exit = (
                cur.close < prev.sma20 or
                cur.close >= entry * tp or
                (maxp - cur.close) / maxp >= tsl or
                (use_rsi and getattr(prev, f'rsi{rsi_p}') > ob_l)
            )
            if cond_exit:
                ret = (cur.open - entry) - (entry + cur.open) * fee
                rets.append(ret)
                equity.append(equity[-1] + ret)
                in_pos = False

    if not rets:
        return {'pnl': 0.0, 'sharpe': np.nan, 'mdd': 0.0, 'trades': 0, 'win_rate': 0.0}
    rets = np.array(rets)
    return {
        'pnl': rets.sum(),
        'sharpe': sharpe_ratio(rets),
        'mdd': max_drawdown(np.cumsum(rets)),
        'trades': len(rets),
        'win_rate': np.mean(rets > 0)
    }

# --- build hyperparameter combos ---
fee_val = trade.get('fee_rate', FEE_DEFAULT)
base = {
    'fee': [fee_val],
    'use_rsi': [True, False], 'rsi_period': [14, 28, 56], 'rsi_os': [20, 30], 'rsi_ob': [70, 80],
    'use_ema': [True, False],
    'bb_std_mult': [1.0, 1.5, 2.0],
    'use_macd': [True, False], 'macd_fast': [5, 8], 'macd_slow': [13, 21], 'macd_signal': [3, 5],
    'vol_spike': [strat['vol_spike_mult']], 'imb_thresh': [strat['imbalance_threshold']],
    'tp_mult': [trade['tp_mult']], 'tsl_mult': [trade['trailing_stop_mult']],
    'ml_weight': [strat['ml_weight']], 'ml_thresh': [strat['ml_threshold']]
}
combos = [dict(zip(base, vals)) for vals in itertools.product(*base.values())]

# --- define walk-forward windows ---
n = len(df)
train_n = int(0.5 * n)
test_n = int(0.1 * n)
step = test_n
indices = [(i, i+train_n, i+train_n, i+train_n+test_n) for i in range(0, n-(train_n+test_n), step)]

# --- perform walk-forward optimization ---
all_oos_metrics = []
selected_hyperparams = []
for w, (start, split, tst_start, end) in enumerate(indices, 1):
    train_range = (df.index[start], df.index[split-1])
    test_range = (df.index[tst_start], df.index[end-1])
    logging.info(f"Window {w}: TRAIN {train_range[0]}-{train_range[1]}, TEST {test_range[0]}-{test_range[1]}")

    train_results = Parallel(n_jobs=N_JOBS)(delayed(lambda c: {**c, **backtest_slice(df.iloc[start:split], c)})(c) for c in combos)
    df_trp = pd.DataFrame(train_results)
    top_combos = df_trp.sort_values(['sharpe', 'pnl'], ascending=False).head(TOP_N)

    metrics_list = []
    for idx, best in top_combos.iterrows():
        params = best.to_dict()
        metrics = backtest_slice(df.iloc[tst_start:end], params)
        metrics_list.append(metrics)
        logging.info(f"  Combo {idx+1}/{TOP_N}: OOS Sharpe={metrics['sharpe']:.3f}, PnL={metrics['pnl']:.2f}")

    avg = {k: np.nanmean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    all_oos_metrics.append(avg)
    logging.info(f"  Averaged OOS: {avg}")
    selected_hyperparams.append(top_combos.iloc[0].to_dict())

# --- aggregate overall out-of-sample ---
final = {k: np.nanmean([m[k] for m in all_oos_metrics]) for k in all_oos_metrics[0]}

# print aggregated performance
print('
=== Aggregate Out-of-Sample Performance Across All Windows ===')
print(final)

# report selected hyperparameter sets per window
print('
=== Selected Hyperparameter Sets Per Window ===')
for i, params in enumerate(selected_hyperparams, 1):
    hp = {k: params[k] for k in params if k not in ['pnl','sharpe','mdd','trades','win_rate']}
    print(f"Window {i}: {hp}")
