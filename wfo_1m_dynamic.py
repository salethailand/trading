#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file 1-Minute/5-Minute Walk-Forward Optimization (STRICT, Regime-Adaptive)

- Realistic fills: next-bar open fills; intrabar TP/SL via high/low; SMA/time exits
- Costs: fee + slippage (fixed or searched), dynamic slip components (ATR%, spread%)
- Optional volume participation cap (<= X% of prior-bar volume)
- Objectives: return/pnl/sharpe/sharpe_ann/sortino/calmar/blend; single or multi
- Indicators: ATR/ADX/RSI, EMA trend filter, Bollinger offset
- ML gating (optional via config) + probability temperature calibration
- Risk: equity-based sizing (risk-per-trade), optional prob-weighted sizing
- Guardrails: daily loss halt + cooldown, consecutive-loss lockout, session filter
- Expanding WFO (train→folded OOS) with embargo gap
- Artifacts: trials_metrics.csv, pareto_return_vs_drawdown.png, strategy_profile.json, best_params.pkl
- Optional: Monte-Carlo bootstrap on OOS trades; MLflow logging
- Regime-adaptive: bull/bear/sideways overrides (config or searched)

Changes in this build:
- Conservative tick rounding: long-entry->ceil, short-entry->floor; optional tick rounding on exits
- Optional stop-only extra slippage (--stop-extra-slip)
- Trailing fix: evaluate exits using the prior trail; update trail after exit checks
- FIX: CLI flags now override JSON for fee, fixed_slippage, and other exchange/regime knobs
- FIX: Data sanitation — UTC index, monotonic sort, duplicate-bar collapse (OHLCV), NaN-safe hour filter
- FIX: Replaced deprecated fillna(method="bfill") with .bfill()
"""

import sys, os, json, math, argparse, logging, statistics, ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_datetime64_any_dtype

# ------------------ Helper: detect if a CLI flag was passed -------------------
def _cli_passed(name: str) -> bool:
    """
    Return True if --<name> (or --<name>=...) is present in sys.argv.
    Example: _cli_passed("fixed-slippage")
    """
    flag = f"--{name}"
    for a in sys.argv[1:]:
        if a == flag or a.startswith(flag + "="):
            return True
    return False

# --------------------- Optional deps ---------------------
try:
    import optuna
    from optuna.samplers import TPESampler
except ModuleNotFoundError:
    print("Missing dependency: optuna", file=sys.stderr); sys.exit(1)
try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None
try:
    import joblib
except ModuleNotFoundError:
    joblib = None
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("wfo-1m-dynamic")

# =========================================================
#                     Data loading
# =========================================================

def _normalize_price_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapper = {}
    lower = {c.lower(): c for c in df.columns}
    for want, alts in {
        "open":  ["open", "o"],
        "high":  ["high", "h"],
        "low":   ["low", "l"],
        "close": ["close", "c", "price"],
        "volume":["volume","vol","base_volume","base volume","basevolume"],
        "ask":   ["ask"],
        "bid":   ["bid"],
        "spread":["spread"],
    }.items():
        for a in alts:
            if a in df.columns:
                mapper[a] = want
                break
            if a.lower() in lower:
                mapper[ lower[a.lower()] ] = want
                break
    dfx = df.rename(columns=mapper)
    return dfx

def _choose_ts_unit_from_numeric(arr: np.ndarray) -> str:
    # Avoid RuntimeWarning on all-NaN via finite-mask
    arr = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr)]
    med = float(np.median(valid)) if valid.size else 0.0
    # Unix seconds ~1e9, ms ~1e12
    return "ms" if med > 1e12 else "s"

def _ensure_timestamp_index(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
    ts_unit: Optional[str] = None,   # 's','ms','us','ns' or None to auto
) -> pd.DataFrame:
    """
    Make a UTC DatetimeIndex. If ts_col is provided, use it (case-insensitive).
    Otherwise try common names: timestamp/time/date/datetime. If already indexed
    by DatetimeIndex, normalize to UTC.
    """
    colmap = {c.lower(): c for c in df.columns}

    def _to_utc_idx(ts_like):
        if is_datetime64_any_dtype(ts_like):
            idx = pd.DatetimeIndex(ts_like)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        else:
            arr = pd.to_numeric(ts_like, errors="coerce")
            unit = ts_unit or _choose_ts_unit_from_numeric(arr)
            idx = pd.to_datetime(arr, unit=unit, utc=True)
        return idx

    # If user told us a column, honor it (case-insensitive)
    if ts_col:
        key = colmap.get(str(ts_col).lower())
        if key is None:
            raise ValueError(f"--ts-col '{ts_col}' not found in columns {list(df.columns)}")
        idx = _to_utc_idx(df[key])
        dfx = df.drop(columns=[key]).copy()
        dfx.index = idx
        dfx.index.name = "timestamp"
        return dfx

    # Else, try well-known names automatically (case-insensitive)
    for want in ("timestamp", "time", "date", "datetime"):
        key = colmap.get(want)
        if key is not None:
            idx = _to_utc_idx(df[key])
            dfx = df.drop(columns=[key]).copy()
            dfx.index = idx
            dfx.index.name = "timestamp"
            return dfx

    # Already indexed by time?
    if isinstance(df.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        dfi = df.copy()
        dfi.index = idx
        dfi.index.name = "timestamp"
        return dfi

    raise ValueError("No timestamp/Time/Timestamp column found.")

def _collapse_duplicate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate timestamp rows using OHLCV semantics (stable & pandas-2.x safe)."""
    if not df.index.has_duplicates:
        return df

    # Stable sort so 'first'/'last' respect original row order within each timestamp
    dfs = df.sort_index(kind="mergesort")

    # Build a simple column->func aggregation map
    agg = {}
    for c in dfs.columns:
        lc = c.lower()
        if lc == "open":
            agg[c] = "first"
        elif lc == "high":
            agg[c] = "max"
        elif lc == "low":
            agg[c] = "min"
        elif lc == "close":
            agg[c] = "last"
        elif lc == "volume":
            agg[c] = "sum"
        elif lc in ("ask", "bid"):
            agg[c] = "last"     # keep the latest quote
        elif lc == "spread":
            agg[c] = "median"   # fallback when no bid/ask; use 'last' if you prefer
        else:
            agg[c] = "last"     # conservative default

    out = dfs.groupby(level=0, sort=False).agg(agg).sort_index()

    # Prefer a consistent spread definition when bid/ask exist
    if "ask" in out.columns and "bid" in out.columns:
        out["spread"] = out["ask"] - out["bid"]

    out.index.name = "timestamp"
    return out



def load_market_file(path: str, ts_col: Optional[str] = None, ts_unit: Optional[str] = None) -> pd.DataFrame:
    # Load
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Normalize columns and ensure UTC DatetimeIndex
    df = _normalize_price_cols(df)
    df = _ensure_timestamp_index(df, ts_col=ts_col, ts_unit=ts_unit)

    # Coerce numerics
    for c in ["open","high","low","close","volume","ask","bid","spread"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop NaT and stable-sort by time
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.isna()].sort_index(kind="mergesort")

    # Collapse duplicate timestamps with OHLCV semantics
    if df.index.has_duplicates:
        df = _collapse_duplicate_bars(df)

    # Validate columns & drop rows with missing OHLCV
    need = {"open","high","low","close","volume"}
    if not need.issubset(df.columns):
        raise ValueError(f"File must contain columns at least {need}, got {list(df.columns)}")
    cols = [c for c in ["open","high","low","close","volume","ask","bid","spread"] if c in df.columns]
    df = df[cols].dropna(subset=["open","high","low","close","volume"])

    return df


# =========================================================
#                     Indicators
# =========================================================

def compute_atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def compute_adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    up = h.diff(); down = -l.diff()
    dm_pos = np.where((up > down) & (up > 0), up, 0.0)
    dm_neg = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    di_pos = 100 * pd.Series(dm_pos, index=h.index).rolling(n, min_periods=n).mean() / (atr + 1e-12)
    di_neg = 100 * pd.Series(dm_neg, index=h.index).rolling(n, min_periods=n).mean() / (atr + 1e-12)
    dx = ((di_pos - di_neg).abs() / (di_pos + di_neg + 1e-12)) * 100
    return dx.rolling(n, min_periods=n).mean()

def compute_rsi(close: pd.Series, periods: int) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(periods, min_periods=periods).mean()
    loss = (-d.clip(upper=0)).rolling(periods, min_periods=periods).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def prepare_indicators(df: pd.DataFrame, p_sma: int, p_std: int, p_atr: int, p_adx: int,
                       rsi_periods: List[int], ema_periods: List[int]) -> pd.DataFrame:
    dfx = df.copy()
    dfx['sma'] = dfx['close'].rolling(p_sma, min_periods=p_sma).mean()
    dfx['std'] = dfx['close'].rolling(p_std, min_periods=p_std).std()
    dfx['atr'] = compute_atr(dfx['high'], dfx['low'], dfx['close'], p_atr)
    dfx['adx'] = compute_adx(dfx['high'], dfx['low'], dfx['close'], p_adx)
    for rp in rsi_periods:
        dfx[f'rsi{rp}'] = compute_rsi(dfx['close'], rp)
    for ep in ema_periods:
        dfx[f'ema{ep}'] = dfx['close'].ewm(span=ep, adjust=False).mean()
    return dfx.dropna()

# =========================================================
#                     Modeling (optional)
# =========================================================

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith('.pkl'):
        if joblib is None:
            raise RuntimeError("joblib not installed")
        m = joblib.load(path)
        return m, int(getattr(m, 'n_features_in_', 0) or 0)
    if XGBClassifier is None:
        raise RuntimeError("xgboost not installed for non-pkl model")
    m = XGBClassifier(); m.load_model(path)
    try:
        n_features = int(m.get_booster().num_features())
    except Exception:
        n_features = 0
    return m, n_features

def predict_proba_safe(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if getattr(proba, 'ndim', 1) == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if getattr(proba, 'ndim', 1) == 1:
            return proba
    if hasattr(model, 'predict'):
        y = model.predict(X)
        if getattr(y, 'ndim', 1) == 1:
            if np.all((y >= 0.0) & (y <= 1.0)):
                return y
            return 1.0 / (1.0 + np.exp(-y))
        if getattr(y, 'ndim', 2) == 2 and y.shape[1] >= 2:
            return y[:, 1]
    return np.ones(len(X))

def _logit_arr(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1 - p))

def calibrate_proba_temperature(p: np.ndarray, temp: float) -> np.ndarray:
    t = float(temp)
    if t == 1.0:
        return np.asarray(p, dtype=float)
    return 1.0 / (1.0 + np.exp(-_logit_arr(np.asarray(p, dtype=float)) / max(t, 1e-6)))

def build_features(dfx: pd.DataFrame, bb_mult: float, rsi_key: str) -> np.ndarray:
    return np.column_stack([
        dfx[rsi_key].values,
        dfx['atr'].values,
        dfx['adx'].values,
        dfx['std'].values,
        dfx['close'].values - (dfx['sma'].values - bb_mult * dfx['std'].values),
        dfx['adx'].values / (dfx['atr'].values + 1e-12),
    ])

# =========================================================
#                     Regime detection
# =========================================================

def detect_regime(dfx: pd.DataFrame) -> str:
    sma50 = dfx['close'].rolling(50).mean().iloc[-1]
    sma200 = dfx['close'].rolling(200).mean().iloc[-1]
    if sma50 > sma200: return 'bull'
    if sma50 < sma200: return 'bear'
    return 'sideways'

def load_models_from_config(config_path: str) -> Tuple[Dict[str, object], int]:
    if not os.path.exists(config_path):
        return {}, 0
    with open(config_path) as f:
        cfg = json.load(f)
    mcfg = cfg.get('model', {})
    models = {}; nfeat = 0
    path = mcfg.get('path')
    if path:
        try:
            m, nf = load_model(path); models['default'] = m; nfeat = max(nfeat, nf)
        except Exception as e:
            log.warning("Failed to load default model: %s", e)
    for reg, p in (mcfg.get('paths_per_regime') or {}).items():
        if not p: continue
        try:
            m, nf = load_model(p); models[reg] = m; nfeat = max(nfeat, nf)
        except Exception as e:
            log.warning("Failed to load regime model '%s': %s", reg, e)
    return models, nfeat

# =========================================================
#                     Backtest engine
# =========================================================

def _dyn_slip(base_slip: float, atr_pct: float, spread_pct: float, k_atr: float, k_spread: float) -> float:
    slip = float(base_slip) + float(k_atr) * float(atr_pct) + float(k_spread) * float(spread_pct)
    if not np.isfinite(slip) or slip < 0:
        slip = float(base_slip)
    return slip

def run_backtest(dfx: pd.DataFrame, params: Dict, probs: Optional[np.ndarray],
                 trade_log: Optional[list] = None) -> np.ndarray:
    # --- Frictions ---
    fee = float(params.get('fee', 0.0))
    base_slip = float(params.get('slippage', 0.0))
    k_atr = float(params.get('slip_atr_k', 0.0))
    k_spread = float(params.get('slip_spread_k', 0.0))
    spread_hl_frac = float(params.get('spread_hl_frac', 0.05))

    # --- Globals / gates ---
    bbm = float(params.get('bb_std_mult', 2.0))
    rsi_os_base = int(params.get('rsi_os', 30))
    rsi_ob_base = int(params.get('rsi_ob', 70))
    vol_sp = float(params.get('vol_spike', 0.0))
    adx_thresh_base = float(params.get('adx_thresh', 0.0))
    atr_min_pct_base = float(params.get('atr_min_pct', 0.0))
    atr_max_pct_base = float(params.get('atr_max_pct', 1.0))
    tp_mult_base = float(params.get('tp_mult', 1.10))
    tsl_mult_base = float(params.get('tsl_mult', 0.0))
    strat_base = params.get('strategy', 'breakout')
    tstop_base = int(params.get('time_stop', 60))

    use_sma_base = bool(params.get('use_sma', False))
    use_atr_base = bool(params.get('use_atr', False))
    use_adx_base = bool(params.get('use_adx', False))
    use_ml  = bool(params.get('use_ml', False))
    ml_thresh = float(params.get('ml_thresh', 0.0))
    use_ema_base = bool(params.get('use_ema', False))
    use_spike_base = bool(params.get('use_spike', False))
    rsi_required_base = bool(params.get('rsi_required', True))

    # NEW: exit rounding & stop-only extra slippage
    round_exits_flag = bool(params.get('round_exits', True))
    stop_extra_slip = float(params.get('stop_extra_slip', 0.0))

    ema1 = int(params.get('ema1', 5))
    ema2 = int(params.get('ema2', 12))
    if ema2 <= ema1:
        ema2 = ema1 + 1

    rsi_key = params.get('rsi_key') or f"rsi{int(params.get('rsi_period', 14))}"

    fill_timing = params.get('fill_timing', 'next_open')
    bar_priority = params.get('bar_priority', 'stop_first')
    session_start = int(params.get('session_start', 0))
    session_end = int(params.get('session_end', 24))
    include_hours = params.get('include_hours_set')  # None or set[int]

    sides = params.get('sides', 'both')  # 'both'|'long'|'short'
    allow_long = (sides in ('both','long'))
    allow_short = (sides in ('both','short'))

    max_consec_losses = int(params.get('max_consec_losses', 0))
    lockout_bars = int(params.get('lockout_bars', 0))

    start_equity = float(params.get('start_equity', 0.0))
    risk_per = float(params.get('risk_per_trade', 0.0))
    risk_stop_atr_mult = float(params.get('risk_stop_atr_mult', 1.0))
    risk_min_stop_frac = float(params.get('risk_min_stop_frac', 0.001))
    max_notional_pct_equity = float(params.get('max_notional_pct_equity', 1.0))

    price_step = float(params.get('price_step', 0.0001))
    qty_step = float(params.get('qty_step', 0.0001))
    min_notional = float(params.get('min_notional', 5.0))
    participation_cap = float(params.get('participation_cap', 1.0))

    regime_adapt = bool(params.get('regime_adapt', False))
    regime_fast = int(params.get('regime_fast', 50))
    regime_slow = int(params.get('regime_slow', 200))
    regime_band = float(params.get('regime_band', 0.0))
    regime_overrides = params.get('regime_overrides') or {}

    op = dfx['open'].values
    hi = dfx['high'].values
    lo = dfx['low'].values
    cl = dfx['close'].values
    std = dfx['std'].values
    sma = dfx['sma'].values
    adx = dfx['adx'].values
    atr = dfx['atr'].values
    rsi = dfx[rsi_key].values
    ema_fast = dfx.get(f'ema{ema1}', pd.Series(index=dfx.index, data=np.nan)).values
    ema_slow = dfx.get(f'ema{ema2}', pd.Series(index=dfx.index, data=np.nan)).values
    vol = dfx.get('volume', pd.Series(index=dfx.index, data=np.nan)).values

    if 'spread' in dfx.columns:
        spread = dfx['spread'].values
    elif {'ask','bid'}.issubset(set(dfx.columns)):
        spread = (dfx['ask'] - dfx['bid']).values
    else:
        spread = (hi - lo) * spread_hl_frac

    if regime_adapt:
        reg_fast = pd.Series(cl, index=dfx.index).rolling(regime_fast, min_periods=regime_fast).mean().values
        reg_slow = pd.Series(cl, index=dfx.index).rolling(regime_slow, min_periods=regime_slow).mean().values
    else:
        reg_fast = reg_slow = None

    idx = pd.DatetimeIndex(dfx.index)
    days = idx.normalize().to_numpy()
    hours_arr = idx.hour.to_numpy()  # int64, robust after sanitation

    cur_day = days[0] if len(days)>0 else None
    day_start_equity = start_equity if start_equity>0 else 0.0
    day_pnl = 0.0; halt_today = False

    proba_temp = float(params.get('proba_temp', 1.0))
    if use_ml and probs is not None:
        probs_adj = calibrate_proba_temperature(np.asarray(probs, dtype=float), proba_temp)
    else:
        probs_adj = np.ones(len(cl), dtype=float)

    def hour_ok(i: int) -> bool:
        """Session + include-hours gate; NaN/bounds safe."""
        if i >= len(hours_arr):
            return True
        # Attempt fast path
        try:
            h = int(hours_arr[i])
        except Exception:
            # Fallback via timestamp
            try:
                h = int(pd.Timestamp(dfx.index[i]).hour)
            except Exception:
                return True
        if session_start <= session_end:
            sess = (session_start <= h < session_end)
        else:
            sess = (h >= session_start) or (h < session_end)
        return sess and ((include_hours is None) or (h in include_hours))

    def get_regime(i) -> str:
        if not regime_adapt:
            return 'sideways'
        k = i - 1
        if k < 0 or not np.isfinite(reg_fast[k]) or not np.isfinite(reg_slow[k]):
            return 'sideways'
        upper = reg_slow[k] * (1.0 + regime_band)
        lower = reg_slow[k] * (1.0 - regime_band)
        if reg_fast[k] > upper: return 'bull'
        if reg_fast[k] < lower: return 'bear'
        return 'sideways'

    def effective_params_for_regime(reg: str) -> Dict:
        eff = {
            'strategy': strat_base, 'bb_std_mult': bbm,
            'rsi_os': rsi_os_base, 'rsi_ob': rsi_ob_base,
            'adx_thresh': adx_thresh_base, 'atr_min_pct': atr_min_pct_base, 'atr_max_pct': atr_max_pct_base,
            'tp_mult': tp_mult_base, 'tsl_mult': tsl_mult_base, 'time_stop': tstop_base,
            'use_sma': use_sma_base, 'use_ema': use_ema_base, 'use_atr': use_atr_base,
            'use_adx': use_adx_base, 'use_spike': use_spike_base, 'rsi_required': rsi_required_base
        }
        if regime_adapt and isinstance(regime_overrides, dict):
            od = regime_overrides.get(reg)
            if isinstance(od, dict):
                for k, v in od.items():
                    if k in eff:
                        eff[k] = v
        return eff

    rets = []
    in_pos = False
    pos_side = 0    # +1 long, -1 short
    entry_price = 0.0
    entry_i = 0
    trail = 0.0
    qty = 0.0
    equity = start_equity if start_equity > 0 else None
    cooldown_left = 0
    consec_losses = 0
    lockout_left = 0
    pending_exit_i = -1
    pending_exit_side = 0

    exit_tp_mult = max(tp_mult_base, 1.000001)
    exit_tsl_mult = tsl_mult_base
    exit_time_stop = tstop_base
    exit_use_sma = use_sma_base
    exit_strat = strat_base

    # ---------------------------------- main loop ----------------------------------
    for i in range(1, len(cl)):
        if cur_day is not None and days[i] != cur_day:
            cur_day = days[i]
            day_start_equity = (equity if equity is not None else 0.0)
            day_pnl = 0.0; halt_today = False
            consec_losses = 0
            lockout_left = 0
            cooldown_left = 0

        # process deferred next-open exits
        if in_pos and pending_exit_i == i:
            prev_p = cl[i-1]
            atr_pct = atr[i-1] / (prev_p + 1e-12)
            spread_pct = spread[i-1] / (prev_p + 1e-12)
            slip = _dyn_slip(base_slip, atr_pct, spread_pct, k_atr, k_spread)

            # direction-aware exit slippage
            if pending_exit_side == 1:  # long exit (sell)
                exec_price = op[i] * (1 - slip)
                if round_exits_flag and price_step > 0:
                    exec_price = math.floor(exec_price / price_step) * price_step
                pnl_per_share = exec_price - entry_price - (entry_price + exec_price) * fee
            else:  # short exit (buy)
                exec_price = op[i] * (1 + slip)
                if round_exits_flag and price_step > 0:
                    exec_price = math.ceil(exec_price / price_step) * price_step
                pnl_per_share = (entry_price - exec_price) - (entry_price + exec_price) * fee

            trade_pnl = pnl_per_share * (qty if qty > 0 else 1.0)
            rets.append(trade_pnl)
            if trade_log is not None:
                trade_log.append({'entry_time': dfx.index[entry_i], 'exit_time': dfx.index[i],
                                  'side': 'long' if pending_exit_side==1 else 'short',
                                  'entry': entry_price, 'exit': exec_price,
                                  'qty': qty, 'pnl': trade_pnl, 'reason': 'time/sma-nextopen'})
            if equity is not None: equity += trade_pnl
            day_pnl += trade_pnl
            consec_losses = consec_losses + 1 if trade_pnl < 0 else 0
            if max_consec_losses > 0 and consec_losses >= max_consec_losses:
                lockout_left = max(lockout_left, lockout_bars)
            in_pos = False; qty = 0.0; pending_exit_i = -1; pending_exit_side = 0
            continue

        prev_p = cl[i-1]
        atr_pct = atr[i-1] / (prev_p + 1e-12)
        spread_pct = spread[i-1] / (prev_p + 1e-12)
        slip = _dyn_slip(base_slip, atr_pct, spread_pct, k_atr, k_spread)

        prob = probs_adj[i] if (use_ml and i < len(probs_adj)) else 1.0
        if not np.isfinite(prob): prob = 0.0

        can_enter = (not halt_today) and (cooldown_left <= 0) and (lockout_left <= 0) and hour_ok(i)

        # ENTRY
        if not in_pos:
            reg = get_regime(i)
            eff = effective_params_for_regime(reg)

            trend_ok_long = trend_ok_short = True
            if eff['use_ema']:
                trend_ok_long  = (ema_fast[i-1] > ema_slow[i-1])
                trend_ok_short = (ema_fast[i-1] < ema_slow[i-1])

            adx_ok = (adx[i-1] >= float(eff['adx_thresh'])) if eff['use_adx'] else True
            vol_ok = ((float(eff['atr_min_pct']) <= atr_pct <= float(eff['atr_max_pct']))) if eff['use_atr'] else True

            upper = sma[i-1] + float(eff['bb_std_mult']) * std[i-1]
            lower = sma[i-1] - float(eff['bb_std_mult']) * std[i-1]

            # core signals per side/strategy
            if eff['strategy'] == 'breakout':
                core_long  = (prev_p > upper); rsi_ok_long  = (rsi[i-1] > int(eff['rsi_ob']))
                core_short = (prev_p < lower); rsi_ok_short = (rsi[i-1] < int(eff['rsi_os']))
            else:  # mean_reversion
                core_long  = (prev_p < lower); rsi_ok_long  = (rsi[i-1] < int(eff['rsi_os']))
                core_short = (prev_p > upper); rsi_ok_short = (rsi[i-1] > int(eff['rsi_ob']))

            if not eff.get('rsi_required', True):
                rsi_ok_long = rsi_ok_short = True

            spike_ok = True
            if eff['use_spike'] and vol_sp > 0:
                band_ref_up = upper; band_ref_dn = lower
                spike_ok = (abs(prev_p - band_ref_up) >= (vol_sp * std[i-1])) or (abs(prev_p - band_ref_dn) >= (vol_sp * std[i-1]))

            enter_long  = (allow_long  and can_enter and (prob >= ml_thresh) and trend_ok_long  and adx_ok and vol_ok and spike_ok and core_long  and rsi_ok_long)
            enter_short = (allow_short and can_enter and (prob >= ml_thresh) and trend_ok_short and adx_ok and vol_ok and spike_ok and core_short and rsi_ok_short)

            # choose exactly one side if both true (prefer stronger distance from band)
            if enter_long or enter_short:
                choose_short = False
                if enter_long and enter_short:
                    # pick the one with larger standardized distance beyond band
                    d_long = (prev_p - upper) / (std[i-1] + 1e-12) if eff['strategy']=='breakout' else (lower - prev_p) / (std[i-1] + 1e-12)
                    d_short= (lower - prev_p) / (std[i-1] + 1e-12) if eff['strategy']=='breakout' else (prev_p - upper) / (std[i-1] + 1e-12)
                    choose_short = (d_short > d_long)
                elif enter_short:
                    choose_short = True

                # fill
                fill_price = op[i] if fill_timing == 'next_open' else cl[i]
                if choose_short:
                    # entering short = sell → adverse slip lowers price
                    entry_price = fill_price * (1 - slip)
                    # tick rounding (conservative): selling → floor
                    if price_step > 0:
                        entry_price = math.floor(entry_price / price_step) * price_step
                    pos_side = -1
                else:
                    # entering long = buy → adverse slip increases price
                    entry_price = fill_price * (1 + slip)
                    # tick rounding (conservative): buying → ceil
                    if price_step > 0:
                        entry_price = math.ceil(entry_price / price_step) * price_step
                    pos_side = +1

                tp_mult_eff = max(float(eff['tp_mult']), 1.000001)
                exit_tp_mult = tp_mult_eff
                exit_tsl_mult = float(eff['tsl_mult'])
                exit_time_stop = int(eff['time_stop'])
                exit_use_sma = bool(eff['use_sma'])
                exit_strat = eff['strategy']

                # stop sizing
                stop_frac_guess = (risk_stop_atr_mult * atr_pct)
                if exit_tsl_mult > 0:
                    stop_frac_guess = max(stop_frac_guess, float(exit_tsl_mult))
                if not np.isfinite(stop_frac_guess) or stop_frac_guess <= 0:
                    stop_frac_guess = float(params.get('risk_min_stop_frac', 0.001))
                stop_frac_guess = max(stop_frac_guess, float(params.get('risk_min_stop_frac', 0.001)))

                # prob-edge sizing
                edge = max(0.0, prob - params.get('ml_thresh', 0.0)) if use_ml else 1.0
                edge_size = min(float(params.get('edge_k', 1.0)) * edge,
                                float(params.get('edge_cap', 0.5))) if use_ml else 1.0
                if (equity is not None) and risk_per > 0 and stop_frac_guess > 0:
                    qty = (equity * risk_per * max(edge_size, 1e-6)) / (entry_price * stop_frac_guess)
                    if not np.isfinite(qty) or qty <= 0:
                        qty = 0.0
                else:
                    qty = 1.0

                # volume participation cap
                if np.isfinite(vol[i-1]) and participation_cap < 1.0:
                    qty_cap = max(0.0, participation_cap * float(vol[i-1]))
                    qty = min(qty, qty_cap)

                # notional cap
                if equity is not None and max_notional_pct_equity < 1.0:
                    max_notional = equity * max_notional_pct_equity
                    qty = min(qty, max_notional / max(entry_price, 1e-12))

                # round qty
                if qty_step > 0:
                    qty = math.floor(max(qty, 0.0) / qty_step) * qty_step

                if (qty <= 0) or (entry_price * qty < min_notional):
                    qty = 0.0; pos_side = 0
                else:
                    in_pos = True
                    entry_i = i
                    trail = hi[i] if pos_side==+1 else lo[i]

            if cooldown_left > 0: cooldown_left -= 1
            if lockout_left > 0: lockout_left -= 1
            continue

        # POSITION MGMT
        atr_now = atr[i-1]

        # --- use the existing trail to evaluate exits ---
        if pos_side == +1:
            tp_level = entry_price + exit_tp_mult * atr_now
            sl_level = (trail - exit_tsl_mult * atr_now) if exit_tsl_mult > 0 else None
            hit_sl = (sl_level is not None) and (lo[i] <= sl_level)
            hit_tp = (hi[i] >= tp_level)
        else:
            tp_level = entry_price - exit_tp_mult * atr_now
            sl_level = (trail + exit_tsl_mult * atr_now) if exit_tsl_mult > 0 else None
            hit_sl = (sl_level is not None) and (hi[i] >= sl_level)
            hit_tp = (lo[i] <= tp_level)

        exit_now = False
        exit_price = None
        reason = None
        if hit_sl and hit_tp:
            if bar_priority == 'tp_first':
                exit_price = tp_level; reason = 'tp'
            else:
                exit_price = sl_level; reason = 'sl'
            exit_now = True
        elif hit_sl:
            exit_price = sl_level; reason = 'sl'; exit_now = True
        elif hit_tp:
            exit_price = tp_level; reason = 'tp'; exit_now = True

        if exit_now:
            # direction-aware exit slippage (with optional stop-only extra)
            slip_use = slip + (stop_extra_slip if reason == 'sl' else 0.0)
            slip_use = max(slip_use, 0.0)

            if pos_side == +1:  # sell to exit long
                exec_price = exit_price * (1 - slip_use)
                if round_exits_flag and price_step > 0:
                    exec_price = math.floor(exec_price / price_step) * price_step
                pnl_per_share = exec_price - entry_price - (entry_price + exec_price) * fee
            else:               # buy to cover short
                exec_price = exit_price * (1 + slip_use)
                if round_exits_flag and price_step > 0:
                    exec_price = math.ceil(exec_price / price_step) * price_step
                pnl_per_share = (entry_price - exec_price) - (entry_price + exec_price) * fee

            trade_pnl = pnl_per_share * (qty if qty > 0 else 1.0)
            rets.append(trade_pnl)
            if trade_log is not None:
                trade_log.append({'entry_time': dfx.index[entry_i], 'exit_time': dfx.index[i],
                                  'side': 'long' if pos_side==1 else 'short',
                                  'entry': entry_price, 'exit': exec_price,
                                  'qty': qty, 'pnl': trade_pnl, 'reason': reason})
            if equity is not None: equity += trade_pnl
            day_pnl += trade_pnl
            consec_losses = consec_losses + 1 if trade_pnl < 0 else 0
            if trade_pnl < 0 and int(params.get('cooldown_bars',0)) > 0:
                cooldown_left = int(params.get('cooldown_bars',0))
            if max_consec_losses > 0 and consec_losses >= max_consec_losses:
                lockout_left = max(lockout_left, lockout_bars)
            in_pos = False; qty = 0.0; pos_side = 0
        else:
            # only update the trail after exit checks
            if pos_side == +1:
                trail = max(trail, hi[i])
            else:
                trail = min(trail, lo[i])

            t_exit = (i - entry_i) >= exit_time_stop
            sma_exit = ((cl[i] < sma[i-1]) if pos_side==+1 else (cl[i] > sma[i-1]))
            if t_exit or (exit_use_sma and sma_exit):
                if fill_timing == 'next_open' and (i+1) < len(cl):
                    pending_exit_i = i + 1
                    pending_exit_side = pos_side
                else:
                    if pos_side==+1:  # sell
                        exec_price = cl[i] * (1 - slip)
                        if round_exits_flag and price_step > 0:
                            exec_price = math.floor(exec_price / price_step) * price_step
                        pnl_per_share = exec_price - entry_price - (entry_price + exec_price) * fee
                    else:             # buy
                        exec_price = cl[i] * (1 + slip)
                        if round_exits_flag and price_step > 0:
                            exec_price = math.ceil(exec_price / price_step) * price_step
                        pnl_per_share = (entry_price - exec_price) - (entry_price + exec_price) * fee
                    trade_pnl = pnl_per_share * (qty if qty > 0 else 1.0)
                    rets.append(trade_pnl)
                    if trade_log is not None:
                        trade_log.append({'entry_time': dfx.index[entry_i], 'exit_time': dfx.index[i],
                                          'side': 'long' if pos_side==1 else 'short',
                                          'entry': entry_price, 'exit': exec_price,
                                          'qty': qty, 'pnl': trade_pnl, 'reason': 'time/sma'})
                    if equity is not None: equity += trade_pnl
                    day_pnl += trade_pnl
                    consec_losses = consec_losses + 1 if trade_pnl < 0 else 0
                    if trade_pnl < 0 and int(params.get('cooldown_bars',0)) > 0:
                        cooldown_left = int(params.get('cooldown_bars',0))
                    if max_consec_losses > 0 and consec_losses >= max_consec_losses:
                        lockout_left = max(lockout_left, lockout_bars)
                    in_pos = False; qty = 0.0; pos_side = 0

        if float(params.get('daily_loss_limit_pct', 0.0)) > 0 and day_start_equity > 0:
            if day_pnl <= -float(params.get('daily_loss_limit_pct',0.0)) * day_start_equity:
                halt_today = True

        if cooldown_left > 0: cooldown_left -= 1
        if lockout_left > 0: lockout_left -= 1

    # close last position at close
    if in_pos:
        prev_p = cl[-2] if len(cl)>1 else cl[-1]
        atr_pct = atr[-2] / (prev_p + 1e-12) if len(cl)>1 else 0.0
        spread_pct = spread[-2] / (prev_p + 1e-12) if len(cl)>1 else 0.0
        slip = _dyn_slip(base_slip, atr_pct, spread_pct, k_atr, k_spread)
        if pos_side==+1:  # sell
            exec_price = cl[-1] * (1 - slip)
            if round_exits_flag and price_step > 0:
                exec_price = math.floor(exec_price / price_step) * price_step
            pnl_per_share = exec_price - entry_price - (entry_price + exec_price) * fee
        else:             # buy
            exec_price = cl[-1] * (1 + slip)
            if round_exits_flag and price_step > 0:
                exec_price = math.ceil(exec_price / price_step) * price_step
            pnl_per_share = (entry_price - exec_price) - (entry_price + exec_price) * fee
        trade_pnl = pnl_per_share * (qty if qty > 0 else 1.0)
        rets.append(trade_pnl)
        if trade_log is not None:
            trade_log.append({'entry_time': dfx.index[entry_i], 'exit_time': dfx.index[-1],
                              'side': 'long' if pos_side==1 else 'short',
                              'entry': entry_price, 'exit': exec_price,
                              'qty': qty, 'pnl': trade_pnl, 'reason': 'eod'})

    return np.asarray(rets, dtype=float)

# =========================================================
#                     Metrics & WFO
# =========================================================

def compute_metrics(rets, start_equity: float = 0.0) -> dict:
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    trades = int(len(rets))
    total = float(np.nansum(rets)) if trades else 0.0
    wins = int(np.sum(rets > 0)) if trades else 0
    losses = trades - wins
    win_rate = float(wins) / trades if trades else 0.0
    avg_trade = total / trades if trades else 0.0
    if start_equity and start_equity > 0:
        eq = start_equity + np.cumsum(rets)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = float(np.min(dd)) if trades else 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            dd_pct = np.where(peak > 0, dd / peak, 0.0)
        max_dd_pct = float(np.min(dd_pct)) if trades else 0.0
        final_eq = float(eq[-1]) if trades else float(start_equity)
        return_pct = final_eq / start_equity - 1.0
        eq_before = start_equity + np.concatenate(([0.0], np.cumsum(rets[:-1]))) if trades else np.array([start_equity])
        with np.errstate(divide='ignore', invalid='ignore'):
            trade_pct = np.where(eq_before > 0, rets / eq_before[:len(rets)], 0.0)
        avg_trade_pct = float(np.nanmean(trade_pct)) if trades else 0.0
    else:
        eq = np.cumsum(rets); peak = np.maximum.accumulate(eq); dd = eq - peak
        max_dd = float(np.min(dd)) if trades else 0.0
        max_dd_pct = 0.0; final_eq = total; return_pct = 0.0; avg_trade_pct = 0.0
    return {
        'pnl': total, 'return_pct': return_pct, 'trades': trades, 'win_rate': win_rate,
        'avg_trade': avg_trade, 'avg_trade_pct': avg_trade_pct, 'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct, 'final_equity': final_eq, 'wins': wins, 'losses': losses
    }

def walk_forward_folds(df: pd.DataFrame, folds: int, train_frac: float, embargo_bars: int = 0):
    n = len(df)
    if folds <= 1:
        split = int(n * train_frac)
        tr_end = max(0, split - int(embargo_bars))
        te_start = min(n, split + int(embargo_bars))
        yield df.iloc[:tr_end], df.iloc[te_start:]
        return
    train_start = 0
    initial_train = int(n * train_frac)
    initial_train = max(100, initial_train)
    test_total = max(0, n - initial_train)
    fold_size = max(10, int(test_total / folds))
    start_test = initial_train
    while start_test < n:
        end_test = min(start_test + fold_size, n)
        if end_test - start_test < 10: break
        tr_end = max(train_start, start_test - int(embargo_bars))
        te_start = min(end_test, start_test + int(embargo_bars))
        if (tr_end - train_start) >= 100 and (end_test - te_start) >= 10:
            yield df.iloc[train_start:tr_end], df.iloc[te_start:end_test]
        start_test = end_test

# =========================================================
#                     Config helpers
# =========================================================

def load_exchange_from_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    ex = cfg.get("exchange", {}) or {}
    out = {}
    for k in ("name","symbol","price_step","qty_step","min_notional",
              "participation_cap","fee","fixed_slippage","slippage"):
        if ex.get(k) is not None:
            out[k] = ex[k]
    return out

def kucoin_symbol_filters(symbol: str) -> dict:
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen("https://api.kucoin.com/api/v2/symbols", timeout=10) as r:
            data = _json.load(r)
        row = next(x for x in data["data"] if x["symbol"] == symbol)
        return {
            "price_step": float(row["priceIncrement"]),
            "qty_step": float(row["baseIncrement"]),
            "min_notional": float(row["minFunds"]),
        }
    except Exception as e:
        log.warning("KuCoin symbol filter fetch failed: %s", e)
        return {}

def load_regime_from_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    reg = cfg.get("regime", {}) or {}
    out = {}
    for k in ("adapt","fast","slow","band","overrides"):
        if reg.get(k) is not None:
            out[k] = reg[k]
    return out

# =========================================================
#                     Objective builder
# =========================================================

def make_objective(df_train: pd.DataFrame, current_model_fn, nfeat: int, risk_defaults: dict,
                   objective: str, tdays_per_year: int, blend_metric: str, blend_lambda: float,
                   multi_objective: bool, min_trades: int, dd_cap_pct: float, *,
                   has_ml: bool, fee: float, discovery_minimal: bool,
                   fixed_slippage: Optional[float], min_slippage: float,
                   exec_defaults: dict):
    def objective_fn(trial: 'optuna.trial.Trial'):
        # core indicator lengths
        p_sma = trial.suggest_int('p_sma', 5, 60)
        p_std = trial.suggest_int('p_std', 5, 60)
        p_atr = trial.suggest_int('p_atr', 7, 30)
        p_adx = trial.suggest_int('p_adx', 7, 30)
        rsi_period = trial.suggest_int('rsi_period', 7, 84)
        ema1 = trial.suggest_int('ema1', 5, 30)
        ema2 = trial.suggest_int('ema2', 10, 60)
        if ema2 <= ema1: ema2 = ema1 + 1
        rsi_key = f'rsi{rsi_period}'

        use_ml_flag = trial.suggest_categorical('use_ml', [True, False]) if (has_ml and not discovery_minimal) else False
        ml_thresh = trial.suggest_float('ml_thresh', 0.0, 1.0) if (has_ml and not discovery_minimal) else 0.0

        use_spike_choices    = [False] if discovery_minimal else [False, True]
        rsi_required_choices = [False] if discovery_minimal else [True, False]
        use_ema_choices      = [False] if discovery_minimal else [True, False]
        use_atr_choices      = [False] if discovery_minimal else [True, False]
        use_adx_choices      = [False] if discovery_minimal else [True, False]
        use_sma_choices      = [False] if discovery_minimal else [True, False]

        slip_hi   = 5e-4
        spread_hi = 0.05
        bb_lo, bb_hi = (0.5, 3.6)

        params = {
            'fee': float(fee),
            'slippage': (float(fixed_slippage) if fixed_slippage is not None
                         else trial.suggest_float('slippage', float(min_slippage), slip_hi)),
            'slip_atr_k':     trial.suggest_float('slip_atr_k', 0.0, 0.25),
            'slip_spread_k':  trial.suggest_float('slip_spread_k', 0.0, 1.0),
            'spread_hl_frac': trial.suggest_float('spread_hl_frac', 0.0, spread_hi),

            'bb_std_mult': trial.suggest_float('bb_std_mult', bb_lo, bb_hi),
            'rsi_os': trial.suggest_int('rsi_os', 5, 55),
            'rsi_ob': trial.suggest_int('rsi_ob', 45, 95),
            'vol_spike': trial.suggest_float('vol_spike', 0.0, 0.30),

            'adx_thresh':  trial.suggest_float('adx_thresh', 5.0, 35.0),
            'atr_min_pct': trial.suggest_float('atr_min_pct', 0.0, 0.03),
            'atr_max_pct': trial.suggest_float('atr_max_pct', 0.01, 0.15),

            'tp_mult':   trial.suggest_float('tp_mult', 1.01, 3.0),
            'tsl_mult':  trial.suggest_float('tsl_mult', 0.0, 1.0),
            'strategy':  trial.suggest_categorical('strategy', ['breakout', 'mean_reversion']),
            'time_stop': trial.suggest_int('time_stop', 5, 300),

            'use_sma':      trial.suggest_categorical('use_sma', use_sma_choices),
            'use_atr':      trial.suggest_categorical('use_atr', use_atr_choices),
            'use_adx':      trial.suggest_categorical('use_adx', use_adx_choices),
            'use_ml':       use_ml_flag,
            'use_ema':      trial.suggest_categorical('use_ema', use_ema_choices),
            'use_spike':    trial.suggest_categorical('use_spike', use_spike_choices),
            'rsi_required': trial.suggest_categorical('rsi_required', rsi_required_choices),
            'ml_thresh':    ml_thresh,

            'edge_k':   trial.suggest_float('edge_k', 0.0, 2.0),
            'edge_cap': trial.suggest_float('edge_cap', 0.05, 1.0),

            'rsi_key': rsi_key, 'rsi_periods': [rsi_period], 'ema_periods': [ema1, ema2],
            'p_sma': p_sma, 'p_std': p_std, 'p_atr': p_atr, 'p_adx': p_adx, 'ema1': ema1, 'ema2': ema2,

            # exec/risk defaults (propagated)
            'start_equity':        float(risk_defaults.get('start_equity', 0.0)),
            'risk_per_trade':      float(risk_defaults.get('risk_per_trade', 0.0)),
            'risk_stop_atr_mult':  float(risk_defaults.get('risk_stop_atr_mult', 1.0)),
            'risk_min_stop_frac':  float(risk_defaults.get('risk_min_stop_frac', 0.001)),
            'daily_loss_limit_pct':float(risk_defaults.get('daily_loss_limit_pct', 0.0)),
            'cooldown_bars':       int(risk_defaults.get('cooldown_bars', 0)),

            'fill_timing':    exec_defaults.get('fill_timing', 'next_open'),
            'bar_priority':   exec_defaults.get('bar_priority', 'stop_first'),
            'session_start':  int(exec_defaults.get('session_start', 0)),
            'session_end':    int(exec_defaults.get('session_end', 24)),
            'include_hours_set': exec_defaults.get('include_hours_set'),
            'max_consec_losses': int(exec_defaults.get('max_consec_losses', 0)),
            'lockout_bars':   int(exec_defaults.get('lockout_bars', 0)),
            'proba_temp':     float(exec_defaults.get('proba_temp', 1.0)),
            'participation_cap': float(exec_defaults.get('participation_cap', 1.0)),
            'price_step': float(exec_defaults.get('price_step', 0.0001)),
            'qty_step': float(exec_defaults.get('qty_step', 0.0001)),
            'min_notional': float(exec_defaults.get('min_notional', 5.0)),
            'max_notional_pct_equity': float(exec_defaults.get('max_notional_pct_equity', 1.0)),
            'sides': exec_defaults.get('sides', 'both'),

            # NEW: exit rounding & stop extra slip propagated from exec_defaults
            'round_exits': bool(exec_defaults.get('round_exits', True)),
            'stop_extra_slip': float(exec_defaults.get('stop_extra_slip', 0.0)),
        }

        # Regime setup
        regime_adapt_flag = bool(exec_defaults.get('regime_adapt', False))
        cfg_overrides = exec_defaults.get('regime_overrides')
        regime_search = bool(exec_defaults.get('regime_search', False))

        if regime_adapt_flag:
            params['regime_adapt'] = True
            params['regime_fast']  = int(exec_defaults.get('regime_fast', 50))
            params['regime_slow']  = int(exec_defaults.get('regime_slow', 200))
            params['regime_band']  = float(exec_defaults.get('regime_band', 0.0))
            if cfg_overrides and not regime_search:
                params['regime_overrides'] = cfg_overrides
            else:
                regs = ['bull','bear','sideways']
                reg_overrides = {}
                for r in regs:
                    reg_overrides[r] = {
                        'strategy':  trial.suggest_categorical(f'{r}_strategy', ['breakout','mean_reversion']),
                        'bb_std_mult': trial.suggest_float(f'{r}_bb_std_mult', 0.5, 3.6),
                        'rsi_os':    trial.suggest_int(f'{r}_rsi_os', 5, 55),
                        'rsi_ob':    trial.suggest_int(f'{r}_rsi_ob', 45, 95),
                        'adx_thresh': trial.suggest_float(f'{r}_adx_thresh', 5.0, 35.0),
                        'atr_min_pct': trial.suggest_float(f'{r}_atr_min_pct', 0.0, 0.03),
                        'atr_max_pct': trial.suggest_float(f'{r}_atr_max_pct', 0.01, 0.15),
                        'tp_mult':   trial.suggest_float(f'{r}_tp_mult', 1.01, 3.0),
                        'tsl_mult':  trial.suggest_float(f'{r}_tsl_mult', 0.0, 1.0),
                        'time_stop': trial.suggest_int(f'{r}_time_stop', 5, 300),
                        'use_sma':   trial.suggest_categorical(f'{r}_use_sma', [True, False]),
                        'use_ema':   trial.suggest_categorical(f'{r}_use_ema', [True, False]),
                        'use_atr':   trial.suggest_categorical(f'{r}_use_atr', [True, False]),
                        'use_adx':   trial.suggest_categorical(f'{r}_use_adx', [True, False]),
                        'use_spike': trial.suggest_categorical(f'{r}_use_spike', [True, False]),
                        'rsi_required': trial.suggest_categorical(f'{r}_rsi_required', [True, False]),
                    }
                if cfg_overrides:
                    for r, od in cfg_overrides.items():
                        if r in reg_overrides and isinstance(od, dict):
                            reg_overrides[r].update(od)
                        else:
                            reg_overrides[r] = od
                params['regime_overrides'] = reg_overrides
        else:
            params['regime_adapt'] = False

        # quick invalid prune
        if params['strategy'] == 'mean_reversion' and not (params['rsi_os'] < 50 < params['rsi_ob']):
            return (-1e9, 1e9) if multi_objective else -1e9
        if params['atr_min_pct'] > params['atr_max_pct']:
            return (-1e9, 1e9) if multi_objective else -1e9

        dfi = prepare_indicators(df_train, p_sma, p_std, p_atr, p_adx, params['rsi_periods'], params['ema_periods'])
        if params['use_ml']:
            X = build_features(dfi, params['bb_std_mult'], params['rsi_key'])
            if nfeat > 0:
                if X.shape[1] < nfeat:
                    X = np.pad(X, ((0,0),(0, nfeat - X.shape[1])), mode='constant')
                elif X.shape[1] > nfeat:
                    X = X[:, :nfeat]
            model = current_model_fn(dfi)
            probs = predict_proba_safe(model, X)
        else:
            probs = np.ones(len(dfi), dtype=float)

        rets = run_backtest(dfi, params, probs)

        eq0 = float(risk_defaults.get('start_equity', 0.0)) if risk_defaults else 0.0
        trades = int(len(rets))
        if eq0 > 0 and trades > 0:
            equity_curve = eq0 + np.cumsum(rets)
            ret_pct = float(equity_curve[-1] / eq0 - 1.0)
            peak = np.maximum.accumulate(equity_curve)
            with np.errstate(divide='ignore', invalid='ignore'):
                dd_pct_series = np.where(peak > 0, (equity_curve - peak) / peak, 0.0)
            max_dd_pct = float(np.min(dd_pct_series))
        else:
            ret_pct = float(np.nansum(rets)) if trades > 0 else 0.0
            max_dd_pct = float('nan')

        trial.set_user_attr('ret_pct', ret_pct)
        trial.set_user_attr('max_dd_pct', max_dd_pct)
        trial.set_user_attr('trades', trades)

        if trades < int(min_trades):
            return (-1e9, 1e9) if multi_objective else -1e9
        if np.isfinite(max_dd_pct) and abs(max_dd_pct) > float(dd_cap_pct):
            return (-1e9, abs(max_dd_pct)) if multi_objective else -1e9

        if multi_objective:
            return ret_pct, abs(max_dd_pct) if np.isfinite(max_dd_pct) else 1e9
        if objective == 'ret':
            return ret_pct
        if objective == 'pnl':
            return float(np.nansum(rets))

        eq_before = eq0 + np.concatenate(([0.0], np.cumsum(rets[:-1]))) if trades > 0 else np.array([eq0])
        trade_pct = np.divide(rets, eq_before[:trades], out=np.zeros_like(rets), where=eq_before[:trades] > 0) if trades > 0 else np.array([0.0])
        mu = float(np.nanmean(trade_pct)); sigma = float(np.nanstd(trade_pct))
        if objective == 'sharpe':
            return (mu / (sigma + 1e-12)) if sigma > 0 else -1e-9
        if objective == 'sortino':
            downside = np.minimum(trade_pct, 0.0); dvar = float(np.nanmean(np.square(downside))); ddev = math.sqrt(max(dvar, 0.0))
            return (mu / (ddev + 1e-12)) if ddev > 0 else (-1e-9 if mu <= 0 else 1e3)
        if objective == 'calmar':
            denom = abs(max_dd_pct) if np.isfinite(max_dd_pct) else 0.0
            return (ret_pct/denom) if denom>1e-12 else -1e-9
        if objective == 'sharpe_ann':
            try:
                days = int(pd.Index(df_train.index).normalize().nunique())
            except Exception:
                days = max(1, trades)
            tpd = (trades / max(days, 1)) if trades > 0 else 0.0
            tpy = tpd * max(int(tdays_per_year), 1)
            ann = math.sqrt(max(tpy, 1e-12))
            return ((mu/(sigma+1e-12)) * ann) if sigma>0 else -1e-9
        if objective == 'blend':
            if blend_metric == 'sharpe':
                comp = (mu/(sigma+1e-12)) if sigma>0 else -1e-9
            elif blend_metric == 'sortino':
                downside = np.minimum(trade_pct, 0.0)
                dvar = float(np.nanmean(np.square(downside)))
                ddev = math.sqrt(max(dvar, 0.0))
                comp = (mu/(ddev+1e-12)) if ddev>0 else -1e-9
            else:
                try:
                    days = int(pd.Index(df_train.index).normalize().nunique())
                except Exception:
                    days = max(1, trades)
                tpd = (trades / max(days, 1)) if trades > 0 else 0.0
                tpy = tpd * max(int(tdays_per_year), 1)
                ann = math.sqrt(max(tpy, 1e-12))
                comp = ((mu/(sigma+1e-12)) * ann) if sigma>0 else -1e-9
            return ret_pct + float(blend_lambda)*comp

        return ret_pct
    return objective_fn

# =========================================================
#                     Selection helpers
# =========================================================

def _resolve_ret_dd(df: pd.DataFrame):
    """
    Returns (ret_col_name, dd_abs_series) supporting old and new schemas.
    - Old: 'ret_pct', 'max_dd_pct' (negative)
    - New: 'ret_pct_true', 'max_dd_pct_true' (negative), plus 'objective_value', 'objective_dd_abs'
    """
    # Return column
    if 'ret_pct' in df.columns:
        ret_col = 'ret_pct'
    elif 'ret_pct_true' in df.columns:
        ret_col = 'ret_pct_true'
    else:
        # fallback to the optimized objective value if nothing else present
        ret_col = 'objective_value'

    # Drawdown absolute magnitude
    if 'max_dd_pct' in df.columns:
        dd_abs = df['max_dd_pct'].astype(float).abs()
    elif 'max_dd_pct_true' in df.columns:
        dd_abs = df['max_dd_pct_true'].astype(float).abs()
    elif 'objective_dd_abs' in df.columns:
        dd_abs = df['objective_dd_abs'].astype(float)
    else:
        raise KeyError("No drawdown column found (need max_dd_pct[_true] or objective_dd_abs).")

    return ret_col, dd_abs

def select_best_trial(trials_df: pd.DataFrame, dd_cap_pct: float):
    df = trials_df.copy()
    ret_col, dd_abs = _resolve_ret_dd(df)
    df['abs_dd'] = dd_abs
    df_f = df[df['abs_dd'] <= float(dd_cap_pct)]
    if len(df_f) == 0:
        df_f = df
    idx = df_f[ret_col].astype(float).idxmax()
    return int(df_f.loc[idx, 'number'])

def _trial_by_num(study, num: int):
    for t in study.trials:
        if t.number == num:
            return t
    raise KeyError(num)

def select_top_k_params(study, trials_df: pd.DataFrame, dd_cap_pct: float, k: int, aggregate: str = 'median'):
    df = trials_df.copy()
    ret_col, dd_abs = _resolve_ret_dd(df)
    df['abs_dd'] = dd_abs
    cond = df['abs_dd'] <= float(dd_cap_pct)
    df = df[cond] if cond.any() else df
    df = df.sort_values(ret_col, ascending=False)
    top = df.head(max(1, int(k)))
    nums = top['number'].astype(int).tolist()
    if len(nums) == 0:
        raise RuntimeError('No trials to select from')
    if len(nums) == 1 or aggregate == 'best':
        t = _trial_by_num(study, nums[0])
        return nums[0], dict(t.params), nums
    params_list = [dict(_trial_by_num(study, n).params) for n in nums]
    keys = set().union(*[p.keys() for p in params_list])
    import numpy as _np, statistics as _stat
    agg = {}
    for key in keys:
        vals = [p[key] for p in params_list if key in p]
        if not vals: continue
        v0 = vals[0]
        if isinstance(v0, (bool, str)):
            try: agg[key] = _stat.mode(vals)
            except Exception: agg[key] = vals[0]
        elif isinstance(v0, int):
            agg[key] = int(_np.median(vals))
        elif isinstance(v0, float):
            agg[key] = float(_np.median(vals))
        else:
            agg[key] = vals[0]
    return nums[0], agg, nums

# =========================================================
#                     MC bootstrap
# =========================================================

def mc_bootstrap_trade_seq(rets: np.ndarray, start_equity: float, sims: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return None
    ret_pcts, dd_pcts = [], []
    for _ in range(int(sims)):
        sample = rng.choice(rets, size=len(rets), replace=True)
        m = compute_metrics(sample, start_equity=start_equity)
        ret_pcts.append(float(m['return_pct']))
        dd_pcts.append(abs(float(m['max_drawdown_pct'])))
    ret_pcts = np.asarray(ret_pcts); dd_pcts = np.asarray(dd_pcts)
    return {
        'ret_pct_p5': np.percentile(ret_pcts, 5),
        'ret_pct_p50': np.percentile(ret_pcts, 50),
        'ret_pct_p95': np.percentile(ret_pcts, 95),
        'dd_abs_pct_p5': np.percentile(dd_pcts, 5),
        'dd_abs_pct_p50': np.percentile(dd_pcts, 50),
        'dd_abs_pct_p95': np.percentile(dd_pcts, 95),
        'sims': int(sims),
    }

# =========================================================
#                     Main
# =========================================================

def parse_hours_list(s: Optional[str]) -> Optional[set]:
    if not s: return None
    try:
        # Allow commas and whitespace
        parts = [p for p in (x.strip() for x in s.replace(',', ' ').split()) if p]
        vals = [int(x) for x in parts]
        return set([h for h in vals if 0 <= h <= 23])
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="1m/5m walk-forward optimization (STRICT, Regime-Adaptive)")
    ap.add_argument('--data-file', default='10k.csv')
    ap.add_argument('--trials', type=int, default=1000)
    ap.add_argument('--n-jobs', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--train-frac', type=float, default=0.6)
    ap.add_argument('--config', default='config.mexc_1.json')
    ap.add_argument('--mlflow-experiment', default=None)
    ap.add_argument('--wfo-folds', type=int, default=3)
    ap.add_argument('--out-dir', default='artifacts')
    # timestamp handling
    ap.add_argument('--ts-col', default=None, help='Timestamp column name (case-insensitive). If omitted, auto-detect.')
    ap.add_argument('--ts-unit', default=None, choices=['s','ms','us','ns'],
                    help='Unit for numeric --ts-col. If omitted, auto-detect by magnitude.')
    # risk & frictions
    ap.add_argument('--start-equity', type=float, default=20000.0)
    ap.add_argument('--risk-per-trade', type=float, default=0.01)
    ap.add_argument('--risk-stop-atr-mult', type=float, default=1.0)
    ap.add_argument('--risk-min-stop-frac', type=float, default=0.001)
    ap.add_argument('--fee', type=float, default=0.001, help='Per-trade fee fraction on entry+exit combined.')
    ap.add_argument('--fixed-slippage', type=float, default=None, help='If set, use this slippage fraction and do not search over slippage.')
    ap.add_argument('--min-slippage', type=float, default=0.0, help='Lower bound for slippage search (ignored if fixed-slippage is set).')
    # realism extras
    ap.add_argument('--participation-cap', type=float, default=1.0, help='Max fraction of previous bar volume allowed on entry (e.g., 0.05 = 5%).')
    ap.add_argument('--max-notional-pct-equity', type=float, default=1.0, help='Max notional at entry as a fraction of equity (1.0 = 100%).')
    ap.add_argument('--embargo-bars', type=int, default=0, help='Gap (in bars) removed between train and test to reduce leakage.')
    ap.add_argument('--mc-sims', type=int, default=0, help='If >0, run Monte Carlo bootstrap sims on OOS trades and print percentiles.')
    ap.add_argument('--dump-trades', default=None, help='If set, dump OOS test trades to CSV at this path.')
    # objectives
    ap.add_argument('--objective', choices=['ret','pnl','sharpe','sharpe_ann','sortino','calmar','blend'], default='ret')
    ap.add_argument('--blend-metric', choices=['sharpe','sharpe_ann','sortino'], default='sharpe')
    ap.add_argument('--blend-lambda', type=float, default=0.5)
    ap.add_argument('--trading-days-per-year', type=int, default=252)
    mo = ap.add_mutually_exclusive_group()
    mo.add_argument('--multi-objective', dest='multi_objective', action='store_true')
    mo.add_argument('--single-objective', '--no-multi-objective', dest='multi_objective', action='store_false')
    ap.set_defaults(multi_objective=True)
    # guardrails
    ap.add_argument('--min-trades', type=int, default=75)
    ap.add_argument('--dd-cap-pct', type=float, default=0.20)
    ap.add_argument('--daily-loss-limit-pct', type=float, default=0.03)
    ap.add_argument('--cooldown-bars', type=int, default=20)
    ap.add_argument('--max-consec-losses', type=int, default=0)
    ap.add_argument('--lockout-bars', type=int, default=0)
    # execution realism
    ap.add_argument('--fill-timing', choices=['close','next_open'], default='next_open')
    ap.add_argument('--bar-priority', choices=['stop_first','tp_first'], default='stop_first')
    ap.add_argument('--session-start', type=int, default=0)
    ap.add_argument('--session-end', type=int, default=24)
    ap.add_argument('--include-hours', default=None, help='Comma/space-separated list of hours (0..23) allowed to enter.')
    ap.add_argument('--sides', choices=['both','long','short'], default='both')
    # NEW: exit rounding toggle (default ON) and stop-only extra slip
    ap.add_argument('--round-exits', dest='round_exits', action='store_true')
    ap.add_argument('--no-round-exits', dest='round_exits', action='store_false')
    ap.set_defaults(round_exits=True)
    ap.add_argument('--stop-extra-slip', type=float, default=0.0, help='Additional slippage applied only on stop exits (e.g., 0.0002 = 2 bps).')
    # ML & discovery
    ap.add_argument('--proba-temp', type=float, default=1.0)
    ap.add_argument('--discovery-minimal', action='store_true',
                    help='Turn off most gates during search to encourage entries; more realistic in eval.')
    # selection
    ap.add_argument('--select-top-k', type=int, default=1)
    ap.add_argument('--aggregate-params', choices=['median','best'], default='median')
    # exchange constraints
    ap.add_argument('--min-notional', type=float, default=5.0)
    ap.add_argument('--price-step', type=float, default=0.0001)
    ap.add_argument('--qty-step', type=float, default=0.0001)
    # regime adaptation
    ap.add_argument('--regime-adapt', action='store_true', help='Enable regime-adaptive overrides.')
    ap.add_argument('--regime-fast', type=int, default=50)
    ap.add_argument('--regime-slow', type=int, default=200)
    ap.add_argument('--regime-band', type=float, default=0.0)
    ap.add_argument('--regime-search', action='store_true', help='If set, Optuna will search per-regime overrides (config can still override fields).')

    # harmless flag to satisfy old runners that pass it
    ap.add_argument('--echo-params', action='store_true',
                    help='If set, echo selected params JSON to stdout near the end (no other behavior change).')

    args = ap.parse_args()

    # Load
    df_base = load_market_file(args.data_file, ts_col=args.ts_col, ts_unit=args.ts_unit)

    # ---------- Data sanity checks (warn-only) ----------
    def _data_sanity(df: pd.DataFrame):
        issues = []
        if not df.index.is_monotonic_increasing:
            issues.append("Index not monotonic increasing")
        dups = df.index.duplicated().sum()
        if dups:
            issues.append(f"Duplicate timestamps: {dups}")
        if (df[['open','high','low','close']] <= 0).any().any():
            issues.append("Non-positive prices present")
        try:
            pr = df['close']
            atr200 = compute_atr(df['high'], df['low'], df['close'], 200).bfill()
            jumps = ((pr.diff().abs()) > (20 * atr200)).sum()
            if jumps > 0: issues.append(f"Extreme jumps >20*ATR200: {int(jumps)}")
        except Exception:
            pass
        if issues:
            log.warning("DATA SANITY: " + " | ".join(issues))
    _data_sanity(df_base)

    # Split train/test with embargo
    n = len(df_base)
    split = int(n * float(args.train_frac))
    emb = max(0, int(args.embargo_bars))
    tr_end = max(0, split - emb)
    te_start = min(n, split + emb)
    df_train = df_base.iloc[:tr_end].copy()
    df_test  = df_base.iloc[te_start:].copy()
    if len(df_train) < 500 or len(df_test) < 200:
        log.warning("Very small train/test after embargo. Train=%d Test=%d", len(df_train), len(df_test))

    # Models
    models, nfeat = load_models_from_config(args.config)
    has_ml = bool(models) and (nfeat > 0)

    def current_model_fn(dfi: pd.DataFrame):
        if not models:
            class Dummy:
                def predict_proba(self, X):
                    return np.column_stack([1-np.ones(len(X)), np.ones(len(X))])
            return Dummy()
        regime = detect_regime(dfi)
        return models.get(regime, models.get('default', next(iter(models.values()))))

    # Defaults from config
    ex_cfg = load_exchange_from_config(args.config)
    reg_cfg = load_regime_from_config(args.config)

    risk_defaults = {
        'start_equity': args.start_equity,
        'risk_per_trade': args.risk_per_trade,
        'risk_stop_atr_mult': args.risk_stop_atr_mult,
        'risk_min_stop_frac': args.risk_min_stop_frac,
        'daily_loss_limit_pct': args.daily_loss_limit_pct,
        'cooldown_bars': args.cooldown_bars,
    }
    exec_defaults = {
        'fill_timing': args.fill_timing,
        'bar_priority': args.bar_priority,
        'session_start': args.session_start,
        'session_end': args.session_end,
        'include_hours_set': parse_hours_list(args.include_hours),
        'max_consec_losses': args.max_consec_losses,
        'lockout_bars': args.lockout_bars,
        'proba_temp': args.proba_temp,
        'participation_cap': float(args.participation_cap),
        'price_step': args.price_step,
        'qty_step': args.qty_step,
        'min_notional': args.min_notional,
        'max_notional_pct_equity': args.max_notional_pct_equity,
        'sides': args.sides,
        # NEW:
        'round_exits': bool(args.round_exits),
        'stop_extra_slip': float(args.stop_extra_slip),
        # regime defaults
        'regime_adapt': bool(args.regime_adapt),
        'regime_fast': int(args.regime_fast),
        'regime_slow': int(args.regime_slow),
        'regime_band': float(args.regime_band),
        'regime_search': bool(args.regime_search),
    }

    # ------------------------- Merge exchange config --------------------------
    # Honor CLI over JSON (CLI > JSON > defaults)
    if ex_cfg.get("participation_cap") is not None and not _cli_passed("participation-cap"):
        exec_defaults["participation_cap"] = float(ex_cfg["participation_cap"])
    if ex_cfg.get("price_step") is not None and not _cli_passed("price-step"):
        exec_defaults["price_step"] = float(ex_cfg["price_step"])
    if ex_cfg.get("qty_step") is not None and not _cli_passed("qty-step"):
        exec_defaults["qty_step"] = float(ex_cfg["qty_step"])
    if ex_cfg.get("min_notional") is not None and not _cli_passed("min-notional"):
        exec_defaults["min_notional"] = float(ex_cfg["min_notional"])

    if ex_cfg.get("fee") is not None and not _cli_passed("fee"):
        args.fee = float(ex_cfg["fee"])

    # accept either fixed_slippage or slippage from JSON as a fixed-slippage override
    if not _cli_passed("fixed-slippage"):
        if ex_cfg.get("fixed_slippage") is not None:
            args.fixed_slippage = float(ex_cfg["fixed_slippage"])
        elif ex_cfg.get("slippage") is not None and args.fixed_slippage is None:
            args.fixed_slippage = float(ex_cfg["slippage"])

    # Optional KuCoin symbol auto-fetch
    if (ex_cfg.get("name","").lower() == "kucoin") and ex_cfg.get("symbol"):
        need_steps = (
            exec_defaults["price_step"] in (None, 0) or
            exec_defaults["qty_step"] in (None, 0) or
            exec_defaults["min_notional"] in (None, 0)
        )
        if need_steps:
            kf = kucoin_symbol_filters(ex_cfg["symbol"])
            for k in ("price_step","qty_step","min_notional"):
                if kf.get(k) is not None:
                    # Only fill gaps (do not override CLI-provided values)
                    if k == "price_step" and not _cli_passed("price-step") and exec_defaults["price_step"] in (None, 0):
                        exec_defaults[k] = float(kf[k])
                    elif k == "qty_step" and not _cli_passed("qty-step") and exec_defaults["qty_step"] in (None, 0):
                        exec_defaults[k] = float(kf[k])
                    elif k == "min_notional" and not _cli_passed("min-notional") and exec_defaults["min_notional"] in (None, 0):
                        exec_defaults[k] = float(kf[k])

    # -------------------------- Merge regime config ---------------------------
    # Honor CLI over JSON (CLI > JSON > defaults)
    if reg_cfg.get("adapt") is not None and not _cli_passed("regime-adapt"):
        exec_defaults["regime_adapt"] = bool(reg_cfg["adapt"])
    if reg_cfg.get("fast") is not None and not _cli_passed("regime-fast"):
        exec_defaults["regime_fast"] = int(reg_cfg["fast"])
    if reg_cfg.get("slow") is not None and not _cli_passed("regime-slow"):
        exec_defaults["regime_slow"] = int(reg_cfg["slow"])
    if reg_cfg.get("band") is not None and not _cli_passed("regime-band"):
        exec_defaults["regime_band"] = float(reg_cfg["band"])
    if reg_cfg.get("overrides") is not None:
        exec_defaults["regime_overrides"] = reg_cfg["overrides"]

    # Objective
    objective_fn = make_objective(
        df_train,
        current_model_fn,
        nfeat,
        risk_defaults,
        args.objective,
        args.trading_days_per_year,
        args.blend_metric,
        args.blend_lambda,
        args.multi_objective,
        args.min_trades,
        args.dd_cap_pct,
        has_ml=has_ml,
        fee=args.fee,
        discovery_minimal=args.discovery_minimal,
        fixed_slippage=args.fixed_slippage,
        min_slippage=args.min_slippage,
        exec_defaults=exec_defaults,
    )

    # Study (seeded)
    sampler = TPESampler(seed=int(args.seed))
    if args.multi_objective:
        study = optuna.create_study(directions=['maximize','minimize'], sampler=sampler)
    else:
        study = optuna.create_study(direction='maximize', sampler=sampler)
    print(f"Starting optimization ({args.trials} trials, n_jobs={args.n_jobs}) | multi-objective={args.multi_objective}")
    study.optimize(objective_fn, n_trials=args.trials, n_jobs=args.n_jobs)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Save trials CSV (write BOTH legacy & new schema for compatibility)
    rows = []
    for t in study.trials:
        # objective value (what Optuna maximized)
        if args.multi_objective and t.values is not None:
            obj_value = t.values[0]
            obj_dd    = t.values[1]
        else:
            obj_value = (t.value if t.value is not None else t.user_attrs.get('ret_pct'))
            obj_dd    = abs(float(t.user_attrs.get('max_dd_pct') or np.nan))

        # true return% and true max_dd% saved as user attrs by objective_fn
        true_ret = t.user_attrs.get('ret_pct')
        true_dd  = t.user_attrs.get('max_dd_pct')

        rows.append({
            'number': t.number,
            # Legacy fields:
            'ret_pct': true_ret,
            'max_dd_pct': true_dd,
            # New fields:
            'objective_value': obj_value,
            'objective_dd_abs': obj_dd,
            'ret_pct_true': true_ret,
            'max_dd_pct_true': true_dd,
            'trades': t.user_attrs.get('trades'),
            'strategy': (t.params.get('strategy') if t.params else None),
            'state': str(t.state),
            'params': json.dumps(t.params, sort_keys=True, default=str),
        })
    trials_df = pd.DataFrame(rows)
    csv_path = out_dir/'trials_metrics.csv'
    trials_df.to_csv(csv_path, index=False)
    print(f"Saved trials CSV -> {csv_path}")

    # Pareto plot (Return% vs |Max DD%|)
    try:
        ret_col, dd_abs = _resolve_ret_dd(trials_df)
        dd = dd_abs.astype(float) * 100.0
        rr = trials_df[ret_col].astype(float) * 100.0
        strat = trials_df['strategy'] if 'strategy' in trials_df.columns else pd.Series(index=trials_df.index, data=None)
        trades = trials_df['trades'].astype(float) if 'trades' in trials_df.columns else pd.Series(index=trials_df.index, data=np.nan)
        nums = trials_df['number']

        mask = np.isfinite(dd) & np.isfinite(rr)
        dd = dd[mask]; rr = rr[mask]; strat_m = strat[mask]; trades_m = trades[mask]; idxs = nums[mask].values

        def pareto_front(x_dd, y_ret, ids):
            pts = sorted(zip(x_dd, y_ret, ids), key=lambda p: (p[0], -p[1]))
            front = []; best_y = -1e18
            for d, r, i in pts:
                if r > best_y:
                    front.append((d, r, i)); best_y = r
            return front

        front = pareto_front(dd.values.tolist(), rr.values.tolist(), idxs.tolist())

        cmap = {'breakout': 'tab:blue', 'mean_reversion': 'tab:orange'}
        colors = [cmap.get(s, 'tab:gray') for s in strat_m]

        try:
            tmax = float(np.nanmax(trades_m.values))
            sizes = 20 + 80 * (trades_m.values / tmax if tmax > 0 else 0)
        except Exception:
            sizes = np.full(len(dd), 30.0)

        plt.figure()
        plt.scatter(dd, rr, s=sizes, c=colors, alpha=0.7, edgecolors='none')
        if front:
            xf = [p[0] for p in front]; yf = [p[1] for p in front]
            plt.plot(xf, yf, linewidth=1.5, alpha=0.8)

        try:
            best_num = select_best_trial(trials_df, args.dd_cap_pct*1.0)
            if best_num in idxs:
                import numpy as _np
                k = _np.where(idxs == best_num)[0][0]
                plt.scatter(dd.values[k], rr.values[k], s=220, marker='*',
                            facecolors='none', edgecolors='black', linewidths=1.8, label='Best score')
        except Exception:
            pass
        try:
            j = int(np.nanargmax(rr.values))
            plt.scatter(dd.values[j], rr.values[j], s=160, marker='X',
                        facecolors='none', edgecolors='green', linewidths=1.6, label='Best return')
        except Exception:
            pass

        import matplotlib.lines as mlines
        h_breakout = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=6, label='breakout')
        h_meanrev = mlines.Line2D([], [], color='tab:orange', marker='o', linestyle='None', markersize=6, label='mean_reversion')
        h_best = mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=10, label='Best score')
        h_bestret = mlines.Line2D([], [], color='green', marker='X', linestyle='None', markersize=8, label='Best return')
        plt.legend(handles=[h_breakout, h_meanrev, h_best, h_bestret], loc='best')

        plt.xlabel('Max Drawdown % (magnitude)')
        plt.ylabel('Return %')
        title = (f"Optuna Trials: Return vs Max DD ({'multi' if args.multi_objective else args.objective})\n"
                 "Color: strategy | Size: #trades")
        plt.title(title)
        plt.tight_layout()
        pareto_path = out_dir / 'pareto_return_vs_drawdown.png'
        plt.savefig(pareto_path, dpi=150)
        plt.close()
        print(f"Saved Pareto plot -> {pareto_path}")
    except Exception as e:
        print(f"Warning: failed to create Pareto plot: {e}")

    # --- Select top-K under DD cap ---
    top_k = int(getattr(args, 'select_top_k', 1) or 1)
    agg_style = getattr(args, 'aggregate_params', 'best')

    try:
        best_num, best_params, top_nums = select_top_k_params(
            study, trials_df, float(args.dd_cap_pct), max(1, top_k), agg_style
        )
    except Exception:
        best_num = select_best_trial(trials_df, float(args.dd_cap_pct))
        best_params = _trial_by_num(study, best_num).params
        top_nums = [best_num]

    best = dict(best_params)

    # Merge execution + risk + fee (safe defaults)
    rsi_period = int(best.get('rsi_period', 14))
    best.update({
        'fee': float(args.fee),
        'rsi_key': f"rsi{rsi_period}",
        'start_equity': float(args.start_equity),
        'risk_per_trade': float(args.risk_per_trade),
        'risk_stop_atr_mult': float(args.risk_stop_atr_mult),
        'risk_min_stop_frac': float(args.risk_min_stop_frac),
        'daily_loss_limit_pct': float(args.daily_loss_limit_pct),
        'cooldown_bars': int(args.cooldown_bars),
        'fill_timing': getattr(args, 'fill_timing', 'next_open'),
        'bar_priority': getattr(args, 'bar_priority', 'stop_first'),
        'session_start': int(getattr(args, 'session_start', 0)),
        'session_end': int(getattr(args, 'session_end', 24)),
        'include_hours_set': parse_hours_list(args.include_hours),
        'max_consec_losses': int(getattr(args, 'max_consec_losses', 0)),
        'lockout_bars': int(getattr(args, 'lockout_bars', 0)),
        'proba_temp': float(getattr(args, 'proba_temp', 1.0)),
        'participation_cap': float(getattr(args, 'participation_cap', 1.0)),
        'price_step': float(getattr(args, 'price_step', 0.0001)),
        'qty_step': float(getattr(args, 'qty_step', 0.0001)),
        'min_notional': float(getattr(args, 'min_notional', 5.0)),
        'max_notional_pct_equity': float(getattr(args, 'max_notional_pct_equity', 1.0)),
        'sides': args.sides,
        'regime_adapt': bool(exec_defaults.get('regime_adapt', False)),
        'regime_fast': int(exec_defaults.get('regime_fast', 50)),
        'regime_slow': int(exec_defaults.get('regime_slow', 200)),
        'regime_band': float(exec_defaults.get('regime_band', 0.0)),
        # NEW:
        'round_exits': bool(exec_defaults.get('round_exits', True)),
        'stop_extra_slip': float(exec_defaults.get('stop_extra_slip', 0.0)),
    })
    if exec_defaults.get('regime_overrides') is not None:
        best['regime_overrides'] = exec_defaults['regime_overrides']

    if 'slippage' not in best and args.fixed_slippage is not None:
        best['slippage'] = float(args.fixed_slippage)

    # ---- Add meta so your aggregators don't KeyError when reading params ----
    best['select_top_k'] = top_k
    best['aggregate_params'] = agg_style
    best['objective'] = ('multi' if args.multi_objective else args.objective)
    best['wfo_folds'] = int(args.wfo_folds)
    best['embargo_bars'] = int(args.embargo_bars)

    _defaults = {
        'use_ml': False, 'ml_thresh': 0.0,
        'use_sma': False, 'use_atr': False, 'use_adx': False,
        'use_ema': False, 'use_spike': False,
        'edge_k': 1.0, 'edge_cap': 0.5,
    }
    for k, v in _defaults.items():
        best.setdefault(k, v)

    print(f"Top-K under DD cap: {top_nums} (agg='{agg_style}')")

    # Evaluate train/test with selected params
    trade_log_tr = []
    dtr = prepare_indicators(df_train, best['p_sma'], best['p_std'], best['p_atr'], best['p_adx'],
                             [best['rsi_period']], [best['ema1'], best['ema2']])
    if best.get('use_ml', False) and has_ml:
        Xtr = build_features(dtr, best['bb_std_mult'], f"rsi{best['rsi_period']}")
        if nfeat>0:
            if Xtr.shape[1] < nfeat:
                Xtr = np.pad(Xtr, ((0,0),(0, nfeat - Xtr.shape[1])), mode='constant')
            elif Xtr.shape[1] > nfeat:
                Xtr = Xtr[:, :nfeat]
        model_tr = current_model_fn(dtr); probs_tr = predict_proba_safe(model_tr, Xtr)
    else:
        probs_tr = np.ones(len(dtr))
    rets_tr = run_backtest(dtr, best, probs_tr, trade_log_tr); mtr = compute_metrics(rets_tr, start_equity=args.start_equity)

    trade_log_te = []
    dte = prepare_indicators(df_test, best['p_sma'], best['p_std'], best['p_atr'], best['p_adx'],
                             [best['rsi_period']], [best['ema1'], best['ema2']])
    if best.get('use_ml', False) and has_ml:
        Xte = build_features(dte, best['bb_std_mult'], f"rsi{best['rsi_period']}")
        if nfeat>0:
            if Xte.shape[1] < nfeat:
                Xte = np.pad(Xte, ((0, 0), (0, nfeat - Xte.shape[1])), mode='constant')
            elif Xte.shape[1] > nfeat:
                Xte = Xte[:, :nfeat]
        model_te = current_model_fn(dte); probs_te = predict_proba_safe(model_te, Xte)
    else:
        probs_te = np.ones(len(dte))
    rets_te = run_backtest(dte, best, probs_te, trade_log_te); mte = compute_metrics(rets_te, start_equity=args.start_equity)

    print("=== Selected trial(s) ===")
    print(f"top_k={top_k} selected_trial_numbers={top_nums} primary={best_num}")
    print(f"ret%_train={mtr['return_pct']:.2%} dd%_train={mtr['max_drawdown_pct']:.2%} trades={mtr['trades']}")
    print(f"ret%_test={mte['return_pct']:.2%} dd%_test={mte['max_drawdown_pct']:.2%} pnl_test={mte['pnl']:.2f} trades_test={mte['trades']}")

    # Dump test trades if requested
    if args.dump_trades:
        path = Path(args.out_dir) / Path(args.dump_trades).name if not Path(args.dump_trades).is_absolute() else Path(args.dump_trades)
        if trade_log_te:
            tdf = pd.DataFrame(trade_log_te)
            tdf = tdf[['entry_time','exit_time','side','entry','exit','qty','pnl','reason']]
            tdf.to_csv(path, index=False)
            print(f"Dumped TEST trades -> {path}")

    # MC bootstrap on test trades
    if int(args.mc_sims) > 0:
        mc = mc_bootstrap_trade_seq(rets_te, start_equity=args.start_equity, sims=int(args.mc_sims), seed=int(args.seed))
        if mc:
            print("MC bootstrap on OOS trades "
                  f"(sims={mc['sims']}): ret% [p5,p50,p95]=[{mc['ret_pct_p5']:.2%}, {mc['ret_pct_p50']:.2%}, {mc['ret_pct_p95']:.2%}] "
                  f"| |DD|% [p5,p50,p95]=[{mc['dd_abs_pct_p5']:.2%}, {mc['dd_abs_pct_p50']:.2%}, {mc['dd_abs_pct_p95']:.2%}]")

    # WFO
    if args.wfo_folds and args.wfo_folds > 1:
        wfo_pnls, wfo_trades, wfo_win_rates, wfo_avg_trades, wfo_drawdowns, wfo_return_pcts, wfo_dd_pcts = [], [], [], [], [], [], []
        fold_idx = 0
        for tr, te in walk_forward_folds(df_base, args.wfo_folds, args.train_frac, args.embargo_bars):
            fold_idx += 1
            dtr = prepare_indicators(tr, best['p_sma'], best['p_std'], best['p_atr'], best['p_adx'], [best['rsi_period']], [best['ema1'], best['ema2']])
            dte = prepare_indicators(te, best['p_sma'], best['p_std'], best['p_atr'], best['p_adx'], [best['rsi_period']], [best['ema1'], best['ema2']])
            if best.get('use_ml', False) and has_ml:
                Xte = build_features(dte, best['bb_std_mult'], f"rsi{best['rsi_period']}")
                if nfeat > 0:
                    if Xte.shape[1] < nfeat:
                        Xte = np.pad(Xte, ((0, 0), (0, nfeat - Xte.shape[1])), mode='constant')
                    elif Xte.shape[1] > nfeat:
                        Xte = Xte[:, :nfeat]
                model = current_model_fn(dtr)
                probs = predict_proba_safe(model, Xte)
            else:
                probs = np.ones(len(dte))
            rets = run_backtest(dte, best, probs)
            m = compute_metrics(rets, start_equity=args.start_equity)
            wfo_pnls.append(m['pnl']); wfo_trades.append(m['trades']); wfo_win_rates.append(m['win_rate'])
            wfo_avg_trades.append(m['avg_trade']); wfo_drawdowns.append(m['max_drawdown'])
            wfo_return_pcts.append(m['return_pct']); wfo_dd_pcts.append(m['max_drawdown_pct'])
            print(f"WFO fold {fold_idx}: OOS PnL={m['pnl']:.2f} trades={m['trades']} win_rate={m['win_rate']:.2%} avg={m['avg_trade']:.6f} ret%={m['return_pct']:.2%} maxDD={m['max_drawdown']:.6f} maxDD%={m['max_drawdown_pct']:.2%}")

        if wfo_pnls:
            mean_pnl = float(np.nanmean(wfo_pnls)); std_pnl = float(np.nanstd(wfo_pnls))
            mean_wr = float(np.nanmean(wfo_win_rates)); mean_avg = float(np.nanmean(wfo_avg_trades))
            worst_dd = float(np.nanmin(wfo_drawdowns))
            mean_ret_pct = float(np.nanmean(wfo_return_pcts)) if wfo_return_pcts else 0.0
            worst_dd_pct = float(np.nanmin(wfo_dd_pcts)) if wfo_dd_pcts else 0.0
            total_trades = int(np.nansum(wfo_trades))
            print(f"WFO summary: meanPnL={mean_pnl:.2f} stdPnL={std_pnl:.2f} meanWR={mean_wr:.2%} meanAvg={mean_avg:.6f} meanRet%={mean_ret_pct:.2%} worstDD={worst_dd:.6f} worstDD%={worst_dd_pct:.2%} folds={len(wfo_pnls)} trades={total_trades}")

    # ---- begin: echo run settings into the profile ----
    run_config = {
        'trials': int(args.trials),
        'n_jobs': int(args.n_jobs),
        'seed': int(args.seed),
        'objective': ('multi' if args.multi_objective else args.objective),
        'multi_objective': bool(args.multi_objective),
        'train_frac': float(args.train_frac),
        'wfo_folds': int(args.wfo_folds),
        'embargo_bars': int(args.embargo_bars),
        'select_top_k': int(args.select_top_k),
        'aggregate_params': str(args.aggregate_params),
        'start_equity': float(args.start_equity),
        'fee': float(args.fee),
        'fixed_slippage': (None if args.fixed_slippage is None else float(args.fixed_slippage)),
        'min_slippage': float(args.min_slippage),
        'fill_timing': str(args.fill_timing),
        'bar_priority': str(args.bar_priority),
        'session_start': int(args.session_start),
        'session_end': int(args.session_end),
        'include_hours': args.include_hours,
        'sides': str(args.sides),
        'regime_adapt': bool(args.regime_adapt),
        'regime_fast': int(args.regime_fast),
        'regime_slow': int(args.regime_slow),
        'regime_band': float(args.regime_band),
        'round_exits': bool(args.round_exits),
        'stop_extra_slip': float(args.stop_extra_slip),
    }
    # ---- end: echo run settings into the profile ----

    # ---- consistent return metric + suspicious flag ----
    def _ret_consistent(m, eq0):
        try:
            eq0f = float(eq0)
            return (float(m.get('pnl', 0.0)) / eq0f) if eq0f > 0 else None
        except Exception:
            return None

    ret_train_cons = _ret_consistent(mtr, args.start_equity)
    ret_test_cons  = _ret_consistent(mte, args.start_equity)
    suspicious = False
    try:
        suspicious = (
            (ret_test_cons is not None and ret_test_cons > 1.0) or
            (mte.get('win_rate') is not None and mte.get('trades') is not None and
             float(mte['win_rate']) >= 0.95 and int(mte['trades']) >= 100)
        )
    except Exception:
        pass
    # ----------------------------------------------------

    # Save profile
    profile = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'objective': 'multi' if args.multi_objective else args.objective,
        'selection': {'dd_cap_pct': args.dd_cap_pct, 'best_trial': best_num, 'top_k': top_k, 'top_trials': top_nums},
        'train_bars': int(len(df_train)), 'test_bars': int(len(df_test)),
        'train_metrics': mtr, 'test_metrics': mte,
        'params': best,
        'guardrails': {
            'max_dd_pct_allowed': args.dd_cap_pct,
            'min_trades_lookback': args.min_trades,
            'daily_loss_limit_pct': args.daily_loss_limit_pct,
            'cooldown_bars': args.cooldown_bars,
            'fill_timing': args.fill_timing,
            'bar_priority': args.bar_priority,
            'session_hours': [args.session_start, args.session_end],
            'include_hours': sorted(list(parse_hours_list(args.include_hours) or [])),
            'consec_loss_lockout': [args.max_consec_losses, args.lockout_bars],
            'sides': args.sides,
        },
        'regime': {
            'enabled': bool(best.get('regime_adapt', False)),
            'fast': int(best.get('regime_fast', 50)),
            'slow': int(best.get('regime_slow', 200)),
            'band': float(best.get('regime_band', 0.0)),
            'overrides': best.get('regime_overrides', None)
        },
        'run_config': run_config,
        'metrics_consistent': {
            'return_pct_train_consistent': ret_train_cons,
            'return_pct_test_consistent': ret_test_cons,
            'suspicious_run': suspicious
        }
    }
    prof_path = out_dir/'strategy_profile.json'
    with open(prof_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, default=str)
    print(f"Wrote {prof_path}")

    # Echo params if requested (no-op otherwise)
    if args.echo_params:
        try:
            print("=== ECHO SELECTED PARAMS (JSON) ===")
            print(json.dumps(best, indent=2, sort_keys=True, default=str))
        except Exception:
            pass

    # Save best params as pkl for reuse
    if joblib is not None:
        try:
            pkl_path = out_dir / 'best_params.pkl'
            joblib.dump(best, pkl_path)
            print(f"Wrote {pkl_path}")
        except Exception as e:
            print(f"Warning: failed to save best_params.pkl: {e}")

    # MLflow (optional) — param safety for non-scalars
    if args.mlflow_experiment and mlflow is not None:
        safe_params = {}
        for k, v in best.items():
            if isinstance(v, (list, tuple, set, dict)):
                try:
                    safe_params[k] = json.dumps(v, default=str)
                except Exception:
                    safe_params[k] = str(v)
            else:
                safe_params[k] = v
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name="selected-wfo"):
            mlflow.log_params(safe_params)
            mlflow.log_metrics({
                'bars_train': len(df_train), 'bars_test': len(df_test),
                'return_pct_train': float(mtr['return_pct']), 'max_dd_pct_train': float(mtr['max_drawdown_pct']),
                'trades_train': int(mtr['trades']), 'wins_train': int(mtr.get('wins',0)), 'losses_train': int(mtr.get('losses',0)),
                'return_pct_test': float(mte['return_pct']), 'max_dd_pct_test': float(mte['max_drawdown_pct']),
                'trades_test': int(mte['trades']), 'wins_test': int(mte.get('wins',0)), 'losses_test': int(mte.get('losses',0)),
            })
            if csv_path.exists(): mlflow.log_artifact(str(csv_path))
            pareto_path = out_dir / 'pareto_return_vs_drawdown.png'
            if pareto_path.exists(): mlflow.log_artifact(str(pareto_path))
            mlflow.log_artifact(str(prof_path))

if __name__ == '__main__':
    main()
