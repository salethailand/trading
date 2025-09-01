import sys
import pandas as pd
import numpy as np

# — Load & Prep —
df = pd.read_csv('JS_5m.csv', parse_dates=['open_time']).set_index('open_time')
df.rename(columns={'open_price':'open','high_price':'high','low_price':'low','close_price':'close'}, inplace=True)
df.dropna(subset=['open','high','low','close','volume'], inplace=True)

# — Indicators —
df['sma20'] = df['close'].rolling(20).mean()
std20      = df['close'].rolling(20).std()
df['bb_h'] = df['sma20'] + 1.0*std20   # ← ultra‑narrow
df['bb_l'] = df['sma20'] - 1.0*std20
d = df['close'].diff()
gain = d.clip(lower=0).rolling(14).mean()
loss = (-d.clip(upper=0)).rolling(14).mean()
df['rsi'] = 100 - (100/(1 + gain/loss))
df.dropna(inplace=True)

# — Historical Trades for Kelly —
raw_trades = []
in_pos = False
for prev, cur in zip(df.iloc[:-1].itertuples(), df.iloc[1:].itertuples()):
    if not in_pos and prev.close > prev.bb_h:
        in_pos = True; entry = cur.open
    elif in_pos and cur.close < prev.sma20:
        raw_trades.append((entry, cur.open)); in_pos = False

# — Kelly Calculation (full) —
rets = [(ex-en)/en for en,ex in raw_trades]
W  = np.mean([r>0 for r in rets])
Rw = np.mean([r for r in rets if r>0]) if any(r>0 for r in rets) else 0
Rl = -np.mean([r for r in rets if r<0]) if any(r<0 for r in rets) else 0
K  = W - (1-W)/(Rw/Rl) if Rl>0 else 0   # full Kelly

# — Backtest Aggressively —
cash, pos = 1000.0, 0.0
equity = []
for prev, cur in zip(df.iloc[:-1].itertuples(), df.iloc[1:].itertuples()):
    price = cur.open
    if pos==0 and prev.close > prev.bb_h and K>0:
        alloc = cash * K
        pos, cash = alloc/price, cash-alloc
    elif pos>0 and cur.close < prev.sma20:
        cash += pos*price; pos = 0.0
    equity.append(cash + pos*price)

# — Finalize & Metrics —
if pos>0:
    cash += pos*df['close'].iloc[-1]; equity[-1]=cash

es = pd.Series(equity, index=df.index[1:])
mrets     = es.resample('M').last().pct_change().dropna()*100
total_ret = (es.iloc[-1]/1000 - 1)*100
max_dd    = ((es - es.cummax())/es.cummax()*100).min()

print("Ultra‑Turbo BB(1σ) + Kelly100:")
print(f"Avg Monthly Return: {mrets.mean():.2f}%")
print(f"Total Return:       {total_ret:.1f}%")
print(f"Max Drawdown:       {max_dd:.1f}%")
