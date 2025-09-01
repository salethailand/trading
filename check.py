from binance.client import Client
import pandas as pd
import os      

api_key = 'd7SzkI4oBBjgAVb17pPfIW9d62XugJksL8w1LvjllQcp5ZUOMf4khkoMHfmDNrjq'
api_secret = 'rKiR4wfxt1QMg4Elj8QllLGeqYDO82eJad2mtAElu3w4rwVviSacTILX9PYe3MFa'
client = Client(api_key, api_secret)


# ─── Parameters ─────────────────────────────────────────────────────────────────
symbol     = 'SOLUSDT'
intervals  = ['5m', '15m']
limit      = 100000           # number of most‑recent candles to fetch per interval
output_dir = 'data'        # directory to save CSVs

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ─── Fetch & Export Loop ────────────────────────────────────────────────────────
for interval in intervals:
    klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=limit
    )

    df = pd.DataFrame(
        klines,
        columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_vol', 'trades',
            'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
        ]
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp','open','high','low','close','volume']]

    filename = os.path.join(output_dir, f"{symbol.lower()}_{interval}.csv")
    df.to_csv(
        filename,
        index=False,
        date_format='%Y-%m-%d %H:%M:%S'
    )
    print(f"Exported {len(df)} rows of {symbol} {interval} data to {filename}")

print("All intervals fetched and exported.")