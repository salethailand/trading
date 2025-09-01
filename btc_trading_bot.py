import asyncio
import os
import time
import pandas as pd
import schedule
from datetime import datetime
from dotenv import load_dotenv
import ccxt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from telegram import Bot

# === Load secrets from .env ===
load_dotenv()

KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET')
KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# === Exchange & Telegram Setup ===
exchange = ccxt.kucoin({
    'apiKey': KUCOIN_API_KEY,
    'secret': KUCOIN_API_SECRET,
    'password': KUCOIN_API_PASSPHRASE,
    'enableRateLimit': True,
})

telegram_bot = Bot(token=TELEGRAM_TOKEN)
SYMBOL = 'BTC/USDT'
TRADE_AMOUNT_USDT = 50  # ← You can change this
in_position = False  # ← Global trade state

# === Telegram async function ===
async def send_telegram(message):
    try:
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        print("Telegram Error:", e)

# === Fetch data and indicators ===
def fetch_data():
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='1h', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    return df

# === Strategy Logic ===
def signal_generator(df):
    last = df.iloc[-1]
    breakout = last['close'] > df['high'].rolling(20).max().iloc[-2]
    volume_spike = last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5
    momentum = 55 < last['rsi'] < 75
    trend = last['ema9'] > last['ema21']
    return breakout and volume_spike and momentum and trend

# === Main Trading Logic ===
async def place_trade():
    global in_position
    df = fetch_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        if not in_position and signal_generator(df):
            balance = exchange.fetch_balance()
            usdt = balance['USDT']['free']
            if usdt >= TRADE_AMOUNT_USDT:
                price = exchange.fetch_ticker(SYMBOL)['ask']
                amount = round(TRADE_AMOUNT_USDT / price, 6)
                exchange.create_market_buy_order(SYMBOL, amount)
                in_position = True
                msg = (
                    f"🟢 BUY {SYMBOL}\n"
                    f"Amount: {amount} BTC @ ${price:.2f}\n"
                    f"Balance (USDT): ${usdt:.2f}\n"
                    f"Time: {now}"
                )
                await send_telegram(msg)

        elif in_position:
            price = exchange.fetch_ticker(SYMBOL)['last']
            entry_price = df['close'].iloc[-2]
            gain = (price - entry_price) / entry_price
            if gain >= 0.03 or gain <= -0.015:
                balance = exchange.fetch_balance()
                amount = balance['BTC']['free']
                if amount > 0.0001:
                    exchange.create_market_sell_order(SYMBOL, amount)
                    in_position = False
                    msg = (
                        f"🔴 SELL {SYMBOL}\n"
                        f"Amount: {amount:.6f} BTC @ ${price:.2f}\n"
                        f"Gain: {gain*100:+.2f}%\n"
                        f"Balance (BTC): {amount:.6f}\n"
                        f"Time: {now}"
                    )
                    await send_telegram(msg)

    except Exception as e:
        print("Trade Error:", e)
        await send_telegram(f"⚠️ Trade Error: {e}")

# === Bot Runner ===
async def run_bot():
    try:
        await place_trade()
    except Exception as e:
        print("Bot Error:", e)
        await send_telegram(f"⚠️ Bot Error: {e}")

# === Scheduler ===
def start_scheduler():
    loop = asyncio.get_event_loop()

    def job():
        loop.create_task(run_bot())

    schedule.every(1).hours.do(job)
    loop.run_until_complete(send_telegram("🤖 KuCoin BTC Bot Started!"))
    while True:
        schedule.run_pending()
        time.sleep(10)

# === Entry Point ===
if __name__ == "__main__":
    start_scheduler()
