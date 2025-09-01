import os
import asyncio
import time
from datetime import datetime, timezone
import ccxt
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
from dotenv import load_dotenv
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import schedule

# Load .env config
load_dotenv()

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))

exchange = ccxt.kucoin({
    'apiKey': KUCOIN_API_KEY,
    'secret': KUCOIN_API_SECRET,
    'password': KUCOIN_API_PASSPHRASE,
    'enableRateLimit': True,
})

telegram_bot = Bot(token=TELEGRAM_TOKEN)

PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TRADE_RISK_PERCENT = 0.02
MAX_TRADES_PER_PAIR_PER_DAY = 2
MIN_USDT_BALANCE = 100

in_position = {pair: False for pair in PAIRS}
trades_today = {pair: 0 for pair in PAIRS}
last_trade_date = datetime.now(timezone.utc).date()

# Telegram alerts
async def send_telegram(message):
    try:
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        print(f"Telegram error: {e}")

# Add indicators
def fetch_ohlcv_with_indicators(symbol, timeframe='1h', limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    return df

def check_entry_signal(symbol):
    df_1h = fetch_ohlcv_with_indicators(symbol, '1h', 100)
    df_4h = fetch_ohlcv_with_indicators(symbol, '4h', 100)
    last_1h = df_1h.iloc[-1]
    last_4h = df_4h.iloc[-1]

    trend_1h = last_1h['ema9'] > last_1h['ema21']
    trend_4h = last_4h['ema9'] > last_4h['ema21']
    momentum = 50 < last_1h['rsi'] < 70
    strong_trend = last_4h['adx'] > 25
    breakout = last_1h['close'] > df_1h['high'].rolling(20).max().iloc[-2]

    return all([trend_1h, trend_4h, momentum, strong_trend, breakout]), df_1h

def calculate_position_size(symbol, price):
    balance = exchange.fetch_balance()
    usdt_available = balance['USDT']['free']
    if usdt_available < MIN_USDT_BALANCE:
        return 0
    return round((usdt_available * TRADE_RISK_PERCENT) / price, 6)

def place_buy_order(symbol, amount):
    return exchange.create_market_buy_order(symbol, amount)

def place_sell_order(symbol, amount):
    return exchange.create_market_sell_order(symbol, amount)

def get_current_price(symbol):
    return exchange.fetch_ticker(symbol)['last']

# Trade logic
async def trade_pair(symbol):
    global in_position, trades_today, last_trade_date

    if datetime.now(timezone.utc).date() != last_trade_date:
        for p in PAIRS:
            trades_today[p] = 0
        last_trade_date = datetime.now(timezone.utc).date()

    if trades_today[symbol] >= MAX_TRADES_PER_PAIR_PER_DAY:
        return

    signal, df_1h = check_entry_signal(symbol)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not in_position[symbol] and signal:
        price = get_current_price(symbol)
        amount = calculate_position_size(symbol, price)
        if amount <= 0:
            await send_telegram(f"⚠️ Not enough USDT balance to trade {symbol}.")
            return
        try:
            place_buy_order(symbol, amount)
            in_position[symbol] = True
            trades_today[symbol] += 1
            await send_telegram(
                f"🟢 BUY {symbol}\nAmount: {amount:.6f} @ ${price:.2f}\n"
                f"Balance: ${exchange.fetch_balance()['USDT']['free']:.2f}\nTime: {now_str}"
            )
        except Exception as e:
            await send_telegram(f"❌ Buy order failed: {e}")

    elif in_position[symbol]:
        price = get_current_price(symbol)
        entry_price = df_1h['close'].iloc[-2]
        gain = (price - entry_price) / entry_price
        take_profit = 0.03
        stop_loss = -0.015

        coin = symbol.split('/')[0]
        amount = exchange.fetch_balance()[coin]['free']

        if gain >= take_profit or gain <= stop_loss:
            try:
                place_sell_order(symbol, amount)
                in_position[symbol] = False
                trades_today[symbol] += 1
                await send_telegram(
                    f"🔴 SELL {symbol}\nAmount: {amount:.6f} @ ${price:.2f}\n"
                    f"Gain: {gain*100:.2f}%\nTime: {now_str}"
                )
            except Exception as e:
                await send_telegram(f"❌ Sell order failed: {e}")

# Telegram command /status
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    balance = exchange.fetch_balance()
    usdt = balance.get('USDT', {}).get('free', 0)
    lines = [f"💰 USDT Balance: ${usdt:.2f}"]
    for pair in PAIRS:
        coin = pair.split('/')[0]
        amt = balance.get(coin, {}).get('free', 0)
        pos = "In Position" if in_position.get(pair, False) else "No Position"
        lines.append(f"{pair}: {amt:.6f} ({pos}), Trades: {trades_today.get(pair, 0)}")
    await update.message.reply_text('\n'.join(lines))


# Scheduler loop
async def job_runner():
    tasks = [trade_pair(pair) for pair in PAIRS]
    await asyncio.gather(*tasks)

def start_scheduler(app):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def reset_trades_daily():
        global last_trade_date
        today = datetime.now(timezone.utc).date()
        if today != last_trade_date:
            for p in PAIRS:
                trades_today[p] = 0
            last_trade_date = today

    def job():
        reset_trades_daily()
        loop.create_task(job_runner())

    schedule.every(1).hours.do(job)
    loop.run_until_complete(send_telegram("🤖 Bot started and running."))
    while True:
        schedule.run_pending()
        time.sleep(10)

# Entry point
if __name__ == "__main__":
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("status", status_command))

    import threading
    threading.Thread(target=start_scheduler, args=(application,), daemon=True).start()

    application.run_polling()
