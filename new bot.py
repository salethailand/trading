import asyncio
from telegram import Bot
# Replace these with your actual values
TELEGRAM_TOKEN= '7016560702:AAEmOUfu6z-3LCVyUsmftYJekkHQIZeiOFw'
TELEGRAM_CHAT_ID= 981459625

async def send_test_message():
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="✅ Telegram bot test message!")

if __name__ == "__main__":
    asyncio.run(send_test_message())
    print("Test message sent!")