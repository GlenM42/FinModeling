import asyncio

from telegram import Update
from telegram.ext import ConversationHandler, ContextTypes

from finmodeling import commands

# Conversation states — high range to avoid collision with existing handlers (0-9)
DEPOSIT_TICKER, DEPOSIT_AMOUNT = 20, 21


# ---------------------------------------------------------------------------
# /deposit — two-step conversation: ask ticker, then amount
# ---------------------------------------------------------------------------

async def deposit_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Which ticker would you like to deposit into?")
    return DEPOSIT_TICKER


async def deposit_ticker_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['deposit_ticker'] = update.message.text.upper()
    await update.message.reply_text(
        f"How many dollars are you depositing into {context.user_data['deposit_ticker']}?"
    )
    return DEPOSIT_AMOUNT


async def deposit_amount_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    ticker = context.user_data['deposit_ticker']

    try:
        result = await asyncio.to_thread(commands.deposit_cmd, user_id, ticker, update.message.text)
    except commands.CommandError as e:
        await update.message.reply_text(str(e))
        return DEPOSIT_AMOUNT

    await update.message.reply_text(result.message)
    context.user_data.pop('deposit_ticker', None)
    return ConversationHandler.END
