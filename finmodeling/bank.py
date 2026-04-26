import asyncio

from telegram import Update
from telegram.ext import ConversationHandler, ContextTypes

from finmodeling.database import bank_deposit, bank_get_balance

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
    try:
        amount = float(update.message.text)
        if amount <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Please enter a valid positive dollar amount.")
        return DEPOSIT_AMOUNT

    user_id = update.effective_user.id
    ticker = context.user_data['deposit_ticker']

    await asyncio.to_thread(bank_deposit, user_id, ticker, amount)
    new_balance = await asyncio.to_thread(bank_get_balance, user_id, ticker)
    reply = f"Deposited ${amount:.2f} into {ticker}. New balance: ${new_balance:.2f}."

    await update.message.reply_text(reply)
    context.user_data.pop('deposit_ticker', None)
    return ConversationHandler.END
