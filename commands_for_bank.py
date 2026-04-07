import asyncio

from telegram import Update
from telegram.ext import ConversationHandler, ContextTypes

from commands_for_database import bank_deposit, bank_get_balance, bank_get_all
from commands_for_management import find_current_price

# Conversation states — high range to avoid collision with existing handlers (0-9)
DEPOSIT_TICKER, DEPOSIT_AMOUNT = 20, 21


# ---------------------------------------------------------------------------
# Pure logic helpers
# ---------------------------------------------------------------------------

def deposit(user_id, ticker, amount):
    """Add amount to bank balance. Returns new balance."""
    bank_deposit(user_id, ticker, amount)
    return bank_get_balance(user_id, ticker)


def get_bank(user_id):
    """Return list of (ticker, balance, last_updated) rows with balance > 0."""
    return bank_get_all(user_id)


def get_purchase_suggestion(user_id, ticker, current_price):
    """Return (whole_shares, remainder) based on bank balance and current price."""
    balance = bank_get_balance(user_id, ticker)
    if not current_price or current_price <= 0:
        return 0, balance
    whole_shares = int(balance // current_price)
    remainder = round(balance - whole_shares * current_price, 2)
    return whole_shares, remainder


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

    new_balance = await asyncio.to_thread(deposit, user_id, ticker, amount)
    current_price = await asyncio.to_thread(find_current_price, ticker)

    reply = f"Deposited ${amount:.2f} into {ticker}. New balance: ${new_balance:.2f}."
    if current_price:
        whole_shares = int(new_balance // current_price)
        remainder = round(new_balance - whole_shares * current_price, 2)
        reply += (
            f"\n\nCurrent {ticker} price: ${current_price:.2f}"
            f"\nYou could buy {whole_shares} share(s) now, with ${remainder:.2f} remaining."
        )
    else:
        reply += f"\n(Could not fetch current price for {ticker}.)"

    await update.message.reply_text(reply)
    context.user_data.pop('deposit_ticker', None)
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# /bank — simple command, no conversation
# ---------------------------------------------------------------------------

async def bank_show(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    rows = await asyncio.to_thread(get_bank, user_id)

    if not rows:
        await update.message.reply_text("Your bank is empty. Use /deposit to add funds.")
        return

    lines = ["<b>Bank balances:</b>\n"]
    for ticker, balance, last_updated in rows:
        current_price = await asyncio.to_thread(find_current_price, ticker)
        if current_price:
            whole_shares = int(balance // current_price)
            remainder = round(balance - whole_shares * current_price, 2)
            lines.append(
                f"<b>{ticker}</b>: ${balance:.2f} | Price: ${current_price:.2f} "
                f"→ buy {whole_shares} share(s), ${remainder:.2f} remaining"
            )
        else:
            lines.append(f"<b>{ticker}</b>: ${balance:.2f} (price unavailable)")

    await update.message.reply_text("\n".join(lines), parse_mode='HTML')

