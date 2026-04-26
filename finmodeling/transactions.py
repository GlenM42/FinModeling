import asyncio

from telegram import Update
from telegram.ext import ConversationHandler, ContextTypes

from finmodeling.database import remove_transactions, add_transaction, bank_get_balance, bank_deduct

TICKER, CONFIRM_TICKER, QUANTITY, PURCHASE_DATE, PURCHASE_PRICE, CONFIRMATION = range(6)
REMOVE_TICKER, REMOVE_DATE, REMOVE_CONFIRMATION = range(6, 9)
BANK_DEDUCT = 9


async def remove_transaction_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("You've chosen to remove a transaction. "
                                    "Please tell me the ticker of the transaction you want to remove.")
    return REMOVE_TICKER


async def remove_ticker_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    ticker = update.message.text.upper()
    context.user_data['remove_ticker'] = ticker
    await update.message.reply_text(f"Ticker {ticker} received. Now, please tell me the purchase date of "
                                    f"the transaction you want to remove (in format YYYY-MM-DD).")
    return REMOVE_DATE


async def remove_date_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    purchase_date = update.message.text
    context.user_data['remove_date'] = purchase_date
    await update.message.reply_text(f"Purchase date {purchase_date} received for ticker "
                                    f"{context.user_data['remove_ticker']}. "
                                    f"Are you sure you want to remove this transaction? (yes/no)")
    return REMOVE_CONFIRMATION


async def remove_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    response = update.message.text.lower()
    if response == 'yes':
        # Extract details from context.user_data
        user_id = update.effective_user.id
        ticker = context.user_data['remove_ticker']
        purchase_date = context.user_data['remove_date']

        # Call the function to remove the transaction from the database
        await asyncio.to_thread(remove_transactions, user_id, ticker, purchase_date)

        await update.message.reply_text("Transaction removed successfully.")
    else:
        await update.message.reply_text("Operation cancelled. No transactions were removed.")

    # Clear user_data related to removal
    context.user_data.pop('remove_ticker', None)
    context.user_data.pop('remove_date', None)

    return ConversationHandler.END


async def add_transaction_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Which ticker are you buying?")
    return TICKER


async def ticker_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    ticker = update.message.text.upper()
    context.user_data['ticker'] = ticker
    await update.message.reply_text(f"How many shares of {ticker}?")
    return QUANTITY


async def quantity_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        float_quantity = float(update.message.text)
    except ValueError:
        await update.message.reply_text("Please enter a valid number.")
        return QUANTITY
    context.user_data['quantity'] = float_quantity
    await update.message.reply_text(
        f"{float_quantity} shares of {context.user_data['ticker']} — what was the purchase date? (YYYY-MM-DD)"
    )
    return PURCHASE_DATE


async def purchase_date_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['purchase_date'] = update.message.text
    await update.message.reply_text("Price per share?")
    return PURCHASE_PRICE


async def purchase_price_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        float_price = float(update.message.text)
    except ValueError:
        await update.message.reply_text("Please enter a valid number.")
        return PURCHASE_PRICE
    context.user_data['purchase_price'] = float_price

    ticker = context.user_data['ticker']
    quantity = context.user_data['quantity']
    purchase_date = context.user_data['purchase_date']
    total = round(quantity * float_price, 2)

    await update.message.reply_text(
        f"Here's the transaction:\n\n"
        f"  {ticker} — {quantity} shares\n"
        f"  Date: {purchase_date}\n"
        f"  Price: ${float_price:.2f}\n"
        f"  Total: ${total:.2f}\n\n"
        f"Confirm? (yes/no)"
    )
    return CONFIRMATION


async def final_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    response = update.message.text.lower()
    if response in ['yes', 'proceed']:
        user_id = update.effective_user.id
        ticker = context.user_data['ticker']
        quantity = context.user_data['quantity']
        purchase_date = context.user_data['purchase_date']
        purchase_price = context.user_data['purchase_price']
        total_cost = round(quantity * purchase_price, 2)

        await asyncio.to_thread(add_transaction, user_id, ticker, quantity, purchase_price, purchase_date)

        await update.message.reply_text(
            f"Recorded. Deduct ${total_cost:.2f} from your {ticker} bank balance? (yes/no)"
        )
        return BANK_DEDUCT
    else:
        await update.message.reply_text("Cancelled.")
        context.user_data.clear()
        return ConversationHandler.END


async def bank_deduct_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    response = update.message.text.lower()
    if response == 'yes':
        user_id = update.effective_user.id
        ticker = context.user_data['ticker']
        quantity = context.user_data['quantity']
        purchase_price = context.user_data['purchase_price']
        total_cost = round(quantity * purchase_price, 2)

        current_balance = await asyncio.to_thread(bank_get_balance, user_id, ticker)
        if total_cost > current_balance:
            await update.message.reply_text(
                f"Note: purchase (${total_cost:.2f}) exceeds current bank balance (${current_balance:.2f}) — "
                f"balance will go negative."
            )
        await asyncio.to_thread(bank_deduct, user_id, ticker, total_cost)
        new_balance = await asyncio.to_thread(bank_get_balance, user_id, ticker)
        await update.message.reply_text(f"{ticker} bank balance: ${new_balance:.2f}.")
    else:
        await update.message.reply_text("Bank unchanged.")

    context.user_data.clear()
    return ConversationHandler.END