import asyncio

from telegram import Update
from telegram.ext import ConversationHandler, ContextTypes

from commands_for_database import remove_transactions, add_transaction

TICKER, CONFIRM_TICKER, QUANTITY, PURCHASE_DATE, PURCHASE_PRICE, CONFIRMATION = range(6)
REMOVE_TICKER, REMOVE_DATE, REMOVE_CONFIRMATION = range(6, 9)
(OPTION_TICKER, OPTION_QUANTITY, OPTION_PURCHASE_DATE, OPTION_PURCHASE_PRICE, OPTION_CONFIRMATION) = range(5)
(OPTION_REMOVE_TICKER, OPTION_REMOVE_PURCHASE_DATE, OPTION_REMOVE_CONFIRMATION) = range(2, 5)


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
        await asyncio.to_thread(remove_transactions, 'portfolio.db', user_id, ticker, purchase_date)

        await update.message.reply_text("Transaction removed successfully.")
    else:
        await update.message.reply_text("Operation cancelled. No transactions were removed.")

    # Clear user_data related to removal
    context.user_data.pop('remove_ticker', None)
    context.user_data.pop('remove_date', None)

    return ConversationHandler.END


async def add_transaction_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Let's add the new transaction. Before we do that, I would like you "
                                    "to tell me the following information. First, tell me the ticker we're buying.")
    return TICKER


async def ticker_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    ticker = update.message.text.upper()  # Convert ticker to uppercase
    context.user_data['ticker'] = ticker  # Store ticker in user_data

    # Fetch ticker description (Placeholder function)
    description = "This is a placeholder description for " + ticker
    await update.message.reply_text(
        f"Oh great. I have found the ticker. Here is a brief description: {description}. Is this the security you're "
        f"interested in? (Reply with 'proceed' to continue or 'abort' to stop)")

    return CONFIRM_TICKER


async def confirm_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    response = update.message.text.lower()
    if response == 'proceed':
        await update.message.reply_text(
            f"Great. Now how many securities of {context.user_data['ticker']} have you bought?")
        return QUANTITY
    else:
        await update.message.reply_text("Transaction aborted.")
        return ConversationHandler.END


async def quantity_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    quantity = update.message.text
    # Simple validation to check if quantity is a number
    if not quantity.isdigit():
        await update.message.reply_text("Please enter a valid number for the quantity.")
        return QUANTITY
    context.user_data['quantity'] = int(quantity)
    await update.message.reply_text(
        f"Great. We're talking about {quantity} securities of {context.user_data['ticker']}. Can you tell me when did you bought it in a format YYYY-MM-DD?")
    return PURCHASE_DATE


async def purchase_date_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    purchase_date = update.message.text
    # Here you might want to add more sophisticated date validation
    context.user_data['purchase_date'] = purchase_date
    await update.message.reply_text(f"Wonderful. Can you tell me what was the price back then?")
    return PURCHASE_PRICE


async def purchase_price_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    purchase_price = update.message.text
    # Simple validation to check if price is a float number
    try:
        float_price = float(purchase_price)
    except ValueError:
        await update.message.reply_text("Please enter a valid number for the price.")
        return PURCHASE_PRICE
    context.user_data['purchase_price'] = float_price
    await update.message.reply_text(f"Boss, I believe the transaction to be the following:"
                                    f"\nNumber of securities: {context.user_data['quantity']}"
                                    f"\nTicker we're buying: {context.user_data['ticker']}"
                                    f"\nPurchase date: {context.user_data['purchase_date']}"
                                    f"\nPurchase price: ${context.user_data['purchase_price']}."
                                    f"\nIf everything is correct, I will put it down. Shall I proceed?")
    return CONFIRMATION


async def final_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    response = update.message.text.lower()
    if response in ['yes', 'proceed']:
        # Extract transaction details from context.user_data
        user_id = update.effective_user.id
        ticker = context.user_data['ticker']
        quantity = context.user_data['quantity']
        purchase_date = context.user_data['purchase_date']
        purchase_price = context.user_data['purchase_price']

        # Insert transaction into the database (ensure add_transaction is async or wrapped with asyncio.to_thread)
        await asyncio.to_thread(add_transaction, 'portfolio.db', user_id, ticker, quantity, purchase_price,
                                purchase_date)

        await update.message.reply_text("Transaction recorded successfully.")
    else:
        await update.message.reply_text("Transaction aborted. No data recorded.")

    # Clear user_data for the current transaction
    context.user_data.clear()

    return ConversationHandler.END