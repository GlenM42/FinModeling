from telegram import Update
import asyncio
from telegram.ext import (
    CallbackContext
)
from telegram.ext import ConversationHandler

from commands_for_database import add_option_transaction, remove_option_transaction

(OPTION_TICKER, OPTION_QUANTITY, OPTION_PURCHASE_DATE, OPTION_PURCHASE_PRICE, OPTION_CONFIRMATION) = range(5)
(OPTION_REMOVE_TICKER, OPTION_REMOVE_PURCHASE_DATE, OPTION_REMOVE_CONFIRMATION) = range(2, 5)


async def add_option_start(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Please tell me the option symbol you've traded.")
    return OPTION_TICKER


# Handler to receive the option ticker
async def option_ticker_received(update: Update, context: CallbackContext) -> int:
    context.user_data['option_ticker'] = update.message.text.upper()
    await update.message.reply_text(
        f"Option symbol {context.user_data['option_ticker']} received. How many contracts did you buy/sell?")
    return OPTION_QUANTITY


# Handler to receive the quantity
async def option_quantity_received(update: Update, context: CallbackContext) -> int:
    text = update.message.text
    if text.isdigit():
        context.user_data['option_quantity'] = int(text)
        await update.message.reply_text("What was the purchase price per contract?")
        return OPTION_PURCHASE_PRICE
    else:
        await update.message.reply_text("Please enter a valid number for the quantity.")
        return OPTION_QUANTITY


# Handler to receive the purchase price
async def option_purchase_price_received(update: Update, context: CallbackContext) -> int:
    text = update.message.text
    try:
        context.user_data['option_purchase_price'] = float(text)
        await update.message.reply_text("On what date did you buy/sell the contracts? Please use YYYY-MM-DD format.")
        return OPTION_PURCHASE_DATE
    except ValueError:
        await update.message.reply_text("Please enter a valid price.")
        return OPTION_PURCHASE_PRICE


# Handler to receive the purchase date
async def option_purchase_date_received(update: Update, context: CallbackContext) -> int:
    # Here you should validate the date format
    text = update.message.text
    if True:  # Replace this with actual validation
        context.user_data['option_purchase_date'] = text
        # Confirm all details before proceeding
        await update.message.reply_text(
            "Please confirm the details of your options transaction:\n"
            f"Symbol: {context.user_data['option_ticker']}\n"
            f"Quantity: {context.user_data['option_quantity']}\n"
            f"Purchase price: {context.user_data['option_purchase_price']}\n"
            f"Purchase date: {context.user_data['option_purchase_date']}\n"
            "If all the details are correct, please type 'confirm' to proceed or 'cancel' to abort."
        )
        return OPTION_CONFIRMATION
    else:
        await update.message.reply_text("Please enter a valid date in YYYY-MM-DD format.")
        return OPTION_PURCHASE_DATE


# Final confirmation handler
async def option_final_confirmation(update: Update, context: CallbackContext) -> int:
    text = update.message.text.lower()
    if text == 'confirm':
        # Insert the transaction into the database
        await asyncio.to_thread(
            add_option_transaction,
            'portfolio.db',
            update.effective_user.id,
            context.user_data['option_ticker'],
            context.user_data['option_quantity'],
            context.user_data['option_purchase_price'],
            context.user_data['option_purchase_date']
        )
        await update.message.reply_text("Option transaction added successfully.")
    else:
        await update.message.reply_text("Option transaction canceled.")

    # Clear context.user_data and end the conversation
    context.user_data.clear()
    return ConversationHandler.END


async def remove_option_start(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Please tell me the option symbol for the transaction you want to remove.")
    return OPTION_REMOVE_TICKER


# Handler to receive the option ticker for removal
async def option_remove_ticker_received(update: Update, context: CallbackContext) -> int:
    context.user_data['option_remove_ticker'] = update.message.text.upper()
    await update.message.reply_text(
        "Please enter the purchase date of the transaction you want to remove (in format YYYY-MM-DD).")
    return OPTION_REMOVE_PURCHASE_DATE


# Handler to receive the purchase date of the option transaction for removal
async def option_remove_purchase_date_received(update: Update, context: CallbackContext) -> int:
    # Here you should validate the date format
    text = update.message.text
    if True:  # Replace this with actual validation
        context.user_data['option_remove_purchase_date'] = text
        await update.message.reply_text("You are about to remove the transaction with the following details:\n"
                                        f"Symbol: {context.user_data['option_remove_ticker']}\n"
                                        f"Purchase date: {context.user_data['option_remove_purchase_date']}\n"
                                        "Please confirm removal by typing 'confirm'. Type 'cancel' to abort.")
        return OPTION_REMOVE_CONFIRMATION
    else:
        await update.message.reply_text("Please enter a valid date in YYYY-MM-DD format.")
        return OPTION_REMOVE_PURCHASE_DATE


# Final confirmation handler for removing the option transaction
async def option_remove_final_confirmation(update: Update, context: CallbackContext) -> int:
    text = update.message.text.lower()
    if text == 'confirm':
        # Remove the transaction from the database
        await asyncio.to_thread(
            remove_option_transaction,
            'portfolio.db',
            update.effective_user.id,
            context.user_data['option_remove_ticker'],
            context.user_data['option_remove_purchase_date']
        )
        await update.message.reply_text("Option transaction removed successfully.")
    else:
        await update.message.reply_text("Option transaction removal canceled.")

    # Clear context.user_data and end the conversation
    context.user_data.clear()
    return ConversationHandler.END
