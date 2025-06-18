import asyncio
import os
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, filters, ContextTypes

from commands_for_calendar import find_next_first_friday
from commands_for_management import initialize_portfolio, calculate_performance, show_portfolio_as_image, \
    plot_portfolio_performance, find_current_price, find_previous_price
from commands_for_transactions import add_transaction_start, ticker_received, quantity_received, \
    purchase_date_received, purchase_price_received, final_confirmation, remove_transaction_start, \
    remove_ticker_received, remove_date_received, remove_confirmation

# Define states
TICKER, CONFIRM_TICKER, QUANTITY, PURCHASE_DATE, PURCHASE_PRICE, CONFIRMATION = range(6)
REMOVE_TICKER, REMOVE_DATE, REMOVE_CONFIRMATION = range(6, 9)

load_dotenv()
ADMIN_USER_IDS = [
    int(os.getenv('ADMIN_TELEGRAM_ID_1')),
    int(os.getenv('ADMIN_TELEGRAM_ID_2'))
]


# Define your async start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Accessing the message object
    message = update.message

    # Building a response string that includes some details of the message
    response = (
        f"Hello, Mr. {message.from_user.full_name if message.from_user else 'N/A'}! "
        f"I would like to help you with your portfolio management."
    )

    # Sending the details back as a message
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


# Define your async portfolio command handler
async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id  # Get the user ID of the person sending the command

    if user_id not in ADMIN_USER_IDS:
        # If the user is not in the list of admins, send an unauthorized message
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text=f"You are not authorized to view the portfolio. Contact "
                                            f"admin in case you needed to be added (your user ID is: {user_id})")
    else:
        # If the user is an admin, proceed with showing the portfolio
        portfolio_df = await asyncio.to_thread(initialize_portfolio)
        portfolio_perf = await asyncio.to_thread(calculate_performance, portfolio_df)

        # Use the modified functions to save images
        await asyncio.to_thread(show_portfolio_as_image, portfolio_perf, 'portfolio_table.png')
        await asyncio.to_thread(plot_portfolio_performance, portfolio_perf, 'portfolio_performance.png')

        # Send the images
        with open('portfolio_table.png', 'rb') as file:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=file)
        with open('portfolio_performance.png', 'rb') as file:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=file)

        # Delete the images after sending
        os.remove('portfolio_table.png')
        os.remove('portfolio_performance.png')


async def month_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Fetching data for VOO... Please wait.")

    current_price = await asyncio.to_thread(find_current_price, "VOO")
    previous_price = await asyncio.to_thread(find_previous_price, "VOO")

    if current_price is not None and previous_price is not None:
        percentage_change = round(((current_price - previous_price) / previous_price) * 100, 2)

        # Determine the quantity to suggest
        if percentage_change > -10:
            suggestion = "to buy one"
        elif -20 < percentage_change <= -10:
            suggestion = "to think about buying a second"
        else:
            suggestion = "to buy two"

        next_first_friday = find_next_first_friday()

        summary_message = (
            f"Hello, Boss. It is time to purchase another VOO.\n"
            f"\nPrevious price: ${previous_price:.2f}\n"
            f"Current price: ${current_price:.2f}\n"
            f"The change is {percentage_change:.2f}%, therefore we suggest <i>{suggestion}</i> VOO.\n"
            f"\nThe next first Friday of the month will be {next_first_friday.strftime('%Y-%m-%d')}. You can set up a reminder on that day!"
        )
        await update.message.reply_text(summary_message, parse_mode='HTML')
    else:
        await update.message.reply_text("Failed to retrieve price data for VOO.")


async def abort_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Operation aborted. If you need help or wish to perform "
                                    "another operation, just let me know.")
    return ConversationHandler.END


def main():
    application = Application.builder().token(os.getenv('TELEGRAM_API')).build()

    add_transaction_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('add_transaction', add_transaction_start)],
        states={
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, ticker_received)],
            QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, quantity_received)],
            PURCHASE_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, purchase_date_received)],
            PURCHASE_PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, purchase_price_received)],
            CONFIRMATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, final_confirmation)]
        },
        fallbacks=[
            CommandHandler('abort', abort_conversation),
        ]
    )

    remove_transaction_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('remove_transaction', remove_transaction_start)],
        states={
            REMOVE_TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, remove_ticker_received)],
            REMOVE_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, remove_date_received)],
            REMOVE_CONFIRMATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, remove_confirmation)]
        },
        fallbacks=[CommandHandler('abort', abort_conversation)],
    )

    # Add handlers to the application
    application.add_handler(add_transaction_conv_handler)
    application.add_handler(remove_transaction_conv_handler)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("portfolio", portfolio))
    application.add_handler(CommandHandler("month_summary", month_summary))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == '__main__':
    main()
