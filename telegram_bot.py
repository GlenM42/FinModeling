import asyncio
import os
from dotenv import load_dotenv
import logging

import pandas as pd

from finmodeling.helpers import find_current_price, find_previous_price
from finmodeling.plots import plot_portfolio_history_by_ticker, plot_portfolio_history_total, plot_portfolio_performance, plot_prices_with_purchase_markers, show_portfolio_as_image

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv() # We should import .env before the database (there is connection to db based on ENVs)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, filters, ContextTypes

from finmodeling.calendar import find_next_first_friday
from finmodeling.database import bank_get_all, create_bank_table, initialize_portfolio, retrieve_admins
from finmodeling.management import calculate_aggregate_performance, calculate_performance, compute_portfolio_history, \
    build_purchase_events
from finmodeling.transactions import add_transaction_start, ticker_received, quantity_received, \
    purchase_date_received, purchase_price_received, final_confirmation, bank_deduct_response, \
    remove_transaction_start, remove_ticker_received, remove_date_received, remove_confirmation
from finmodeling.bank import (
    deposit_start, deposit_ticker_received, deposit_amount_received,
    DEPOSIT_TICKER, DEPOSIT_AMOUNT
)

# Define states
TICKER, CONFIRM_TICKER, QUANTITY, PURCHASE_DATE, PURCHASE_PRICE, CONFIRMATION = range(6)
REMOVE_TICKER, REMOVE_DATE, REMOVE_CONFIRMATION = range(6, 9)
BANK_DEDUCT = 9

ADMIN_IDS = retrieve_admins()
FRACTIONABLE_ASSETS = ["VOO"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message

    response = (
        f"Hello, Mr. {message.from_user.full_name if message.from_user else 'N/A'}! "
        f"I would like to help you with your portfolio management."
    )

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    portfolio_df = await asyncio.to_thread(initialize_portfolio)
    portfolio_perf = await asyncio.to_thread(calculate_performance, portfolio_df)
    aggr_performance = await asyncio.to_thread(calculate_aggregate_performance, portfolio_perf)

    # Use the modified functions to save images
    await asyncio.to_thread(show_portfolio_as_image, portfolio_perf, 'portfolio_table.png')
    await asyncio.to_thread(plot_portfolio_performance, aggr_performance, 'portfolio_performance.png')

    # Send the images
    with open('portfolio_table.png', 'rb') as file:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=file)
    with open('portfolio_performance.png', 'rb') as file:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=file)

    # Delete the images after sending
    os.remove('portfolio_table.png')
    os.remove('portfolio_performance.png')


async def bank_show(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    rows = await asyncio.to_thread(bank_get_all, user_id)

    if not rows:
        await update.message.reply_text("Your bank is empty. Use /deposit to add funds.")
        return

    lines = ["<b>Bank balances:</b>\n"]
    for ticker, balance, _ in rows:
        try:
            current_price = await asyncio.to_thread(find_current_price, ticker)
        except Exception as e:
            logger.error(f"Could not get the current price for {ticker}: {e}", exc_info=True)
            lines.append(f"<b>{ticker}</b>: ${balance:.2f} (price unavailable)")
            continue

        if balance < 0:
            lines.append(
                f"<b>{ticker}</b>: ${balance:.2f} | Price: ${current_price:.2f} "
                f"→ wait for positive balance"
            )
            continue

        if ticker in FRACTIONABLE_ASSETS:
            shares = round(balance / current_price, 2)
            lines.append(
                f"<b>{ticker}</b>: ${balance:.2f} | Price: ${current_price:.2f} "
                f"→ buy {shares} share(s)"
            )
        else:
            whole_shares = int(balance // current_price)
            remainder = round(balance - whole_shares * current_price, 2)
            lines.append(
                f"<b>{ticker}</b>: ${balance:.2f} | Price: ${current_price:.2f} "
                f"→ buy {whole_shares} share(s), ${remainder:.2f} remaining"
            )

    await update.message.reply_text("\n".join(lines), parse_mode='HTML')


async def month_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Fetching data... Please wait.")

    try:
        current_price = await asyncio.to_thread(find_current_price, "VOO")
        previous_price = await asyncio.to_thread(find_previous_price, "VOO")
    except Exception as e:
        logger.error(f"Could not fetch the info for VOO: {e}", exc_info=True)
        await update.message.reply_text("Failed to retrieve price data for VOO.")
        return

    percentage_change = round(((current_price - previous_price) / previous_price) * 100, 2)

    next_first_friday = find_next_first_friday()

    summary_message = (
        f"Hello, Boss. It is time to make another deposit.\n"
        f"\nFor your information, VOO rose {percentage_change:.2f}%."
        f"\nThe next first Friday of the month will be {next_first_friday.strftime('%Y-%m-%d')}. You can set up a reminder on that day."
    )
    await update.message.reply_text(summary_message, parse_mode='HTML')

    # Since bank exists for the monthly purchases, let's just use it
    # in the monthly summary command, and not create something new.
    await bank_show(update, context)

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    portfolio_df = await asyncio.to_thread(initialize_portfolio)
    portfolio_df['purchase_date'] = pd.to_datetime(portfolio_df['purchase_date']).dt.tz_localize(None)

    try:
        prices_df, holdings_df, total_value = await asyncio.to_thread(
            compute_portfolio_history,
            portfolio_df,
        )
    except Exception as e:
        logger.error(f"Could not fetch portfolio history: {e}", exc_info=True)
        response = "I am sorry, I could not fetch the portfolio history."
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
        return

    events_by_date, events_per_ticker = build_purchase_events(portfolio_df, prices_df.index)

    # Generate and save plots
    await asyncio.to_thread(
        plot_portfolio_history_total,
        total_value,
        'portfolio_history_total.png',
        events_by_date=events_by_date
    )
    await asyncio.to_thread(
        plot_portfolio_history_by_ticker,
        holdings_df,
        prices_df,
        'portfolio_history_by_ticker.png',
        events_per_ticker=events_per_ticker
    )

    await asyncio.to_thread(
        plot_prices_with_purchase_markers,
        prices_df,
        events_per_ticker,
        'portfolio_prices_with_buys.png',
        normalize=False  # raw dollar prices
    )

    await asyncio.to_thread(
        plot_prices_with_purchase_markers,
        prices_df,
        events_per_ticker,
        'portfolio_prices_with_buys2.png',
        normalize=True  # percentage rises
    )

    # Send the images
    with open('portfolio_history_total.png', 'rb') as f1:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f1)
    with open('portfolio_history_by_ticker.png', 'rb') as f2:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f2)
    with open('portfolio_prices_with_buys.png', 'rb') as f3:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f3)
    with open('portfolio_prices_with_buys2.png', 'rb') as f4:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f4)

    # Clean up generated images if they exist
    for path in ['portfolio_history_total.png', 'portfolio_history_by_ticker.png',
                    'portfolio_prices_with_buys.png', 'portfolio_prices_with_buys2.png']:
        os.remove(path)


async def abort_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Operation aborted. If you need help or wish to perform "
                                    "another operation, just let me know.")
    return ConversationHandler.END


def main():
    create_bank_table()

    application = Application.builder().token(os.getenv('TELEGRAM_API')).build()

    add_transaction_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('add_transaction', add_transaction_start, filters=filters.User(user_id=ADMIN_IDS))],
        states={
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, ticker_received)],
            QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, quantity_received)],
            PURCHASE_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, purchase_date_received)],
            PURCHASE_PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, purchase_price_received)],
            CONFIRMATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, final_confirmation)],
            BANK_DEDUCT: [MessageHandler(filters.TEXT & ~filters.COMMAND, bank_deduct_response)],
        },
        fallbacks=[
            CommandHandler('abort', abort_conversation),
        ],
    )

    remove_transaction_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('remove_transaction', remove_transaction_start, filters=filters.User(user_id=ADMIN_IDS))],
        states={
            REMOVE_TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, remove_ticker_received)],
            REMOVE_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, remove_date_received)],
            REMOVE_CONFIRMATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, remove_confirmation)]
        },
        fallbacks=[CommandHandler('abort', abort_conversation)],
    )

    deposit_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('deposit', deposit_start, filters=filters.User(user_id=ADMIN_IDS))],
        states={
            DEPOSIT_TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, deposit_ticker_received)],
            DEPOSIT_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, deposit_amount_received)],
        },
        fallbacks=[CommandHandler('abort', abort_conversation)],
    )

    # Add handlers to the application
    application.add_handler(add_transaction_conv_handler)
    application.add_handler(remove_transaction_conv_handler)
    application.add_handler(deposit_conv_handler)
    application.add_handler(CommandHandler("bank", bank_show, filters=filters.User(user_id=ADMIN_IDS)))
    application.add_handler(CommandHandler("start", start, filters=filters.User(user_id=ADMIN_IDS)))
    application.add_handler(CommandHandler("portfolio", portfolio, filters=filters.User(user_id=ADMIN_IDS)))
    application.add_handler(CommandHandler("month_summary", month_summary, filters=filters.User(user_id=ADMIN_IDS)))
    application.add_handler(CommandHandler("history", history, filters=filters.User(user_id=ADMIN_IDS)))

    logger.info("Bot is up!")
    application.run_polling()


if __name__ == '__main__':
    main()
