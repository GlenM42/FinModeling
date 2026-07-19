import asyncio
import os

from telegram import Update
from telegram.ext import ConversationHandler, ContextTypes

from finmodeling import commands
from finmodeling.plots import (
    plot_portfolio_history_by_ticker,
    plot_portfolio_history_total,
    plot_portfolio_performance,
    plot_prices_with_purchase_markers,
    show_portfolio_as_image,
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message

    response = (
        f"Hello, Mr. {message.from_user.full_name if message.from_user else 'N/A'}! "
        f"I would like to help you with your portfolio management."
    )

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    filenames = [f"portfolio_{uid}.png", f"portfolio_perf_{uid}.png"]

    try:
        status = await commands.get_portfolio_status()
    except commands.CommandError as e:
        await update.message.reply_text(str(e))
        return

    try:
        await asyncio.gather(
            asyncio.to_thread(show_portfolio_as_image, status.performance_df, filenames[0]),
            asyncio.to_thread(plot_portfolio_performance, status.aggregate_df, filenames[1]),
        )

        for path in filenames:
            with open(path, 'rb') as f:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f)
    finally:
        for path in filenames:
            if os.path.exists(path):
                os.remove(path)


async def _send_bank_status(update: Update, bank_status) -> None:
    if not bank_status.rows:
        await update.message.reply_text("Your bank is empty. Use /deposit to add funds.")
        return

    lines = ["<b>Bank balances:</b>\n"]
    for row in bank_status.rows:
        if row["price"] is None:
            lines.append(f"<b>{row['ticker']}</b>: ${row['balance']:.2f} (price unavailable)")
        else:
            lines.append(
                f"<b>{row['ticker']}</b>: ${row['balance']:.2f} | Price: ${row['price']:.2f} "
                f"→ {row['suggestion']}"
            )

    await update.message.reply_text("\n".join(lines), parse_mode='HTML')


async def bank_show(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    status = await asyncio.to_thread(commands.get_bank_status, user_id)
    await _send_bank_status(update, status)


async def month_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Fetching data... Please wait.")
    user_id = update.effective_user.id

    try:
        status = await asyncio.to_thread(commands.get_month_summary, user_id)
    except commands.CommandError as e:
        await update.message.reply_text(str(e))
        return

    summary_message = (
        f"Hello, Boss. It is time to make another deposit.\n"
        f"\nFor your information, VOO rose {status.voo_change_pct:.2f}%."
        f"\nThe next first Friday of the month will be {status.next_first_friday}. You can set up a reminder on that day."
    )
    await update.message.reply_text(summary_message, parse_mode='HTML')

    # Reuse the bank status already fetched by get_month_summary instead of querying twice
    await _send_bank_status(update, status.bank_status)


async def history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    filenames = [
        f"portfolio_history_total_{uid}.png",
        f"portfolio_history_by_ticker_{uid}.png",
        f"portfolio_prices_with_buys_{uid}.png",
        f"portfolio_prices_with_buys2_{uid}.png",
    ]

    try:
        status = await asyncio.to_thread(commands.get_history_status)
    except commands.CommandError as e:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=str(e))
        return

    try:
        await asyncio.gather(
            asyncio.to_thread(
                plot_portfolio_history_total, status.total_value_series, filenames[0],
                events_by_date=status.events_by_date,
            ),
            asyncio.to_thread(
                plot_portfolio_history_by_ticker, status.holdings_df, status.prices_df, filenames[1],
                events_per_ticker=status.events_per_ticker,
            ),
            asyncio.to_thread(
                plot_prices_with_purchase_markers, status.prices_df, status.events_per_ticker, filenames[2],
                normalize=False,
            ),
            asyncio.to_thread(
                plot_prices_with_purchase_markers, status.prices_df, status.events_per_ticker, filenames[3],
                normalize=True,
            ),
        )

        for path in filenames:
            with open(path, 'rb') as f:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f)
    finally:
        for path in filenames:
            if os.path.exists(path):
                os.remove(path)


async def abort_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Operation aborted. If you need help or wish to perform "
                                    "another operation, just let me know.")
    return ConversationHandler.END
