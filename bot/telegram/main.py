import logging
import os

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()  # Must precede importing finmodeling.database (engine is built from env vars at import time)

from telegram.error import NetworkError
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, filters, ContextTypes

from finmodeling.database import create_bank_table, retrieve_admins
from finmodeling.transactions import add_transaction_start, ticker_received, quantity_received, \
    purchase_date_received, purchase_price_received, final_confirmation, bank_deduct_response, \
    remove_transaction_start, remove_ticker_received, remove_date_received, remove_confirmation
from finmodeling.bank import (
    deposit_start, deposit_ticker_received, deposit_amount_received,
    DEPOSIT_TICKER, DEPOSIT_AMOUNT
)

from bot.secretary.mcp_client import MCPClient
from bot.telegram.handlers import start, portfolio, bank_show, month_summary, history, abort_conversation
from bot.telegram.secretary_handler import handle_free_text

# Define states
TICKER, CONFIRM_TICKER, QUANTITY, PURCHASE_DATE, PURCHASE_PRICE, CONFIRMATION = range(6)
REMOVE_TICKER, REMOVE_DATE, REMOVE_CONFIRMATION = range(6, 9)
BANK_DEDUCT = 9

ADMIN_IDS = retrieve_admins()


async def post_init(application: Application) -> None:
    application.bot_data["mcp_client"] = await MCPClient().start()
    logger.info("MCP client started")


async def post_shutdown(application: Application) -> None:
    mcp_client = application.bot_data.get("mcp_client")
    if mcp_client is not None:
        await mcp_client.close()


def main():
    create_bank_table()

    application = (
        Application.builder()
        .token(os.getenv('TELEGRAM_API'))
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

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

    # Added LAST, same handler group: PTB tries handlers in registration order and
    # stops at the first match, so a user mid-ConversationHandler keeps being routed
    # to that conversation — only idle free text (no leading slash) reaches the
    # AI secretary.
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & filters.User(user_id=ADMIN_IDS), handle_free_text)
    )

    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        if isinstance(context.error, NetworkError):
            logger.warning(f"Transient network error (retrying): {context.error}")
            return
        logger.error("Unexpected error", exc_info=context.error)

    application.add_error_handler(error_handler)

    logger.info("Bot is up!")
    application.run_polling()


if __name__ == '__main__':
    main()
