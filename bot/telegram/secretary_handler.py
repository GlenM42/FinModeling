import logging

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from bot.secretary import agent

logger = logging.getLogger(__name__)


async def handle_free_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    mcp = context.application.bot_data["mcp_client"]

    try:
        reply = await agent.handle_user_message(user_id, update.message.text, mcp)
    except Exception:
        logger.exception("Secretary failed for user %s", user_id)
        reply = "Sorry, I hit an error processing that — try again or use a slash command."

    try:
        await update.message.reply_text(reply, parse_mode="HTML")
    except BadRequest:
        # The model is instructed to use Telegram HTML, but a cheap model can
        # still emit invalid markup (e.g. a stray '<' or unescaped '&'). Fall
        # back to plain text rather than losing the reply.
        logger.warning("Secretary reply failed HTML parsing for user %s, resending as plain text", user_id)
        await update.message.reply_text(reply)
