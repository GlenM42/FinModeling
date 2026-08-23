"""
The AI portfolio secretary: a bounded OpenAI tool-calling loop backed by our
own MCP tools. Never lets an OpenAI/MCP exception escape — tool failures are
fed back to the model as a JSON error string, and the loop always returns a
string (falling back to an apology if MAX_TOOL_HOPS is exhausted).
"""

import json
import logging
from datetime import date

from openai import AsyncOpenAI

from bot.secretary import config, sessions
from bot.secretary.mcp_client import MCPClient

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


def _system_prompt() -> str:
    return (
        "You are the user's portfolio secretary for a personal finance Telegram bot.\n"
        "Never use emojis.\n"
        "Use the available tools to check real data — never guess balances, prices, or "
        "portfolio positions. When asked about purchase/transaction dates (e.g. \"when did "
        "I last buy X\"), use get_history_status — it has exact per-ticker purchase dates.\n"
        "When the user describes a transaction or deposit in natural language (e.g. \"I "
        "bought 3 shares of VOO at $500 yesterday\"), extract the fields and call the "
        "tool directly in one shot; only ask a clarifying question if a required field "
        "is truly missing or ambiguous.\n"
        f"Resolve relative dates (\"yesterday\", \"today\") to YYYY-MM-DD using the "
        f"current date: {date.today().isoformat()}.\n"
        "Format replies using Telegram's HTML message formatting ONLY — the tags <b>, <i>, "
        "<u>, <s>, <code>, and <pre> are allowed; nothing else. Never use Markdown syntax "
        "(no **, no _, no #, no bullet dashes as list markers). Do not write a literal "
        "'<' or '&' character outside of one of the allowed tags."
    )


async def handle_user_message(user_id: int, user_text: str, mcp: MCPClient) -> str:
    session = sessions.get_session(user_id, _system_prompt())
    sessions.append_user(session, user_text)

    tools = mcp.tools_for_llm()

    for _ in range(config.MAX_TOOL_HOPS):
        response = await _client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=sessions.get_context_for_api(session),
            tools=tools,
        )
        choice = response.choices[0]

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            reply = choice.message.content or ""
            sessions.append_raw(session, {"role": "assistant", "content": reply})
            return reply

        sessions.append_raw(session, choice.message.model_dump(exclude_none=True))

        for tool_call in choice.message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
                result_text = await mcp.call_tool(tool_call.function.name, args, user_id=user_id)
            except Exception:
                logger.exception("MCP tool call failed: %s", tool_call.function.name)
                result_text = json.dumps({"error": "tool call failed"})

            sessions.append_raw(session, {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

    fallback = "Sorry, I'm having trouble completing that — try rephrasing, or use the slash commands directly."
    sessions.append_raw(session, {"role": "assistant", "content": fallback})
    return fallback
