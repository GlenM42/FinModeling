"""
Our own MCP server: exposes the finmodeling.commands layer as MCP tools.

Every tool takes `user_id` at the MCP level, even the read-only ones, so the
secretary's user_id-injection wrapper (bot.secretary.mcp_client) never needs a
per-tool exception. Run standalone via stdio:

    uv run python -m bot.mcp_server.server
"""

import asyncio
import json
from dataclasses import asdict

from dotenv import load_dotenv

load_dotenv()  # Must precede importing finmodeling.commands (engine is built from env vars at import time)

from mcp.server.fastmcp import FastMCP

from finmodeling import commands

mcp = FastMCP("finmodeling")


@mcp.tool()
async def get_portfolio_status(user_id: int) -> str:
    """Get current portfolio positions and performance (gain/loss per ticker and in aggregate)."""
    try:
        status = await commands.get_portfolio_status()
    except commands.CommandError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(status.summary)


@mcp.tool()
async def get_bank_status(user_id: int) -> str:
    """Get the user's bank balances per ticker and how many shares they can currently afford to buy."""
    status = await asyncio.to_thread(commands.get_bank_status, user_id)
    return json.dumps({"rows": status.rows})


@mcp.tool()
async def get_month_summary(user_id: int) -> str:
    """Get this month's VOO price change, the next scheduled deposit-reminder date, and bank status."""
    try:
        status = await asyncio.to_thread(commands.get_month_summary, user_id)
    except commands.CommandError as e:
        return json.dumps({"error": str(e)})
    return json.dumps({
        "voo_change_pct": status.voo_change_pct,
        "next_first_friday": status.next_first_friday,
        "bank_status": {"rows": status.bank_status.rows},
    })


@mcp.tool()
async def get_history_status(user_id: int) -> str:
    """Get portfolio value over time, current per-ticker holdings, and the exact date of
    every past purchase transaction (overall and per ticker). Use this whenever the user
    asks about purchase/transaction dates, e.g. "when did I last buy X" or "when did I buy
    X" (no charts — for charts, tell the user to run /history)."""
    try:
        status = await asyncio.to_thread(commands.get_history_status)
    except commands.CommandError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(status.summary)


@mcp.tool()
async def add_transaction(
    user_id: int,
    ticker: str,
    quantity: float,
    purchase_price: float,
    purchase_date: str,
    deduct_from_bank: bool = True,
) -> str:
    """Record a stock/ETF purchase. purchase_date must be YYYY-MM-DD. Set deduct_from_bank=False
    if the user explicitly does not want the cost taken out of their bank balance."""
    try:
        result = await asyncio.to_thread(
            commands.add_transaction_cmd,
            user_id, ticker, quantity, purchase_price, purchase_date, deduct_from_bank,
        )
    except commands.CommandError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(asdict(result))


@mcp.tool()
async def remove_transaction(user_id: int, ticker: str, purchase_date: str) -> str:
    """Remove a previously recorded transaction matching ticker + purchase_date exactly."""
    try:
        result = await asyncio.to_thread(commands.remove_transaction_cmd, user_id, ticker, purchase_date)
    except commands.CommandError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(asdict(result))


@mcp.tool()
async def deposit(user_id: int, ticker: str, amount: float) -> str:
    """Deposit cash into the user's bank balance for a given ticker, to save toward a future purchase."""
    try:
        result = await asyncio.to_thread(commands.deposit_cmd, user_id, ticker, amount)
    except commands.CommandError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(asdict(result))


if __name__ == "__main__":
    mcp.run(transport="stdio")
