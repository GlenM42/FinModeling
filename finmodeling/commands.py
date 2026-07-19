"""
Shared, Telegram/AI-agnostic command layer.

This is the single source of truth for "what does /portfolio, /bank,
/add_transaction, /remove_transaction, /deposit, /month_summary and /history
actually do." Both the Telegram handlers/conversations and the MCP tools call
only into this module (never directly into database.py/management.py), so
behavior can't silently fork between the slash-command path and the AI
secretary path.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from finmodeling.calendar import find_next_first_friday
from finmodeling.database import (
    add_transaction as db_add_transaction,
    bank_deduct,
    bank_deposit,
    bank_get_all,
    bank_get_balance,
    initialize_portfolio,
    remove_transactions,
)
from finmodeling.helpers import find_current_price, find_previous_price
from finmodeling.management import (
    build_purchase_events,
    calculate_aggregate_performance,
    calculate_performance,
    compute_portfolio_history,
)

FRACTIONABLE_ASSETS = ["VOO"]


class CommandError(Exception):
    """Expected, user-facing failure (bad input, external API failure, etc).
    Callers should show str(e) verbatim rather than a generic error."""


# ---------------------------------------------------------------------------
# Read-only status
# ---------------------------------------------------------------------------

@dataclass
class PortfolioStatus:
    performance_df: pd.DataFrame
    aggregate_df: pd.DataFrame
    summary: dict


async def get_portfolio_status() -> PortfolioStatus:
    portfolio_df = await asyncio.to_thread(initialize_portfolio)
    try:
        performance_df = await calculate_performance(portfolio_df)
    except RuntimeError as e:
        raise CommandError("Failed to calculate portfolio performance.") from e

    aggregate_df = await asyncio.to_thread(calculate_aggregate_performance, performance_df)

    total_row = performance_df.iloc[-1]
    positions = [
        {
            "ticker": row["ticker"],
            "quantity": float(row["quantity"]),
            "total_return": float(row["total return"]),
            "percentage_return": float(row["percentage return"]),
        }
        for _, row in aggregate_df.iloc[:-1].iterrows()
    ]
    summary = {
        "positions": positions,
        "total_value": float(total_row["current value"]),
        "total_investment": float(total_row["investment value"]),
        "total_return": float(total_row["total return"]),
        "total_percentage_return": float(total_row["percentage return"]),
    }
    return PortfolioStatus(performance_df=performance_df, aggregate_df=aggregate_df, summary=summary)


@dataclass
class BankStatus:
    rows: list[dict]


def get_bank_status(user_id: int) -> BankStatus:
    rows = []
    for ticker, balance, _ in bank_get_all(user_id):
        balance = float(balance)
        try:
            price = find_current_price(ticker)
        except Exception:
            rows.append({"ticker": ticker, "balance": balance, "price": None, "suggestion": "price unavailable"})
            continue

        if balance < 0:
            suggestion = "wait for positive balance"
        elif ticker in FRACTIONABLE_ASSETS:
            suggestion = f"buy {round(balance / price, 2)} share(s)"
        else:
            whole_shares = int(balance // price)
            remainder = round(balance - whole_shares * price, 2)
            suggestion = f"buy {whole_shares} share(s), ${remainder:.2f} remaining"

        rows.append({"ticker": ticker, "balance": balance, "price": float(price), "suggestion": suggestion})

    return BankStatus(rows=rows)


@dataclass
class MonthSummary:
    voo_change_pct: float
    next_first_friday: str
    bank_status: BankStatus


def get_month_summary(user_id: int) -> MonthSummary:
    try:
        current_price = find_current_price("VOO")
        previous_price = find_previous_price("VOO")
    except Exception as e:
        raise CommandError("Failed to retrieve price data for VOO.") from e

    voo_change_pct = round(((current_price - previous_price) / previous_price) * 100, 2)
    next_first_friday = find_next_first_friday().strftime("%Y-%m-%d")

    return MonthSummary(
        voo_change_pct=voo_change_pct,
        next_first_friday=next_first_friday,
        bank_status=get_bank_status(user_id),
    )


@dataclass
class HistoryStatus:
    prices_df: pd.DataFrame
    holdings_df: pd.DataFrame
    total_value_series: pd.Series
    events_by_date: dict
    events_per_ticker: dict
    summary: dict


def get_history_status() -> HistoryStatus:
    portfolio_df = initialize_portfolio()
    portfolio_df["purchase_date"] = pd.to_datetime(portfolio_df["purchase_date"]).dt.tz_localize(None)

    try:
        prices_df, holdings_df, total_value = compute_portfolio_history(portfolio_df)
    except Exception as e:
        raise CommandError("I am sorry, I could not fetch the portfolio history.") from e

    events_by_date, events_per_ticker = build_purchase_events(portfolio_df, prices_df.index)

    last_holdings = holdings_df.iloc[-1]
    last_prices = prices_df.iloc[-1]
    positions = [
        {
            "ticker": ticker,
            "shares": float(last_holdings[ticker]),
            "value": float(last_holdings[ticker] * last_prices[ticker]),
        }
        for ticker in holdings_df.columns
    ]

    purchase_events = [
        {"date": pd.Timestamp(dt).strftime("%Y-%m-%d"), "tickers": label}
        for dt, label in sorted(events_by_date.items())
    ]
    purchase_dates_by_ticker = {
        ticker: [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in sorted(dates)]
        for ticker, dates in events_per_ticker.items()
    }

    summary = {
        "history_start": prices_df.index[0].strftime("%Y-%m-%d"),
        "history_end": prices_df.index[-1].strftime("%Y-%m-%d"),
        "current_total_value": float(total_value.iloc[-1]),
        "positions": positions,
        "purchase_events": purchase_events,
        "purchase_dates_by_ticker": purchase_dates_by_ticker,
    }

    return HistoryStatus(
        prices_df=prices_df,
        holdings_df=holdings_df,
        total_value_series=total_value,
        events_by_date=events_by_date,
        events_per_ticker=events_per_ticker,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

@dataclass
class BankAdjustment:
    new_balance: float
    warning: str | None


def apply_bank_deduction(user_id: int, ticker: str, total_cost: float) -> BankAdjustment:
    """The one place 'deduct total_cost, warn if it goes negative, return new
    balance' lives. Called both by add_transaction_cmd's one-shot path
    (deduct_from_bank=True) and by the /add_transaction conversation's separate
    bank-deduct step, so the rule can never fork between the two entry points."""
    current_balance = bank_get_balance(user_id, ticker)
    warning = None
    if total_cost > current_balance:
        warning = (
            f"purchase (${total_cost:.2f}) exceeds current bank balance "
            f"(${current_balance:.2f}) — balance will go negative."
        )
    bank_deduct(user_id, ticker, total_cost)
    new_balance = bank_get_balance(user_id, ticker)
    return BankAdjustment(new_balance=new_balance, warning=warning)


@dataclass
class TransactionResult:
    ticker: str
    quantity: float
    purchase_price: float
    purchase_date: str
    total_cost: float
    bank_adjustment: BankAdjustment | None
    message: str


def add_transaction_cmd(
    user_id: int,
    ticker: str,
    quantity: float,
    purchase_price: float,
    purchase_date: str,
    deduct_from_bank: bool = True,
) -> TransactionResult:
    ticker = ticker.strip().upper()

    try:
        quantity = float(quantity)
        if quantity <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise CommandError("Please enter a valid positive quantity.")

    try:
        purchase_price = float(purchase_price)
        if purchase_price <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise CommandError("Please enter a valid positive price.")

    try:
        datetime.strptime(purchase_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise CommandError("purchase_date must be in YYYY-MM-DD format.")

    db_add_transaction(user_id, ticker, quantity, purchase_price, purchase_date)

    total_cost = round(quantity * purchase_price, 2)
    bank_adjustment = apply_bank_deduction(user_id, ticker, total_cost) if deduct_from_bank else None

    message = (
        f"Recorded {quantity} share(s) of {ticker} at ${purchase_price:.2f} "
        f"on {purchase_date} (total ${total_cost:.2f})."
    )
    if bank_adjustment:
        message += f" {ticker} bank balance: ${bank_adjustment.new_balance:.2f}."
        if bank_adjustment.warning:
            message += f" Note: {bank_adjustment.warning}"

    return TransactionResult(
        ticker=ticker,
        quantity=quantity,
        purchase_price=purchase_price,
        purchase_date=purchase_date,
        total_cost=total_cost,
        bank_adjustment=bank_adjustment,
        message=message,
    )


@dataclass
class RemovalResult:
    message: str


def remove_transaction_cmd(user_id: int, ticker: str, purchase_date: str) -> RemovalResult:
    ticker = ticker.strip().upper()
    remove_transactions(user_id, ticker, purchase_date)
    return RemovalResult(message=f"Removed transaction(s) for {ticker} on {purchase_date}.")


@dataclass
class DepositResult:
    new_balance: float
    message: str


def deposit_cmd(user_id: int, ticker: str, amount: float) -> DepositResult:
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise CommandError("Please enter a valid positive dollar amount.")

    ticker = ticker.strip().upper()
    bank_deposit(user_id, ticker, amount)
    new_balance = bank_get_balance(user_id, ticker)
    return DepositResult(
        new_balance=new_balance,
        message=f"Deposited ${amount:.2f} into {ticker}. New balance: ${new_balance:.2f}.",
    )
