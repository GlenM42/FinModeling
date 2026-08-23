import asyncio
import json
from types import SimpleNamespace

from bot.mcp_server import server
from bot.secretary.mcp_client import MCPClient, _to_openai_tool
from finmodeling import commands


def run(coro):
    return asyncio.run(coro)


def test_get_portfolio_status_returns_summary_json(monkeypatch):
    async def fake_get_portfolio_status():
        return commands.PortfolioStatus(performance_df=None, aggregate_df=None, summary={"total_value": 100.0})

    monkeypatch.setattr(commands, "get_portfolio_status", fake_get_portfolio_status)

    result = run(server.get_portfolio_status(user_id=1))

    assert json.loads(result) == {"total_value": 100.0}


def test_get_portfolio_status_returns_error_json_on_command_error(monkeypatch):
    async def fake_get_portfolio_status():
        raise commands.CommandError("boom")

    monkeypatch.setattr(commands, "get_portfolio_status", fake_get_portfolio_status)

    result = run(server.get_portfolio_status(user_id=1))

    assert json.loads(result) == {"error": "boom"}


def test_get_bank_status_returns_rows_json(monkeypatch):
    monkeypatch.setattr(
        commands, "get_bank_status",
        lambda user_id: commands.BankStatus(rows=[{"ticker": "VOO", "balance": 1.0, "price": 2.0, "suggestion": "x"}]),
    )

    result = run(server.get_bank_status(user_id=1))

    assert json.loads(result) == {"rows": [{"ticker": "VOO", "balance": 1.0, "price": 2.0, "suggestion": "x"}]}


def test_get_month_summary_returns_json(monkeypatch):
    bank_status = commands.BankStatus(rows=[])
    monkeypatch.setattr(
        commands, "get_month_summary",
        lambda user_id: commands.MonthSummary(
            voo_change_pct=1.23, next_first_friday="2024-02-02", bank_status=bank_status,
        ),
    )

    result = run(server.get_month_summary(user_id=1))

    assert json.loads(result) == {
        "voo_change_pct": 1.23,
        "next_first_friday": "2024-02-02",
        "bank_status": {"rows": []},
    }


def test_get_month_summary_returns_error_json_on_command_error(monkeypatch):
    def boom(user_id):
        raise commands.CommandError("no VOO data")

    monkeypatch.setattr(commands, "get_month_summary", boom)

    result = run(server.get_month_summary(user_id=1))

    assert json.loads(result) == {"error": "no VOO data"}


def test_get_history_status_returns_summary_json(monkeypatch):
    monkeypatch.setattr(
        commands, "get_history_status",
        lambda: commands.HistoryStatus(
            prices_df=None, holdings_df=None, total_value_series=None,
            events_by_date={}, events_per_ticker={}, summary={"current_total_value": 5.0, "positions": []},
        ),
    )

    result = run(server.get_history_status(user_id=1))

    assert json.loads(result) == {"current_total_value": 5.0, "positions": []}


def test_add_transaction_returns_result_json(monkeypatch):
    def fake_add(user_id, ticker, quantity, purchase_price, purchase_date, deduct_from_bank):
        return commands.TransactionResult(
            ticker=ticker, quantity=quantity, purchase_price=purchase_price,
            purchase_date=purchase_date, total_cost=quantity * purchase_price,
            bank_adjustment=None, message="ok",
        )

    monkeypatch.setattr(commands, "add_transaction_cmd", fake_add)

    result = run(server.add_transaction(
        user_id=1, ticker="voo", quantity=2, purchase_price=100.0,
        purchase_date="2024-01-01", deduct_from_bank=False,
    ))

    payload = json.loads(result)
    assert payload["ticker"] == "voo"
    assert payload["total_cost"] == 200.0
    assert payload["bank_adjustment"] is None


def test_add_transaction_returns_error_json_on_command_error(monkeypatch):
    def boom(*a, **k):
        raise commands.CommandError("bad quantity")

    monkeypatch.setattr(commands, "add_transaction_cmd", boom)

    result = run(server.add_transaction(
        user_id=1, ticker="VOO", quantity=-1, purchase_price=100.0, purchase_date="2024-01-01",
    ))

    assert json.loads(result) == {"error": "bad quantity"}


def test_remove_transaction_returns_result_json(monkeypatch):
    monkeypatch.setattr(
        commands, "remove_transaction_cmd",
        lambda user_id, ticker, purchase_date: commands.RemovalResult(message=f"removed {ticker}"),
    )

    result = run(server.remove_transaction(user_id=1, ticker="VOO", purchase_date="2024-01-01"))

    assert json.loads(result) == {"message": "removed VOO"}


def test_deposit_returns_result_json(monkeypatch):
    monkeypatch.setattr(
        commands, "deposit_cmd",
        lambda user_id, ticker, amount: commands.DepositResult(new_balance=150.0, message="deposited"),
    )

    result = run(server.deposit(user_id=1, ticker="VOO", amount=100.0))

    assert json.loads(result) == {"new_balance": 150.0, "message": "deposited"}


def test_deposit_returns_error_json_on_command_error(monkeypatch):
    def boom(user_id, ticker, amount):
        raise commands.CommandError("bad amount")

    monkeypatch.setattr(commands, "deposit_cmd", boom)

    result = run(server.deposit(user_id=1, ticker="VOO", amount=-5))

    assert json.loads(result) == {"error": "bad amount"}


# ---------------------------------------------------------------------------
# Security regression: the LLM must never see or control user_id
# ---------------------------------------------------------------------------

def _fake_tool(name, properties, required):
    return SimpleNamespace(
        name=name,
        description=f"{name} description",
        inputSchema={"type": "object", "properties": properties, "required": required, "title": f"{name}Arguments"},
    )


def test_to_openai_tool_strips_user_id_from_schema():
    tool = _fake_tool(
        "add_transaction",
        {"user_id": {"type": "integer"}, "ticker": {"type": "string"}},
        ["user_id", "ticker"],
    )

    openai_tool = _to_openai_tool(tool)

    params = openai_tool["function"]["parameters"]
    assert "user_id" not in params["properties"]
    assert "user_id" not in params["required"]
    assert "ticker" in params["properties"]
    assert params["required"] == ["ticker"]


def test_to_openai_tool_handles_read_only_tool_with_only_user_id():
    tool = _fake_tool("get_bank_status", {"user_id": {"type": "integer"}}, ["user_id"])

    openai_tool = _to_openai_tool(tool)

    params = openai_tool["function"]["parameters"]
    assert params["properties"] == {}
    assert params["required"] == []


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, dict(args)))
        return SimpleNamespace(content=[SimpleNamespace(text='{"ok": true}')])


def test_call_tool_always_overwrites_a_spoofed_user_id():
    client = MCPClient()
    fake_session = _FakeSession()
    client._session = fake_session

    result = run(client.call_tool("add_transaction", {"ticker": "VOO", "user_id": 999}, user_id=42))

    assert result == '{"ok": true}'
    assert fake_session.calls == [("add_transaction", {"ticker": "VOO", "user_id": 42})]


def test_call_tool_injects_user_id_even_when_absent_from_llm_args():
    client = MCPClient()
    fake_session = _FakeSession()
    client._session = fake_session

    run(client.call_tool("get_bank_status", {}, user_id=7))

    assert fake_session.calls == [("get_bank_status", {"user_id": 7})]
