import pytest

from finmodeling import commands


def test_add_transaction_cmd_rejects_non_positive_quantity(monkeypatch):
    monkeypatch.setattr(commands, "db_add_transaction", lambda *a, **k: None)
    with pytest.raises(commands.CommandError):
        commands.add_transaction_cmd(1, "VOO", 0, 100.0, "2024-01-01", deduct_from_bank=False)


def test_add_transaction_cmd_rejects_non_positive_price(monkeypatch):
    monkeypatch.setattr(commands, "db_add_transaction", lambda *a, **k: None)
    with pytest.raises(commands.CommandError):
        commands.add_transaction_cmd(1, "VOO", 1, 0, "2024-01-01", deduct_from_bank=False)


def test_add_transaction_cmd_rejects_bad_date(monkeypatch):
    monkeypatch.setattr(commands, "db_add_transaction", lambda *a, **k: None)
    with pytest.raises(commands.CommandError):
        commands.add_transaction_cmd(1, "VOO", 1, 100.0, "not-a-date", deduct_from_bank=False)


def test_add_transaction_cmd_records_without_deduction(monkeypatch):
    calls = {}
    monkeypatch.setattr(commands, "db_add_transaction", lambda *a, **k: calls.setdefault("add", a))

    def fail_if_called(*a, **k):
        raise AssertionError("apply_bank_deduction should not be called")

    monkeypatch.setattr(commands, "apply_bank_deduction", fail_if_called)

    result = commands.add_transaction_cmd(1, "voo", 2, 100.0, "2024-01-01", deduct_from_bank=False)

    assert calls["add"] == (1, "VOO", 2.0, 100.0, "2024-01-01")
    assert result.bank_adjustment is None
    assert result.total_cost == 200.0


def test_add_transaction_cmd_deducts_when_requested(monkeypatch):
    monkeypatch.setattr(commands, "db_add_transaction", lambda *a, **k: None)
    seen = {}

    def fake_apply(user_id, ticker, total_cost):
        seen["args"] = (user_id, ticker, total_cost)
        return commands.BankAdjustment(new_balance=50.0, warning=None)

    monkeypatch.setattr(commands, "apply_bank_deduction", fake_apply)

    result = commands.add_transaction_cmd(1, "VOO", 2, 100.0, "2024-01-01", deduct_from_bank=True)

    assert seen["args"] == (1, "VOO", 200.0)
    assert result.bank_adjustment.new_balance == 50.0


def test_apply_bank_deduction_warns_when_going_negative(monkeypatch):
    monkeypatch.setattr(commands, "bank_get_balance", lambda user_id, ticker: 50.0)
    monkeypatch.setattr(commands, "bank_deduct", lambda *a, **k: None)

    adjustment = commands.apply_bank_deduction(1, "VOO", 200.0)

    assert adjustment.warning is not None
    assert "negative" in adjustment.warning


def test_apply_bank_deduction_no_warning_when_sufficient(monkeypatch):
    monkeypatch.setattr(commands, "bank_get_balance", lambda user_id, ticker: 500.0)
    monkeypatch.setattr(commands, "bank_deduct", lambda *a, **k: None)

    adjustment = commands.apply_bank_deduction(1, "VOO", 200.0)

    assert adjustment.warning is None


def test_deposit_cmd_rejects_non_positive_amount():
    with pytest.raises(commands.CommandError):
        commands.deposit_cmd(1, "VOO", "0")


def test_deposit_cmd_rejects_non_numeric_amount():
    with pytest.raises(commands.CommandError):
        commands.deposit_cmd(1, "VOO", "not-a-number")


def test_deposit_cmd_deposits_and_reports_new_balance(monkeypatch):
    monkeypatch.setattr(commands, "bank_deposit", lambda *a, **k: None)
    monkeypatch.setattr(commands, "bank_get_balance", lambda user_id, ticker: 150.0)

    result = commands.deposit_cmd(1, "voo", "100")

    assert result.new_balance == 150.0
    assert "150.00" in result.message


def test_remove_transaction_cmd(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        commands, "remove_transactions",
        lambda user_id, ticker, date: seen.setdefault("args", (user_id, ticker, date)),
    )

    result = commands.remove_transaction_cmd(1, "voo", "2024-01-01")

    assert seen["args"] == (1, "VOO", "2024-01-01")
    assert "VOO" in result.message


def test_get_bank_status_fractionable_asset(monkeypatch):
    monkeypatch.setattr(commands, "bank_get_all", lambda user_id: [("VOO", 150.0, None)])
    monkeypatch.setattr(commands, "find_current_price", lambda ticker: 100.0)

    status = commands.get_bank_status(1)

    assert status.rows[0]["suggestion"] == "buy 1.5 share(s)"


def test_get_bank_status_non_fractionable_asset(monkeypatch):
    monkeypatch.setattr(commands, "bank_get_all", lambda user_id: [("AAPL", 250.0, None)])
    monkeypatch.setattr(commands, "find_current_price", lambda ticker: 100.0)

    status = commands.get_bank_status(1)

    assert status.rows[0]["suggestion"] == "buy 2 share(s), $50.00 remaining"


def test_get_bank_status_negative_balance(monkeypatch):
    monkeypatch.setattr(commands, "bank_get_all", lambda user_id: [("AAPL", -10.0, None)])
    monkeypatch.setattr(commands, "find_current_price", lambda ticker: 100.0)

    status = commands.get_bank_status(1)

    assert status.rows[0]["suggestion"] == "wait for positive balance"


def test_get_bank_status_price_unavailable(monkeypatch):
    monkeypatch.setattr(commands, "bank_get_all", lambda user_id: [("AAPL", 100.0, None)])

    def boom(ticker):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(commands, "find_current_price", boom)

    status = commands.get_bank_status(1)

    assert status.rows[0]["price"] is None
    assert status.rows[0]["suggestion"] == "price unavailable"
