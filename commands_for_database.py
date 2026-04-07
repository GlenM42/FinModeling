from sqlalchemy import text
from db import engine


# Bank table DDL (run once at startup via create_bank_table()):
#   CREATE TABLE IF NOT EXISTS bank (
#       user_id     INTEGER NOT NULL,
#       ticker      VARCHAR(10) NOT NULL,
#       balance     DOUBLE NOT NULL DEFAULT 0,
#       last_updated DATE,
#       PRIMARY KEY (user_id, ticker)
#   );

def create_bank_table():
    """Create the bank table if it does not already exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bank (
                user_id      INTEGER NOT NULL,
                ticker       VARCHAR(10) NOT NULL,
                balance      DOUBLE NOT NULL DEFAULT 0,
                last_updated DATE,
                PRIMARY KEY (user_id, ticker)
            )
        """))

def add_transaction(user_id, ticker, quantity, purchase_price, purchase_date):
    """
    Inserts the ticker into assets (if not already there),
    then records the transaction.
    """
    # engine.begin() provides a transactional connection that commits on exit
    with engine.begin() as conn:
        # 1) ensure ticker exists
        conn.execute(
            text("INSERT IGNORE INTO assets (ticker) VALUES (:ticker)"),
            {"ticker": ticker}
        )

        # 2) insert the transaction
        conn.execute(
            text("""
                INSERT INTO transactions
                    (user_id, asset_id, quantity, purchase_price, purchase_date)
                VALUES (
                    :user_id,
                    (SELECT id FROM assets WHERE ticker = :ticker),
                    :quantity,
                    :purchase_price,
                    :purchase_date
                )
            """),
            {
                "user_id": user_id,
                "ticker": ticker,
                "quantity": quantity,
                "purchase_price": purchase_price,
                "purchase_date": purchase_date,
            }
        )


def bank_deposit(user_id, ticker, amount):
    """Upsert: add amount to bank balance for user+ticker."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO bank (user_id, ticker, balance, last_updated)
                VALUES (:user_id, :ticker, :amount, CURDATE())
                ON DUPLICATE KEY UPDATE
                    balance      = balance + :amount,
                    last_updated = CURDATE()
            """),
            {"user_id": user_id, "ticker": ticker, "amount": amount},
        )


def bank_get_balance(user_id, ticker):
    """Return current balance for user+ticker, or 0.0 if no row exists."""
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT balance FROM bank WHERE user_id = :user_id AND ticker = :ticker"),
            {"user_id": user_id, "ticker": ticker},
        )
        row = result.fetchone()
        return float(row[0]) if row else 0.0


def bank_get_all(user_id):
    """Return all (ticker, balance, last_updated) rows with balance > 0 for a user."""
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                SELECT ticker, balance, last_updated
                FROM bank
                WHERE user_id = :user_id AND balance > 0
                ORDER BY ticker
            """),
            {"user_id": user_id},
        )
        return result.fetchall()


def bank_deduct(user_id, ticker, amount):
    """Subtract amount from balance, flooring at 0."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE bank
                SET balance      = balance - :amount,
                    last_updated = CURDATE()
                WHERE user_id = :user_id AND ticker = :ticker
            """),
            {"user_id": user_id, "ticker": ticker, "amount": amount},
        )


def remove_transactions(user_id, ticker, purchase_date):
    """
    Deletes any transactions for a given user/ticker/date.
    """
    with engine.begin() as conn:
        conn.execute(
            text("""
                DELETE FROM transactions
                 WHERE user_id      = :user_id
                   AND asset_id     = (SELECT id FROM assets WHERE ticker = :ticker)
                   AND purchase_date = :purchase_date
            """),
            {
                "user_id": user_id,
                "ticker": ticker,
                "purchase_date": purchase_date,
            }
        )
