from sqlalchemy import text
from db import engine

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
