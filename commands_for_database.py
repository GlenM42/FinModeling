import sqlite3


def ensure_user_exists(db_path, user_id, username=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check if user already exists
    cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
    if cursor.fetchone() is None:
        # Insert new user
        cursor.execute("INSERT INTO users (user_id, username) VALUES (?, ?)", (user_id, username))
        conn.commit()
    conn.close()


def add_transaction(db_path, user_id, ticker, quantity, purchase_price, purchase_date):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure the asset exists in the assets table
    cursor.execute("INSERT INTO assets (ticker) VALUES (?) ON CONFLICT(ticker) DO NOTHING", (ticker,))

    # Insert the transaction, linking to the asset via a subquery to get the asset ID
    cursor.execute("""
        INSERT INTO transactions (user_id, asset_id, quantity, purchase_price, purchase_date) 
        VALUES (
            ?,
            (SELECT id FROM assets WHERE ticker = ?),
            ?,
            ?,
            ?
        )
    """, (user_id, ticker, quantity, purchase_price, purchase_date))

    conn.commit()
    conn.close()


def remove_transactions(db_path, user_id, ticker, purchase_date_arg):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Delete transactions for a specific user and asset
    cursor.execute("""
        DELETE FROM transactions
        WHERE user_id = ? AND asset_id = (SELECT id FROM assets WHERE ticker = ?) AND purchase_date = ?
    """, (user_id, ticker, purchase_date_arg))

    conn.commit()
    conn.close()


def add_option_transaction(db_path, user_id, options_symbol, quantity, purchase_price, purchase_date):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Убедимся, что пользователь существует
    ensure_user_exists(db_path, user_id)
    # Добавим транзакцию с опционом
    cursor.execute("""
        INSERT INTO options (options_symbol, user_id, quantity, purchase_price, purchase_date) 
        VALUES (?, ?, ?, ?, ?)
    """, (options_symbol, user_id, quantity, purchase_price, purchase_date))
    conn.commit()
    conn.close()


def remove_option_transaction(db_path, user_id, options_symbol, purchase_date):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Удаление транзакции с опционом для конкретного пользователя и опциона
    cursor.execute("""
        DELETE FROM options
        WHERE user_id = ? AND options_symbol = ? AND purchase_date = ?
    """, (user_id, options_symbol, purchase_date))
    conn.commit()
    conn.close()
