from datetime import datetime
from retry import retry

import pytz
import seaborn as sns
import yfinance as yf

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
from matplotlib import cm
from yahoo_fin import options

# Ensure all DataFrame content is displayed and set visual style
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set(style='whitegrid')


def initialize_portfolio():
    """
    Initializes the portfolio DataFrame from the SQLite database.
    """
    # Connect to SQLite database
    conn = sqlite3.connect('portfolio.db')
    query = """
    SELECT a.ticker, t.quantity, t.purchase_price, t.purchase_date
    FROM transactions t
    JOIN assets a ON t.asset_id = a.id
    """
    # Execute query and load the results into a DataFrame
    portfolio = pd.read_sql_query(query, conn)
    # print(portfolio.columns)  # Add this line to debug
    conn.close()
    return portfolio


@retry(tries=3, delay=2)
def fetch_stock_data(ticker, start_date_str):
    """
    Fetches the current stock/ETF price and calculates dividends issued after the start_date_str and before the current
    date for a given ticker.
    """
    # Convert start_date_str to a datetime object and ensure it's timezone-aware
    start_date = pd.to_datetime(start_date_str).tz_localize(pytz.UTC)
    # Current datetime also made timezone-aware
    end_date = pd.to_datetime(datetime.now()).tz_localize(pytz.UTC)

    try:
        stock_data = yf.Ticker(ticker)
        if stock_data.info.get('currentPrice') is not None:
            price = stock_data.info.get('currentPrice')
        else:
            price = stock_data.info.get('bid')

        # Check if the dividends index is tz-aware or not
        if stock_data.dividends.index.tz is None:
            # Index is tz-naive, localize it to UTC
            dividends_index_aware = stock_data.dividends.index.tz_localize(pytz.UTC, ambiguous='infer')
        else:
            # Index is already tz-aware, convert it to UTC
            dividends_index_aware = stock_data.dividends.index.tz_convert(pytz.UTC)

        # Correctly filter dividends based on the adjusted dates
        dividends = stock_data.dividends[(dividends_index_aware >= start_date) &
                                         (dividends_index_aware < end_date)].sum()

        return price, dividends
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")
        return None, pd.Series(dtype=float)


def fetch_option_data_and_show_returns():
    # Connect to the SQLite database
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()

    # Query the database to get all options data along with user information
    query = """
        SELECT o.options_symbol, u.username, o.quantity, o.purchase_price, o.purchase_date
        FROM options o
        JOIN users u ON o.user_id = u.user_id
        """
    c.execute(query)
    options_data = c.fetchall()
    conn.close()

    # Create a DataFrame from the options data
    df = pd.DataFrame(options_data,
                      columns=['options_symbol', 'username', 'quantity', 'purchase_price', 'purchase_date'])

    # Initialize column for current prices
    df['current_price'] = None

    for index, row in df.iterrows():
        try:
            option_symbol = row['options_symbol']
            underlying_symbol = option_symbol[:4]  # Simplified extraction, adjust as necessary

            # Get all expiration dates for the underlying symbol
            expiration_dates = options.get_expiration_dates(underlying_symbol)

            found = False
            for date in expiration_dates:
                if found:
                    break

                option_chain = options.get_options_chain(underlying_symbol, date)
                calls_data = option_chain['calls']
                puts_data = option_chain['puts']

                # Find the correct option in the calls and puts DataFrame
                matching_call = calls_data[calls_data['Contract Name'] == option_symbol]
                matching_put = puts_data[puts_data['Contract Name'] == option_symbol]

                if not matching_call.empty:
                    current_price = matching_call.iloc[0]['Bid']
                    found = True
                elif not matching_put.empty:
                    current_price = matching_put.iloc[0]['Bid']
                    found = True

            if found:
                df.at[index, 'current_price'] = current_price
            else:
                print(f"Option {option_symbol} not found in any expiration date.")
                df.at[index, 'current_price'] = None
        except Exception as e:
            print(f"Error fetching option price for {option_symbol}: {e}")
            df.at[index, 'current_price'] = None

    # Calculate investment value and returns
    df['investment_value'] = round(df['quantity'] * df['purchase_price'] * 100, 2)
    df['return_dollars'] = (df['current_price'] - df['purchase_price']) * df['quantity'] * 100
    df['return_percent'] = ((df['current_price'] - df['purchase_price']) / df['purchase_price']) * 100
    df['current_value'] = df['investment_value'] + df['return_dollars']

    # Round the columns to 2 decimal places
    df = df.round({'return_dollars': 1, 'return_percent': 1, 'current_value': 1})

    # Group by username and calculate totals
    totals_df = df.groupby('username').agg(
        total_quantity=('quantity', 'sum'),
        total_investment_value=('investment_value', 'sum'),
        total_current_value=('current_value', 'sum'),
        total_return_dollars=('return_dollars', 'sum')
    ).reset_index()

    # Calculate total return percent for each user
    totals_df['total_return_percent'] = (totals_df['total_current_value'] / totals_df['total_investment_value'] - 1) * 100

    # Round the totals DataFrame
    totals_df = totals_df.round({'total_return_dollars': 2, 'total_return_percent': 2})

    # Prepare the totals_df for appending by adjusting columns to match the main df
    totals_df['options_symbol'] = 'N/A'  # For purchase_price and current_price
    totals_df['purchase_price'] = 'N/A'
    totals_df['current_price'] = 'N/A'
    totals_df = totals_df.rename(columns={
        'username': 'username',
        'total_quantity': 'quantity',
        'total_investment_value': 'investment_value',
        'total_current_value': 'current_value',
        'total_return_dollars': 'return_dollars',
        'total_return_percent': 'return_percent'
    })

    # Append totals to the main DataFrame
    final_df = pd.concat([df, totals_df], ignore_index=True)
    final_df.update(final_df[['current_value', 'return_dollars', 'return_percent']].map('{:.2f}'.format))

    # Visualize the table as an image
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.3 + 2))  # Extra space for the header
    ax.axis('off')
    table_data = final_df[
        ['username', 'options_symbol', 'quantity', 'purchase_price', 'investment_value', 'current_value',
         'current_price', 'return_dollars', 'return_percent']]
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.savefig('option_returns_table.png')
    plt.close()

    # Group by username for plotting returns
    grouped = df.groupby('username')
    colors = cm.tab10(np.linspace(0, 1, len(grouped)))

    # Create a separate DataFrame for the portfolio totals
    portfolio_totals = totals_df[totals_df['options_symbol'] == 'N/A']

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Plot the individual options returns
    for (name, group), color in zip(grouped, colors):
        axs[0].bar(group['options_symbol'], group['return_dollars'], color=color, label=name)
        axs[1].bar(group['options_symbol'], group['return_percent'], color=color, label=name)

    # Plot the portfolio totals
    axs[0].bar(portfolio_totals['username'], portfolio_totals['return_dollars'], color='k', label='Portfolio Totals')
    axs[1].bar(portfolio_totals['username'], portfolio_totals['return_percent'], color='k', label='Portfolio Totals')

    axs[0].tick_params(axis='x', rotation=45)
    axs[1].tick_params(axis='x', rotation=45)

    axs[0].set_title('Return in Dollars by Option')
    axs[0].set_xlabel('Options Symbol')
    axs[0].set_ylabel('Return ($)')
    axs[0].legend()

    axs[1].set_title('Return Percentage by Option')
    axs[1].set_xlabel('Options Symbol')
    axs[1].set_ylabel('Return (%)')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('option_returns.png')
    plt.close()

    return 'option_returns_table.png', 'option_returns.png'


def calculate_performance(portfolio):
    """
    Calculates performance metrics for the portfolio based on current prices and dividends received after purchase dates.
    """
    portfolio['current price'] = 0.0
    portfolio['dividends'] = 0.0
    for index, row in portfolio.iterrows():
        # Adjust column names here to match the DataFrame
        result = fetch_stock_data(row['ticker'], row['purchase_date'])
        if result[0] is None:  # Assuming that no price data found returns None as the first item
            # print(f"Error for ticker {row['ticker']}: {result[2]}")  # Log the error message
            continue  # Skip further processing for this row
        current_price, dividends = result[:2]
        portfolio.at[index, 'current price'] = round(current_price, 2)
        portfolio.at[index, 'dividends'] = round(dividends, 2)

    # Ensure data types are correct for calculations
    portfolio['quantity'] = portfolio['quantity'].astype(float)
    portfolio['purchase_price'] = portfolio['purchase_price'].astype(float)  # Adjusted to match DataFrame
    portfolio['current price'] = portfolio['current price'].astype(float)
    portfolio['dividends'] = portfolio['dividends'].astype(float)

    portfolio['investment value'] = (portfolio['quantity'] * portfolio['purchase_price']).round(2)
    portfolio['current value'] = (portfolio['quantity'] * portfolio['current price']).round(2)
    portfolio['dividend return'] = (portfolio['dividends'] * portfolio['quantity']).round(2)
    portfolio['total return'] = (
            (portfolio['current value'] - portfolio['investment value']) + portfolio['dividend return']).round(2)
    portfolio['percentage return'] = ((portfolio['total return'] / portfolio['investment value']) * 100).round(2)

    # Append total row using pd.concat to avoid AttributeError
    totals = pd.DataFrame(portfolio.sum(numeric_only=True)).T
    totals['ticker'] = 'total'  # Ensure this matches the actual DataFrame if needed
    totals['current price'] = 'N/A'
    totals['dividends'] = 'N/A'
    totals['purchase_price'] = 'N/A'
    totals['percentage return'] = (totals['total return'] / totals['investment value'] * 100).round(2)
    portfolio = pd.concat([portfolio, totals], ignore_index=True)
    return portfolio


def plot_portfolio_performance(portfolio, filename='portfolio_performance.png'):
    """
    Plots the portfolio's absolute and percentage returns, including a bar for the total portfolio performance.

    Args:
        portfolio (pd.DataFrame): The portfolio DataFrame.
        filename (str): The filename to save the plot.
    """
    if 'percentage return' not in portfolio.columns:
        portfolio['percentage return'] = (portfolio['total return'] / portfolio['investment value']) * 100
        print('Percentage was not included previously. Had to calculate it here')

    # Convert 'dividends' column to numeric, coercing errors
    portfolio['dividends'] = pd.to_numeric(portfolio['dividends'], errors='coerce')

    # Aggregate total returns and percantage returns for each ticker
    aggregated_portfolio = portfolio.groupby('ticker').agg(
        {
            'quantity': 'sum',
            'total return': 'sum',
            'percentage return': 'mean'
        }
    ).reset_index()

    total_quantity = portfolio['quantity'].sum()
    total_return = portfolio['total return'].sum()
    total_investment_value = portfolio['investment value'].sum()
    total_dividends = portfolio['dividends'].sum()
    total_percentage_return = ((total_return + total_dividends) / total_investment_value) * 100 if total_investment_value != 0 else 0

    # Remove old total rows if they exist
    portfolio = portfolio[portfolio['ticker'] != 'total']

    # Append the total row to the aggregated portfolio
    total_row = pd.DataFrame({
        'ticker': ['total'],
        'quantity': [total_quantity],
        'total return': [total_return],
        'percentage return': [total_percentage_return]
    })
    aggregated_portfolio = pd.concat([aggregated_portfolio, total_row], ignore_index=True)

    print("Aggregated Portfolio DataFrame:")
    print(aggregated_portfolio)

    aggregated_portfolio = aggregated_portfolio.drop_duplicates(subset=['ticker'], keep='first')
    print("New Aggregated Portfolio DataFrame:")
    print(aggregated_portfolio)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # print("Total return is ", total_return)
    # print("Total return from aggregated portfolio is ", aggregated_portfolio['total return'][:-1])

    # Absolute total return plot
    bars = axs[0].bar(aggregated_portfolio['ticker'], aggregated_portfolio['total return'], color='skyblue', label='Individual Stocks')
    total_bar = axs[0].bar(['total'], [total_return/2], color='steelblue', label='Portfolio Total')
    axs[0].set_title('Absolute Total Return (Including Dividends)')
    axs[0].set_ylabel('Total Return ($)')
    axs[0].legend()
    # Add numeric labels on top of the bars

    for bar in bars:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, yval, f'${round(yval, 2)}', ha='center', va='bottom')

    for bar in total_bar:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, yval, f'${round(yval, 2)}', ha='center', va='bottom')


    # Percentage return plot
    bars = axs[1].bar(aggregated_portfolio['ticker'], aggregated_portfolio['percentage return'], color='lightgreen', label='Individual Stocks')
    # axs[1].bar(aggregated_portfolio['ticker'], aggregated_portfolio['percentage return'], color='lightgreen', label='Individual Stocks')
    axs[1].bar(['total'], [total_percentage_return], color='lime', label='Portfolio Total')
    axs[1].set_title('Percentage Return (Including Dividends)')
    axs[1].set_ylabel('Return (%)')
    axs[1].legend()

    for bar in bars:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, yval, f'{round(yval, 2)}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    plt.show()

def show_portfolio_as_image(portfolio, filename='portfolio_table.png'):
    """
    Generates an image representation of the portfolio DataFrame with adjustments for 'Total' row.
    """
    fig, ax = plt.subplots(figsize=(12, (len(portfolio) + 1) * 0.65))
    ax.axis('tight')
    ax.axis('off')
    table_data = portfolio.round(2).fillna("N/A")
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center',
                     colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def find_current_price(ticker_symbol):
    """Fetch the current price of the stock."""
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1mo")
    if not hist.empty:
        return hist['Close'].iloc[-1]
    else:
        return None


def find_previous_price(ticker_symbol):
    """Fetch the previous price of the stock from a month ago."""
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1mo")
    if not hist.empty:
        return hist['Close'].iloc[0]
    else:
        return None


# Portfolio information
db_path = 'portfolio.db'
