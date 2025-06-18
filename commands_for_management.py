from datetime import datetime

from requests import HTTPError
from retry import retry

import pytz
import seaborn as sns
import yfinance as yf

import matplotlib

from db import engine

matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

# Ensure all DataFrame content is displayed and set visual style
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set(style='whitegrid')


def initialize_portfolio():
    """
    Initializes the portfolio DataFrame the database.
    """
    query = """
    SELECT a.ticker, t.quantity, t.purchase_price, t.purchase_date
    FROM transactions t
    JOIN assets a ON t.asset_id = a.id
    ORDER BY t.purchase_date
    """

    portfolio = pd.read_sql_query(query, con=engine)
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

        # Check if the dividend index is tz-aware or not
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
    except HTTPError as he:
        status = he.response.status_code if hasattr(he, 'response') else "?"
        url = he.request.url if hasattr(he, 'request') else "unknown"
        print(f"HTTPError {status} fetching {ticker} from {url}: {he}")
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")
        return None, pd.Series(dtype=float)


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

    # Append the total row to the aggregated portfolio
    total_row = pd.DataFrame({
        'ticker': ['total'],
        'quantity': [total_quantity],
        'total return': [total_return],
        'percentage return': [total_percentage_return]
    })
    aggregated_portfolio = pd.concat([aggregated_portfolio, total_row], ignore_index=True)
    aggregated_portfolio = aggregated_portfolio.drop_duplicates(subset=['ticker'], keep='first')

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

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
    axs[1].bar(['total'], [total_percentage_return], color='darkgreen', label='Portfolio Total')
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
