from datetime import datetime
from typing import Tuple

import numpy as np
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

def compute_portfolio_history(
    portfolio: pd.DataFrame,
    start_buffer_days: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Build daily time series of (1) adjusted prices per ticker (AdjClose-equivalent via auto_adjust),
    (2) shares held per ticker, and (3) total portfolio value across time.

    Returns (prices_df, holdings_df, total_value_series)
    - prices_df: index=dates, columns=tickers, values=Close (auto-adjusted)
    - holdings_df: index=dates, columns=tickers, values=shares held that day
    - total_value_series: index=dates, values=sum(holdings * prices)
    """
    if portfolio.empty:
        raise ValueError("Portfolio transactions are empty.")

    df = portfolio.copy()
    df['purchase_date'] = pd.to_datetime(df['purchase_date']).dt.tz_localize(None)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0.0)

    # Price window: a small buffer before earliest purchase up to tomorrow (exclusive)
    earliest = df['purchase_date'].min()
    start = (earliest - pd.Timedelta(days=start_buffer_days)).date()
    end = (pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(days=1)).date()

    tickers = sorted(df['ticker'].unique())

    # Download auto-adjusted prices and get a flat "Close" table with tickers as columns
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,       # yields adj-close-equivalent in "Close"
        progress=False,
        group_by='column'       # columns like: ('Close', 'AAPL', 'MSFT', ...)
    )

    # Handle single- vs multi-ticker shapes
    if isinstance(raw.columns, pd.MultiIndex):
        # Multi: select the 'Close' slice → columns are tickers
        prices_df = raw['Close'].copy()
    else:
        # Single: keep the single 'Close' column and rename to the ticker
        if 'Close' not in raw.columns:
            raise ValueError("Expected 'Close' column in downloaded data.")
        prices_df = raw[['Close']].copy()
        prices_df.columns = [tickers[0]]

    # Sort, drop all-NaN rows just in case
    prices_df = prices_df.sort_index().dropna(how='all')

    # Make sure the columns cover exactly our tickers (in case of delistings / typos)
    have = {c.upper(): c for c in prices_df.columns}
    keep_cols = [have[t.upper()] for t in tickers if t.upper() in have]
    if not keep_cols:
        raise ValueError("No price columns matched the requested tickers.")
    prices_df = prices_df[keep_cols]

    # Build holdings matrix on trading days in prices_df
    holdings_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)

    # Apply each transaction from its first trading day onward
    for _, r in df.iterrows():
        t = r['ticker']
        if t not in holdings_df.columns:
            # Skip transactions for tickers with no price history returned
            continue
        q = float(r['quantity'])
        d = r['purchase_date']
        idx_pos = prices_df.index.searchsorted(pd.Timestamp(d), side='left')
        if idx_pos < len(prices_df.index):
            holdings_df.loc[holdings_df.index[idx_pos]:, t] += q

    # Per-ticker values and total equity curve
    per_ticker_values = holdings_df * prices_df
    total_value = per_ticker_values.sum(axis=1)

    return prices_df, holdings_df, total_value

def build_purchase_events(
    portfolio: pd.DataFrame,
    prices_index: pd.DatetimeIndex
):
    """
    Returns:
      events_by_date: dict[pd.Timestamp -> str]  # labels joined by comma for total plot
      events_per_ticker: dict[str -> list[pd.Timestamp]]  # for per-ticker markers
    """
    df = portfolio.copy()
    df['purchase_date'] = pd.to_datetime(df['purchase_date']).dt.tz_localize(None)

    # Map each transaction to the first trading date in prices_index (next/eq day)
    trading_dates = prices_index
    events_by_date = {}
    events_per_ticker = {}

    for _, r in df.iterrows():
        tkr = str(r['ticker'])
        d = r['purchase_date']
        pos = trading_dates.searchsorted(pd.Timestamp(d), side='left')
        if pos >= len(trading_dates):
            continue  # purchase after last known price (unlikely); skip
        trade_day = trading_dates[pos]

        # total-plot labels (join tickers that share the same day)
        if trade_day not in events_by_date:
            events_by_date[trade_day] = tkr
        else:
            # avoid duplicate names on same day for same ticker
            labels = set(map(str.strip, events_by_date[trade_day].split(',')))
            labels.add(tkr)
            events_by_date[trade_day] = ", ".join(sorted(labels))

        # per-ticker markers
        events_per_ticker.setdefault(tkr, []).append(trade_day)

    return events_by_date, events_per_ticker

def plot_portfolio_history_total(
    total_value: pd.Series,
    filename: str = 'portfolio_history_total.png',
    events_by_date: dict | None = None
) -> None:
    """
    Plot total portfolio value through time. Optionally draw vertical lines at purchase dates
    with ticker labels at the top.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(total_value.index, total_value.values, label='Portfolio Value')
    ax.set_title('Portfolio Value Over Time', pad=30)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value ($)')
    ax.legend()

    if events_by_date:
        ymin, ymax = ax.get_ylim()
        y_text = ymax  # annotate at the top
        for dt, label in sorted(events_by_date.items()):
            ax.axvline(dt, linestyle='--', linewidth=1, alpha=0.5)
            # place a small label at the top; slight vertical offset in axes coords
            ax.annotate(
                label,
                xy=(dt, y_text),
                xycoords=('data', 'data'),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=8, rotation=90
            )

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_portfolio_history_by_ticker(
    holdings_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    filename: str = 'portfolio_history_by_ticker.png',
    events_per_ticker: dict | None = None
) -> None:
    """
    Plot each ticker's contribution to portfolio value as lines.
    Optionally mark purchase dates with points and small labels.
    """
    per_ticker_values = holdings_df * prices_df

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in per_ticker_values.columns:
        series = per_ticker_values[col]
        ax.plot(series.index, series.values, label=col)

        if events_per_ticker and col in events_per_ticker:
            event_dates = events_per_ticker[col]
            # Find the y values at those dates (align on index)
            valid_dates = [d for d in event_dates if d in series.index]
            if valid_dates:
                yvals = series.loc[valid_dates].values
                ax.scatter(valid_dates, yvals, s=18, zorder=5)
                # Tiny labels just above each dot
                for x, y in zip(valid_dates, yvals):
                    ax.annotate(col, xy=(x, y), xytext=(0, 6),
                                textcoords='offset points', ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_title('Per-Ticker Position Value Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value ($)')
    ax.legend(ncols=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_prices_with_purchase_markers(
    prices_df: pd.DataFrame,
    events_per_ticker: dict[str, list[pd.Timestamp]] | None = None,
    filename: str = 'portfolio_prices_with_buys.png',
    normalize: bool = False,
    min_label_gap_days: int = 14,
    trim_pre_purchase: bool = True,   # hide days before first buy when normalize=True
) -> None:
    """
    Plot adjusted prices for all tickers (no quantity effect) and mark buy dates.
    If normalize=True, each ticker is rebased to 100 at its FIRST BUY date (per-ticker).
    """
    P = prices_df.copy()
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in P.columns:
        s = P[col].dropna()
        if s.empty:
            continue

        # --- per-ticker normalization at first buy date ---
        if normalize:
            # pick the first buy date that exists in the price index; else fall back to first valid date
            anchor = None
            if events_per_ticker and col in events_per_ticker:
                aligned = [d for d in events_per_ticker[col] if d in s.index]
                if aligned:
                    anchor = min(aligned)
            if anchor is None:
                anchor = s.index[0]

            base = s.loc[anchor]
            if pd.isna(base) or base == 0:
                # Safety fallback
                base = s.dropna().iloc[0]

            s_norm = (s / float(base)) * 100.0

            if trim_pre_purchase and events_per_ticker and col in events_per_ticker:
                # hide dates before first buy so the line starts at the purchase
                s_norm = s_norm.where(s.index >= anchor)

            plot_series = s_norm
        else:
            plot_series = s

        ax.plot(plot_series.index, plot_series.values, label=col)

        if events_per_ticker and col in events_per_ticker:
            ev_dates = [d for d in events_per_ticker[col] if d in plot_series.index]
            if ev_dates:
                yvals = plot_series.loc[ev_dates].values
                ax.scatter(ev_dates, yvals, s=18, zorder=5)

                last_label_dt = None
                for x, y in zip(ev_dates, yvals):
                    if np.isnan(y):
                        continue
                    if last_label_dt is None or (x - last_label_dt).days >= min_label_gap_days:
                        ax.annotate(col, xy=(x, y), xytext=(0, 6),
                                    textcoords='offset points', ha='center', va='bottom', fontsize=7)
                        last_label_dt = x

    ax.set_title('Price History With Buy Markers' + (' — Rebased to First Buy (100)' if normalize else ''))
    ax.set_xlabel('Date')
    ax.set_ylabel('Index (100 at first buy)' if normalize else 'Price ($)')
    ax.legend(ncols=2, fontsize=9)
    plt.tight_layout()
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
