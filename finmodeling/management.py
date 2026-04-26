from datetime import datetime

from retry import retry

import pytz
import yfinance as yf

import logging

logger = logging.getLogger(__name__)

import pandas as pd

# Ensure all DataFrame content is displayed and set visual style
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


@retry(tries=3, delay=2)
def fetch_stock_data(ticker: str, start_date: str) -> tuple[float, float]:
    """
    Fetches the current stock/ETF price and calculates dividends issued after the start_date_str and before the current
    date for a given ticker.

    Due to unstability of the Yahoo API, this method can raise exceptions. Some of them can be
    solved due to retry. It is the responsibility of the caller to catch them. 
    """
    # Convert start_date_str to a datetime object and ensure it's timezone-aware
    start_date = pd.to_datetime(start_date).tz_localize(pytz.UTC)
    # Current datetime also made timezone-aware
    end_date = pd.to_datetime(datetime.now()).tz_localize(pytz.UTC)

    stock_data = yf.Ticker(ticker)
    
    # For unknown reasons, some tickers do not have 'currentPrice',
    # so we get the next best value -- the bid
    price = stock_data.info.get('currentPrice') or stock_data.info.get('bid')
    if price is None:
        raise ValueError(f"Could not fetch stock data. price is '{price}'")

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


def calculate_performance(portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates performance metrics for the portfolio based on current prices and dividends received after purchase dates.
    
    Takes as an input pandas dataframe with the following columns: 
    - ticker
    - quantity
    - purchase_price
    - purchase_date

    Adds the following columns by doing appropriate calculations:
    - current price
    - dividends
    - investment value
    - current value
    - dividend return
    - total return

    Returns updated pandas dataframe
    """
    portfolio['current price'] = 0.0
    portfolio['dividends'] = 0.0
    for index, row in portfolio.iterrows():
        # Adjust column names here to match the DataFrame
        try:
            current_price, dividends = fetch_stock_data(row['ticker'], row['purchase_date'])
        except Exception as e:
            logger.error(f"We could not fetch data for the ticker {row['ticker']}: {e}", exc_info=True)
            continue # skip this row

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
    totals['ticker'] = 'Total'
    totals['current price'] = 'N/A'
    totals['dividends'] = 'N/A'
    totals['purchase_price'] = 'N/A'
    totals['percentage return'] = (totals['total return'] / totals['investment value'] * 100).round(2)

    portfolio = pd.concat([portfolio, totals], ignore_index=True)
    
    return portfolio


def calculate_aggregate_performance(portfolio_performance: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the portfolio performance table and creates a dataframe with aggregate performance.
    """
    # We need to drop the last row (Total), as it is not needed for the
    # aggregation calculations.
    portfolio_performance = portfolio_performance[:-1]

    # Aggregate total returns and percantage returns for each ticker
    aggregated_portfolio = portfolio_performance.groupby('ticker').agg(
        {
            'quantity': 'sum',
            'total return': 'sum',
            'percentage return': 'mean'
        }
    ).reset_index()

    total_quantity = portfolio_performance['quantity'].sum()
    total_return = portfolio_performance['total return'].sum()
    total_investment_value = portfolio_performance['investment value'].sum()
    total_dividends = portfolio_performance['dividends'].sum()
    total_percentage_return = ((total_return + total_dividends) / total_investment_value) * 100

    # Append the total row to the aggregated portfolio
    total_row = pd.DataFrame({
        'ticker': ['Total'],
        'quantity': [total_quantity],
        'total return': [total_return],
        'percentage return': [total_percentage_return]
    })
    aggregated_portfolio = pd.concat([aggregated_portfolio, total_row], ignore_index=True)

    return aggregated_portfolio.round(2)


@retry(tries=3, delay=2)
def compute_portfolio_history(
    portfolio: pd.DataFrame,
    start_buffer_days: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Build daily time series of (1) adjusted prices per ticker (AdjClose-equivalent via auto_adjust),
    (2) shares held per ticker, and (3) total portfolio value across time.

    Returns (prices_df, holdings_df, total_value_series)
    - prices_df: index=dates, columns=tickers, values=Close (auto-adjusted)
    - holdings_df: index=dates, columns=tickers, values=shares held that day
    - total_value_series: index=dates, values=sum(holdings * prices)
    """
    df = portfolio.copy()

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
        idx_pos = prices_df.index.searchsorted(d, side='left')
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

    # Map each transaction to the first trading date in prices_index (next/eq day)
    trading_dates = prices_index
    events_by_date = {}
    events_per_ticker = {}

    for _, r in df.iterrows():
        tkr = str(r['ticker'])
        d = r['purchase_date']
        pos = trading_dates.searchsorted(d, side='left')
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
