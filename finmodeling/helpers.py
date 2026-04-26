import yfinance as yf
from retry import retry

@retry(tries=3, delay=2)
def find_current_price(ticker_symbol) -> float:
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1mo")
    return hist['Close'].iloc[-1]


@retry(tries=3, delay=2)
def find_previous_price(ticker_symbol) -> float:
    """Fetch the previous price of the stock from a month ago."""
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1mo")
    return hist['Close'].iloc[0]
