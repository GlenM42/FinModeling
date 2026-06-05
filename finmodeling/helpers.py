import math

import yfinance as yf
from retry import retry


@retry(tries=3, delay=2)
def find_current_price(ticker_symbol) -> float:
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1mo")
    price = float(hist['Close'].iloc[-1])
    if not math.isfinite(price):
        raise ValueError(f"Invalid price for {ticker_symbol}: {price}")
    return price


@retry(tries=3, delay=2)
def find_previous_price(ticker_symbol) -> float:
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1mo")
    price = float(hist['Close'].iloc[0])
    if not math.isfinite(price):
        raise ValueError(f"Invalid price for {ticker_symbol}: {price}")
    return price
