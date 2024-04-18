import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# Fetch historical data for the new set of assets
symbols = ['SBUX', 'PEP', 'MCD']
data = yf.download(symbols, start="2020-01-01", end="2024-03-25")['Adj Close']

# Calculate daily returns
daily_returns = data.pct_change()

# Calculate expected returns and covariance matrix
annual_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252

# Fetch current prices
current_prices = {symbol: yf.Ticker(symbol).info.get("currentPrice") for symbol in symbols}

total_budget = 2000  # Updated budget

print("\n======HERE=WE=DO=REGULAR=PORTFOLIO=OPTIMIZATION=======\n")


# Portfolio optimization function to minimize volatility
def optimize_portfolio(returns, cov_matrix):
    num_assets = len(returns)
    args = (cov_matrix,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum to 1
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets, ]

    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    result = minimize(portfolio_volatility, initial_guess, args=args, method='SLSQP', bounds=bounds,
                      constraints=constraints)
    return result.x


optimal_weights = optimize_portfolio(annual_returns, cov_matrix)

print("Optimal weights by regular optimization:", optimal_weights)

# Calculate the dollar amount to allocate to each stock
dollar_allocation = optimal_weights * total_budget

# Calculate how many shares to buy for each stock
shares_to_buy = {symbol: np.floor(dollar_allocation[i] / current_prices[symbol]) for i, symbol in enumerate(symbols)}

print("Shares to buy:", shares_to_buy)

print("\n======HERE=WE=DO==PORTFOLIO=OPTIMIZATION=BY=SHARPE=RATIO======\n")

# Risk-free rate (assuming a typical value; adjust based on current data)
risk_free_rate = 0.0522


# Function to calculate the Sharpe Ratio for a portfolio
def sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, returns) - risk_free_rate
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return / portfolio_volatility


# Objective function to maximize (negative Sharpe Ratio for minimization function)
def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)


# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in symbols)
initial_guess = [1. / len(symbols)] * len(symbols)

# Portfolio optimization to maximize the Sharpe Ratio
opt_result = minimize(neg_sharpe_ratio, initial_guess, args=(annual_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_result.x

print("Optimal weights judging by Sharpe Ration:", optimal_weights)

# Calculate the dollar amount to allocate to each stock
dollar_allocation = optimal_weights * total_budget

# Calculate how many shares to buy for each stock
shares_to_buy = {symbol: np.floor(dollar_allocation[i] / current_prices[symbol]) for i, symbol in enumerate(symbols)}

print("Shares to buy:", shares_to_buy)

print("\n====HERE=WE=DO=PORTFOLIO=OPTIMIZATION=BY=MINIMIZING=LEFT=TAIL====\n")

# Calculate the downside volatility
target_return = 0.0
downside_returns = daily_returns.copy()
# we are setting all the returns above zero to be null
downside_returns[downside_returns > target_return] = 0

# Calculate semi-variance (only negative returns)
semi_variance = downside_returns.var() * 252


# Custom objective function (Maximize Returns - Penalty for Downside Risk)
def objective(weights, returns, semi_variance):
    portfolio_return = np.dot(weights, returns)
    portfolio_semi_vol = np.sqrt(np.dot(weights.T, np.dot(np.diag(semi_variance), weights)))
    penalty_factor = 5  # Adjust this to control the penalty for downside risk
    return -(portfolio_return - penalty_factor * portfolio_semi_vol)


# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in symbols)
initial_guess = [1. / len(symbols)] * len(symbols)

# Portfolio optimization
opt_result = minimize(objective, initial_guess, args=(annual_returns, semi_variance),
                      method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_result.x

print("Optimal weights:", optimal_weights)

dollar_allocation = optimal_weights * total_budget

shares_to_buy = {symbol: np.floor(dollar_allocation[i] / current_prices[symbol]) for i, symbol in enumerate(symbols)}

print("Shares to buy: ", shares_to_buy)

# portfolio_info = {
#     'SBUX': [6, 91.79],
#     'PEP': [3, 174.76],
#     'MCD': [1, 282.25]
# }
