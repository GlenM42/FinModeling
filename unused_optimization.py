import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# Fetch historical data
symbols = ['VOO', 'AVUV', 'AVDV']
data = yf.download(symbols, start="2020-01-01", end="2026-03-31", auto_adjust=True)['Close']

# --- Monthly returns (more stable for long-horizon factor portfolios) --------
monthly_data    = data.resample("ME").last()
monthly_returns = monthly_data.pct_change().dropna()

# Annualised stats from monthly returns
annual_returns = monthly_returns.mean() * 12
cov_matrix     = monthly_returns.cov()  * 12

print("Annualised returns:")
print((annual_returns * 100).round(2))
print("\nAnnualised volatilities:")
print((np.sqrt(np.diag(cov_matrix)) * 100).round(2))

# Risk-free rate
risk_free_rate = 0.045

# Shared constraints & bounds (no shorts, weights sum to 1)
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds      = tuple((0, 1) for _ in symbols)
w0          = [1. / len(symbols)] * len(symbols)

# =============================================================================
print("\n======HERE=WE=DO=REGULAR=PORTFOLIO=OPTIMIZATION=======\n")
# =============================================================================

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights @ cov_matrix @ weights)

result = minimize(portfolio_volatility, w0,
                  args=(cov_matrix,),
                  method='SLSQP', bounds=bounds, constraints=constraints)

print("Optimal weights (min volatility):")
for s, w in zip(symbols, result.x):
    print(f"  {s}: {w*100:.1f}%")

# =============================================================================
print("\n======HERE=WE=DO=PORTFOLIO=OPTIMIZATION=BY=SHARPE=RATIO======\n")
# =============================================================================

def neg_sharpe(weights, returns, cov_matrix, rf):
    ret = weights @ returns
    vol = np.sqrt(weights @ cov_matrix @ weights)
    return -(ret - rf) / vol

result_sharpe = minimize(neg_sharpe, w0,
                         args=(annual_returns, cov_matrix, risk_free_rate),
                         method='SLSQP', bounds=bounds, constraints=constraints)

print("Optimal weights (max Sharpe):")
for s, w in zip(symbols, result_sharpe.x):
    print(f"  {s}: {w*100:.1f}%")

# =============================================================================
print("\n====HERE=WE=DO=PORTFOLIO=OPTIMIZATION=BY=MINIMIZING=LEFT=TAIL====\n")
# =============================================================================

# --- Fixed semi-variance: E[min(r, 0)^2] per asset, annualised --------------
target_return = 0.0
neg_ret       = monthly_returns.copy()
neg_ret[neg_ret > target_return] = 0          # zero out positive months
semi_variance = (neg_ret ** 2).mean() * 12    # annualised expected squared downside

def objective_downside(weights, returns, semi_variance):
    portfolio_return   = weights @ returns
    # Diagonal approximation of downside covariance (standard for semi-variance)
    portfolio_semi_vol = np.sqrt(weights @ np.diag(semi_variance) @ weights)
    penalty_factor     = 5
    return -(portfolio_return - penalty_factor * portfolio_semi_vol)

result_downside = minimize(objective_downside, w0,
                           args=(annual_returns, semi_variance),
                           method='SLSQP', bounds=bounds, constraints=constraints)

print("Optimal weights (min downside risk):")
for s, w in zip(symbols, result_downside.x):
    print(f"  {s}: {w*100:.1f}%")

# =============================================================================
print("\n====SUMMARY====\n")
# =============================================================================

print(f"{'Asset':<8} {'Min Vol':>10} {'Max Sharpe':>12} {'Min Downside':>14}")
print("-" * 48)
for i, s in enumerate(symbols):
    print(f"{s:<8} "
          f"{result.x[i]*100:>9.1f}%  "
          f"{result_sharpe.x[i]*100:>11.1f}%  "
          f"{result_downside.x[i]*100:>13.1f}%")