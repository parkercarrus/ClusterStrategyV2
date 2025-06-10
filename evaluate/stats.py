import config 
import numpy as np
from typing import Dict

def compute_performance_metrics(portfolio_value):
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    max_drawdown = drawdown.min()
    returns = portfolio_value.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)
    start_value = portfolio_value.iloc[0]
    end_value = portfolio_value.iloc[-1]
    num_years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    cagr = (end_value / start_value) ** (1 / num_years) - 1
    return max_drawdown, sharpe_ratio, cagr