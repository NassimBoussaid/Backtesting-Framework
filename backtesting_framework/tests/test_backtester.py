import numpy as np
import pandas as pd
import pytest

from backtesting_framework.Core.Backtester import Backtester
from backtesting_framework.Strategies.Value import Value
from backtesting_framework.Strategies.MovingAverage import MovingAverage

def generate_random_walk(start_price, size, volatility):
    returns = np.random.normal(loc=0, scale=volatility, size=size)
    price = start_price * (1 + returns).cumprod()
    return price

def generate_prices_dataframe(start_date, end_date, tickers, start_prices, volatility, freq="D"):
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    prices = pd.DataFrame(index=dates)
    for ticker in tickers:
        prices[ticker] = generate_random_walk(start_prices[ticker], len(dates), volatility[ticker])
    return prices

def test_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        Backtester(empty_df)

def test_value_strategy():
    prices = generate_prices_dataframe("2019-01-01", "2020-12-31", ["AAPL", "TSLA", "AMZN"],
                                       {"AAPL": 100, "TSLA": 200, "AMZN": 300}, {"AAPL": 0.01, "TSLA": 0.02, "AMZN": 0.03})

    per_data = generate_prices_dataframe("2019-01-01", "2020-12-31", ["AAPL", "TSLA", "AMZN"],{"AAPL": 15, "TSLA": 20, "AMZN": 25}, {"AAPL": 0.01, "TSLA": 0.02, "AMZN": 0.03})
    pbr_data = generate_prices_dataframe("2019-01-01", "2020-12-31", ["AAPL", "TSLA", "AMZN"],{"AAPL": 2, "TSLA": 3, "AMZN": 4}, {"AAPL": 0.01, "TSLA": 0.02, "AMZN": 0.03})
    metrics_data = {"PER": per_data, "PBR": pbr_data}
    backtester = Backtester(prices)
    strategy = Value()
    strategy.fit(metrics_data)
    result = backtester.run(strategy)

    assert result is not None
    assert isinstance(result.portfolio_returns, pd.Series)
    assert isinstance(result.cumulative_returns, pd.Series)

def test_moving_average():
    prices = generate_prices_dataframe("2019-01-01", "2023-01-03", ["AAPL", "TSLA", "AMZN"],
                                       {"AAPL": 100, "TSLA": 200, "AMZN": 300},
                                       {"AAPL": 0.01, "TSLA": 0.02, "AMZN": 0.03})
    backtester = Backtester(prices, special_start=1, rebalancing_frequency="monthly")
    strategy = MovingAverage(short_window=20, long_window=100)
    result = backtester.run(strategy)

    assert result is not None
    assert isinstance(result.portfolio_returns, pd.Series)
    assert isinstance(result.cumulative_returns, pd.Series)
    assert (len(result.portfolio_returns) == len(prices))


