from backtesting_framework.Core.Backtester import Backtester
import pandas as pd
from backtesting_framework.Strategies.Value import Value
from backtesting_framework.Strategies.MinVariance import MinVariance
from backtesting_framework.Strategies.PairsTrading import PairsTradingStrategy

# historical_data = pd.read_excel("Data_.xlsx", index_col=0)
# ahah = Backtester(historical_data, special_start=100)
# moving_average_strategy = MovingAverage(7, 100, exponential_mode=False)
# result = ahah.run(moving_average_strategy, "daily")

price_data = pd.read_csv("S&P500_PX_LAST_EXAMPLE.csv", index_col=0, parse_dates=True)
backtester = Backtester(price_data, special_start=1, rebalancing_frequency="monthly")

#per_data = pd.read_csv("/datasets/S&P500_PER.csv", index_col=0, parse_dates=True)
#pbr_data = pd.read_csv("/datasets/S&P500_PBR.csv", index_col=0, parse_dates=True)
#metrics_data = {"PER": per_data, "PBR": pbr_data}

strategy = MinVariance(short_sell=False)
#strategy.fit(metrics_data)
result = backtester.run(strategy)

result.display_statistics()
result.plot_cumulative_returns()
result.plot_monthly_returns_heatmap()
result.plot_returns_distribution()

"""
price_data = pd.read_excel("C:/Users/nicoc/Desktop/Doc important/272/Python - POO/ProjetPOO/backtesting_framework/datasets/Data_.xlsx", index_col=0, parse_dates=True)
backtester = Backtester(price_data, special_start=1, rebalancing_frequency="daily")

strategy = PairsTradingStrategy(price_data, z_score_entry=-0.5, z_score_exit=0.5, significant_level=0.05)
result = backtester.run(strategy)

result.display_statistics()
result.plot_cumulative_returns()
result.plot_portfolio_returns()
"""