from Backtester import Backtester
import pandas as pd
from Strategies.Value import Value

# historical_data = pd.read_excel("Data_.xlsx", index_col=0)
# ahah = Backtester(historical_data, special_start=100)
# moving_average_strategy = MovingAverage(7, 100, exponential_mode=False)
# result = ahah.run(moving_average_strategy, "daily")

price_data = pd.read_csv("../Datasets/S&P500_PX_LAST.csv", index_col=0, parse_dates=True)
backtester = Backtester(price_data, special_start=30,
                        rebalancing_frequency="monthly")

per_data = pd.read_csv("../Datasets/S&P500_PER.csv", index_col=0, parse_dates=True)
pbr_data = pd.read_csv("../Datasets/S&P500_PBR.csv", index_col=0, parse_dates=True)
metrics_data = {"PER": per_data, "PBR": pbr_data}

strategy = Value(window=30, assets_picked_long=1000, assets_picked_short=0)
strategy.fit(metrics_data)
result = backtester.run(strategy)

result.display_statistics()
#result.plot_cumulative_returns()
#result.plot_portfolio_returns()
