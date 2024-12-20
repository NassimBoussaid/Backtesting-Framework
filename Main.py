from Backtester import Backtester
import pandas as pd
from Moving_Average import MovingAverage

# Merge historical_data and result into a new DataFrame
historical_data = pd.read_excel("Data_.xlsx", index_col=0)
ahah = Backtester(historical_data, special_start=100)
moving_average_strategy = MovingAverage(7, 100, exponential_mode=False)
result = ahah.run(moving_average_strategy, "daily")
result.display_statistics()
result.plot_cumulative_returns()
result.plot_portfolio_returns()
