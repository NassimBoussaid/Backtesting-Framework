from Backtester import Backtester
import pandas as pd
from Moving_Average import MovingAverage

historical_data = pd.read_excel("Data.xlsx", index_col=0)
ahah = Backtester(historical_data)
moving_average_strategy = MovingAverage()
result = ahah.run(moving_average_strategy, 5, 90)
print(result)