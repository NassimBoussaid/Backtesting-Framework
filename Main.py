from Backtester import Backtester
import pandas as pd
from Moving_Average import MovingAverage

# Merge historical_data and result into a new DataFrame
historical_data = pd.read_excel("Data_.xlsx", index_col=0)
ahah = Backtester(historical_data)
moving_average_strategy = MovingAverage(10, 90, exponential_mode=True)
result = ahah.run(moving_average_strategy, 5, 90)

historical_data['Moving_Average_10'] = historical_data['AAPL'].rolling(window=10).mean()
historical_data['Moving_Average_90'] = historical_data['AAPL'].rolling(window=90).mean()

# Convert the Series 'result' to a DataFrame
merged_data = pd.DataFrame(result, columns=['AAPL'])

# Now, add data from historical_data, aligning by the index of result
merged_data['Moving_Average_10'] = historical_data['Moving_Average_10'].reindex(result.index).fillna(0)
merged_data['Moving_Average_90'] = historical_data['Moving_Average_90'].reindex(result.index).fillna(0)
print(merged_data)

import matplotlib.pyplot as plt

# Plot the signal and the moving averages
fig, ax = plt.subplots(figsize=(12, 6))

# Add secondary_y=True to plot the signal on a separate axis
merged_data['AAPL'].plot(ax=ax, label='Signal (AAPL)', linewidth=1, secondary_y=True)
merged_data['Moving_Average_10'].plot(ax=ax, label='Moving Average 10', linewidth=1)
merged_data['Moving_Average_90'].plot(ax=ax, label='Moving Average 90', linewidth=1)

ax.set_title('AAPL UW Equity Signal with 10-day and 90-day Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
plt.show()
