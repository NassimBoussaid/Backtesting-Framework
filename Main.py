from Backtester import Backtester
import pandas as pd
from Moving_Average import MovingAverage
from RSI import RSI
from MeanReversion import MeanReversion
from BollingerBands import BollingerBands


# Charger les données historiques
historical_data = pd.read_excel("Data_.xlsx", index_col=0)
# Initialiser le backtester
backtester = Backtester(historical_data, special_start=100)

#######################################################################################################################################

######################################  RSI  #############################################


# Définir la stratégie RSI
rsi_strategy = RSI(period=14, overbought_threshold=70, oversold_threshold=30)

# Exécuter la stratégie
result = backtester.run(rsi_strategy, "daily")

######################################  Bollinger Bands #############################################
# Définir la stratégie Bollinger Bands
#bollinger_strategy = BollingerBands(window=20, num_std_dev=2.0)

# Exécuter la stratégie
#result = backtester.run(bollinger_strategy, "daily")

######################################  Mean Reversion #############################################

# Définir la stratégie Mean Reversion
#mean_reversion_strategy = MeanReversion(window=20, zscore_threshold=1.6)

# Exécuter la stratégie
#result = backtester.run(mean_reversion_strategy, "daily")


######################################  Moving Average #############################################
# Définir la stratégie Moving Average
#moving_average = MovingAverage(7, 100, exponential_mode=False)

#Executer la strategie
#result = backtester.run(moving_average, "daily")

#######################################################################################################################################

result.display_statistics()
result.plot_cumulative_returns()
result.plot_portfolio_returns()
