from Core.Strategy import Strategy
import pandas as pd

class KeltnerChannelStrategy(Strategy):
    def __init__(self, atr_period=10, atr_multiplier=2.0, sma_period=20):
        """
        A Keltner Channel strategy that goes long if the price is above the upper band,
        and short if the price is below the lower band.

        Parameters:
        -----------
        atr_period : int
            Number of periods for the ATR calculation.
        atr_multiplier : float
            Multiplier for the ATR to calculate the bands.
        sma_period : int
            Number of periods for the SMA calculation.
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.sma_period = sma_period

    def calculate_atr(self, prices, period):
        """
        Calculate the average true range (ATR) of a given price series.
        :param prices:
        :param period:
        :return:
        """

        delta = prices[1:] - prices[:-1]
        delta = pd.Series(delta)
        atr = delta.abs().rolling(period).mean().iloc[-1]
        return atr

    def calculate_sma(self, prices, period):
        """
        Calculate the simple moving average (SMA) of a given price series.
        :param prices:
        :param period:
        :return:
        """
        prices = pd.Series(prices)
        sma = prices.rolling(period).mean().iloc[-1]
        return sma

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Determine the position based on Keltner Channels.
        :param historical_data:
        :param current_position:
        :return:
        """

        prices = historical_data.values
        atr = self.calculate_atr(prices, self.atr_period)
        sma = self.calculate_sma(prices, self.sma_period)
        upper_band = sma + atr * self.atr_multiplier
        lower_band = sma - atr * self.atr_multiplier

        if prices[-1] > upper_band:
            return 1
        elif prices[-1] < lower_band:
            return -1
        else:
            return current_position
