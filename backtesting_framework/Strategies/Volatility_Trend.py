from backtesting_framework.Core.Strategy import Strategy
import pandas as pd

class VolatilityTrendStrategy(Strategy):
    def __init__(self,atr_period=14,dmi_period=14,atr_threshold=1.0):
        """
        A volatility trend strategy that goes long if the ATR is greater than the threshold
        and the DMI is positive, else short.

        Parameters:
        -----------
        atr_period : int
            Number of periods for the ATR calculation.
        dmi_period : int
            Number of periods for the DMI calculation.
        atr_threshold : float
            Threshold for the ATR.
        """
        super().__init__(multi_asset=False)
        self.atr_period = atr_period
        self.dmi_period = dmi_period
        self.atr_threshold = atr_threshold

    def calculate_atr(self,prices,period):
        """
        Calculate the average true range (ATR) of a given price series.
        Measure the volatility of the price series.
        Doesn't give any signal, just a measure of volatility.
        """
        delta = prices[1:] - prices[:-1]
        delta = pd.Series(delta)
        atr = delta.abs().rolling(period).mean().iloc[-1]
        return atr

    def calculate_dmi(self,prices,period):
        """
        Calculate the directional movement index (DMI) of a given price series.
        Measure the strength of the trend.
        dmi is the absolute difference between the positive and negative directional movement indicators.
        Positive means the trend is up, negative means the trend is down.
        """
        delta = prices[1:] - prices[:-1]
        delta = pd.Series(delta)
        delta = delta.reset_index(drop=True)
        pos_dm = delta.where(delta > 0, 0).rolling(period).sum().iloc[-1]
        neg_dm = -delta.where(delta < 0, 0).rolling(period).sum().iloc[-1]
        atr = self.calculate_atr(prices,period)
        pos_di = 100 * pos_dm / atr
        neg_di = 100 * neg_dm / atr
        dmi = abs(pos_di - neg_di) / (pos_di + neg_di) * 100
        return dmi

    def get_position(self,historical_data:pd.DataFrame,current_position:float)->float:
        """
        Determine the position based on volatility trend.
        """
        prices = historical_data.values
        atr = self.calculate_atr(prices,self.atr_period)
        dmi = self.calculate_dmi(prices,self.dmi_period)
        if atr > self.atr_threshold and dmi > 0:
            return 1
        elif atr > self.atr_threshold and dmi < 0:
            return -1
        else:
            return 0
