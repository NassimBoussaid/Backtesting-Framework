from Core.Strategy import Strategy
import pandas as pd

class MovingAverage(Strategy):
    def __init__(self, short_window: int, long_window: int, exponential_mode=False):
        """
        A moving average strategy that goes long if short MA > long MA, else short.

        Parameters:
        -----------
        short_window : int
            Number of periods for the short moving average.
        long_window : int
            Number of periods for the long moving average.
        exponential_mode : bool
            If True, compute exponential moving averages (EMA) instead of simple MAs.
        column : str
            The column name of the asset prices in the historical DataFrame.
        """
        self.short_window = short_window
        self.long_window = long_window
        self.exponential_mode = exponential_mode

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Determine the position based on moving averages.

        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical price data including the current day.
        current_position : float
            Current position (e.g., 1 for long, -1 for short, 0 for flat).

        Returns:
        --------
        float
            New position (-1, 0, or 1) based on the strategy's logic.
        """
        if len(historical_data) < max(self.short_window, self.long_window):
            # Not enough data to compute strategy signals
            return current_position

        prices = historical_data.values

        if self.exponential_mode:
            # Calculate EMA for short window
            ma_short = self._calculate_ema(prices, self.short_window)
            # Calculate EMA for long window
            ma_long = self._calculate_ema(prices, self.long_window)
        else:
            # Calculate simple MA for short window
            ma_short = self._calculate_sma(prices, self.short_window)
            # Calculate simple MA for long window
            ma_long = self._calculate_sma(prices, self.long_window)

        if ma_short > ma_long:
            return 1
        elif ma_short < ma_long:
            return -1
        else:
            return current_position

    def _calculate_sma(self, prices, window: int) -> float:
        """
        Calculate simple moving average manually.

        Parameters:
        -----------
        prices : array-like
            Array of prices.
        window : int
            Number of periods over which to compute the SMA.

        Returns:
        --------
        float
            The simple moving average over the last `window` periods.
        """
        relevant_prices = prices[-window:]
        return relevant_prices.sum() / window

    def _calculate_ema(self, prices, window: int) -> float:
        """
        Calculate exponential moving average (EMA) manually.

        Parameters:
        -----------
        prices : array-like
            Array of prices.
        window : int
            Number of periods for the EMA.

        Returns:
        --------
        float
            The exponential moving average over the last `window` periods.
        """
        alpha = 2 / (window + 1)
        # Apply EMA formula incrementally from the point where we have full window:
        relevant_prices = prices[-window:]
        # Start with the first price as the EMA seed
        ema = relevant_prices[0]

        # For each subsequent price, update the EMA:
        for price in relevant_prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def fit(self, data):
        """
        Optional fitting method. Not used for this strategy.
        """
        pass