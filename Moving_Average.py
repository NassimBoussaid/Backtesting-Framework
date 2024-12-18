from Strategy import Strategy

class MovingAverage(Strategy):
    def __init__(self, rolling_days_1: int, rolling_days_2: int, exponential_mode=False):
        self.rolling_days_1 = rolling_days_1
        self.rolling_days_2 = rolling_days_2
        self.exponential_mode = exponential_mode

    def get_position(self, historical_data, current_position):
        try:
            ma_short, ma_long = self.calculate_moving_average(historical_data)
        except ValueError as error:
            print("Error")
        if ma_short > ma_long:
            return 1
        elif ma_short < ma_long:
            return -1

    def calculate_moving_average(self, historical_data):
        if self.exponential_mode:
            alpha_short = 2 / (self.rolling_days_1 + 1)
            alpha_long = 2 / (self.rolling_days_2 + 1)
            ma_short = historical_data["AAPL UW Equity"].iloc[-self.rolling_days_1]
            ma_long = historical_data["AAPL UW Equity"].iloc[-self.rolling_days_2]
            for price_short, price_long in zip(historical_data["AAPL UW Equity"].iloc[-self.rolling_days_1 + 1], historical_data["AAPL UW Equity"].iloc[-self.rolling_days_2 + 1]):
                ma_short = alpha_short * price_short + (1 - alpha_short) * ma_short
                ma_long = alpha_long * price_long + (1 - alpha_long) * ma_long
        else:
            ma_short = historical_data["AAPL UW Equity"].iloc[-self.rolling_days_1:].mean()
            ma_long = historical_data["AAPL UW Equity"].iloc[-self.rolling_days_2:].mean()
        return ma_short, ma_long

    def fit(self, data):
        pass