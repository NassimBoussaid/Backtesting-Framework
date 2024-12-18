from Strategy import Strategy
import pandas as pd
import numpy as np

class MovingAverage(Strategy):
    def __init__(self):
        self.ahaha = 1

    def get_position(self, historical_data, current_position):
        MA_10 = historical_data["AAPL UW Equity"].iloc[-10:].mean()
        MA_90 = historical_data["AAPL UW Equity"].iloc[-90:].mean()
        Signal = int(MA_10 > MA_90)
        return Signal

    def fit(self, data):
        pass