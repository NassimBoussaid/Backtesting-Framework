import pandas as pd

import Strategy

class Backtester:
    def __init__(self, data_source):
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source
        else:
            self.data = pd.read_csv(data_source)

    def run(self, tested_strat: Strategy, interval: float, special_start: int):
        nb_dates = len(self.data)
        result = pd.Series(dtype='float64')
        for i in range(special_start, nb_dates, interval):
            current_df = self.data.iloc[:i]
            current_signal = tested_strat.get_position(current_df, 1)
            current_date = current_df.index[-1]
            result.loc[current_date] = current_signal
        return result