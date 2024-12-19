import pandas as pd

import Strategy

class Backtester:
    def __init__(self, data_source):
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source
        else:
            self.data = pd.read_csv(data_source)

    def run(self, strategy: Strategy, rebalancing_frequency: int, special_start: int):
        assets = self.data.columns
        dates = self.data.index
        # Initialize the composition matrix with dates as index and assets as columns
        composition_matrix = pd.DataFrame(index=dates, columns=assets, dtype="float64")

        # Determine rebalancing dates
        rebalancing_dates = dates[special_start::rebalancing_frequency]

        for asset in assets:
            current_position = 1  # Initialize the current position for the asset

            for date_index in range(special_start, len(dates)):
                current_date = dates[date_index]  # Keep current_date inside the loop
                current_df = self.data.iloc[:date_index + 1][asset]  # Slice data up to the current date

                # Rebalance on rebalancing dates
                if current_date in rebalancing_dates:
                    signal = strategy.get_position(current_df, current_position)
                    current_position = signal

                # Fill the composition matrix with the signal
                composition_matrix.at[current_date, asset] = current_position

        return composition_matrix