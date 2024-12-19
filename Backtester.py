import pandas as pd
import Strategy
from Result import Result


class Backtester:
    def __init__(self, data_source, transaction_cost=0.0, slippage=0.0):
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source
        else:
            self.data = pd.read_csv(data_source)
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run(self, strategy: Strategy, rebalancing_frequency: int, special_start: int):
        composition_matrix = self.calculate_composition_matrix(strategy, rebalancing_frequency, special_start)
        asset_contributions, portfolio_returns, cumulative_asset_returns, cumulative_returns = self.calculate_returns(
            composition_matrix)

        result = Result(portfolio_returns, cumulative_returns, risk_free_rate=0.02)

        return result

    def calculate_composition_matrix(self, strategy: Strategy, rebalancing_frequency: float, special_start: int):
        assets = self.data.columns
        dates = self.data.index
        composition_matrix = pd.DataFrame(index=dates, columns=assets, dtype="float64")

        rebalancing_dates = dates[special_start::rebalancing_frequency]

        for asset in assets:
            current_position = 1

            for date_index in range(special_start, len(dates)):
                current_date = dates[date_index]
                current_df = self.data.iloc[:date_index + 1][asset]

                if current_date in rebalancing_dates:
                    signal = strategy.get_position(current_df, current_position)
                    current_position = signal

                composition_matrix.at[current_date, asset] = current_position

        return composition_matrix

    def calculate_transaction_costs(self, shifted_positions):
        """
        Calculate transaction costs based on changes in positions.
        """
        return (shifted_positions.diff().abs().sum(axis=1)) * self.transaction_cost

    def calculate_slippage_costs(self, shifted_positions):
        """
        Calculate slippage costs based on changes in positions.
        """
        return (shifted_positions.diff().abs().sum(axis=1)) * self.slippage

    def calculate_returns(self, composition_matrix):
        asset_returns = self.data.pct_change().fillna(0)
        shifted_positions = composition_matrix.shift(1).fillna(0)

        # Adjust returns for transaction costs and slippage
        transaction_costs = self.calculate_transaction_costs(shifted_positions)
        slippage_costs = self.calculate_slippage_costs(shifted_positions)

        asset_contributions = shifted_positions.multiply(asset_returns, axis=0)

        portfolio_returns = asset_contributions.sum(axis=1) - transaction_costs - slippage_costs

        cumulative_asset_returns = (1 + asset_contributions).cumprod() - 1
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

        return asset_contributions, portfolio_returns, cumulative_asset_returns, cumulative_returns