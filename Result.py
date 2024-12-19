import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Result:
    def __init__(self, portfolio_returns, cumulative_returns, risk_free_rate=0.0, periods_per_year=252):
        """
        Initialize the Result object.

        Parameters:
        portfolio_returns (pd.Series): Time series of daily portfolio returns.
        cumulative_returns (pd.Series): Time series of cumulative portfolio returns.
        risk_free_rate (float): Risk-free rate for Sharpe/Sortino ratio calculations.
        periods_per_year (int): Number of periods in a year (e.g., 252 for daily data).
        """
        if not isinstance(portfolio_returns, pd.Series) or not isinstance(cumulative_returns, pd.Series):
            raise TypeError("portfolio_returns and cumulative_returns must be pandas Series.")
        if not portfolio_returns.index.equals(cumulative_returns.index):
            raise ValueError("portfolio_returns and cumulative_returns must have the same index.")

        self.portfolio_returns = portfolio_returns
        self.cumulative_returns = cumulative_returns
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Compute performance statistics
        self.total_return = self.calculate_total_return()
        self.annualized_return = self.calculate_annualized_return()
        self.volatility = self.calculate_volatility()
        self.sharpe_ratio = self.calculate_sharpe_ratio()
        self.max_drawdown = self.calculate_max_drawdown()
        self.sortino_ratio = self.calculate_sortino_ratio()

    def calculate_total_return(self):
        """
        Calculate the total return over the period.
        """
        return self.cumulative_returns.iloc[-1]

    def calculate_annualized_return(self):
        """
        Calculate the annualized return using actual number of years.
        """
        total_days = len(self.portfolio_returns)
        annualized_return = (1 + self.total_return) ** (self.periods_per_year / total_days) - 1
        return annualized_return

    def calculate_volatility(self):
        """
        Calculate the annualized volatility.
        """
        return self.portfolio_returns.std(ddof=1) * np.sqrt(self.periods_per_year)

    def calculate_sharpe_ratio(self):
        """
        Calculate the Sharpe Ratio.
        """
        excess_returns = self.portfolio_returns - self.risk_free_rate / self.periods_per_year
        annualized_excess_return = excess_returns.mean() * self.periods_per_year
        annualized_volatility = excess_returns.std(ddof=1) * np.sqrt(self.periods_per_year)
        return annualized_excess_return / annualized_volatility if annualized_volatility != 0 else np.nan

    def calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown.
        """
        cumulative_max = self.cumulative_returns.cummax()
        drawdowns = self.cumulative_returns - cumulative_max
        return drawdowns.min()

    def calculate_sortino_ratio(self):
        """
        Calculate the Sortino Ratio.
        """
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        annualized_downside_std = downside_returns.std(ddof=1) * np.sqrt(self.periods_per_year)
        annualized_return = self.calculate_annualized_return()
        if annualized_downside_std == 0:
            return np.nan
        return (annualized_return - self.risk_free_rate) / annualized_downside_std

    def display_statistics(self):
        """
        Display the performance statistics.
        """
        stats = {
            'Total Return': f"{self.total_return:.2%}",
            'Annualized Return': f"{self.annualized_return:.2%}",
            'Volatility': f"{self.volatility:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.4f}",
            'Maximum Drawdown': f"{self.max_drawdown:.2%}",
            'Sortino Ratio': f"{self.sortino_ratio:.4f}"
        }
        print("\nPerformance Statistics:")
        print("-----------------------")
        for k, v in stats.items():
            print(f"{k}: {v}")

    def plot_cumulative_returns(self):
        """
        Plot the cumulative portfolio returns over time.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.cumulative_returns.index, self.cumulative_returns, label="Cumulative Returns")
        plt.title("Cumulative Portfolio Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_portfolio_returns(self):
        """
        Plot the daily portfolio returns.
        """
        plt.figure(figsize=(12, 6))
        plt.bar(self.portfolio_returns.index, self.portfolio_returns, label="Portfolio Returns", color="blue",
                alpha=0.6)
        plt.title("Daily Portfolio Returns")
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
