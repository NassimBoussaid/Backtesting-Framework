import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar

class Result:
    PERIODS_PER_YEAR = 252
    
    def __init__(self, portfolio_returns, cumulative_returns, risk_free_rate=0.0):
        """
        Initialize the Result object.

        Parameters:
        portfolio_returns (pd.Series): Time series of daily portfolio returns.
        cumulative_returns (pd.Series): Time series of cumulative portfolio returns.
        risk_free_rate (float): Risk-free rate for Sharpe/Sortino ratio calculations.
        """
        if not isinstance(portfolio_returns, pd.Series) or not isinstance(cumulative_returns, pd.Series):
            raise TypeError("portfolio_returns and cumulative_returns must be pandas Series.")
        if not portfolio_returns.index.equals(cumulative_returns.index):
            raise ValueError("portfolio_returns and cumulative_returns must have the same index.")

        self.portfolio_returns = portfolio_returns
        self.cumulative_returns = cumulative_returns
        self.risk_free_rate = risk_free_rate

        # Compute performance statistics
        self.total_return = self.calculate_total_return()
        self.annualized_return = self.calculate_annualized_return()
        self.volatility = self.calculate_volatility()
        self.sharpe_ratio = self.calculate_sharpe_ratio()
        self.max_drawdown = self.calculate_max_drawdown()
        self.sortino_ratio = self.calculate_sortino_ratio()

        # Others
        self.monthly_returns_df = self.calculate_monthly_returns()
        self.daily_volatility_series = self.calculate_daily_volatility()

    def calculate_total_return(self):
        """
        Calculate the total return over the period.
        """
        return self.cumulative_returns.iloc[-1]

    def calculate_annualized_return(self):
        """
        Calculate the annualized return using actual number of years.
        """
        total_days = (self.portfolio_returns.index[-1] - self.portfolio_returns.index[0]).days
        years = total_days / 365.25  # Considering leap years
        return (1 + self.total_return) ** (1 / years) - 1

    def calculate_volatility(self):
        """
        Calculate the annualized volatility.
        """
        return self.portfolio_returns.std(ddof=1) * np.sqrt(Result.PERIODS_PER_YEAR)

    def calculate_sharpe_ratio(self):
        """
        Calculate the Sharpe Ratio.
        """
        excess_returns = self.portfolio_returns - self.risk_free_rate / Result.PERIODS_PER_YEAR
        annualized_excess_return = excess_returns.mean() * Result.PERIODS_PER_YEAR
        annualized_volatility = excess_returns.std(ddof=1) * np.sqrt(Result.PERIODS_PER_YEAR)
        return annualized_excess_return / annualized_volatility

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
        annualized_downside_std = downside_returns.std(ddof=1) * np.sqrt(Result.PERIODS_PER_YEAR)
        if annualized_downside_std == 0:
            return np.nan
        return (self.annualized_return - self.risk_free_rate) / annualized_downside_std

    def calculate_monthly_returns(self):
        """
        Calculate the monthly returns as a DataFrame, grouped by year and month.

        Returns:
        pd.DataFrame: Monthly returns with years as rows and months as columns.
        """
        # Resample portfolio returns to monthly frequency and calculate monthly returns
        monthly_returns = self.portfolio_returns.resample('M').sum()

        # Create a DataFrame with year and month columns
        monthly_returns_df = monthly_returns.to_frame(name='Monthly Returns')
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month

        # Pivot the table with years as rows and months as columns
        pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Monthly Returns')

        # Map month numbers to month abbreviations
        pivot.columns = [calendar.month_abbr[int(m)] for m in pivot.columns]

        return pivot

    def calculate_daily_volatility(self, window=30):
        """
        Calculate rolling daily volatility over a rolling window.

        Parameters:
        window (int): Rolling window size in days (e.g., 30 for monthly).

        Returns:
        pd.Series: Rolling daily volatility.
        """
        rolling_volatility = (self.portfolio_returns.rolling(window=window).std(ddof=1) * np.sqrt(Result.PERIODS_PER_YEAR))

        return rolling_volatility

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
