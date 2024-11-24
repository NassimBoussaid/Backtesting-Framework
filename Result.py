import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import calendar


class Result:
    def __init__(self, index_values, risk_free_rate=0.0, periods_per_year=252):
        """
        Initialize the Result object.

        Parameters:
        index_values (pd.Series): Time series of the index levels (portfolio values), with dates as the index.
        risk_free_rate (float): Risk-free rate for Sharpe/Sortino ratio calculations.
        periods_per_year (int): Number of periods in a year (e.g., 252 for daily data).
        """
        if not isinstance(index_values, pd.Series):
            raise TypeError("index_values must be a pandas Series with dates as the index.")
        if not isinstance(index_values.index, pd.DatetimeIndex):
            raise TypeError("index_values must have a DatetimeIndex.")
        self.index_values = index_values
        self.returns = self.index_values.pct_change().dropna()  # daily returns
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

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
        return (self.index_values.iloc[-1] / self.index_values.iloc[0]) - 1

    def calculate_annualized_return(self):
        """
        Calculate the annualized return using actual number of years.
        """
        total_days = (self.index_values.index[-1] - self.index_values.index[0]).days
        years = total_days / 365.25  # Considering leap years
        return (1 + self.total_return) ** (1 / years) - 1

    def calculate_volatility(self):
        """
        Calculate the annualized volatility.
        """
        return self.returns.std(ddof=1) * np.sqrt(self.periods_per_year)  # ddof = 1 to do / N-1 instead of N

    def calculate_sharpe_ratio(self):
        """
        Calculate the Sharpe Ratio.
        """
        excess_returns = self.returns - self.risk_free_rate / self.periods_per_year
        annualized_excess_return = excess_returns.mean() * self.periods_per_year
        annualized_volatility = excess_returns.std(ddof=1) * np.sqrt(self.periods_per_year)
        return annualized_excess_return / annualized_volatility

    def calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown.
        """
        cumulative_max = self.index_values.cummax()
        drawdowns = (self.index_values - cumulative_max) / cumulative_max
        return drawdowns.min()

    def calculate_sortino_ratio(self):
        """
        Calculate the Sortino Ratio.
        """
        downside_returns = self.returns[self.returns < 0]
        expected_return = self.returns.mean() * self.periods_per_year
        downside_std = downside_returns.std(ddof=1) * np.sqrt(self.periods_per_year)
        if downside_std == 0:
            return np.nan
        else:
            return (expected_return - self.risk_free_rate) / downside_std

    def calculate_monthly_returns(self):
        """
        Calculate the monthly returns as a DataFrame, grouped by year and month.

        Returns:
        pd.DataFrame: Monthly returns with years as rows and months as columns.
        """
        monthly_returns = self.index_values.resample('M').ffill().pct_change()
        monthly_returns_df = monthly_returns.to_frame(name='Monthly Returns')
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month

        # Create a pivot table with years as rows and months as columns
        pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Monthly Returns')

        # Map month numbers to month abbreviations
        pivot.columns = [calendar.month_abbr[int(m)] for m in pivot.columns]

        return pivot

    def calculate_daily_volatility(self, window=30):
        """
        Calculate rolling daily volatility over a rolling window.

        Parameters:
        window (int): Rolling window size in days (30 for monthly).

        Returns:
        pd.Series: Rolling daily volatility.
        """
        rolling_volatility = self.returns.rolling(window=window).std() * np.sqrt(self.periods_per_year)
        return rolling_volatility

    def plot_monthly_returns_heatmap(self):
        """
        Plot a heatmap of the monthly returns
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.monthly_returns_df * 100,  # %
            cmap='RdYlGn',
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={'label': 'Monthly Return (%)'},
            center=0  # Neutral value for color scale
        )
        plt.title("Monthly Returns", fontsize=16)
        plt.xlabel("Month")
        plt.ylabel("Year")
        plt.tight_layout()
        plt.show()

    def plot_index_level(self):
        """
        Plot the index levels over time.
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=self.index_values.index,
            y=self.index_values,
            color='blue'
        )
        plt.title("Index Level Over Time", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Index Level")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.tight_layout()
        plt.show()

    def plot_daily_volatility(self):
        """
        Plot the rolling daily volatility.
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x=self.daily_volatility_series.index,
            y=self.daily_volatility_series * 100,  # Convert to percentage
            color='orange'
        )
        plt.title("Daily Volatility", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Volatility (%)")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.tight_layout()
        plt.show()

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
