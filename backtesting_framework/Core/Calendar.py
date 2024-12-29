from datetime import datetime, timedelta
import pandas as pd
from typing import List
from workalendar.usa import UnitedStates


class Calendar:
    """
    A Calendar class to manage all trading dates and rebalancing dates based on specified frequencies,
    accounting for U.S. federal holidays.

    Attributes:
        frequency (str): Frequency of rebalancing ('monthly', 'quarterly', 'yearly', etc.).
        start_date (datetime): The start date for generating dates.
        end_date (datetime): The end date for generating dates.
        all_dates (List[datetime]): A list of all trading dates within the start and end dates.
        rebalancing_dates (List[datetime]): A list of generated rebalancing dates based on frequency.
    """

    FREQUENCY_MAPPING = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME',
        'quarterly': 'Q',
        'yearly': 'A',
    }

    CALENDAR = UnitedStates()

    def __init__(self, frequency: str, start_date: str, end_date: str):
        """
        Initializes the Calendar object with a specified frequency and date range,
        accounting for U.S. federal holidays.

        Args:
            frequency (str): The frequency of rebalancing ('monthly', 'quarterly', etc.).
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Raises:
            ValueError: If the frequency is not supported or date formats are incorrect.
        """
        frequency = frequency.lower()
        if frequency not in self.FREQUENCY_MAPPING:
            raise ValueError(
                f"Unsupported frequency '{frequency}'. Supported frequencies are: {list(self.FREQUENCY_MAPPING.keys())}"
            )

        try:
            self.start_date = pd.to_datetime(start_date)
            self.end_date = pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError(f"Error parsing dates: {e}")

        if self.start_date > self.end_date:
            raise ValueError("start_date must be earlier than end_date.")

        self.frequency = frequency
        self.holidays = set(self.get_holidays_in_range(self.start_date, self.end_date))

        self.all_dates = self._generate_all_trading_dates()
        self.rebalancing_dates = self._generate_rebalancing_dates()

    def _is_trading_day(self, date: datetime) -> bool:
        """
        Checks if a given date is a trading day (i.e., not a weekend or U.S. holiday).

        Args:
            date (datetime): The date to check.

        Returns:
            bool: True if the date is a trading day, False otherwise.
        """
        return date.weekday() < 5 and date.date() not in self.holidays

    @classmethod
    def get_holidays_in_range(cls, start_date: datetime, end_date: datetime) -> List[datetime.date]:
        """
        Retrieves all federal holidays within a specified date range.

        Args:
            start_date (datetime): The start date.
            end_date (datetime): The end date.

        Returns:
            List[datetime.date]: A list of holiday dates.
        """
        holidays = []
        start_year = start_date.year
        end_year = end_date.year

        for year in range(start_year, end_year + 1):
            yearly_holidays = cls.CALENDAR.holidays(year)
            for date, _ in yearly_holidays:
                if start_date.date() <= date <= end_date.date():
                    holidays.append(date)

        return holidays

    def _generate_all_trading_dates(self) -> List[datetime]:
        """
        Generates a list of all trading dates within the start and end dates,
        excluding weekends and U.S. federal holidays.

        Returns:
            List[datetime]: A list of all trading dates.
        """
        all_business_days = pd.bdate_range(start=self.start_date, end=self.end_date, freq='C',
                                           holidays=self.holidays).tolist()
        return all_business_days

    def _adjust_to_next_trading_day(self, date: datetime) -> datetime:
        """
        Adjusts a date to the next available trading day if it falls on a non-trading day.

        Args:
            date (datetime): The date to adjust.

        Returns:
            datetime: The adjusted trading day.
        """
        adjusted_date = date
        while not self._is_trading_day(adjusted_date):
            adjusted_date += timedelta(days=1)
            if adjusted_date > self.end_date:
                raise ValueError("Adjusted date exceeds the end_date range.")
        return adjusted_date

    def _generate_rebalancing_dates(self) -> set[datetime]:
        """
        Generates a list of rebalancing dates based on the frequency and date range,
        adjusted to ensure they fall on trading days.

        Returns:
            List[datetime]: A list of rebalancing dates.
        """
        freq_alias = self.FREQUENCY_MAPPING[self.frequency]
        scheduled_dates = pd.date_range(start=self.start_date, end=self.end_date, freq=freq_alias)
        rebalancing_dates = sorted(
            self._adjust_to_next_trading_day(date.to_pydatetime())
            for date in scheduled_dates
        )
        return set(rebalancing_dates)

    def is_rebalancing_date(self, date: str) -> bool:
        """
        Args:
            date (str): The date to check in 'YYYY-MM-DD' format.

        Returns:
            bool: True if the date is a rebalancing date, False otherwise.
        """
        try:
            check_date = pd.to_datetime(date).to_pydatetime()
        except Exception as e:
            raise ValueError(f"Error parsing date '{date}': {e}")

        return check_date in self.rebalancing_dates

    def add_rebalancing_date(self, date: str):
        """
        Adds a custom rebalancing date to the set, adjusted to the next trading day if necessary.

        Args:
            date (str): The date to add in 'YYYY-MM-DD' format.
        """
        try:
            new_date = pd.to_datetime(date).to_pydatetime()
        except Exception as e:
            raise ValueError(f"Error parsing date '{date}': {e}")

        if not (self.start_date <= new_date <= self.end_date):
            raise ValueError("The new rebalancing date must be within the start_date and end_date range.")

        adjusted_date = self._adjust_to_next_trading_day(new_date)

        # Attempt to add the adjusted_date to the set
        if adjusted_date in self.rebalancing_dates:
            raise ValueError("The date is already a rebalancing date.")
        else:
            self.rebalancing_dates.add(adjusted_date)

    def remove_rebalancing_date(self, date: str):
        """
        Removes a rebalancing date from the list.

        Args:
            date (str): The date to remove in 'YYYY-MM-DD' format.
        """
        try:
            remove_date = pd.to_datetime(date).to_pydatetime()
        except Exception as e:
            raise ValueError(f"Error parsing date '{date}': {e}")

        if remove_date in self.rebalancing_dates:
            self.rebalancing_dates.remove(remove_date)
        else:
            raise ValueError("The date is not in the list of rebalancing dates.")

    def __repr__(self):
        return (f"Calendar(frequency='{self.frequency}', "
                f"start_date='{self.start_date.date()}', "
                f"end_date='{self.end_date.date()}', "
                f"all_dates={len(self.all_dates)} trading days, "
                f"rebalancing_dates={len(self.rebalancing_dates)} dates)")
