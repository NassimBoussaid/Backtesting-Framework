from datetime import datetime, timedelta
import pandas as pd
from typing import List
from workalendar.usa import UnitedStates

class Calendar:
    """
    Classe de gestion du calendrier de trading :
    Gestion des jours de trading et des dates de rebalancement en fonction des fréquences spécifiées,
    en tenant compte des jours fériés fédéraux aux États-Unis.
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
        Initialisation de la classe Calendar.

        :param frequency: Fréquence des rebalancements ('daily', 'monthly', 'quarterly', etc.).
        :param start_date: Date de début au format 'YYYY-MM-DD'.
        :param end_date: Date de fin au format 'YYYY-MM-DD'.
        :raises ValueError: Fréquence non supportée ou format de date incorrect.
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
        Identification d'un jour de trading (pas un week-end ou un jour férié fédéral).

        :param date: Date à vérifier.
        :return: True si c'est un jour de trading, False sinon.
        """
        return date.weekday() < 5 and date.date() not in self.holidays

    @classmethod
    def get_holidays_in_range(cls, start_date: datetime, end_date: datetime) -> List[datetime.date]:
        """
        Extraction des jours fériés fédéraux dans une plage de dates.

        :param start_date: Date de début.
        :param end_date: Date de fin.
        :return: Liste des jours fériés.
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
        Génération des jours de trading dans la plage définie, hors week-ends et jours fériés.

        :return: Liste des jours de trading.
        """
        all_business_days = pd.bdate_range(start=self.start_date, end=self.end_date, freq='C',
                                           holidays=self.holidays).tolist()
        return all_business_days

    def _adjust_to_next_trading_day(self, date: datetime) -> datetime:
        """
        Ajustement d'une date au prochain jour de trading disponible si nécessaire.

        :param date: Date à ajuster.
        :return: Jour de trading ajusté.
        """
        adjusted_date = date
        while not self._is_trading_day(adjusted_date):
            adjusted_date += timedelta(days=1)
            if adjusted_date > self.end_date:
                raise ValueError("Adjusted date exceeds the end_date range.")
        return adjusted_date

    def _generate_rebalancing_dates(self) -> set[datetime]:
        """
        Génération des dates de rebalancement en fonction de la fréquence et de la plage de dates.

        :return: Ensemble des dates de rebalancement.
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
        Vérification si une date est une date de rebalancement.

        :param date: Date au format 'YYYY-MM-DD'.
        :return: True si c'est une date de rebalancement, False sinon.
        """
        try:
            check_date = pd.to_datetime(date).to_pydatetime()
        except Exception as e:
            raise ValueError(f"Error parsing date '{date}': {e}")

        return check_date in self.rebalancing_dates

    def add_rebalancing_date(self, date: str):
        """
        Ajout d'une date de rebalancement personnalisée, ajustée au jour de trading suivant si nécessaire.

        :param date: Date à ajouter au format 'YYYY-MM-DD'.
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
        Suppression d'une date de rebalancement.

        :param date: Date à supprimer au format 'YYYY-MM-DD'.
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
        """
        Représentation textuelle de l'objet Calendar.
        """
        return (f"Calendar(frequency='{self.frequency}', "
                f"start_date='{self.start_date.date()}', "
                f"end_date='{self.end_date.date()}', "
                f"all_dates={len(self.all_dates)} trading days, "
                f"rebalancing_dates={len(self.rebalancing_dates)} dates)")
