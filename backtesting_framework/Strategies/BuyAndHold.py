from backtesting_framework.Core.Strategy import Strategy
import pandas as pd

class BuyAndHold(Strategy):
    """
    Stratégie Buy and Hold : Achète tous les actifs à parts égales au début
    et les conserve sans effectuer de rebalancement.
    """

    def __init__(self):
        """
        Initialisation de la stratégie Buy and Hold.
        """
        super().__init__(multi_asset=False)
        pass

    def fit(self, data):
        """
        Préparation des données pour la stratégie Buy and Hold.
        :param data: DataFrame Pandas contenant les prix des actifs (index = dates, colonnes = tickers).
        """

    def get_position(self, historical_data: pd.Series, current_position: float) -> float:
        """
        Détermine la position à prendre pour un actif donné à la date courante.
        Dans Buy and Hold, la position est toujours long.

        :param historical_data: pd.Series de prix pour un actif donné (index = dates).
        :param current_position: Position actuelle (non utilisé dans Buy and Hold).
        :return: Position fixe à 1.0 (long) pour Buy and Hold.
        """
        return 1.0