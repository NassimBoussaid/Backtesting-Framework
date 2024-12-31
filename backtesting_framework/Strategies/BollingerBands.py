from backtesting_framework.Core.Strategy import Strategy
import pandas as pd
import numpy as np

class BollingerBands(Strategy):
    """
    Stratégie basée sur les Bandes de Bollinger :
    Identification des opportunités d'achat ou de vente lorsque les prix
    atteignent les bandes inférieure ou supérieure respectivement.
    """
    def __init__(self, window: int, num_std_dev: float):
        """
        Initialisation de la stratégie Bollinger Bands.

        :param window: Période de calcul pour la moyenne mobile et l'écart type.
        :param num_std_dev: Nombre d'écarts types pour les bandes supérieure et inférieure.
        """
        super().__init__(multi_asset=False)
        self.window = window
        self.num_std_dev = num_std_dev

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Détermination de la position en fonction des Bandes de Bollinger.

        :param historical_data: pd.DataFrame contenant les prix historiques, y compris la journée actuelle.
        :param current_position: Position actuelle (1 = achat, -1 = vente, 0 = neutre).
        :return: Nouvelle position basée sur le signal des Bandes de Bollinger (-1, 0 ou 1).
        """

        # Vérification de la disponibilité de suffisamment de données
        if len(historical_data) < self.window:
            return current_position

        # Extraction des prix
        prices = historical_data.values

        # Calcul des Bandes de Bollinger
        moving_average, upper_band, lower_band = self._calculate_bollinger_bands(prices)

        # Prix actuel
        current_price = prices[-1]

        # Détermination de la position
        if current_price < lower_band:
            return 1  # Position longue
        elif current_price > upper_band:
            return -1  # Position courte
        else:
            return 0  # Position neutre

    def _calculate_bollinger_bands(self, prices: np.ndarray):
        """
        Calcul des Bandes de Bollinger.

        :param prices: np.ndarray contenant les prix historiques.
        :return: Tuple contenant la moyenne mobile, la bande supérieure et la bande inférieure.
        """

        # Calcul de la moyenne mobile
        moving_average = np.mean(prices[-self.window:])
        # Calcul de l'écart type
        standard_deviation = np.std(prices[-self.window:])

        # Calcul des bandes
        upper_band = moving_average + self.num_std_dev * standard_deviation
        lower_band = moving_average - self.num_std_dev * standard_deviation

        return moving_average, upper_band, lower_band

    def fit(self, data):
        """
        Méthode d'ajustement optionnelle. Non utilisée pour cette stratégie.
        """
        pass
