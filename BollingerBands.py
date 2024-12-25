from Strategy import Strategy
import pandas as pd
import numpy as np

class BollingerBands(Strategy):
    def __init__(self, window: int, num_std_dev: float):
        """
        Une stratégie basée sur les Bandes de Bollinger.

        Paramètres :
        -----------
        window : int
            Période de calcul pour la moyenne mobile et l'écart type.
        num_std_dev : float
            Nombre d'écarts types pour les bandes supérieure et inférieure.
        """
        self.window = window
        self.num_std_dev = num_std_dev

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Détermine la position basée sur les Bandes de Bollinger.

        Paramètres :
        -----------
        historical_data : pd.DataFrame
            Données historiques des prix, incluant le jour actuel.
        current_position : float
            Position actuelle (par exemple, 1 pour achat, -1 pour vente, 0 pour neutre).

        Retourne :
        --------
        float
            Nouvelle position (-1, 0 ou 1) basée sur le signal des Bandes de Bollinger.
        """
        if len(historical_data) < self.window:
            # Pas assez de données pour calculer les Bandes de Bollinger
            return current_position

        prices = historical_data.values

        # Calcul des Bandes de Bollinger
        moving_average, upper_band, lower_band = self._calculate_bollinger_bands(prices)

        current_price = prices[-1]

        if current_price < lower_band:
            return 1  # Position d'achat
        elif current_price > upper_band:
            return -1  # Position de vente
        else:
            return 0  # Position neutre

    def _calculate_bollinger_bands(self, prices: np.ndarray):
        """
        Calcul des Bandes de Bollinger.

        Paramètres :
        -----------
        prices : array-like
            Tableau des prix historiques.

        Retourne :
        --------
        tuple
            Moyenne mobile, bande supérieure et bande inférieure.
        """
        moving_average = np.mean(prices[-self.window:])
        standard_deviation = np.std(prices[-self.window:])

        upper_band = moving_average + self.num_std_dev * standard_deviation
        lower_band = moving_average - self.num_std_dev * standard_deviation

        return moving_average, upper_band, lower_band

    def fit(self, data):
        """
        Méthode d'ajustement optionnelle. Non utilisée pour cette stratégie.
        """
        pass
