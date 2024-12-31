from backtesting_framework.Core.Strategy import Strategy
import pandas as pd


class MeanReversion(Strategy):
    """
    Stratégie de Mean Reversion basée sur le z-score :
    Identification des anomalies de prix par rapport à la moyenne.
    """
    def __init__(self, window: int, zscore_threshold: int):
        """
        Initialisation de la stratégie Mean Reversion.

        :param window: Nombre de périodes pour le calcul de la moyenne et de l'écart-type.
        :param zscore_threshold: Seuil du z-score pour prendre une position.
        """
        super().__init__(multi_asset=False)
        self.window = window
        self.zscore_threshold = zscore_threshold

    def get_position(self, historical_data: pd.Series, current_position: float) -> float:
        """
        Détermination de la position à prendre en fonction du z-score.

        :param historical_data: pd.Series contenant les données de prix historiques.
        :param current_position: Position actuelle sur l'actif (1 = long, -1 = short, 0 = neutre).
        :return: Nouvelle position (-1 = short, 1 = long, 0 = neutre).
        """
        # Vérification du nombre suffisant de données pour les calculs
        if len(historical_data) < self.window:
            return current_position

        # Sélection des dernières valeurs pour les calculs
        recent_data = historical_data.iloc[-self.window:]

        # Calcul de la moyenne et de l'écart-type
        mean = recent_data.mean()
        std = recent_data.std()

        # Extraction du prix actuel
        last_price = historical_data.iloc[-1]

        # Vérification de l'écart-type pour éviter les divisions par zéro
        if std == 0:
            return current_position

        # Calcul du z-score
        zscore = (last_price - mean) / std

        # Définition de la position en fonction du z-score
        if zscore > self.zscore_threshold:
            return -1  # Prix trop élevé, position courte
        elif zscore < -self.zscore_threshold:
            return 1  # Prix trop bas, position longue
        else:
            return 0  # Pas de position si le prix est dans une plage normale

    def fit(self, data):
        """
        Méthode optionnelle pour l'ajustement. Non utilisée dans cette stratégie.
        """
        pass