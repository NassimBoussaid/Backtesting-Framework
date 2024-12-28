from Core.Strategy import Strategy
import pandas as pd


class MeanReversion(Strategy):
    def __init__(self, window: int, zscore_threshold: int):
        """
        Stratégie de Mean Reversion basée sur le z-score.

        Paramètres:
        -----------
        window : int
            Nombre de périodes pour la moyenne glissante et l'écart-type.
        zscore_threshold : float
            Seuil du z-score pour prendre une position. Exemple: 1.0, 1.5, 2.0
        """
        self.window = window
        self.zscore_threshold = zscore_threshold

    def get_position(self, historical_data: pd.Series, current_position: float) -> float:
        """
        Détermine la position à prendre en fonction du z-score.

        Paramètres:
        -----------
        historical_data : pd.Series
            Données de prix historiques (y compris la valeur actuelle).
        current_position : float
            Position actuelle sur l'actif (1 = long, -1 = short, 0 = flat).

        Retour:
        -------
        float
            Nouvelle position (-1, 0, ou 1).
        """
        # S'il n'y a pas assez de données, on conserve la position actuelle
        if len(historical_data) < self.window:
            return current_position

        # On prend les dernières 'window' valeurs pour calculer la moyenne et l'écart-type
        recent_data = historical_data.iloc[-self.window:]
        mean = recent_data.mean()
        std = recent_data.std()

        # Prix actuel (dernière valeur du jeu de données)
        last_price = historical_data.iloc[-1]

        # Calcul du z-score
        # z-score = (valeur_actuelle - moyenne) / écart_type
        if std == 0:
            # Si l'écart-type est nul, on ne peut pas calculer le z-score
            return current_position

        zscore = (last_price - mean) / std

        # Si le z-score est plus grand que le seuil => trop cher => short
        if zscore > self.zscore_threshold:
            return -1
        # Si le z-score est plus petit que -seuil => trop bas => long
        elif zscore < -self.zscore_threshold:
            return 1
        else:
            # Pas d'anomalie de prix => pas de position
            return 0

    def fit(self, data):
        """
        Méthode d'entraînement optionnelle.
        Pour une stratégie de mean reversion simple, il n'y a rien à entraîner.
        """
        pass