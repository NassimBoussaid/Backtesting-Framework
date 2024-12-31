from backtesting_framework.Core.Strategy import Strategy
import pandas as pd

class MovingAverage(Strategy):
    """
    Stratégie de moyenne mobile :
    Prend une position longue si la moyenne mobile courte (MA courte) dépasse
    la moyenne mobile longue (MA longue), sinon une position courte.
    """
    def __init__(self, short_window: int, long_window: int, exponential_mode=False):
        """
        Initialisation de la stratégie de moyenne mobile.

        :param short_window: Nombre de périodes pour la moyenne mobile courte.
        :param long_window: Nombre de périodes pour la moyenne mobile longue.
        :param exponential_mode: Booléen, si True, calcule les moyennes mobiles exponentielles (EMA) au lieu des moyennes simples (SMA).
        """
        super().__init__(multi_asset=False)
        self.short_window = short_window
        self.long_window = long_window
        self.exponential_mode = exponential_mode

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Détermine la position à prendre (longue, courte ou neutre) en fonction des moyennes mobiles.

        :param historical_data: pd.DataFrame contenant les données de prix historiques, y compris la journée actuelle.
        :param current_position: Position actuelle (1 pour long, -1 pour short, 0 pour neutre).
        :return: La nouvelle position à prendre (1 = long, -1 = short, 0 = neutre).
        """
        if len(historical_data) < max(self.short_window, self.long_window):
            # Pas assez de données pour calculer les signaux de la stratégie
            return current_position

        prices = historical_data.values

        if self.exponential_mode:
            # Calcul des EMA pour les fenêtres courte et longue
            ma_short = self._calculate_ema(prices, self.short_window)
            ma_long = self._calculate_ema(prices, self.long_window)
        else:
            # Calcul des SMA pour les fenêtres courte et longue
            ma_short = self._calculate_sma(prices, self.short_window)
            ma_long = self._calculate_sma(prices, self.long_window)

        if ma_short > ma_long:
            return 1  # Position longue
        elif ma_short < ma_long:
            return -1  # Position courte
        else:
            return current_position  # Maintenir la position actuelle

    def _calculate_sma(self, prices, window: int) -> float:
        """
        Calcule manuellement la moyenne mobile simple (SMA).

        :param prices: Tableau de prix.
        :param window: Nombre de périodes pour calculer la SMA.
        :return: Valeur de la SMA sur les dernières `window` périodes.
        """
        relevant_prices = prices[-window:]
        return relevant_prices.sum() / window

    def _calculate_ema(self, prices, window: int) -> float:
        """
        Calcule manuellement la moyenne mobile exponentielle (EMA).

        :param prices: Tableau de prix.
        :param window: Nombre de périodes pour calculer l'EMA.
        :return: Valeur de l'EMA sur les dernières `window` périodes.
        """
        alpha = 2 / (window + 1)
        # On utilise la première valeur comme point de départ de l'EMA
        relevant_prices = prices[-window:]
        ema = relevant_prices[0]

        # Calcul incrémental de l'EMA pour les prix suivants
        for price in relevant_prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def fit(self, data):
        """
        Méthode optionnelle d'ajustement (fit). Non utilisée pour cette stratégie.
        """
        pass