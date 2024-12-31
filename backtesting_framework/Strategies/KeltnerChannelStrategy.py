from backtesting_framework.Core.Strategy import Strategy
import pandas as pd

class KeltnerChannelStrategy(Strategy):
    """
    Stratégie basée sur les canaux de Keltner :
    Position longue si le prix dépasse la bande supérieure,
    position courte si le prix est en dessous de la bande inférieure.
    """
    def __init__(self, atr_period=10, atr_multiplier=2.0, sma_period=20):
        """
        Initialisation de la stratégie Keltner Channel.

        :param atr_period: Nombre de périodes pour le calcul de l'ATR.
        :param atr_multiplier: Multiplicateur pour l'ATR pour définir les bandes.
        :param sma_period: Nombre de périodes pour le calcul de la SMA.
        """
        super().__init__(multi_asset=False)
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.sma_period = sma_period

    def calculate_atr(self, prices, period):
        """
        Calcul de l'Average True Range (ATR).

        :param prices: Série de prix.
        :param period: Nombre de périodes pour le calcul de l'ATR.
        :return: Valeur de l'ATR calculée.
        """
        delta = prices[1:] - prices[:-1]
        delta = pd.Series(delta)
        atr = delta.abs().rolling(period).mean().iloc[-1]
        return atr

    def calculate_sma(self, prices, period):
        """
        Calcul de la Simple Moving Average (SMA).

        :param prices: Série de prix.
        :param period: Nombre de périodes pour le calcul de la SMA.
        :return: Valeur de la SMA calculée.
        """
        prices = pd.Series(prices)
        sma = prices.rolling(period).mean().iloc[-1]
        return sma

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Détermination de la position en fonction des canaux de Keltner.

        :param historical_data: pd.DataFrame contenant les prix historiques.
        :param current_position: Position actuelle (1 = long, -1 = short, 0 = neutre).
        :return: Nouvelle position à prendre (1 = long, -1 = short, 0 = neutre).
        """

        # Extraction des prix à partir des données historiques
        prices = historical_data.values

        # Calcul de l'ATR et de la SMA
        atr = self.calculate_atr(prices, self.atr_period)
        sma = self.calculate_sma(prices, self.sma_period)

        # Calcul des bandes supérieure et inférieure
        upper_band = sma + atr * self.atr_multiplier
        lower_band = sma - atr * self.atr_multiplier

        # Détermination de la position
        if prices[-1] > upper_band:
            return 1  # Position longue
        elif prices[-1] < lower_band:
            return -1  # Position courte
        else:
            return current_position  # Maintien de la position actuelle

    def fit(self, data):
        """
        Méthode optionnelle pour l'ajustement. Non utilisée dans cette stratégie.
        """
        pass