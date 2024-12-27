from Core.Strategy import Strategy
import pandas as pd


class RSI(Strategy):
    """
    Stratégie RSI : achète si RSI < oversold_threshold, vend si RSI > overbought_threshold.
    """

    def __init__(self, period:int, oversold_threshold: int, overbought_threshold: int):
        """
        Initialise la stratégie RSI avec ses paramètres.

        :param period: Période de calcul du RSI (par défaut 14).
        :param oversold_threshold: Seuil de survente (par défaut 30).
        :param overbought_threshold: Seuil de surachat (par défaut 70).
        """
        self.period = period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Détermine la position à prendre (long/short/flat) en fonction de la valeur du RSI.

        :param historical_data: Série Pandas (ou DataFrame d'une seule colonne) des prix historiques de l'actif
                                jusqu'à la date actuelle (incluse).
        :param current_position: Position courante (1 pour long, -1 pour short, 0 pour flat, etc.)
        :return: Nouvelle position : 1, -1 ou 0 (selon la logique RSI).
        """

        # On s'assure qu'on a suffisamment d'historique pour calculer le RSI
        if len(historical_data) < self.period:
            return current_position

        # Extraire uniquement les prix (sous forme de numpy array)
        prices = historical_data.values

        # Calcul du RSI
        rsi_value = self.compute_rsi(prices, self.period)

        # Logique d'entrée/sortie basée sur le RSI
        if rsi_value < self.oversold_threshold:
            # Zone de survente => on achète (long)
            return 1
        elif rsi_value > self.overbought_threshold:
            # Zone de surachat => on vend (short)
            return -1
        else:
            # Sinon, on peut choisir de rester à 0 ou de conserver la position courante
            # return current_position
            return 0

    def compute_rsi(self, prices, period: int):
        """
        Calcule la valeur du RSI (Relative Strength Index) sur la fenêtre donnée.

        :param prices: Numpy array contenant l'historique de prix (Close ou autre).
        :param period: Nombre de périodes pour le calcul du RSI (par défaut 14).
        :return: Valeur du RSI (entre 0 et 100).
        """
        # Convertir en Series pour utiliser diff() facilement
        prices_series = pd.Series(prices)

        # Calcul des variations
        delta = prices_series.diff(1).dropna()

        # On sépare les gains et les pertes
        gains = delta.where(delta > 0, 0.0)
        losses = delta.where(delta < 0, 0.0).abs()

        # Moyenne des gains et pertes sur "period" périodes (méthode exponentielle lissée, standard pour le RSI)
        # On démarre par une moyenne arithmétique des "period" premières valeurs,
        # puis on utilise la méthode lissée (Wilders).

        avg_gain = gains.rolling(window=period, min_periods=period).mean().iloc[period - 1]
        avg_loss = losses.rolling(window=period, min_periods=period).mean().iloc[period - 1]

        # Ensuite, on fait le calcul progressif lissé en partant de la période "period+1"
        # pour chaque nouvelle valeur de gain/perte, on met à jour selon la formule de Wilders.
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period

        # Éviter la division par zéro
        if avg_loss == 0:
            return 100.0  # RSI = 100 => surachat extrême

        # Calcul du RSI
        rs = avg_gain / avg_loss
        rsi_value = 100 - (100 / (1 + rs))
        return rsi_value

    def fit(self, data):
        """
        Méthode de fit (optionnelle). Peut être laissée vide si la stratégie ne nécessite pas d'entraînement.
        """
        pass
