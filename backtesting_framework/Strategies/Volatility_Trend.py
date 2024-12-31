from backtesting_framework.Core.Strategy import Strategy
import pandas as pd

class VolatilityTrendStrategy(Strategy):
    """
    Stratégie basée sur la volatilité et la tendance :
    Prend une position long si l'ATR est supérieur au seuil et que le DMI est positif,
    sinon prend une position courte ou neutre.
    """
    def __init__(self, atr_period=14, dmi_period=14, atr_threshold=1.0):
        """
        Initialisation de la stratégie VolatilityTrendStrategy.

        :param atr_period: Nombre de périodes pour le calcul de l'ATR.
        :param dmi_period: Nombre de périodes pour le calcul du DMI.
        :param atr_threshold: Seuil utilisé pour déterminer la volatilité.
        """
        super().__init__(multi_asset=False)
        self.atr_period = atr_period
        self.dmi_period = dmi_period
        self.atr_threshold = atr_threshold

    def calculate_atr(self, prices, period):
        """
        Calcule l'Average True Range (ATR) pour une série de prix donnée.
        L'ATR mesure la volatilité des prix sur une période donnée.

        :param prices: Série de prix historiques.
        :param period: Nombre de périodes pour le calcul de l'ATR.
        :return: Valeur de l'ATR calculée.
        """
        delta = prices[1:] - prices[:-1]
        delta = pd.Series(delta)
        atr = delta.abs().rolling(period).mean().iloc[-1]
        return atr

    def calculate_dmi(self,prices,period):
        """
        Calcule le Directional Movement Index (DMI) pour une série de prix donnée.
        Le DMI mesure la force de la tendance basée sur les mouvements directionnels positifs et négatifs.

        :param prices: Série de prix historiques.
        :param period: Nombre de périodes pour le calcul du DMI.
        :return: Valeur du DMI calculée.
        """
        delta = prices[1:] - prices[:-1]
        delta = pd.Series(delta)
        delta = delta.reset_index(drop=True)

        # Calcul des mouvements directionnels positifs et négatifs
        pos_dm = delta.where(delta > 0, 0).rolling(period).sum().iloc[-1]
        neg_dm = -delta.where(delta < 0, 0).rolling(period).sum().iloc[-1]
        atr = self.calculate_atr(prices,period)

        # Calcul de l'ATR pour normaliser les mouvements directionnels
        pos_di = 100 * pos_dm / atr
        neg_di = 100 * neg_dm / atr

        # Calcul du DMI basé sur les indices directionnels
        dmi = abs(pos_di - neg_di) / (pos_di + neg_di) * 100
        return dmi

    def get_position(self,historical_data:pd.DataFrame,current_position:float)->float:
        """
        Détermine la position à prendre (longue, courte ou neutre)
        en fonction de la volatilité et de la tendance.

        :param historical_data: pd.DataFrame contenant les prix historiques pour un actif donné.
        :param current_position: Position actuelle sur l'actif (1 = long, -1 = short, 0 = neutre).
        :return: La nouvelle position à prendre (1 = long, -1 = short, 0 = neutre).
        """
        prices = historical_data.values

        # Calcul de l'ATR et du DMI pour les données historiques
        atr = self.calculate_atr(prices,self.atr_period)
        dmi = self.calculate_dmi(prices,self.dmi_period)
        if atr > self.atr_threshold and dmi > 0:
            return 1  # Position longue
        elif atr > self.atr_threshold and dmi < 0:
            return -1  # Position courte
        else:
            return 0  # Position neutre

    def fit(self, data):
        """
        Méthode optionnelle d'ajustement (fit). Non utilisée pour cette stratégie.
        """
        pass