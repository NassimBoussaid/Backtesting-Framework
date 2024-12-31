from backtesting_framework.Core.Strategy import Strategy
import pandas as pd
import numpy as np

class Size(Strategy):
    """
    Stratégie Size : Achète les actions avec la plus petite capitalisation boursière
    et vend les actions avec la plus grande capitalisation boursière.

    L'attribution des scores se fait sur une seule métrique : la capitalisation boursière.
    """

    def __init__(self, window: int = 30, assets_picked_long: int = 5, assets_picked_short: int = 5):
        """
        Initialisation de la stratégie Size.

        :param window: Période, en nombre de jours, de la fenêtre glissante pour lisser les métriques.
        :param assets_picked_long: Nombre d'actifs à acheter.
        :param assets_picked_short: Nombre d'actifs à vendre.
        """
        super().__init__(multi_asset=False)
        self.window = window
        self.assets_picked_long = assets_picked_long
        self.assets_picked_short = assets_picked_short
        # DataFrame qui contiendra les scores finaux pour chaque actif et chaque date.
        self.ranking_df = None

    def fit(self, market_cap_data):
        """
        Calcul du rang de chaque actif selon sa capitalisation boursière sur la dernière fenêtre glissante.
        Plus la capitalisation est basse, plus le score est élevé.

        :param market_cap_data: DataFrame contenant les capitalisations boursières des actifs.
        """
        # Vérification des données
        if not isinstance(market_cap_data, pd.DataFrame):
            raise TypeError("Les données doivent être un DataFrame Pandas contenant les capitalisations boursières.")

        # Remplacement des valeurs invalides "#N/A N/A" ou autres par NaN
        market_cap_data.replace("#N/A N/A", np.nan, inplace=True)

        # Calcul des moyennes glissantes pour lisser les capitalisations boursières
        market_cap_rolling = market_cap_data.rolling(self.window).mean()

        # Calcul des rangs pour la capitalisation boursière
        market_cap_score = market_cap_rolling.rank(axis=1, method="first", ascending=False)

        # Calcul du score final (basé uniquement sur le rang de la capitalisation boursière)
        self.ranking_df = market_cap_score.rank(axis=1, method="min", ascending=True)

    def get_position(self, historical_data: pd.Series, current_position: float) -> float:
        """
        Détermine la position à prendre (long, short ou neutre) pour un actif donné à une date donnée.

        :param historical_data: pd.Series contenant les prix historiques pour un actif donné.
        :param current_position: Position actuelle sur l'actif (1 pour long, -1 pour short, 0 pour neutral).
        :return: La nouvelle position à prendre (1 = long, -1 = short, 0 = neutral).
        """
        # Récupération du ticker et de la date
        current_ticker = historical_data.name
        current_date = historical_data.index[-1]

        # Absence du ticker ou de la date dans le DataFrame de ranking
        if (current_ticker not in self.ranking_df.columns) or (current_date not in self.ranking_df.index):
            return 0.0

        # Récupération du rang du ticker à la date donnée
        rank_value = self.ranking_df.loc[current_date, current_ticker]

        # Si le rang est NaN, renvoie une position neutral
        if pd.isna(rank_value):
            return 0.0

        # Exclusion des NaN pour calcul le nombre total d'actifs à la date donnée
        valid_tickers = self.ranking_df.loc[current_date].dropna()
        total_tickers = len(valid_tickers)

        # Détermination de la position en fonction des paramètres
        if rank_value > (total_tickers - self.assets_picked_long):
            return 1.0  # Position long
        elif rank_value <= self.assets_picked_short:
            return -1.0  # Position short
        else:
            return 0.0  # Position neutre