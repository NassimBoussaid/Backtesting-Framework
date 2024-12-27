from Core.Strategy import Strategy
import pandas as pd
import numpy as np

class Quality(Strategy):
    """
    Stratégie Quality : Achète les actions de meilleure qualité (ROE élevé et ROA élevé)
    et vend les actions de moins bonne qualité.

    L'attribution des scores se fait selon deux métriques : le ROE (Return on Equity)
    et le ROA (Return on Assets).
    """

    def __init__(self, window: int = 30, assets_picked_long: int = 5, assets_picked_short: int = 5):
        """
        Initialisation de la stratégie Quality.

        :param window: Période, en nombre de jours, de la fenêtre glissante pour lisser les métriques.
        :param assets_picked_long: Nombre d'actifs à acheter.
        :param assets_picked_short: Nombre d'actifs à vendre.
        """
        self.window = window
        self.assets_picked_long = assets_picked_long
        self.assets_picked_short = assets_picked_short
        # DataFrame qui contiendra les scores finaux pour chaque actif et chaque date.
        self.ranking_df = None

    def fit(self, data):
        """
        Calcul du rang de chaque actif selon son ROE et ROA sur la dernière fenêtre glissante.
        Plus le ROE est haut, plus l'actif est de qualité et ainsi plus le score est élevé.
        Plus le ROA est haut, plus l'actif est de qualité et ainsi plus le score est élevé.

        :param data: Dictionnaire contenant deux DataFrames pour les métriques ROE et ROA.
        """
        # Vérification que les données soient bien un dictionnaire avec les clés "ROE" et "ROA"
        if not isinstance(data, dict):
            raise TypeError("Les données doivent être passées sous forme d'un dictionnaire {'ROE': df_roe, 'ROA': df_roa}.")
        if "ROE" not in data or "ROA" not in data:
            raise KeyError("Le dictionnaire 'data' doit contenir les clés 'ROE' et 'ROA'.")

        roe_df = data["ROE"]
        roa_df = data["ROA"]

        # Remplacement des valeurs invalides "#N/A N/A" par NaN
        roe_df.replace("#N/A N/A", np.nan, inplace=True)
        roa_df.replace("#N/A N/A", np.nan, inplace=True)

        # Calcul des moyennes glissantes pour le ROE et le ROA
        roe_rolling = roe_df.rolling(self.window).mean()
        roa_rolling = roa_df.rolling(self.window).mean()

        # Calcul des rangs pour le ROE et le ROA
        roe_score = roe_rolling.rank(axis=1, method="first", ascending=True)
        roa_score = roa_rolling.rank(axis=1, method="first", ascending=True)

        # Calcul du score moyen en combinant les deux rangs
        combined_score = (roe_score + roa_score) / 2

        # Calcul du score final : plus le score est élevé, plus le rang est élevé
        self.ranking_df = combined_score.rank(axis=1, method="min", ascending=True)

    def get_position(self, historical_data: pd.Series, current_position: float) -> float:
        """
        Détermine la position à prendre (long, short ou neutral) pour un actif donné à une date donnée.

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

        # Exclusion des NaN pour calculer le nombre total d'actifs à la date donnée
        valid_tickers = self.ranking_df.loc[current_date].dropna()
        total_tickers = len(valid_tickers)

        # Détermination de la position en fonction des paramètres
        if rank_value > (total_tickers - self.assets_picked_long):
            return 1.0  # Position long
        elif rank_value <= self.assets_picked_short:
            return -1.0  # Position short
        else:
            return 0.0  # Position neutral
