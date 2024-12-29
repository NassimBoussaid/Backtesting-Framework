from Core.Strategy import Strategy
import pandas as pd
import numpy as np

class Value(Strategy):
    """
    Stratégie Value : Achète les actions les plus sous-évaluées et vend les actions
    les plus surévaluées.

    L'attribution des scores se fait selon deux métriques : le PER (Price Earning Ratio)
    et le PBR (Price to Book Ratio).
    """

    def __init__(self, window: int = 30, assets_picked_long: int = 5, assets_picked_short: int = 5):
        """
        Initialisation de la stratégie Value.

        :param window: Période, en nombre de jours, de la fenêtre glissante pour lisser les métriques.
        :param assets_picked_long: Nombre d'actifs à acheter.
        :param assets_picked_short: Nombre d'actifs à vendre.
        """
        self.window = window
        self.assets_picked_long = assets_picked_long
        self.assets_picked_short = assets_picked_short
        # DataFrame qui contiendra les scores finaux pour chaque actif et chaque date.
        # Les actifs avec les plus hauts scores sont achetés, ceux avec les plus bas sont vendus.
        self.ranking_df = None

    def fit(self, data):
        """
        Calcul du rang de chaque actif selon son PER et PBR sur la dernière fenêtre glissante.
        Plus le PER est bas, plus l'actif est sous-évalué et ainsi plus le score est élevé.
        Plus le PBR est bas, plus l'actif est surévalué et ainsi plus le score est élevé.

        :param data: Dictionnaire contenant deux DataFrames pour les métriques PER et PBR.
        """
        # Vérification que les données soient bien un dictionnaire avec les clés "PER" et "PBR"
        if not isinstance(data, dict):
            raise TypeError("Les données doivent être passées sous forme d'un dictionnaire {'PER': df_per, 'PBR': df_pbr}.")
        if "PER" not in data or "PBR" not in data:
            raise KeyError("Le dictionnaire 'data' doit contenir les clés 'PER' et 'PBR'.")

        per_df = data["PER"]
        pbr_df = data["PBR"]
        # Remplacement des valeurs invalides "#N/A N/A" (format Bloomberg) par NaN
        per_df.replace("#N/A N/A", np.nan, inplace=True)
        pbr_df.replace("#N/A N/A", np.nan, inplace=True)

        # Calcul des moyennes glissantes pour le PER et le PBR
        per_rolling = per_df.rolling(self.window).mean()
        pbr_rolling = pbr_df.rolling(self.window).mean()

        # Calcul des rangs pour le PER et le PBR
        per_score = per_rolling.rank(axis=1, method="first", ascending=False)
        pbr_score = pbr_rolling.rank(axis=1, method="first", ascending=False)

        # Calcul du score moyen en combinant les deux rangs
        combined_score = (per_score + pbr_score) / 2

        # Calcul du score final (basé sur le score combiné)
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

        # Exclusion des NaN pour calcul le nombre total d'actifs à la date donnée
        valid_tickers = self.ranking_df.loc[current_date].dropna()
        total_tickers = len(valid_tickers)

        # Détermination de la position en fonction des paramètres
        if rank_value > (total_tickers - self.assets_picked_long):
            return 1.0  # Position long
        elif rank_value <= self.assets_picked_short:
            return -1.0  # Position short
        else:
            return 0.0  # Position neutral