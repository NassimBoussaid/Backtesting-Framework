import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import CollinearityWarning
from statsmodels.tsa.stattools import coint

from backtesting_framework.Core.Strategy import Strategy


class PairsTradingStrategy(Strategy):
    """
    Classe de stratégie pour le trading de paires.
    """

    def __init__(self,data, z_score_upper=1.0, z_score_lower=-1.0,significant_level=0.05):
        """
        Initialise la stratégie de trading de paires.

        :param asset1: Nom du premier actif.
        :param asset2: Nom du deuxième actif.
        :param z_score_entry: Seuil de z-score pour entrer en position.
        :param z_score_exit: Seuil de z-score pour sortir de position.
        """

        super().__init__(multi_asset=True)
        self.z_score = None
        self.z_score_lower = z_score_lower
        self.z_score_upper = z_score_upper
        self.significant_level = significant_level
        self.pairs = self.find_cointegrated_pairs(data, self.significant_level)

    def find_cointegrated_pairs(self,data,significance_level=0.05):
        """
        Trouve les paires d'actifs cointégrées (correlation) dans les données.
        :param data: nos stocks et leurs prix respectifs
        :param significance_level: level de la p_value pour le test de cointegration
        :return: pairs, score_matrix, pvalue_matrix
        """

        data_valid = data.dropna(axis=1)
        data_valid = data_valid.loc[:, data_valid.nunique() > 10]
        warnings.filterwarnings("ignore", category=CollinearityWarning)
        keys = data_valid.keys()
        n = len(keys)
        pairs = []
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        indices = np.triu_indices(n, 1)
        # Appliquer les tests de cointégration pour chaque paire
        for i, j in zip(*indices):
            S1 = data_valid[keys[i]]
            S2 = data_valid[keys[j]]
            score, pvalue, _ = coint(S1, S2)
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance_level:
                pairs.append((keys[i], keys[j]))
        return pairs

    def calculate_z_score(self, series):
        """
        Calcule le z-score.

        :param series: Serie de prix.
        :return: Z-score.
        """
        return (series.iloc[-1] - series.mean()) / series.std()

    def get_position(self, historical_data, current_position):
        """
            Génère les signaux de trading pour une paire d'actifs.

            :param current_position: Position actuelle.
            :param historical_data: pd.DataFrame de prix pour un notre panier d'actifs (index = dates).
            :return: liste de signaux pour la date de rebalancement donnée.
        """

        valid_columns = historical_data.columns[historical_data.notna().sum() >= 2]
        historical_data_valid = historical_data[valid_columns]

        if historical_data_valid.shape[1] == 0:
            # Si aucune colonne valide n'est trouvée, retourne une liste de positions nulles pour tous les actifs
            return [0] * len(historical_data.columns)

        # On trouve les paires cointegrées (corrélées)

        nb_assets = len(historical_data_valid.columns)

        # On initialise la position à 0 pour tous les actifs
        position = pd.DataFrame(data=[[0] * nb_assets],columns=historical_data_valid.columns)

        for pair in self.pairs:
            asset1, asset2 = pair
            S1 = historical_data_valid[asset1]
            S2 = historical_data_valid[asset2]

            # Calcul du ratio de prix et z-score
            spread = S1 - S2
            self.z_score = self.calculate_z_score(spread)

            if self.z_score >= self.z_score_upper:
                # Short S1, Long S2
                position[asset1] = -1
                position[asset2] = 1
            elif self.z_score >= self.z_score_lower:
                # Long S1, Short S2
                position[asset1] = 1
                position[asset2] = -1
            else:
                # Close positions
                position[asset1] = 0
                position[asset2] = 0

        current_position = position.values.tolist()[0]
        return current_position

