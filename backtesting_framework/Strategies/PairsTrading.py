import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import CollinearityWarning
from statsmodels.tsa.stattools import coint

from backtesting_framework.Core.Strategy import Strategy


class PairsTradingStrategy(Strategy):
    """
    Stratégie de trading de paires :
    Identifie des paires d'actifs co-intégrées et génère des signaux de trading
    basés sur les écarts de prix (spread) entre ces paires.
    """

    def __init__(self, data, z_score_upper=1.0, z_score_lower=-1.0, significant_level=0.05):
        """
        Initialisation de la stratégie de trading de paires.

        :param data: Données des actifs et leurs prix respectifs.
        :param z_score_upper: Seuil supérieur pour le z-score, déclenchant une position courte sur une paire.
        :param z_score_lower: Seuil inférieur pour le z-score, déclenchant une position longue sur une paire.
        :param significant_level: Niveau de significativité pour le test de cointégration (p-value).
        """

        super().__init__(multi_asset=True)
        self.z_score = None
        self.z_score_lower = z_score_lower
        self.z_score_upper = z_score_upper
        self.significant_level = significant_level
        self.pairs = self.find_cointegrated_pairs(data, self.significant_level)

    def find_cointegrated_pairs(self,data,significance_level=0.05):
        """
        Identifie les paires d'actifs co-intégrées dans les données.

        :param data: pd.DataFrame contenant les prix des actifs.
        :param significance_level: Seuil de significativité pour la p-value du test de co-intégration.
        :return: Liste des paires co-intégrées.
        """
        # Suppression des colonnes avec des valeurs manquantes
        data_valid = data.dropna(axis=1)
        data_valid = data_valid.loc[:, data_valid.nunique() > 10]
        warnings.filterwarnings("ignore", category=CollinearityWarning)

        keys = data_valid.keys()
        n = len(keys)
        pairs = []
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        indices = np.triu_indices(n, 1)

        # Application du test de cointégration pour chaque paire d'actifs
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
        Calcule le z-score pour une série de données.

        :param series: pd.Series représentant la série des écarts (spread).
        :return: Valeur du z-score.
        """
        return (series.iloc[-1] - series.mean()) / series.std()

    def get_position(self, historical_data, current_position):
        """
        Génère les positions de trading pour une paire d'actifs en fonction de l'écart
        (spread) entre leurs prix et des seuils définis.

        :param historical_data: pd.DataFrame contenant les données de prix historiques.
        :param current_position: Liste des positions actuelles sur les actifs.
        :return: Liste des nouvelles positions pour chaque actif.
        """
        # Filtrage des colonnes valides (au moins 2 valeurs non nulles)
        valid_columns = historical_data.columns[historical_data.notna().sum() >= 2]
        historical_data_valid = historical_data[valid_columns]

        if historical_data_valid.shape[1] == 0:
            # Si aucune colonne valide n'est trouvée, retourne une liste de positions nulles
            return [0] * len(historical_data.columns)

        # Initialisation de la position à 0 pour tous les actifs
        nb_assets = len(historical_data_valid.columns)
        position = pd.DataFrame(data=[[0] * nb_assets],columns=historical_data_valid.columns)

        # Itération sur les paires co-intégrées
        for pair in self.pairs:
            asset1, asset2 = pair
            S1 = historical_data_valid[asset1]
            S2 = historical_data_valid[asset2]

            # Calcul du spread et du z-score
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
                # Ferme les positions
                position[asset1] = 0
                position[asset2] = 0

        # Conversion des positions en liste
        current_position = position.values.tolist()[0]
        return current_position

    def fit(self, data):
        """
        Méthode optionnelle d'ajustement (fit). Non utilisée pour cette stratégie.
        """
        pass