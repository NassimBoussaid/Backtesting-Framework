from backtesting_framework.Core.Strategy import Strategy
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class MinVariance(Strategy):
    """
    Stratégie Min Variance : Pondère les actifs de manière à minimiser la
    variance du portefeuille
    """

    def __init__(self, short_sell=False):
        """
        Initialisation de la stratégie Min Variance.

        :param short_sell: Booléen, si True, autorise la vente à découvert (poids négatifs).
        """
        super().__init__(multi_asset=True)
        self.short_sell = short_sell

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> np.ndarray:
        """
        Détermine les positions et les pondérations des actifs à la date courante.

        :param historical_data: pd.DataFrame contenant les données de prix historiques (index = dates).
        :param current_position: Position actuelle (non utilisée dans cette stratégie).
        :return: np.ndarray contenant les pondérations optimisées, avec 0 pour les actifs sans données suffisantes.
        """
        # Identification des colonnes valides (actifs) avec au moins deux points de données valides
        valid_columns = historical_data.columns[historical_data.notna().sum() >= 2]
        historical_data_valid = historical_data[valid_columns]

        if historical_data_valid.shape[1] == 0:
            # Retour d'un tableau de poids nuls si aucune colonne valide
            return np.zeros(historical_data.shape[1])

        # Calcul de la matrice des rendements
        return_matrix = historical_data_valid.pct_change()
        return_matrix = return_matrix.iloc[1:]

        # Calcul de la matrice de covariance
        cov_matrix = return_matrix.cov()

        # Initialisation des poids et définition du problème d'optimisation
        nb_assets = return_matrix.shape[1]
        initial_weights = np.full(nb_assets, 1 / nb_assets)  # Equal weight initialization

        # Définition de la fonction de variance du portefeuille
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Définition de la contrainte de budget (somme des poids = 1)
        def budget_constraint(weights):
            return np.sum(weights) - 1

        constraints = ({'type': 'eq', 'fun': budget_constraint})  # Equality constraint

        # Définition des bornes pour les poids
        if self.short_sell:
            bounds = [(-1, 1) for _ in range(nb_assets)]
        else:
            bounds = [(0, 1) for _ in range(nb_assets)]

        # Optimisation de la variance du portefeuille
        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 100, 'disp': True}  # Tolérance plus stricte et affichage des détails
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Extraction des poids optimisés pour les colonnes valides
        optimized_weights_valid = result.x

        # Mapping des poids optimisés sur les colonnes originales
        optimized_weights = np.zeros(historical_data.shape[1])  # Initialize all weights to 0
        optimized_weights[np.isin(historical_data.columns, valid_columns)] = optimized_weights_valid

        return optimized_weights

    def fit(self, data):
        """
        Méthode optionnelle d'ajustement (fit). Non utilisée pour cette stratégie.
        """
        pass