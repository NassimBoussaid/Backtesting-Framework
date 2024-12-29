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
        """
        self.short_sell = short_sell

    def fit(self, data):
        """
        Préparation des données pour la stratégie Min Variance.
        Actuellement non implémentée.
        """
        pass

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> np.ndarray:
        """
        Détermine les positions et les pondérations des actifs à la date courante.

        :param historical_data: pd.DataFrame de prix pour un actif donné (index = dates).
        :param current_position: Position actuelle.
        :return: np.ndarray contenant les pondérations optimisées, avec 0 pour les actifs sans données suffisantes.
        """
        # Step 1: Identify valid columns (assets) with at least two valid data points
        valid_columns = historical_data.columns[historical_data.notna().sum() >= 2]
        historical_data_valid = historical_data[valid_columns]

        if historical_data_valid.shape[1] == 0:
            # If no valid columns remain, return a zero-weight array for all assets
            return np.zeros(historical_data.shape[1])

        # Step 2: Compute the return matrix
        return_matrix = historical_data_valid.pct_change()  # Drop the first row with NaN

        return_matrix = return_matrix.iloc[1:]

        # Step 3: Calculate the covariance matrix
        cov_matrix = return_matrix.cov()

        # Step 4: Initialize weights and define the optimization problem
        nb_assets = return_matrix.shape[1]
        initial_weights = np.full(nb_assets, 1 / nb_assets)  # Equal weight initialization

        # Define the portfolio variance function
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Budget constraint: weights must sum to 1
        def budget_constraint(weights):
            return np.sum(weights) - 1

        constraints = ({'type': 'eq', 'fun': budget_constraint})  # Equality constraint

        # Bounds: no short selling (0 <= weights <= 1) or allow short selling (-1 <= weights <= 1)
        if self.short_sell:
            bounds = [(-1, 1) for _ in range(nb_assets)]
        else:
            bounds = [(0, 1) for _ in range(nb_assets)]

        # Step 5: Optimization with precise convergence tolerance
        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-7, 'maxiter': 100, 'disp': True}  # Tolérance plus stricte et affichage des détails
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Step 6: Extract results
        optimized_weights_valid = result.x

        # Step 7: Map optimized weights back to original columns
        optimized_weights = np.zeros(historical_data.shape[1])  # Initialize all weights to 0
        optimized_weights[np.isin(historical_data.columns, valid_columns)] = optimized_weights_valid

        return optimized_weights
