import pandas as pd
import Strategy
from Result import Result
from Calendar import Calendar
from backtesting_framework.Utils.Tools import load_data


class Backtester:
    """
    Classe permettant de backtester une stratégie financière sur un ensemble de données.
    """

    def __init__(self, data_source, weight_scheme='EqualWeight', market_cap_source=None,
                 multi_assets=False, special_start=1, transaction_cost=0.0,
                 slippage=0.0, risk_free_rate=0.0, rebalancing_frequency='monthly'):
        """
        Initialise l'objet Backtester.

        :param data_source: Fichier CSV ou DataFrame Pandas contenant les données à backtester.
        :param weight_scheme: Schéma de pondération à utiliser ('EqualWeight' ou 'MarketCapWeight').
                              Par défaut 'EqualWeight'.
        :param market_cap_source: Chemin vers le fichier CSV des capitalisations boursières. Requis si
                                  weight_scheme est 'MarketCapWeight'.
        :param multi_assets: Booléen indiquant si le backtest concerne plusieurs actifs simultanément.
        :param special_start: Indice à partir duquel le backtest commence.
        :param transaction_cost: Montant des coûts de transaction par rebalancement (par défaut : 0.0).
        :param slippage: Montant des coûts de slippage (exécution différente de l'ordre) par rebalancement
                         (par défaut : 0.0).
        :param risk_free_rate: Taux sans risque marché (par défaut : 0.0).
        :param rebalancing_frequency: Fréquence de rebalancement ('monthly', 'weekly', etc.).
        """
        self.data = load_data(data_source)
        self.weight_scheme = weight_scheme
        self.market_cap_source = market_cap_source
        self.multi_assets = multi_assets
        self.special_start = special_start
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.rfr = risk_free_rate

        # Charger et aligner les données de capitalisation boursière si nécessaire
        if self.weight_scheme == 'MarketCapWeight':
            self.market_caps = None
            self.load_market_caps()

        # Détermination des bornes pour le calendrier
        self.start_date = self.data.index[0].strftime('%Y-%m-%d')
        self.end_date = self.data.index[-1].strftime('%Y-%m-%d')

        self.calendar = Calendar(
            frequency=rebalancing_frequency,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Initialisation de la weight_matrix
        self.weight_matrix = None

    def load_market_caps(self):
        """
        Charge les données de capitalisation boursière et les aligne avec les données de marché.
        """
        if self.market_cap_source is None:
            raise ValueError("market_cap_source doit être fourni si weight_scheme est 'MarketCapWeight'")

        self.market_caps = load_data(self.market_cap_source)

        # Réindexer avec self.data + forward filling si data manquante
        if not self.market_caps.index.equals(self.data.index):
            self.market_caps = self.market_caps.reindex(self.data.index).ffill()

        # Réindexer avec self.data en gardant uniquement l'intersection
        common_columns = self.data.columns.intersection(self.market_caps.columns)
        if common_columns.empty:
            raise ValueError(
                "Il n'y a aucune colonne en commun entre les données de marché et les capitalisations boursières.")
        self.market_caps = self.market_caps[common_columns]

    def run(self, strategy: Strategy):
        """
        Exécute la stratégie donnée sur les données de marché.

        :param strategy: Instance de la classe Strategy définissant les signaux d'achat/vente.
        :return: Instance de la classe Result contenant les résultats du backtest.
        """
        composition_matrix = self.calculate_composition_matrix(strategy)
        self.weight_matrix = self.calculate_weight_matrix(composition_matrix)
        asset_contributions, portfolio_returns, cumulative_asset_returns, cumulative_returns, result_trade = \
            self.calculate_returns(composition_matrix)

        result = Result(
            portfolio_returns=portfolio_returns,
            cumulative_returns=cumulative_returns,
            risk_free_rate=self.rfr,
            trade_stats=result_trade
        )
        return result

    def calculate_composition_matrix(self, strategy: Strategy):
        """
        Calcule la matrice des positions du portefeuille au cours du temps.

        :param strategy: Instance de la classe Strategy définissant les signaux d'achat/vente.
        :return: DataFrame Pandas représentant les positions du portefeuille.
        """
        assets = self.data.columns
        trading_dates = self.data.index
        rebalancing_dates = self.calendar.rebalancing_dates
        composition_matrix = pd.DataFrame(
            index=trading_dates,
            columns=assets,
            dtype="float64"
        )

        if self.multi_assets:
            current_position = 0
            for date_index in range(self.special_start, len(trading_dates)):
                current_date = trading_dates[date_index]
                current_df = self.data.loc[:current_date]
                # Mise à jour des positions aux dates de rebalancement
                if current_date in rebalancing_dates:
                    current_position = strategy.get_position(current_df, current_position)

                composition_matrix.loc[current_date] = current_position
        else:
            # Initialisation des positions pour chaque actif
            for asset in assets:
                current_position = 0

                for date_index in range(self.special_start, len(trading_dates)):
                    current_date = trading_dates[date_index]
                    current_df = self.data.loc[:current_date][asset]

                    # Mise à jour des positions aux dates de rebalancement
                    if current_date in rebalancing_dates:
                        current_position = strategy.get_position(current_df, current_position)

                    composition_matrix.at[current_date, asset] = current_position

        return composition_matrix

    def calculate_weight_matrix(self, composition_matrix):
        """
        Calcule la matrice des pondérations du portefeuille au cours du temps en fonction du schéma de pondération.

        :param composition_matrix: DataFrame des positions du portefeuille.
        :return: DataFrame Pandas représentant les pondérations du portefeuille.
        """
        if self.weight_scheme == 'EqualWeight':
            selected_counts = composition_matrix.abs().sum(axis=1).replace(0, pd.NA)
            weight_matrix = composition_matrix.divide(selected_counts, axis=0).fillna(0)  # Normalisation des poids
        elif self.weight_scheme == 'MarketCapWeight':
            weighted_market_caps = composition_matrix * self.market_caps
            sum_market_caps = weighted_market_caps.abs().sum(axis=1).replace(0, pd.NA)
            weight_matrix = weighted_market_caps.divide(sum_market_caps, axis=0).fillna(0)  # Normalisation des poids
        else:
            raise ValueError(f"Schéma de pondération inconnu : {self.weight_scheme}")

        return weight_matrix

    def calculate_transaction_costs(self, shifted_positions):
        """
        Calcule les coûts de transaction en fonction des changements de positions.

        :param shifted_positions: Positions décalées pour le calcul.
        :return: Série Pandas représentant les coûts de transaction par période.
        """
        return (shifted_positions.diff().abs().sum(axis=1)) * self.transaction_cost

    def calculate_slippage_costs(self, shifted_positions):
        """
        Calcule les coûts de slippage en fonction des changements de positions.

        :param shifted_positions: Positions décalées pour le calcul.
        :return: Série Pandas représentant les coûts de slippage par période.
        """
        return (shifted_positions.diff().abs().sum(axis=1)) * self.slippage

    def evaluate_trade(self, shifted_positions):
        trade_count = 0
        win_trade_count = 0

        for asset in shifted_positions.columns:
            last_position = shifted_positions.iloc[0][asset]  # Initialize with the first position for the asset
            last_trade_value = self.data.iloc[0][asset]  # Initialize with the first asset value

            for date in shifted_positions.index:
                current_position = shifted_positions.at[date, asset]
                if last_position != current_position:  # Detect a trade when the position changes
                    trade_count += 1

                    current_value = self.data.at[date, asset]

                    # Check if the trade was a winning trade
                    if (last_position > 0 and current_value > last_trade_value) or \
                            (last_position < 0 and current_value < last_trade_value):
                        win_trade_count += 1

                    # Update last_trade_value and last_position
                    last_trade_value = current_value
                    last_position = current_position

        return trade_count, win_trade_count

    def calculate_returns(self, composition_matrix):
        """
        Calcule les rendements du portefeuille et les rendements cumulés.

        :param composition_matrix: DataFrame des positions du portefeuille.
        :return: Tuple contenant les contributions des actifs, les rendements du portefeuille,
                 les rendements cumulés des actifs et les rendements cumulés du portefeuille.
        """
        # Calcul des rendements des actifs
        asset_returns = self.data.pct_change().fillna(0)

        # Décalage des positions pour éviter un biais : les positions utilisées en t sont décidées en t-1
        shifted_weights = self.weight_matrix.shift(1).fillna(0)

        # Calcul des coûts de transaction et de slippage
        transaction_costs = self.calculate_transaction_costs(shifted_weights)
        slippage_costs = self.calculate_slippage_costs(shifted_weights)

        # Contribution de chaque actif au portefeuille
        asset_contributions = shifted_weights.multiply(asset_returns, axis=0)

        # Rendements du portefeuille
        portfolio_returns = asset_contributions.sum(axis=1) - transaction_costs - slippage_costs

        # Calcul des rendements cumulés
        cumulative_asset_returns = (1 + asset_contributions).cumprod() - 1
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

        if self.special_start != 1:
            shifted_weights = shifted_weights.iloc[self.special_start + 1:]
            asset_contributions = asset_contributions.iloc[self.special_start + 1:]
            portfolio_returns = portfolio_returns.iloc[self.special_start + 1:]
            cumulative_asset_returns = cumulative_asset_returns.iloc[self.special_start + 1:]
            cumulative_returns = cumulative_returns.iloc[self.special_start + 1:]

        result_trade = self.evaluate_trade(shifted_weights)

        return asset_contributions, portfolio_returns, cumulative_asset_returns, cumulative_returns, result_trade
