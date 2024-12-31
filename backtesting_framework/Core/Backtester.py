import pandas as pd
from backtesting_framework.Core.Strategy import Strategy
from backtesting_framework.Core.Result import Result
from backtesting_framework.Core.Calendar import Calendar
from backtesting_framework.Utils.Tools import load_data


class Backtester:
    """
    Classe permettant de backtester une stratégie financière sur un ensemble de données.
    """

    def __init__(
            self,
            data_source,
            weight_scheme='EqualWeight',
            market_cap_source=None,
            special_start=1,
            transaction_cost=0.0,
            slippage=0.0,
            risk_free_rate=0.0,
            rebalancing_frequency='monthly',
            plot_librairy="matplotlib"
    ):
        """
        Initialise l'objet Backtester.

        :param data_source: Fichier CSV ou DataFrame Pandas contenant les données à backtester.
        :param weight_scheme: Schéma de pondération à utiliser ('EqualWeight' ou 'MarketCapWeight'). Par défaut 'EqualWeight'.
        :param market_cap_source: Chemin vers le fichier CSV des capitalisations boursières.
                                  Requis si weight_scheme='MarketCapWeight'.
        :param special_start: Indice à partir duquel le backtest commence (pour ignorer un certain historique initial).
        :param transaction_cost: Montant des coûts de transaction par rebalancement (par défaut : 0.0).
        :param slippage: Montant des coûts de slippage (exécution différente de l'ordre) par rebalancement (par défaut : 0.0).
        :param risk_free_rate: Taux sans risque du marché (annualisé, par défaut : 0.0).
        :param rebalancing_frequency: Fréquence de rebalancement ('monthly', 'weekly', etc.).
        :param plot_librairy: Bibliothèque d'affichage à utiliser (par défaut : "matplotlib").
        """
        self.data = load_data(data_source)
        self.weight_scheme = weight_scheme
        self.market_cap_source = market_cap_source
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

        # Initialisation de la matrice de poids
        self.weight_matrix = None
        self.plot_library = plot_librairy

    def load_market_caps(self):
        """
        Charge les données de capitalisation boursière et les aligne avec les données de marché.

        :raises ValueError: Si market_cap_source n'est pas fourni alors que weight_scheme='MarketCapWeight'.
        :raises ValueError: S'il n'y a aucune colonne commune entre les données de marché et les capitalisations boursières.
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
                "Il n'y a aucune colonne en commun entre les données de marché et les capitalisations boursières."
            )
        self.market_caps = self.market_caps[common_columns]

    def run(self, strategy: Strategy, is_VT=False, target_vol=None):
        """
        Exécute la stratégie donnée sur les données de marché.

        :param strategy: Instance de la classe Strategy définissant les signaux d'achat/vente.
        :param is_VT: Booléen indiquant si on souhaite activer le Vol Targeting (par défaut False).
        :param target_vol: Volatilité cible (annualisée). Optionnel si is_VT=True.
        :return: Instance de la classe Result contenant les résultats du backtest.
        """
        composition_matrix = self.calculate_composition_matrix(strategy)
        self.weight_matrix = self.calculate_weight_matrix(composition_matrix)
        asset_contributions, portfolio_returns, cumulative_asset_returns, cumulative_returns, result_trade = \
            self.calculate_returns(is_VT, target_vol)

        result = Result(
            portfolio_returns=portfolio_returns,
            cumulative_returns=cumulative_returns,
            risk_free_rate=self.rfr,
            trade_stats=result_trade,
            plot_library=self.plot_library
        )
        return result

    def apply_vol_targeting(
            self,
            portfolio_returns: pd.Series,
            target_vol: float,
            rolling_window: int = 20,
            leverage_cap: float = 1.5
    ) -> pd.Series:
        """
        Ajuste les rendements du portefeuille pour atteindre une volatilité cible (target_vol),
        tout en bornant le levier entre 0% et 150%.

        :param portfolio_returns: Série Pandas des rendements quotidiens du portefeuille.
        :param target_vol: Volatilité cible annualisée.
        :param rolling_window: Nombre de jours pour le calcul de la volatilité glissante (par défaut : 20).
        :param leverage_cap: Borne maximum du levier (par défaut : 1.5, soit 150%).
        :return: Série Pandas des rendements ajustés (après Vol Target).
        """
        import numpy as np
        from math import sqrt

        # Calcul de la volatilité réalisée glissante, annualisée
        realized_vol = portfolio_returns.rolling(rolling_window).std() * sqrt(252)
        realized_vol = realized_vol.replace(0, np.nan).ffill()

        daily_leverage = target_vol / realized_vol  # Calcul de l'exposition (levier)
        daily_leverage = daily_leverage.clip(lower=0, upper=leverage_cap)  # Borne: 0% -> 150%

        # r(t) = L(t) * r_port(t) + [1 - L(t)] * (rfr / 252)
        portfolio_returns_vt = daily_leverage * portfolio_returns + (1 - daily_leverage) * (self.rfr / 252)

        return portfolio_returns_vt

    def calculate_composition_matrix(self, strategy: Strategy) -> pd.DataFrame:
        """
        Calcule la matrice des positions du portefeuille au cours du temps pour chaque actif.

        :param strategy: Instance de la classe Strategy définissant les règles d'achat/vente.
        :return: DataFrame Pandas représentant les positions (nombre d'unités) du portefeuille dans chaque actif.
        """
        assets = self.data.columns
        trading_dates = self.data.index
        rebalancing_dates = self.calendar.rebalancing_dates
        composition_matrix = pd.DataFrame(index=trading_dates, columns=assets, dtype="float64")

        if strategy.multi_asset:
            current_position = 0
            for date_index in range(self.special_start, len(trading_dates)):
                current_date = trading_dates[date_index]
                current_df = self.data.loc[:current_date]
                # Mise à jour des positions aux dates de rebalancement
                if current_date in rebalancing_dates:
                    current_position = strategy.get_position(current_df, current_position)

                composition_matrix.loc[current_date] = current_position

        else:
            # Initialisation des positions pour chaque actif (mono-actif)
            for asset in assets:
                current_position = 0
                for date_index in range(self.special_start, len(trading_dates)):
                    current_date = trading_dates[date_index]
                    current_df = self.data.loc[:current_date, asset]

                    # Mise à jour des positions aux dates de rebalancement
                    if current_date in rebalancing_dates:
                        current_position = strategy.get_position(current_df, current_position)

                    composition_matrix.at[current_date, asset] = current_position

        return composition_matrix

    def calculate_weight_matrix(self, composition_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule la matrice des pondérations du portefeuille au cours du temps
        en fonction du schéma de pondération spécifié.

        :param composition_matrix: DataFrame des positions du portefeuille.
        :return: DataFrame Pandas représentant les pondérations du portefeuille (entre -1 et +1, ou autre).
        :raises ValueError: Si le schéma de pondération n'est pas reconnu.
        """
        pd.set_option('future.no_silent_downcasting', True)
        if self.weight_scheme == 'EqualWeight':
            # Comptage du nombre d'actifs en position pour chaque date
            selected_counts = composition_matrix.abs().sum(axis=1).replace(0, pd.NA)
            # Normalisation des poids
            weight_matrix = composition_matrix.divide(selected_counts, axis=0).fillna(0)

        elif self.weight_scheme == 'MarketCapWeight':
            # Pondération par la capitalisation boursière
            weighted_market_caps = composition_matrix * self.market_caps
            sum_market_caps = weighted_market_caps.abs().sum(axis=1).replace(0, pd.NA)
            weight_matrix = weighted_market_caps.divide(sum_market_caps, axis=0).fillna(0)

        else:
            raise ValueError(f"Schéma de pondération inconnu : {self.weight_scheme}")

        return weight_matrix

    def calculate_transaction_costs(self, shifted_positions: pd.DataFrame) -> pd.Series:
        """
        Calcule les coûts de transaction en fonction des changements de positions.

        :param shifted_positions: DataFrame des positions décalées pour calculer la variation de position.
        :return: Série Pandas représentant les coûts de transaction par période.
        """
        return (shifted_positions.diff().abs().sum(axis=1)) * self.transaction_cost

    def calculate_slippage_costs(self, shifted_positions: pd.DataFrame) -> pd.Series:
        """
        Calcule les coûts de slippage en fonction des changements de positions.

        :param shifted_positions: DataFrame des positions décalées pour calculer la variation de position.
        :return: Série Pandas représentant les coûts de slippage par période.
        """
        return (shifted_positions.diff().abs().sum(axis=1)) * self.slippage

    def evaluate_trade(self, shifted_positions: pd.DataFrame) -> tuple:
        """
        Évalue le nombre de trades et le nombre de trades gagnants sur la période.

        :param shifted_positions: DataFrame des positions décalées dans le temps.
        :return: Tuple (trade_count, win_trade_count).
        """
        trade_count = 0
        win_trade_count = 0

        for asset in shifted_positions.columns:
            last_position = shifted_positions.iloc[0][asset]  # Position initiale pour l'actif
            last_trade_value = self.data.iloc[0][asset]  # Prix initial de l'actif

            for date in shifted_positions.index:
                current_position = shifted_positions.at[date, asset]
                # Détecter un trade quand la position change
                if last_position != current_position:
                    trade_count += 1
                    current_value = self.data.at[date, asset]

                    # Vérifier si le trade est gagnant
                    if ((last_position > 0 and current_value > last_trade_value) or
                            (last_position < 0 and current_value < last_trade_value)):
                        win_trade_count += 1

                    # Mise à jour des valeurs de référence
                    last_trade_value = current_value
                    last_position = current_position

        return trade_count, win_trade_count

    def calculate_returns(self, is_VT: bool = False, target_vol: float = None):
        """
        Calcule les rendements du portefeuille et les rendements cumulés,
        avec ou sans Vol Targeting.

        :param is_VT: Booléen indiquant si on souhaite activer le Vol Targeting (par défaut : False).
        :param target_vol: Volatilité cible annualisée (utile si is_VT=True).
        :return:
            - asset_contributions (pd.DataFrame) : contribution quotidienne de chaque actif
            - portfolio_returns (pd.Series) : rendement quotidien du portefeuille
            - cumulative_asset_returns (pd.DataFrame) : rendements cumulés de chaque actif
            - cumulative_returns (pd.Series) : rendements cumulés du portefeuille
            - result_trade (tuple) : (nombre total de trades, nombre de trades gagnants)
        """
        # 1) Calcul des rendements des actifs
        asset_returns = self.data.pct_change().fillna(0)

        # 2) Shift des positions pour éviter le biais (positions en t décidées en t-1)
        shifted_weights = self.weight_matrix.shift(1).fillna(0)

        # 3) Calcul des coûts de transaction et de slippage
        transaction_costs = self.calculate_transaction_costs(shifted_weights)
        slippage_costs = self.calculate_slippage_costs(shifted_weights)

        # 4) Contribution de chaque actif
        asset_contributions = shifted_weights.multiply(asset_returns, axis=0)

        # 5) Rendement brut du portefeuille
        portfolio_returns = asset_contributions.sum(axis=1) - transaction_costs - slippage_costs

        # 6) Application du Vol Target (si demandé)
        if is_VT and (target_vol is not None):
            portfolio_returns = self.apply_vol_targeting(portfolio_returns, target_vol)

        # 7) Calcul des rendements cumulés
        cumulative_asset_returns = (1 + asset_contributions).cumprod() - 1
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

        # 8) Gestion du special_start
        if self.special_start != 1:
            shifted_weights = shifted_weights.iloc[self.special_start + 1:]
            asset_contributions = asset_contributions.iloc[self.special_start + 1:]
            portfolio_returns = portfolio_returns.iloc[self.special_start + 1:]
            cumulative_asset_returns = cumulative_asset_returns.iloc[self.special_start + 1:]
            cumulative_returns = cumulative_returns.iloc[self.special_start + 1:]

        # 9) Évaluation des statistiques de trades
        result_trade = self.evaluate_trade(shifted_weights)

        return asset_contributions, portfolio_returns, cumulative_asset_returns, cumulative_returns, result_trade
