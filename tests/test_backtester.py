import pytest
import pandas as pd
from backtesting_framework.Core.Backtester import Backtester
from backtesting_framework.Core.Strategy import Strategy

def test_backtester_empty_dataframe():
    # Teste que le Backtester lève une erreur avec un DataFrame vide.
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError, match="Le DataFrame fourni est vide ou invalide."):
        Backtester(
            data_source=empty_data,
            weight_scheme="EqualWeight",
            transaction_cost=0.01,
            slippage=0.005,
            risk_free_rate=0.02,
            rebalancing_frequency="daily"
        )

class SimpleStrategy(Strategy):
    def get_position(self, historical_data, current_position):
        # Retourne toujours 1 (position longue).
        return 1

def test_backtester_initialization():
    # Vérifie que le Backtester est correctement initialisé avec des données valides.
    sample_data = pd.DataFrame({
        "Asset1": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
        "Asset2": [200, 202, 204, 206, 208, 210, 212, 214, 216, 218]
    }, index=pd.date_range("2022-01-01", "2022-01-10"))
    backtester = Backtester(
        data_source=sample_data,
        weight_scheme="EqualWeight",
        transaction_cost=0.01,
        slippage=0.005,
        risk_free_rate=0.02,
        rebalancing_frequency="daily"
    )
    assert backtester.data is not None
    assert backtester.weight_scheme == "EqualWeight"
    assert backtester.transaction_cost == 0.01
    assert backtester.slippage == 0.005
    assert backtester.rfr == 0.02
    pd.testing.assert_frame_equal(backtester.data, sample_data)

def test_backtester_run():
    # Teste que le Backtester exécute une stratégie et retourne un résultat.
    sample_data = pd.DataFrame({
        "Asset1": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
        "Asset2": [200, 202, 204, 206, 208, 210, 212, 214, 216, 218]
    }, index=pd.date_range("2022-01-01", "2022-01-10"))
    backtester = Backtester(
        data_source=sample_data,
        weight_scheme="EqualWeight",
        transaction_cost=0.01,
        slippage=0.005,
        risk_free_rate=0.02,
        rebalancing_frequency="daily"
    )
    strategy = SimpleStrategy()
    result = backtester.run(strategy=strategy)
    assert result is not None
    assert hasattr(result, "portfolio_returns")
    assert hasattr(result, "cumulative_returns")

def test_backtester_market_cap_source_error():
    # Teste que le Backtester lève une erreur si `market_cap_source` est manquant.
    sample_data = pd.DataFrame({
        "Asset1": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
        "Asset2": [200, 202, 204, 206, 208, 210, 212, 214, 216, 218]
    }, index=pd.date_range("2022-01-01", "2022-01-10"))
    with pytest.raises(ValueError, match="market_cap_source doit être fourni si weight_scheme est 'MarketCapWeight'"):
        Backtester(
            data_source=sample_data,
            weight_scheme="MarketCapWeight",
            transaction_cost=0.01,
            slippage=0.005
        )

def test_backtester_load_market_caps_error(tmp_path):
    # Teste que le Backtester lève une erreur si les colonnes ne correspondent pas.
    sample_data = pd.DataFrame({
        "Asset1": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
        "Asset2": [200, 202, 204, 206, 208, 210, 212, 214, 216, 218]
    }, index=pd.date_range("2022-01-01", "2022-01-10"))
    market_cap_path = tmp_path / "market_cap.csv"
    market_cap_data = pd.DataFrame({
        "UnrelatedAsset": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }, index=sample_data.index)
    market_cap_data.to_csv(market_cap_path)
    with pytest.raises(ValueError, match="Il n'y a aucune colonne en commun entre les données de marché et les capitalisations boursières."):
        Backtester(
            data_source=sample_data,
            weight_scheme="MarketCapWeight",
            market_cap_source=str(market_cap_path)
        )

def test_calculate_composition_matrix():
    # Teste que la matrice de composition est calculée correctement.
    sample_data = pd.DataFrame({
        "Asset1": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
        "Asset2": [200, 202, 204, 206, 208, 210, 212, 214, 216, 218]
    }, index=pd.date_range("2022-01-01", "2022-01-10"))
    backtester = Backtester(
        data_source=sample_data,
        weight_scheme="EqualWeight",
        transaction_cost=0.01,
        slippage=0.005,
        risk_free_rate=0.02,
        rebalancing_frequency="daily"
    )
    strategy = SimpleStrategy()
    composition_matrix = backtester.calculate_composition_matrix(strategy)
    assert composition_matrix is not None
    non_nan_matrix = composition_matrix.dropna(how='any')
    assert not non_nan_matrix.isnull().values.any()
    assert (non_nan_matrix >= 0).all().all()

def test_calculate_weight_matrix():
    # Teste que la matrice des pondérations est correcte et que les pondérations somment à 1.
    sample_data = pd.DataFrame({
        "Asset1": [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
        "Asset2": [200, 202, 204, 206, 208, 210, 212, 214, 216, 218]
    }, index=pd.date_range("2022-01-01", "2022-01-10"))
    backtester = Backtester(
        data_source=sample_data,
        weight_scheme="EqualWeight",
        transaction_cost=0.01,
        slippage=0.005,
        risk_free_rate=0.02,
        rebalancing_frequency="daily"
    )
    strategy = SimpleStrategy()
    composition_matrix = backtester.calculate_composition_matrix(strategy)
    weight_matrix = backtester.calculate_weight_matrix(composition_matrix)
    assert weight_matrix is not None
    non_nan_weights = weight_matrix.dropna(how='any')
    valid_weights = non_nan_weights[non_nan_weights.sum(axis=1) > 0]
    assert not valid_weights.isnull().values.any()
    assert (valid_weights.sum(axis=1).round(6) == 1).all()
