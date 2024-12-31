import pandas as pd
import pytest
from backtesting_framework.Core.Result import Result

def test_result_initialization():
    # Vérifie que l'initialisation de la classe Result fonctionne correctement.
    portfolio_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02],
                                  index=pd.date_range("2023-01-01", periods=5))
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    result = Result(portfolio_returns, cumulative_returns, risk_free_rate=0.01)

    assert result.total_return == pytest.approx(cumulative_returns.iloc[-1])
    assert result.risk_free_rate == 0.01
    assert result.portfolio_returns is not None
    assert result.cumulative_returns is not None

def test_result_initialization_type_error():
    # Vérifie que l'initialisation échoue si les types des paramètres sont incorrects.
    with pytest.raises(TypeError, match="portfolio_returns et cumulative_returns doivent être des séries pandas."):
        Result(portfolio_returns=[0.01, -0.02, 0.03],
               cumulative_returns=pd.Series([1, 1.01, 1.03]),
               risk_free_rate=0.01)

def test_result_initialization_index_mismatch_error():
    # Vérifie que l'initialisation échoue si les index de portfolio_returns et cumulative_returns ne correspondent pas.
    portfolio_returns = pd.Series([0.01, -0.02, 0.03],
                                   index=pd.date_range("2023-01-01", periods=3))
    cumulative_returns = pd.Series([1, 1.01, 1.03],
                                    index=pd.date_range("2023-01-02", periods=3))

    with pytest.raises(ValueError, match="portfolio_returns et cumulative_returns doivent avoir le même index."):
        Result(portfolio_returns, cumulative_returns)

def test_result_invalid_plot_library():
    # Vérifie que l'initialisation échoue si une bibliothèque de visualisation non supportée est fournie.
    portfolio_returns = pd.Series([0.01, -0.02, 0.03],
                                   index=pd.date_range("2023-01-01", periods=3))
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    with pytest.raises(ValueError, match="plot_library doit être l'une des suivantes : 'matplotlib', 'seaborn', 'plotly'."):
        Result(portfolio_returns, cumulative_returns, plot_library="unknown_library")

def test_result_var_and_es():
    # Vérifie que les calculs de Value at Risk (VaR) et Expected Shortfall (ES) sont corrects.
    portfolio_returns = pd.Series([-0.01, -0.02, -0.03, -0.05, -0.01, -0.04],
                                  index=pd.date_range("2023-01-01", periods=6))
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    result = Result(portfolio_returns, cumulative_returns)

    var = result.calculate_var(alpha=0.05)
    es = result.calculate_expected_shortfall(alpha=0.05)

    assert var < 0
    assert es < var

def test_monthly_returns_calculation():
    # Vérifie que le calcul des rendements mensuels fonctionne et retourne un DataFrame.
    portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02],
                                  index=pd.date_range("2023-01-01", periods=6, freq='D'))
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    result = Result(portfolio_returns, cumulative_returns)

    monthly_returns = result.calculate_monthly_returns()
    assert isinstance(monthly_returns, pd.DataFrame)
    assert not monthly_returns.empty

def test_trade_statistics():
    # Vérifie que les statistiques de trading (nombre de trades et taux de réussite) sont calculées correctement.
    portfolio_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02],
                                  index=pd.date_range("2023-01-01", periods=5))
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    trade_stats = (10, 6)
    result = Result(portfolio_returns, cumulative_returns, trade_stats=trade_stats)

    assert result.total_trades == 10
    assert result.winning_trades == 6
    assert result.win_rate == pytest.approx(0.6)
