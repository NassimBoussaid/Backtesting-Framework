#################################### WARNING #####################################
# Pour lancer l'interface entrer la commande suivante en console : poetry run streamlit run backtesting_framework/Core/app.py
##################################################################################
import streamlit as st
import pandas as pd
from backtesting_framework.Core.Backtester import Backtester
from backtesting_framework.Strategies.RSI import RSI
from backtesting_framework.Strategies.BollingerBands import BollingerBands
from backtesting_framework.Strategies.MeanReversion import MeanReversion
from backtesting_framework.Strategies.MovingAverage import MovingAverage
from backtesting_framework.Strategies.Quality import Quality
from backtesting_framework.Strategies.Value import Value
from backtesting_framework.Strategies.Size import Size
from backtesting_framework.Strategies.BuyAndHold import BuyAndHold
from backtesting_framework.Strategies.MinVariance import MinVariance
from backtesting_framework.Strategies.Volatility_Trend import VolatilityTrendStrategy
from backtesting_framework.Strategies.KeltnerChannelStrategy import KeltnerChannelStrategy


@st.cache_data
def load_data(file_path):
    # Vérifier le format du fichier
    if file_path.name.endswith('.csv'):
        data = pd.read_csv(file_path, index_col=0)
    elif file_path.name.endswith('.parquet'):
        data = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Parquet file.")

    # Convertir l'index en DatetimeIndex
    try:
        data.index = pd.to_datetime(data.index)
    except Exception as e:
        raise ValueError(f"Error converting index to datetime: {e}")

    return data

# Interface utilisateur
st.title("Backtesting Interface")

# Charger plusieurs fichiers
st.subheader("Upload Required Data Files")
uploaded_files = st.file_uploader(
    "Upload Files (Excel or Parquet format)", type=["csv", "parquet"], accept_multiple_files=True
)
data_files = {}
if uploaded_files:
    for file in uploaded_files:
        data_files[file.name] = load_data(file)
    st.write("Uploaded Files:", list(data_files.keys()))

# Paramètres globaux pour la stratégie 1
st.sidebar.subheader("Global Settings - Strategy 1")
transaction_cost_1 = st.sidebar.number_input("Transaction Cost (Strategy 1, %):", min_value=0.0, max_value=100.0, value=0.0, step=0.1)/100
slippage_1 = st.sidebar.number_input("Slippage (Strategy 1, %):", min_value=0.0, max_value=100.0, value=0.0, step=0.1)/100
risk_free_rate_1 = st.sidebar.number_input("Risk Free Rate (Strategy 1, %):", min_value=0.0, max_value=100.0, value=0.0, step=0.1)/100
rebalancing_frequency_1 = st.sidebar.selectbox("Rebalancing Frequency (Strategy 1):", ["daily", "weekly", "monthly"], index=2)
weight_scheme_1 = st.sidebar.selectbox("Weighting Scheme (Strategy 1):", ["EqualWeight", "MarketCapWeight"], index=0)
special_start_1 = st.sidebar.number_input("Special Start (Strategy 1, Index):", min_value=1, max_value=1000, value=1)
plot_library_1 = st.sidebar.selectbox("Visualization Library (Strategy 1):", ["matplotlib", "seaborn", "plotly"], index=0)
apply_vol_target_1 = st.sidebar.checkbox("Apply Vol Target (Strategy 1)", value=False)
target_vol_1 = st.sidebar.number_input("Target Volatility (Strategy 1, %):", min_value=0.0, max_value=100.0, value=10.0, step=0.1) if apply_vol_target_1 else None

market_cap_file_1 = None
if weight_scheme_1 == "MarketCapWeight":
    st.sidebar.subheader("Market Cap Data - Strategy 1")
    selected_market_cap_file_1 = st.sidebar.selectbox("Select Market Cap File (Strategy 1):", options=list(data_files.keys()))
    market_cap_file_1 = data_files[selected_market_cap_file_1] if selected_market_cap_file_1 else None

# Choisir la stratégie 1
st.subheader("Choose Strategy 1")
strategy_name_1 = st.selectbox(
    "Select Strategy 1",
    [
        "RSI",
        "Bollinger Bands",
        "Mean Reversion",
        "Moving Average",
        "Quality",
        "Value",
        "Size",
        "Buy and Hold",
        "MinVariance",
        "Volatility Trend",
        "Keltner Channel",
    ]
)

strategy_1 = None
historical_data_file_1 = None
if strategy_name_1:
    # Sélection du fichier de données historiques pour la stratégie 1
    st.subheader("Select Historical Data File (Strategy 1)")
    historical_data_file_1 = st.selectbox("Select Historical Data File (Strategy 1):", options=list(data_files.keys()))

    if historical_data_file_1:
        historical_data_1 = data_files[historical_data_file_1]

    # Paramètres de la stratégie 1
# Paramètres de la stratégie 1
if strategy_name_1 == "RSI":
    rsi_period_1 = st.number_input("RSI Period (Strategy 1)", min_value=5, value=14, step=1)
    rsi_overbought_1 = st.number_input("Overbought Threshold (Strategy 1)", min_value=50, value=70, step=1)
    rsi_oversold_1 = st.number_input("Oversold Threshold (Strategy 1)", min_value=0, value=30, step=1)
    strategy_1 = RSI(period=rsi_period_1, overbought_threshold=rsi_overbought_1, oversold_threshold=rsi_oversold_1)

elif strategy_name_1 == "Bollinger Bands":
    bb_window_1 = st.number_input("Window Period (Strategy 1)", min_value=5, value=20, step=1)
    bb_std_dev_1 = st.number_input("Number of Standard Deviations (Strategy 1)", min_value=1.0, value=2.0, step=0.1)
    strategy_1 = BollingerBands(window=bb_window_1, num_std_dev=bb_std_dev_1)

elif strategy_name_1 == "Mean Reversion":
    mr_window_1 = st.number_input("Window Period (Strategy 1)", min_value=5, value=20, step=1)
    mr_deviation_1 = st.number_input("Threshold (Number of Standard Deviations) (Strategy 1)", min_value=1.0, value=2.0, step=0.1)
    strategy_1 = MeanReversion(window=mr_window_1, zscore_threshold=mr_deviation_1)

elif strategy_name_1 == "Moving Average":
    short_window_1 = st.number_input("Short Window (Strategy 1)", min_value=5, value=14, step=1)
    long_window_1 = st.number_input("Long Window (Strategy 1)", min_value=20, value=50, step=1)
    exponential_mode_1 = st.checkbox("Exponential Moving Average (EMA) (Strategy 1)", value=False)
    strategy_1 = MovingAverage(short_window=short_window_1, long_window=long_window_1, exponential_mode=exponential_mode_1)

elif strategy_name_1 == "Quality":
    selected_roe_file_1 = st.selectbox("Select ROE File (Strategy 1)", options=list(data_files.keys()))
    selected_roa_file_1 = st.selectbox("Select ROA File (Strategy 1)", options=list(data_files.keys()))
    window_1 = st.number_input("Window Period (Days) (Strategy 1)", min_value=5, value=30, step=1)
    assets_picked_long_1 = st.number_input("Number of Long Positions (Strategy 1)", min_value=0, value=10, step=1)
    assets_picked_short_1 = st.number_input("Number of Short Positions (Strategy 1)", min_value=0, value=10, step=1)
    strategy_1 = Quality(window=window_1, assets_picked_long=assets_picked_long_1, assets_picked_short=assets_picked_short_1)
    if selected_roe_file_1 and selected_roa_file_1:
        strategy_1.fit({"ROE": data_files[selected_roe_file_1], "ROA": data_files[selected_roa_file_1]})

elif strategy_name_1 == "Value":
    selected_per_file_1 = st.selectbox("Select PER File (Strategy 1)", options=list(data_files.keys()))
    selected_pbr_file_1 = st.selectbox("Select PBR File (Strategy 1)", options=list(data_files.keys()))
    window_1 = st.number_input("Window Period (Days) (Strategy 1)", min_value=5, value=30, step=1)
    assets_picked_long_1 = st.number_input("Number of Long Positions (Strategy 1)", min_value=0, value=10, step=1)
    assets_picked_short_1 = st.number_input("Number of Short Positions (Strategy 1)", min_value=0, value=10, step=1)
    strategy_1 = Value(window=window_1, assets_picked_long=assets_picked_long_1, assets_picked_short=assets_picked_short_1)
    if selected_per_file_1 and selected_pbr_file_1:
        strategy_1.fit({"PER": data_files[selected_per_file_1], "PBR": data_files[selected_pbr_file_1]})

elif strategy_name_1 == "Size":
    selected_market_cap_file_1 = st.selectbox("Select Market Cap File (Strategy 1)", options=list(data_files.keys()))
    window_1 = st.number_input("Window Period (Days) (Strategy 1)", min_value=5, value=30, step=1)
    assets_picked_long_1 = st.number_input("Number of Long Positions (Strategy 1)", min_value=0, value=10, step=1)
    assets_picked_short_1 = st.number_input("Number of Short Positions (Strategy 1)", min_value=0, value=10, step=1)
    strategy_1 = Size(window=window_1, assets_picked_long=assets_picked_long_1, assets_picked_short=assets_picked_short_1)
    if selected_market_cap_file_1:
        strategy_1.fit(data_files[selected_market_cap_file_1])

elif strategy_name_1 == "Buy and Hold":
    strategy_1 = BuyAndHold()

elif strategy_name_1 == "MinVariance":
    short_sell_1 = st.checkbox("Allow Short Selling", value=False)
    strategy_1 = MinVariance(short_sell=short_sell_1)

elif strategy_name_1 == "Volatility Trend":
    atr_period_1 = st.number_input("ATR Period (Strategy 1)", min_value=5, value=14, step=1)
    dmi_period_1 = st.number_input("DMI Period (Strategy 1)", min_value=5, value=14, step=1)
    atr_threshold_1 = st.number_input("ATR Threshold (Strategy 1)", min_value=0.1, value=1.0, step=0.1)
    strategy_1 = VolatilityTrendStrategy(atr_period=atr_period_1, dmi_period=dmi_period_1, atr_threshold=atr_threshold_1)

elif strategy_name_1 == "Keltner Channel":
    atr_period_1 = st.number_input("ATR Period (Strategy 1)", min_value=5, value=10, step=1)
    atr_multiplier_1 = st.number_input("ATR Multiplier (Strategy 1)", min_value=1.0, value=2.0, step=0.1)
    sma_period_1 = st.number_input("SMA Period (Strategy 1)", min_value=5, value=20, step=1)
    strategy_1 = KeltnerChannelStrategy(atr_period=atr_period_1, atr_multiplier=atr_multiplier_1, sma_period=sma_period_1)


# Comparaison de stratégies
compare_strategies = st.sidebar.checkbox("Compare Two Strategies", value=False)
if compare_strategies:
    # Paramètres globaux pour la stratégie 2
    st.sidebar.subheader("Global Settings - Strategy 2")
    transaction_cost_2 = st.sidebar.number_input("Transaction Cost (Strategy 2, %):", min_value=0.0, max_value=100.0, value=0.0, step=0.1)/100
    slippage_2 = st.sidebar.number_input("Slippage (Strategy 2, %):", min_value=0.0, max_value=100.0, value=0.0, step=0.1)/100
    risk_free_rate_2 = st.sidebar.number_input("Risk Free Rate (Strategy 2, %):", min_value=0.0, max_value=100.0,value=0.0, step=0.1)/100
    rebalancing_frequency_2 = st.sidebar.selectbox("Rebalancing Frequency (Strategy 2):", ["daily", "weekly", "monthly"], index=2)
    weight_scheme_2 = st.sidebar.selectbox("Weighting Scheme (Strategy 2):", ["EqualWeight", "MarketCapWeight"], index=0)
    special_start_2 = st.sidebar.number_input("Special Start (Strategy 2, Index):", min_value=1, max_value=1000, value=1)
    plot_library_2 = st.sidebar.selectbox("Visualization Library (Strategy 2):", ["matplotlib", "seaborn", "plotly"], index=0)
    apply_vol_target_2 = st.sidebar.checkbox("Apply Vol Target (Strategy 2)", value=False)
    target_vol_2 = st.sidebar.number_input("Target Volatility (Strategy 2, %):", min_value=0.0, max_value=100.0, value=10.0, step=0.1)/100 if apply_vol_target_2 else None

    market_cap_file_2 = None
    if weight_scheme_2 == "MarketCapWeight":
        st.sidebar.subheader("Market Cap Data - Strategy 2")
        selected_market_cap_file_2 = st.sidebar.selectbox("Select Market Cap File (Strategy 2):", options=list(data_files.keys()))
        market_cap_file_2 = data_files[selected_market_cap_file_2] if selected_market_cap_file_2 else None

    # Choisir la stratégie 2
    st.subheader("Choose Strategy 2")
    strategy_name_2 = st.selectbox(
        "Select Strategy 2",
        [
            "RSI",
            "Bollinger Bands",
            "Mean Reversion",
            "Moving Average",
            "Quality",
            "Value",
            "Size",
            "Buy and Hold",
            "MinVariance",
            "Volatility Trend",
            "Keltner Channel",
            "Pairs Trading"
        ]
    )

    strategy_2 = None
    historical_data_file_2 = None
    if strategy_name_2:
        # Sélection du fichier de données historiques pour la stratégie 2
        st.subheader("Select Historical Data File (Strategy 2)")
        historical_data_file_2 = st.selectbox("Select Historical Data File (Strategy 2):", options=list(data_files.keys()))

        if historical_data_file_2:
            historical_data_2 = data_files[historical_data_file_2]

        # Paramètres de la stratégie 2
        if strategy_name_2 == "RSI":
            rsi_period_2 = st.number_input("RSI Period (Strategy 2)", min_value=5, value=14, step=1)
            rsi_overbought_2 = st.number_input("Overbought Threshold (Strategy 2)", min_value=50, value=70, step=1)
            rsi_oversold_2 = st.number_input("Oversold Threshold (Strategy 2)", min_value=0, value=30, step=1)
            strategy_2 = RSI(period=rsi_period_2, overbought_threshold=rsi_overbought_2,
                             oversold_threshold=rsi_oversold_2)

        elif strategy_name_2 == "Bollinger Bands":
            bb_window_2 = st.number_input("Window Period (Strategy 2)", min_value=5, value=20, step=1)
            bb_std_dev_2 = st.number_input("Number of Standard Deviations (Strategy 2)", min_value=1.0, value=2.0,
                                           step=0.1)
            strategy_2 = BollingerBands(window=bb_window_2, num_std_dev=bb_std_dev_2)

        elif strategy_name_2 == "Mean Reversion":
            mr_window_2 = st.number_input("Window Period (Strategy 2)", min_value=5, value=20, step=1)
            mr_deviation_2 = st.number_input("Threshold (Number of Standard Deviations) (Strategy 2)", min_value=1.0,
                                             value=2.0, step=0.1)
            strategy_2 = MeanReversion(window=mr_window_2, zscore_threshold=mr_deviation_2)

        elif strategy_name_2 == "Moving Average":
            short_window_2 = st.number_input("Short Window (Strategy 2)", min_value=5, value=14, step=1)
            long_window_2 = st.number_input("Long Window (Strategy 2)", min_value=20, value=50, step=1)
            exponential_mode_2 = st.checkbox("Exponential Moving Average (EMA) (Strategy 2)", value=False)
            strategy_2 = MovingAverage(short_window=short_window_2, long_window=long_window_2,
                                       exponential_mode=exponential_mode_2)

        elif strategy_name_2 == "Quality":
            selected_roe_file_2 = st.selectbox("Select ROE File (Strategy 2)", options=list(data_files.keys()))
            selected_roa_file_2 = st.selectbox("Select ROA File (Strategy 2)", options=list(data_files.keys()))
            window_2 = st.number_input("Window Period (Days) (Strategy 2)", min_value=5, value=30, step=1)
            assets_picked_long_2 = st.number_input("Number of Long Positions (Strategy 2)", min_value=0, value=10,
                                                   step=1)
            assets_picked_short_2 = st.number_input("Number of Short Positions (Strategy 2)", min_value=0, value=10,
                                                    step=1)
            strategy_2 = Quality(window=window_2, assets_picked_long=assets_picked_long_2,
                                 assets_picked_short=assets_picked_short_2)
            if selected_roe_file_2 and selected_roa_file_2:
                strategy_2.fit({"ROE": data_files[selected_roe_file_2], "ROA": data_files[selected_roa_file_2]})

        elif strategy_name_2 == "Value":
            selected_per_file_2 = st.selectbox("Select PER File (Strategy 2)", options=list(data_files.keys()))
            selected_pbr_file_2 = st.selectbox("Select PBR File (Strategy 2)", options=list(data_files.keys()))
            window_2 = st.number_input("Window Period (Days) (Strategy 2)", min_value=5, value=30, step=1)
            assets_picked_long_2 = st.number_input("Number of Long Positions (Strategy 2)", min_value=0, value=10,
                                                   step=1)
            assets_picked_short_2 = st.number_input("Number of Short Positions (Strategy 2)", min_value=0, value=10,
                                                    step=1)
            strategy_2 = Value(window=window_2, assets_picked_long=assets_picked_long_2,
                               assets_picked_short=assets_picked_short_2)
            if selected_per_file_2 and selected_pbr_file_2:
                strategy_2.fit({"PER": data_files[selected_per_file_2], "PBR": data_files[selected_pbr_file_2]})

        elif strategy_name_2 == "Size":
            selected_market_cap_file_2 = st.selectbox("Select Market Cap File (Strategy 2)",
                                                      options=list(data_files.keys()))
            window_2 = st.number_input("Window Period (Days) (Strategy 2)", min_value=5, value=30, step=1)
            assets_picked_long_2 = st.number_input("Number of Long Positions (Strategy 2)", min_value=0, value=10,
                                                   step=1)
            assets_picked_short_2 = st.number_input("Number of Short Positions (Strategy 2)", min_value=0, value=10,
                                                    step=1)
            strategy_2 = Size(window=window_2, assets_picked_long=assets_picked_long_2,
                              assets_picked_short=assets_picked_short_2)
            if selected_market_cap_file_2:
                strategy_2.fit(data_files[selected_market_cap_file_2])

        elif strategy_name_2 == "Buy and Hold":
            strategy_2 = BuyAndHold()

        elif strategy_name_2 == "MinVariance":
            short_sell_2 = st.checkbox("Allow Short Selling", value=False)
            strategy_2 = MinVariance(short_sell=short_sell_2)

        elif strategy_name_2 == "Volatility Trend":
            atr_period_2 = st.number_input("ATR Period (Strategy 2)", min_value=5, value=14, step=1)
            dmi_period_2 = st.number_input("DMI Period (Strategy 2)", min_value=5, value=14, step=1)
            atr_threshold_2 = st.number_input("ATR Threshold (Strategy 2)", min_value=0.1, value=1.0, step=0.1)
            strategy_2 = VolatilityTrendStrategy(atr_period=atr_period_2, dmi_period=dmi_period_2,
                                                 atr_threshold=atr_threshold_2)

        elif strategy_name_2 == "Keltner Channel":
            atr_period_2 = st.number_input("ATR Period (Strategy 2)", min_value=5, value=10, step=1)
            atr_multiplier_2 = st.number_input("ATR Multiplier (Strategy 2)", min_value=1.0, value=2.0, step=0.1)
            sma_period_2 = st.number_input("SMA Period (Strategy 2)", min_value=5, value=20, step=1)
            strategy_2 = KeltnerChannelStrategy(atr_period=atr_period_2, atr_multiplier=atr_multiplier_2,
                                                sma_period=sma_period_2)

# Exécution du backtest
if st.button("Run Backtest") and strategy_1 is not None and historical_data_file_1:
    backtester_1 = Backtester(
        data_source=historical_data_1,
        weight_scheme=weight_scheme_1,
        market_cap_source=market_cap_file_1,
        special_start=special_start_1,
        transaction_cost=transaction_cost_1/100,
        slippage=slippage_1/100,
        risk_free_rate=risk_free_rate_1/100,
        rebalancing_frequency=rebalancing_frequency_1,
        plot_library=plot_library_1
    )
    result_1 = backtester_1.run(strategy_1, is_VT=apply_vol_target_1, target_vol=target_vol_1)

    if compare_strategies and strategy_2 is not None and historical_data_file_2:
        backtester_2 = Backtester(
            data_source=historical_data_2,
            weight_scheme=weight_scheme_2,
            market_cap_source=market_cap_file_2,
            transaction_cost=transaction_cost_2,
            slippage=slippage_2,
            risk_free_rate=risk_free_rate_2,
            rebalancing_frequency=rebalancing_frequency_2,
            special_start=special_start_2,
            plot_library=plot_library_2
        )
        result_2 = backtester_2.run(strategy_2, is_VT=apply_vol_target_2, target_vol=target_vol_2)

        st.subheader("Strategy Comparison")
        result_1.compare([result_2], strategy_names=[strategy_name_1, strategy_name_2], streamlit_display=True)

    else:
        # Afficher les résultats de la stratégie 1 uniquement
        st.subheader("Results - Strategy 1")
        result_1.display_statistics(streamlit_display=True)
        result_1.plot_cumulative_returns(streamlit_display=True)
        result_1.plot_monthly_returns_heatmap(streamlit_display=True)
        result_1.plot_returns_distribution(streamlit_display=True)




