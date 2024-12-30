#################################### WARNING ###################################
# pour lancer l'interface faut entrer la commande suivante directement dans la console: streamlit run app.py
#si streamlit pas installer : pip install streamlit

##################################################################################
import streamlit as st
import pandas as pd
from Backtester import Backtester
from RSI import RSI
from BollingerBands import BollingerBands
from MeanReversion import MeanReversion
from MovingAverage import MovingAverage
from Quality import Quality
from Value import Value
from Size import Size
from BuyAndHold import BuyAndHold
from MinVariance import MinVariance
from Volatility_Trend import VolatilityTrendStrategy
from Keltner_Channel_Strategy import KeltnerChannelStrategy

@st.cache_data
def load_data(file_path):
    if file_path.name.endswith('.xlsx'):
        data = pd.read_excel(file_path, index_col=0)
    elif file_path.name.endswith('.csv'):
        data = pd.read_csv(file_path, index_col=0)
    else:
        raise ValueError("Unsupported file format. Please upload an Excel or CSV file.")

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
    "Upload Files (Excel or CSV format)", type=["xlsx", "csv"], accept_multiple_files=True
)
data_files = {}
if uploaded_files:
    for file in uploaded_files:
        data_files[file.name] = load_data(file)
    st.write("Uploaded Files:", list(data_files.keys()))

# Paramètres globaux
st.sidebar.subheader("Global Settings")
transaction_cost = st.sidebar.number_input("Transaction Cost (per trade %):", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
slippage = st.sidebar.number_input("Slippage (per trade %):", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
rebalancing_frequency = st.sidebar.selectbox("Rebalancing Frequency:", ["daily", "weekly", "monthly"], index=2)
weight_scheme = st.sidebar.selectbox("Weighting Scheme:", ["EqualWeight", "MarketCapWeight"], index=0)
special_start = st.sidebar.number_input("Special Start (Index):", min_value=1, max_value=1000, value=100)
visualization_library = st.sidebar.selectbox("Visualization Library:", ["matplotlib", "seaborn", "plotly"], index=0)

market_cap_file = None
if weight_scheme == "MarketCapWeight":
    st.sidebar.subheader("Market Cap Data")
    market_cap_file = st.sidebar.file_uploader("Upload Market Cap File (Excel or CSV format):", type=["xlsx", "csv"])

# Comparaison de stratégies
compare_strategies = st.sidebar.checkbox("Compare Two Strategies", value=False)
strategy_2 = None

# Choisir la stratégie principale
st.subheader("Choose a Strategy")
strategy_name = st.selectbox(
    "Select Strategy",
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
        "Keltner Channel"
    ]
)

strategy = None
historical_data_file = None
if strategy_name:
    # Sélection du fichier de données historiques pour toutes les stratégies
    st.subheader("Select Historical Data File")
    historical_data_file = st.selectbox("Select Historical Data File", options=list(data_files.keys()))

    if historical_data_file:
        historical_data = data_files[historical_data_file]

    # Paramètres de la stratégie et sélection des fichiers requis
    if strategy_name == "RSI":
        rsi_period = st.slider("RSI Period", 5, 50, 14)
        rsi_overbought = st.slider("Overbought Threshold", 50, 100, 70)
        rsi_oversold = st.slider("Oversold Threshold", 0, 50, 30)
        strategy = RSI(period=rsi_period, overbought_threshold=rsi_overbought, oversold_threshold=rsi_oversold)

    elif strategy_name == "Bollinger Bands":
        bb_window = st.slider("Window Period", 5, 50, 20)
        bb_std_dev = st.slider("Number of Standard Deviations", 1.0, 3.0, 2.0)
        strategy = BollingerBands(window=bb_window, num_std_dev=bb_std_dev)

    elif strategy_name == "Mean Reversion":
        mr_window = st.slider("Window Period", 5, 50, 20)
        mr_deviation = st.slider("Threshold (Number of Standard Deviations)", 1.0, 3.0, 2.0)
        strategy = MeanReversion(window=mr_window, zscore_threshold=mr_deviation)

    elif strategy_name == "Moving Average":
        short_window = st.slider("Short Window", 5, 50, 14)
        long_window = st.slider("Long Window", 20, 200, 50)
        exponential_mode = st.checkbox("Exponential Moving Average (EMA)", value=False)
        strategy = MovingAverage(short_window=short_window, long_window=long_window, exponential_mode=exponential_mode)

    elif strategy_name == "Quality":
        selected_roe_file = st.selectbox("Select ROE File", options=list(data_files.keys()))
        selected_roa_file = st.selectbox("Select ROA File", options=list(data_files.keys()))
        window = st.slider("Window Period (Days)", 5, 50, 30)
        assets_picked_long = st.slider("Number of Long Positions", 0, 1000, 10)
        assets_picked_short = st.slider("Number of Short Positions", 0, 1000, 10)
        strategy = Quality(window=window, assets_picked_long=assets_picked_long, assets_picked_short=assets_picked_short)
        if selected_roe_file and selected_roa_file:
            strategy.fit({"ROE": data_files[selected_roe_file], "ROA": data_files[selected_roa_file]})

    elif strategy_name == "Value":
        selected_per_file = st.selectbox("Select PER File", options=list(data_files.keys()))
        selected_pbr_file = st.selectbox("Select PBR File", options=list(data_files.keys()))
        window = st.slider("Window Period (Days)", 5, 50, 30)
        assets_picked_long = st.slider("Number of Long Positions", 0, 1000, 10)
        assets_picked_short = st.slider("Number of Short Positions", 0, 1000, 10)
        strategy = Value(window=window, assets_picked_long=assets_picked_long, assets_picked_short=assets_picked_short)
        if selected_per_file and selected_pbr_file:
            strategy.fit({"PER": data_files[selected_per_file], "PBR": data_files[selected_pbr_file]})

    elif strategy_name == "Size":
        selected_market_cap_file = st.selectbox("Select Market Cap File", options=list(data_files.keys()))
        window = st.slider("Window Period (Days)", 5, 50, 30)
        assets_picked_long = st.slider("Number of Long Positions", 0, 1000, 10)
        assets_picked_short = st.slider("Number of Short Positions", 0, 1000, 10)
        strategy = Size(window=window, assets_picked_long=assets_picked_long, assets_picked_short=assets_picked_short)
        if selected_market_cap_file:
            strategy.fit(data_files[selected_market_cap_file])

    elif strategy_name == "Buy and Hold":
        strategy = BuyAndHold()

    elif strategy_name == "MinVariance":
        short_sell = st.checkbox("Allow Short Selling", value=False)
        strategy = MinVariance(short_sell=short_sell)

    elif strategy_name == "Volatility Trend":
        atr_period = st.slider("ATR Period", 5, 50, 14)
        dmi_period = st.slider("DMI Period", 5, 50, 14)
        atr_threshold = st.slider("ATR Threshold", 0.1, 5.0, 1.0, step=0.1)
        strategy = VolatilityTrendStrategy(atr_period=atr_period, dmi_period=dmi_period, atr_threshold=atr_threshold)

    elif strategy_name == "Keltner Channel":
        atr_period = st.slider("ATR Period", 5, 50, 10)
        atr_multiplier = st.slider("ATR Multiplier", 1.0, 5.0, 2.0, step=0.1)
        sma_period = st.slider("SMA Period", 5, 50, 20)
        strategy = KeltnerChannelStrategy(atr_period=atr_period, atr_multiplier=atr_multiplier, sma_period=sma_period)

# Paramètres de la deuxième stratégie si nécessaire
if compare_strategies:
    st.subheader("Choose Second Strategy")
    strategy_name_2 = st.selectbox(
        "Select Second Strategy",
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
            "Keltner Channel"
        ],
        key="strategy_2"
    )

    if strategy_name_2:
        # Paramètres spécifiques pour la deuxième stratégie
        if strategy_name_2 == "RSI":
            rsi_period_2 = st.slider("RSI Period (Strategy 2)", 5, 50, 14)
            rsi_overbought_2 = st.slider("Overbought Threshold (Strategy 2)", 50, 100, 70)
            rsi_oversold_2 = st.slider("Oversold Threshold (Strategy 2)", 0, 50, 30)
            strategy_2 = RSI(period=rsi_period_2, overbought_threshold=rsi_overbought_2, oversold_threshold=rsi_oversold_2)

        elif strategy_name_2 == "Bollinger Bands":
            bb_window_2 = st.slider("Window Period (Strategy 2)", 5, 50, 20)
            bb_std_dev_2 = st.slider("Number of Standard Deviations (Strategy 2)", 1.0, 3.0, 2.0)
            strategy_2 = BollingerBands(window=bb_window_2, num_std_dev=bb_std_dev_2)

        elif strategy_name_2 == "Mean Reversion":
            mr_window_2 = st.slider("Window Period (Strategy 2)", 5, 50, 20)
            mr_deviation_2 = st.slider("Threshold (Number of Standard Deviations) (Strategy 2)", 1.0, 3.0, 2.0)
            strategy_2 = MeanReversion(window=mr_window_2, zscore_threshold=mr_deviation_2)

        elif strategy_name_2 == "Moving Average":
            short_window_2 = st.slider("Short Window (Strategy 2)", 5, 50, 14)
            long_window_2 = st.slider("Long Window (Strategy 2)", 20, 200, 50)
            exponential_mode_2 = st.checkbox("Exponential Moving Average (EMA) (Strategy 2)", value=False)
            strategy_2 = MovingAverage(short_window=short_window_2, long_window=long_window_2, exponential_mode=exponential_mode_2)

        elif strategy_name_2 == "Quality":
            selected_roe_file_2 = st.selectbox("Select ROE File (Strategy 2)", options=list(data_files.keys()))
            selected_roa_file_2 = st.selectbox("Select ROA File (Strategy 2)", options=list(data_files.keys()))
            window_2 = st.slider("Window Period (Days) (Strategy 2)", 5, 50, 30)
            assets_picked_long_2 = st.slider("Number of Long Positions (Strategy 2)", 0, 1000, 10)
            assets_picked_short_2 = st.slider("Number of Short Positions (Strategy 2)", 0, 1000, 10)
            strategy_2 = Quality(window=window_2, assets_picked_long=assets_picked_long_2,
                                 assets_picked_short=assets_picked_short_2)
            if selected_roe_file_2 and selected_roa_file_2:
                strategy_2.fit({"ROE": data_files[selected_roe_file_2], "ROA": data_files[selected_roa_file_2]})

        elif strategy_name_2 == "Value":
            selected_per_file_2 = st.selectbox("Select PER File (Strategy 2)", options=list(data_files.keys()))
            selected_pbr_file_2 = st.selectbox("Select PBR File (Strategy 2)", options=list(data_files.keys()))
            window_2 = st.slider("Window Period (Days) (Strategy 2)", 5, 50, 30)
            assets_picked_long_2 = st.slider("Number of Long Positions (Strategy 2)", 0, 1000, 10)
            assets_picked_short_2 = st.slider("Number of Short Positions (Strategy 2)", 0, 1000, 10)
            strategy_2 = Value(window=window_2, assets_picked_long=assets_picked_long_2,
                               assets_picked_short=assets_picked_short_2)
            if selected_per_file_2 and selected_pbr_file_2:
                strategy_2.fit({"PER": data_files[selected_per_file_2], "PBR": data_files[selected_pbr_file_2]})

        elif strategy_name_2 == "Size":
            selected_market_cap_file_2 = st.selectbox("Select Market Cap File (Strategy 2)",
                                                      options=list(data_files.keys()))
            window_2 = st.slider("Window Period (Days) (Strategy 2)", 5, 50, 30)
            assets_picked_long_2 = st.slider("Number of Long Positions (Strategy 2)", 0, 1000, 10)
            assets_picked_short_2 = st.slider("Number of Short Positions (Strategy 2)", 0, 1000, 10)
            strategy_2 = Size(window=window_2, assets_picked_long=assets_picked_long_2,
                              assets_picked_short=assets_picked_short_2)
            if selected_market_cap_file_2:
                strategy_2.fit(data_files[selected_market_cap_file_2])

        elif strategy_name_2 == "Buy and Hold":
            strategy_2 = BuyAndHold()

        elif strategy_name_2 == "MinVariance":
            short_sell_2 = st.checkbox("Allow Short Selling (Strategy 2)", value=False)
            strategy_2 = MinVariance(short_sell=short_sell_2)

        elif strategy_name_2 == "Volatility Trend":
            atr_period_2 = st.slider("ATR Period (Strategy 2)", 5, 50, 14)
            dmi_period_2 = st.slider("DMI Period (Strategy 2)", 5, 50, 14)
            atr_threshold_2 = st.slider("ATR Threshold (Strategy 2)", 0.1, 5.0, 1.0, step=0.1)
            strategy_2 = VolatilityTrendStrategy(atr_period=atr_period_2, dmi_period=dmi_period_2,
                                                 atr_threshold=atr_threshold_2)

        elif strategy_name_2 == "Keltner Channel":
            atr_period_2 = st.slider("ATR Period (Strategy 2)", 5, 50, 10)
            atr_multiplier_2 = st.slider("ATR Multiplier (Strategy 2)", 1.0, 5.0, 2.0, step=0.1)
            sma_period_2 = st.slider("SMA Period (Strategy 2)", 5, 50, 20)
            strategy_2 = KeltnerChannelStrategy(atr_period=atr_period_2, atr_multiplier=atr_multiplier_2,
                                                sma_period=sma_period_2)

# Exécution du backtest
if st.button("Run Backtest") and strategy is not None and historical_data_file:
    if weight_scheme == "MarketCapWeight" and not market_cap_file:
        st.error("MarketCapWeight selected. Please upload a Market Cap file.")
    else:
        market_cap_data = load_data(market_cap_file) if market_cap_file else None
        backtester = Backtester(
            data_source=historical_data,
            weight_scheme=weight_scheme,
            market_cap_source=market_cap_data,
            transaction_cost=transaction_cost,
            slippage=slippage,
            rebalancing_frequency=rebalancing_frequency,
            special_start=special_start,
            plot_library=visualization_library
        )
        result = backtester.run(strategy)

        # Comparaison de stratégies
        if compare_strategies and strategy_2 is not None:
            backtester_2 = Backtester(
                data_source=historical_data,
                weight_scheme=weight_scheme,
                market_cap_source=market_cap_data,
                transaction_cost=transaction_cost,
                slippage=slippage,
                rebalancing_frequency=rebalancing_frequency,
                special_start=special_start,
                plot_library=visualization_library
            )
            result_2 = backtester_2.run(strategy_2)

            st.subheader("Strategy Comparison")
            comparison = result.compare([result_2], strategy_names=[strategy_name, strategy_name_2], streamlit_display=True)
        else:
            # Afficher les résultats d'une seule stratégie
            st.subheader("Results")
            result.display_statistics(streamlit_display=True)
            result.plot_cumulative_returns(streamlit_display=True)
            result.plot_monthly_returns_heatmap(streamlit_display=True)
            result.plot_returns_distribution(streamlit_display=True)




