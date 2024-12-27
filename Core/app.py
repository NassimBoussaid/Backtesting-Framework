#################################### WARNING ###################################
# pour lancer l'interface faut entrer la commande suivante  directement dans la console: streamlit run app.py
#si streamlit pas installer : pip install streamlit

##################################################################################


import streamlit as st
import pandas as pd
from Backtester import Backtester
from Strategies.RSI import RSI
from Strategies.BollingerBands import BollingerBands
from Strategies.MeanReversion import MeanReversion
from Strategies.MovingAverage import MovingAverage


@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path, index_col=0)

# Interface utilisateur
st.title("Backtesting Interface")

# Charger les données
data_file = st.file_uploader("Upload Historical Data File (Excel format)", type=["xlsx"])
if data_file:
    historical_data = load_data(data_file)
    st.write("Data Preview", historical_data.head())

    # Choisir la stratégie
    strategy_name = st.selectbox(
        "Choose a Strategy",
        ["RSI", "Bollinger Bands", "Mean Reversion", "Moving Average"]
    )

    # Paramètres de la stratégie
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
        mr_threshold = st.slider("Threshold (Number of Standard Deviations)", 1.0, 3.0, 2.0)
        strategy = MeanReversion(window=mr_window, threshold=mr_threshold)

    elif strategy_name == "Moving Average":
        short_window = st.slider("Short Window", 5, 50, 14)
        long_window = st.slider("Long Window", 20, 200, 50)
        exponential_mode = st.checkbox("Exponential Moving Average (EMA)", value=False)
        strategy = MovingAverage(short_window=short_window, long_window=long_window, exponential_mode=exponential_mode)


    if st.button("Run Backtest"):
        backtester = Backtester(historical_data, special_start=100)
        result = backtester.run(strategy, "daily")

        # Afficher les résultats
        st.subheader("Results")
        result.display_statistics(streamlit_display=True)
        result.plot_cumulative_returns(streamlit_display=True)
        result.plot_portfolio_returns(streamlit_display=True)

