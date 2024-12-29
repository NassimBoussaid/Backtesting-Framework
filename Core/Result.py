import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import calendar
from scipy.stats import skew, kurtosis


class Result:
    """
    Classe pour stocker et analyser la performance d'une stratégie de trading ou d'un portefeuille.
    Calcule diverses métriques telles que Sharpe, Sortino, VaR, Expected Shortfall, Calmar Ratio, etc.,
    et fournit des graphiques (rendements cumulés, heatmap mensuelle, distribution) en utilisant la bibliothèque de visualisation choisie.
    """
    PERIODS_PER_YEAR = 252

    def __init__(self, portfolio_returns, cumulative_returns, risk_free_rate=0.0, trade_stats=None,
                 plot_library='matplotlib'):
        """
        Initialise l'objet Result.

        :param portfolio_returns: pd.Series
            Série temporelle des rendements quotidiens (ou périodiques) du portefeuille.
            Doit avoir un DatetimeIndex si un rééchantillonnage (par exemple, mensuel) est souhaité.
        :param cumulative_returns: pd.Series
            Série temporelle des rendements cumulés du portefeuille, avec le même index que portfolio_returns.
            Représente la performance cumulée (par exemple, 1.0 -> 1.05 -> 1.10 ...).
        :param risk_free_rate: float, optionnel
            Taux sans risque annualisé (par défaut = 0.0). Utilisé pour les calculs de Sharpe/Sortino.
        :param trade_stats: tuple, optionnel
            Tuple de (total_trades, winning_trades). Utilisé pour calculer le win_rate.
        :param plot_library: str, optionnel
            Bibliothèque de visualisation à utiliser pour les graphiques. Choix possibles : 'matplotlib', 'seaborn', 'plotly'.
            Par défaut : 'matplotlib'.
        """
        if not isinstance(portfolio_returns, pd.Series) or not isinstance(cumulative_returns, pd.Series):
            raise TypeError("portfolio_returns et cumulative_returns doivent être des séries pandas.")
        if not portfolio_returns.index.equals(cumulative_returns.index):
            raise ValueError("portfolio_returns et cumulative_returns doivent avoir le même index.")
        if not isinstance(plot_library, str):
            raise TypeError("plot_library doit être une chaîne de caractères.")
        if plot_library.lower() not in ['matplotlib', 'seaborn', 'plotly']:
            raise ValueError("plot_library doit être l'une des suivantes : 'matplotlib', 'seaborn', 'plotly'.")

        daily_returns = portfolio_returns.dropna()

        self.portfolio_returns = daily_returns.copy()
        self.portfolio_returns.name = 'Rendement du Portefeuille'
        self.cumulative_returns = cumulative_returns
        self.risk_free_rate = risk_free_rate
        self.plot_library = plot_library.lower()

        # Métriques de performance de base
        self.total_return = self.calculate_total_return()
        self.annualized_return = self.calculate_annualized_return()
        self.volatility = self.calculate_volatility()
        self.sharpe_ratio = self.calculate_sharpe_ratio()
        self.max_drawdown = self.calculate_max_drawdown()
        self.max_drawdown_recovery_time = self.calculate_max_drawdown_recovery_time()
        self.sortino_ratio = self.calculate_sortino_ratio()
        self.calmar_ratio = self.calculate_calmar_ratio()

        # Skewness et Kurtosis
        daily_returns_array = daily_returns.to_numpy()
        self.skewness = skew(daily_returns_array)
        self.kurtosis = kurtosis(daily_returns_array)

        # Statistiques des trades
        self.total_trades = trade_stats[0] if trade_stats else 0
        self.winning_trades = trade_stats[1] if trade_stats else 0
        self.win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0

    def calculate_total_return(self):
        """
        Retourne le rendement cumulatif final (par exemple, 0.30 signifie +30% au total).

        :return: float
            Rendement total du portefeuille.
        """
        return self.cumulative_returns.iloc[-1]

    def calculate_annualized_return(self):
        """
        Calcule le rendement annualisé.

        :return: float
            Rendement annualisé.
        """
        total_days = (self.portfolio_returns.index[-1] - self.portfolio_returns.index[0]).days
        years = total_days / 252
        if years <= 0:
            return np.nan
        return (1 + self.total_return) ** (1 / years) - 1

    def calculate_volatility(self):
        """
        Calcule la volatilité annualisée des rendements.

        :return: float
            Volatilité annualisée.
        """
        return self.portfolio_returns.std(ddof=1) * np.sqrt(self.PERIODS_PER_YEAR)

    def calculate_sharpe_ratio(self):
        """
        Calcule le Sharpe Ratio annualisé.

        :return: float
            Sharpe Ratio.
        """
        daily_rf = self.risk_free_rate / self.PERIODS_PER_YEAR
        excess_returns = self.portfolio_returns - daily_rf
        annual_excess_return = excess_returns.mean() * self.PERIODS_PER_YEAR
        annual_excess_vol = excess_returns.std(ddof=1) * np.sqrt(self.PERIODS_PER_YEAR)
        return annual_excess_return / annual_excess_vol if annual_excess_vol != 0 else np.nan

    def calculate_max_drawdown(self):
        """
        Calcule le drawdown maximum.

        :return: float
            Drawdown maximum.
        """
        cummax_series = self.cumulative_returns.cummax()
        drawdowns = self.cumulative_returns - cummax_series
        return drawdowns.min()

    def calculate_max_drawdown_recovery_time(self):
        """
        Calcule le nombre de jours nécessaires pour que le portefeuille revienne à son plus haut précédent après le drawdown maximum.
        Retourne None si la récupération n'a pas encore eu lieu.

        :return: int or None
            Nombre de jours de récupération ou None si non récupéré.
        """
        cummax_series = self.cumulative_returns.cummax()
        drawdowns = self.cumulative_returns - cummax_series
        trough_idx = drawdowns.idxmin()

        peak_value = cummax_series.loc[trough_idx]

        post_trough = self.cumulative_returns[self.cumulative_returns.index >= trough_idx]
        recovery_idx = post_trough[post_trough >= peak_value].index.min()

        if pd.isna(recovery_idx):
            return None
        else:
            return (recovery_idx - trough_idx).days

    def calculate_sortino_ratio(self):
        """
        Calcule le Sortino Ratio annualisé.

        :return: float
            Sortino Ratio.
        """
        negative_returns = self.portfolio_returns[self.portfolio_returns < 0]
        annual_downside_std = negative_returns.std(ddof=1) * np.sqrt(self.PERIODS_PER_YEAR)
        if annual_downside_std == 0:
            return np.nan
        return (self.annualized_return - self.risk_free_rate) / annual_downside_std

    def calculate_calmar_ratio(self):
        """
        Calcule le Calmar Ratio.

        :return: float
            Calmar Ratio.
        """
        mdd = abs(self.max_drawdown)
        return np.nan if mdd == 0 else (self.annualized_return / mdd)

    def calculate_var(self, alpha=0.05):
        """
        Calcule la Value at Risk (VaR) au niveau de confiance alpha.

        :param alpha: float, optionnel
            Niveau de signification pour la VaR (par défaut = 0.05).
        :return: float
            Seuil de VaR.
        """
        return self.portfolio_returns.quantile(alpha)

    def calculate_expected_shortfall(self, alpha=0.05):
        """
        Calcule l'Expected Shortfall (ES) au niveau de confiance alpha.

        :param alpha: float, optionnel
            Niveau de signification pour l'ES (par défaut = 0.05).
        :return: float
            Expected Shortfall.
        """
        var_threshold = self.calculate_var(alpha)
        tail_losses = self.portfolio_returns[self.portfolio_returns < var_threshold]
        if len(tail_losses) == 0:
            return var_threshold
        return tail_losses.mean()

    def calculate_monthly_returns(self):
        """
        Rééchantillonne les rendements quotidiens en rendements mensuels.

        :return: pd.DataFrame
            Tableau pivot des rendements mensuels.
        """
        monthly_series = self.portfolio_returns.resample('ME').sum()
        df_monthly = monthly_series.to_frame(name='Rendements Mensuels')
        df_monthly['Année'] = df_monthly.index.year
        df_monthly['Mois'] = df_monthly.index.month
        pivot = df_monthly.pivot(index='Année', columns='Mois', values='Rendements Mensuels')
        pivot.columns = [calendar.month_abbr[int(m)] for m in pivot.columns]
        return pivot

    def display_statistics(self, streamlit_display=False):
        """
        Affiche les principales statistiques de performance.
        Inclut la VaR et l'Expected Shortfall à 95% par défaut.

        :param streamlit_display: bool, optionnel
            Si True, affiche les statistiques via Streamlit. Sinon, les imprime dans la console.
        """
        stats = {
            'Rendement Total': f"{self.total_return:.2%}",
            'Rendement Annualisé': f"{self.annualized_return:.2%}",
            'Volatilité': f"{self.volatility:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.4f}",
            'Drawdown Maximum': f"{self.max_drawdown:.2%}",
            'Temps de Récupération (jours)': (
                self.max_drawdown_recovery_time
                if self.max_drawdown_recovery_time is not None
                else "Non récupéré"
            ),
            'Sortino Ratio': f"{self.sortino_ratio:.4f}",
            'Calmar Ratio': f"{self.calmar_ratio:.4f}",
            'Skewness': f"{self.skewness:.4f}",
            'Kurtosis (excess)': f"{self.kurtosis:.4f}",
            'Total Trades': self.total_trades,
            'Winning Trades': self.winning_trades,
            'Win Rate': f"{self.win_rate:.2%}",
            'VaR (95%)': f"{self.calculate_var(0.05):.2%}",
            'Expected Shortfall (95%)': f"{self.calculate_expected_shortfall(0.05):.2%}"
        }

        if streamlit_display:
            import streamlit as st
            st.subheader("Statistiques de Performance")
            for key, val in stats.items():
                st.write(f"**{key} :** {val}")
        else:
            print("\nStatistiques de Performance :")
            print("------------------------------")
            for key, val in stats.items():
                print(f"{key} : {val}")

    def plot_cumulative_returns(self, streamlit_display=False):
        """
        Trace les rendements cumulés au fil du temps en utilisant la bibliothèque de visualisation sélectionnée.

        :param streamlit_display: bool, optionnel
            Si True, affiche le graphique via Streamlit. Sinon, affiche avec plt.show() ou plotly.
        """
        if self.plot_library in ['matplotlib', 'seaborn']:
            plt.figure(figsize=(12, 6))
            if self.plot_library == 'seaborn':
                sns.set(style="darkgrid")
                sns.lineplot(x=self.cumulative_returns.index, y=self.cumulative_returns, label="Rendements Cumulés")
            else:
                plt.plot(self.cumulative_returns.index, self.cumulative_returns, label="Rendements Cumulés")

            plt.title("Rendements Cumulés du Portefeuille")
            plt.xlabel("Date")
            plt.ylabel("Rendement Cumulé")
            plt.legend()
            plt.grid(True)

            if streamlit_display:
                import streamlit as st
                st.pyplot(plt)
                plt.close()
            else:
                plt.show()
                plt.close()

        elif self.plot_library == 'plotly':
            fig = px.line(
                x=self.cumulative_returns.index,
                y=self.cumulative_returns,
                labels={'x': 'Date', 'y': 'Rendement Cumulé'},
                title="Rendements Cumulés du Portefeuille"
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Rendement Cumulé")

            if streamlit_display:
                import streamlit as st
                st.plotly_chart(fig)
            else:
                fig.show(renderer="browser")

    def plot_monthly_returns_heatmap(self, streamlit_display=False):
        """
        Trace une heatmap des rendements mensuels en utilisant la bibliothèque de visualisation sélectionnée.

        :param streamlit_display: bool, optionnel
            Si True, affiche le graphique via Streamlit. Sinon, affiche avec plt.show() ou plotly.
        """
        monthly_returns_pivot = self.calculate_monthly_returns()

        if self.plot_library in ['matplotlib', 'seaborn']:
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                monthly_returns_pivot,
                annot=True,
                fmt=".2%",
                cmap="RdYlGn",
                center=0
            )
            plt.title("Heatmap des Rendements Mensuels")
            plt.xlabel("Mois")
            plt.ylabel("Année")
            plt.tight_layout()

            if streamlit_display:
                import streamlit as st
                st.pyplot(plt)
                plt.close()
            else:
                plt.show()
                plt.close()

        elif self.plot_library == 'plotly':
            heatmap_data = monthly_returns_pivot.reset_index()
            heatmap_data_melt = heatmap_data.melt(id_vars='Année', var_name='Mois', value_name='Rendements Mensuels')
            pivot_plotly = heatmap_data_melt.pivot(index='Année', columns='Mois', values='Rendements Mensuels')
            ordered_months = list(calendar.month_abbr)[1:]
            pivot_plotly = pivot_plotly.reindex(columns=ordered_months)

            fig = px.imshow(
                pivot_plotly,
                labels=dict(x="Mois", y="Année", color="Rendements Mensuels"),
                x=ordered_months,
                y=pivot_plotly.index,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                title="Heatmap des Rendements Mensuels",
                text_auto=True
            )

            fig.update_traces(texttemplate='%{text:.2%}', textfont={"size": 12})

            if streamlit_display:
                import streamlit as st
                st.plotly_chart(fig)
            else:
                fig.show(renderer="browser")

    def plot_returns_distribution(self, alpha=0.05, bins=50, streamlit_display=False):
        """
        Trace un histogramme des rendements quotidiens pour visualiser leur distribution.
        Ajoute des lignes verticales pour la VaR et l'Expected Shortfall au niveau de confiance alpha.

        :param alpha: float, optionnel
            Niveau de signification pour la VaR et l'ES (par défaut = 0.05 = confiance à 95%).
        :param bins: int, optionnel
            Nombre de bins dans l'histogramme (par défaut = 50).
        :param streamlit_display: bool, optionnel
            Si True, affiche le graphique via Streamlit. Sinon, affiche avec plt.show() ou plotly.
        """
        var_value = self.calculate_var(alpha)
        es_value = self.calculate_expected_shortfall(alpha)

        if self.plot_library in ['matplotlib', 'seaborn']:
            plt.figure(figsize=(10, 6))
            if self.plot_library == 'seaborn':
                sns.set(style="darkgrid")
                sns.histplot(self.portfolio_returns, bins=bins, kde=True, color='blue', alpha=0.6)
            else:
                plt.hist(self.portfolio_returns, bins=bins, alpha=0.6, color='blue', density=True, label='Rendements Quotidiens')
                sns.kdeplot(self.portfolio_returns, color='blue', fill=False)

            plt.axvline(var_value, color='red', linestyle='--', label=f"VaR({int((1 - alpha) * 100)}%) = {var_value:.2%}")
            plt.axvline(es_value, color='orange', linestyle='--', label=f"ES({int((1 - alpha) * 100)}%) = {es_value:.2%}")

            plt.title("Distribution des Rendements Quotidiens avec VaR & ES")
            plt.xlabel("Rendement Quotidien")
            plt.ylabel("Fréquence")
            plt.grid(True)
            plt.legend()

            if streamlit_display:
                import streamlit as st
                st.pyplot(plt)
                plt.close()
            else:
                plt.show()
                plt.close()

        elif self.plot_library == 'plotly':
            fig = px.histogram(
                x=self.portfolio_returns,
                nbins=bins,
                labels={'x': 'Rendement Quotidien'},
                title="Distribution des Rendements Quotidiens avec VaR & ES",
                opacity=0.6,
                histnorm='density'
            )
            fig.add_vline(
                x=var_value,
                line=dict(color='red', dash='dash'),
                annotation_text=f"VaR({int((1 - alpha) * 100)}%) = {var_value:.2%}",
                annotation_position="top left",
                annotation=dict(font=dict(color="red"))
            )
            fig.add_vline(
                x=es_value,
                line=dict(color='orange', dash='dash'),
                annotation_text=f"ES({int((1 - alpha) * 100)}%) = {es_value:.2%}",
                annotation_position="bottom left",
                annotation=dict(font=dict(color="orange"))
            )

            fig.update_layout(xaxis_title="Rendement Quotidien", yaxis_title="Densité")

            if streamlit_display:
                import streamlit as st
                st.plotly_chart(fig)
            else:
                fig.show(renderer="browser")
