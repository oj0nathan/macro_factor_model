from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.pipeline import run_full_pipeline


# Streamlit config
st.set_page_config(
    page_title="Macro Factor Model – OLS Timing & Sector Strategy",
    layout="wide",
)


# Helpers
def get_fred_key() -> str | None:
    """Try to pull the FRED key from Streamlit secrets, else env var."""
    key = None
    try:
        key = st.secrets.get("FRED_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None

    if not key:
        key = os.getenv("FRED_API_KEY")

    return key


@st.cache_data(show_spinner="Running macro factor pipeline…", ttl=60 * 60)
def load_data() -> Dict[str, Any]:
    fred_key = get_fred_key()
    if not fred_key:
        st.error(
            "FRED API key not found. Add `FRED_API_KEY` to `.streamlit/secrets.toml` "
            "or export it as an environment variable."
        )
        st.stop()

    data = run_full_pipeline(fred_api_key=fred_key, start_date="1995-01-01", window=84)
    return data


def format_perf_table(df: pd.DataFrame) -> pd.Styler:
    fmt = {
        "Total Return": "{:.2%}".format,
        "CAGR": "{:.2%}".format,
        "AnnVol": "{:.2%}".format,
        "Sharpe": "{:.2f}".format,
        "MaxDD": "{:.2%}".format,
    }
    return df.copy().style.format(fmt)


def apply_last_n_years_filter(
    df_or_series: pd.DataFrame | pd.Series, last_n_years: int
) -> pd.DataFrame | pd.Series:
    """
    Keep only the last `last_n_years` worth of data.

    Works for both Series and DataFrames that have a DateTimeIndex.
    If the series is shorter than N years, this effectively just returns the full series.
    """
    if df_or_series is None or len(df_or_series) == 0:
        return df_or_series

    end_date = df_or_series.index.max()
    start_date = end_date - pd.DateOffset(years=last_n_years)
    return df_or_series.loc[df_or_series.index >= start_date]


# Views
def view_overview(data: Dict[str, Any]) -> None:
    st.header("Strategy vs benchmarks – Summary")

    perf_table = data["perf_table"]
    st.caption("Performance table (monthly data, annualised stats)")
    st.dataframe(format_perf_table(perf_table), use_container_width=True)

    st.subheader("Equity curves")

    equity_curves: pd.DataFrame = data["equity_curves"]

    # Rebase to 1.0 at the start of the available sample
    equity_rebased = equity_curves / equity_curves.iloc[0]
    st.line_chart(equity_rebased, use_container_width=True)


def view_diagnostics(data: Dict[str, Any]) -> None:
    st.header("In-sample fit & predictive power by ETF")

    st.subheader("Rolling OLS fit metrics")
    metrics_df = data["metrics_df"]
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Actual vs predicted excess returns")

    y_actual: pd.DataFrame = data["Y_actual"]
    y_hat: pd.DataFrame = data["Y_hat"]

    etf_list = list(y_actual.columns)
    chosen_etf = st.selectbox("Choose ETF", etf_list, index=0)

    actual_series = y_actual[chosen_etf]
    pred_series = y_hat[chosen_etf].reindex(actual_series.index)

    plot_df = pd.DataFrame(
        {"Actual": actual_series, "Predicted": pred_series}, index=actual_series.index
    )

    st.line_chart(plot_df, use_container_width=True)


def view_macro_and_sensitivities(data: Dict[str, Any]) -> None:
    st.header("Macro dashboard & rolling sector sensitivities")

    tab_macro, tab_beta = st.tabs(["Macro dashboard", "Rolling beta heatmap"])

    # Macro dashboard 
    with tab_macro:
        st.subheader("Macro dashboard")

        last_n_years_macro = st.slider(
            "Show last N years (macro)",
            min_value=3,
            max_value=25,
            value=10,
            step=1,
            help="Filter macro z-score plots to the most recent N years.",
        )

        growth_z: pd.DataFrame = data["growth_z"]
        fin_z: pd.DataFrame = data["fin_z"]

        growth_window = apply_last_n_years_filter(growth_z, last_n_years_macro)
        fin_window = apply_last_n_years_filter(fin_z, last_n_years_macro)

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Growth & Inflation (z-scores)")
            st.line_chart(growth_window, use_container_width=True)

        with col2:
            st.caption("Yield Curve & Real Policy Rate (z-scores)")
            st.line_chart(fin_window, use_container_width=True)

    # Rolling beta heatmap 
    with tab_beta:
        st.subheader("Rolling sector beta to REAL_FF (real policy rate)")

        rolling_betas: pd.DataFrame = data["rolling_betas"]
        all_sectors = list(rolling_betas.columns)

        selected_sectors = st.multiselect(
            "Sectors",
            all_sectors,
            default=all_sectors,
            help="Choose which sector ETFs to show in the heatmap.",
        )

        if not selected_sectors:
            st.warning("Select at least one sector to display the heatmap.")
            return

        beta_df = rolling_betas[selected_sectors].copy()
        if beta_df.empty:
            st.warning("No rolling beta data available.")
            return

        beta_df.index.name = "Date"
        beta_long = (
            beta_df.reset_index()
            .melt(id_vars="Date", var_name="Sector", value_name="Beta")
            .dropna(subset=["Beta"])
        )

        heatmap = (
            alt.Chart(beta_long)
            .mark_rect()
            .encode(
                x=alt.X(
                    "Date:T",
                    title="Date",
                    axis=alt.Axis(format="%Y-%m"),
                ),
                y=alt.Y("Sector:N", title="Sector"),
                color=alt.Color(
                    "Beta:Q",
                    title="Rolling beta",
                    scale=alt.Scale(
                        scheme="redyellowblue", domain=(-0.15, 0.15), reverse=True
                    ),
                ),
                tooltip=[
                    alt.Tooltip("yearmonth(Date):T", title="Month"),
                    alt.Tooltip("Sector:N"),
                    alt.Tooltip("Beta:Q", format=".3f"),
                ],
            )
            .properties(height=320)  # no use_container_width here
        )

        st.altair_chart(heatmap, use_container_width=True)

        with st.expander("How to read this"):
            st.markdown(
                """
- Colors show the rolling beta of each sector's **relative return vs SPY** to the real Fed funds rate (**REAL_FF**).
- Warm colors = more positively exposed; cool colors = more negatively exposed.
- The scale is centred around 0 so positive/negative exposures are comparable.
                """
            )


# Main
def main() -> None:
    st.sidebar.title("Navigation")
    view = st.sidebar.radio(
        "Choose a view:",
        ("Overview", "Factor model diagnostics", "Macro & sensitivities"),
    )

    data = load_data()

    st.markdown(
        """
# Macro Factor Model – OLS Timing & Sector Strategy

This app estimates a macro factor model on US sector ETFs, uses rolling OLS to forecast excess returns, and evaluates a cross-sectional long/short sector strategy and a simple SPY timing rule.

The dashboard is organised into:

- **Overview** – equity curves and high-level performance;
- **Factor model diagnostics** – R² and actual vs predicted returns;
- **Macro & sensitivities** – macro z-score dashboard and rolling sector betas.
        """
    )

    if view == "Overview":
        view_overview(data)
    elif view == "Factor model diagnostics":
        view_diagnostics(data)
    else:
        view_macro_and_sensitivities(data)


if __name__ == "__main__":
    main()
