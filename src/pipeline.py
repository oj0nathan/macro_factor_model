# src/pipeline.py

from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

# --- Universe definitions ----------------------------------------------------

ETF_TICKERS: List[str] = [
    "SPY",  # S&P 500 (broad US market)
    "XLK",  # Technology
    "XLF",  # Financials
    "XLE",  # Energy
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLU",  # Utilities
    "XLRE", # Real Estate
]

FRED_SERIES = {
    "INDPRO":   "INDPRO",     # Industrial Production Index
    "UNRATE":   "UNRATE",     # Unemployment rate
    "CPI":      "CPIAUCSL",   # CPI, all urban consumers
    "FEDFUNDS": "FEDFUNDS",   # Fed Funds Rate
    "T10Y2Y":   "T10Y2Y",     # 10Y - 2Y Treasury yield curve
    "VIX":      "VIXCLS",     # VIX index
    "BAA":      "BAA",        # Moody's Baa corporate yield
    "M2":       "M2SL",       # M2 money stock
    "WALCL":    "WALCL",      # Fed balance sheet total assets
    "USD":      "DTWEXBGS",   # Broad USD index
}


# --- Helpers -----------------------------------------------------------------

def get_fred(api_key: str | None = None) -> Fred:
    """Create a FRED client, using env var if api_key is not provided."""
    if api_key is None:
        api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise ValueError(
            "FRED API key not found. "
            "Set FRED_API_KEY env var or pass fred_api_key to run_full_pipeline()."
        )

    return Fred(api_key=api_key)


def fetch_etf_prices(start_date: str = "1995-01-01"):
    """Download daily ETF prices and convert to monthly returns."""
    data = yf.download(
        ETF_TICKERS,
        start=start_date,
        auto_adjust=True,
        progress=False,
    )

    # MultiIndex columns: first level = field, second = ticker
    prices_daily = data["Close"]
    monthly_prices = prices_daily.resample("ME").last()
    monthly_rets = monthly_prices.pct_change().dropna()

    return monthly_prices, monthly_rets


def fetch_macro_monthly(fred: Fred, start_date: str = "1995-01-01") -> pd.DataFrame:
    """Download macro series from FRED and resample to month-end."""
    macro_raw = pd.DataFrame()

    for col_name, fred_id in FRED_SERIES.items():
        s = fred.get_series(fred_id, observation_start=start_date)
        s.name = col_name
        macro_raw = pd.concat([macro_raw, s], axis=1)

    macro_raw.index = pd.to_datetime(macro_raw.index)
    macro_monthly = macro_raw.resample("ME").last()

    return macro_monthly


def build_macro_factors(macro_monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct your macro factor set from raw monthly macro data."""
    macro = macro_monthly
    mf = pd.DataFrame(index=macro.index)

    # Growth
    mf["IP_YOY"] = macro["INDPRO"].pct_change(12, fill_method=None) * 100.0
    mf["IP_YOY_Δ6m"] = mf["IP_YOY"] - mf["IP_YOY"].shift(6)

    # Labour
    mf["UNRATE_GAP"] = macro["UNRATE"] - macro["UNRATE"].rolling(60).mean()

    # Inflation
    mf["CPI_YOY"] = macro["CPI"].pct_change(12, fill_method=None) * 100.0
    mf["CPI_YOY_Δ6m"] = mf["CPI_YOY"] - mf["CPI_YOY"].shift(6)

    # Policy stance
    mf["REAL_FF"] = macro["FEDFUNDS"] - mf["CPI_YOY"]

    # Yield curve
    t10y2y_roll_mean = macro["T10Y2Y"].rolling(120).mean()
    t10y2y_roll_std = macro["T10Y2Y"].rolling(120).std()
    mf["T10Y2Y_Z"] = (macro["T10Y2Y"] - t10y2y_roll_mean) / t10y2y_roll_std

    # Risk appetite
    vix_roll_mean = macro["VIX"].rolling(24).mean()
    vix_roll_std = macro["VIX"].rolling(24).std()
    mf["VIX_Z"] = (macro["VIX"] - vix_roll_mean) / vix_roll_std

    baa_roll_mean = macro["BAA"].rolling(120).mean()
    baa_roll_std = macro["BAA"].rolling(120).std()
    mf["BAA_Z"] = (macro["BAA"] - baa_roll_mean) / baa_roll_std

    # Liquidity + USD
    mf["M2_YOY"] = macro["M2"].pct_change(12, fill_method=None) * 100.0
    mf["FED_ASSETS_YOY"] = macro["WALCL"].pct_change(12, fill_method=None) * 100.0
    mf["USD_YOY"] = macro["USD"].pct_change(12, fill_method=None) * 100.0

    mf_clean = mf.dropna()
    return mf, mf_clean


def align_for_model(
    monthly_rets: pd.DataFrame,
    macro_monthly: pd.DataFrame,
    macro_factors_clean: pd.DataFrame,
):
    """Align ETF returns, macro data, and factors on a common index."""
    common_index = monthly_rets.index.intersection(macro_monthly.index)
    monthly_rets_aligned = monthly_rets.reindex(common_index)
    macro_monthly_aligned = macro_monthly.reindex(common_index)

    common_index_model = monthly_rets_aligned.index.intersection(macro_factors_clean.index)
    macro_factors_for_model = macro_factors_clean.reindex(common_index_model)
    monthly_rets_for_model = monthly_rets_aligned.reindex(common_index_model)

    return monthly_rets_aligned, macro_monthly_aligned, macro_factors_for_model, monthly_rets_for_model


def build_X_Y(macro_factors_for_model: pd.DataFrame,
              monthly_rets_for_model: pd.DataFrame):
    """Lag factors by 1 month for X, and align Y = ETF returns."""
    X = macro_factors_for_model.shift(1).dropna()
    Y = monthly_rets_for_model.reindex(X.index)
    return X, Y


def compute_excess_returns(Y: pd.DataFrame,
                           macro_monthly_aligned: pd.DataFrame) -> pd.DataFrame:
    """Subtract risk-free rate (Fed Funds) from ETF returns."""
    fedfunds_annual_pct = macro_monthly_aligned["FEDFUNDS"].reindex(Y.index)
    rf_annual = fedfunds_annual_pct / 100.0
    rf_monthly = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0

    Y_excess = Y.sub(rf_monthly, axis=0)
    return Y_excess


def compute_relative_returns(Y_excess: pd.DataFrame):
    """Relative sector returns vs SPY."""
    sector_cols = [c for c in Y_excess.columns if c != "SPY"]
    Y_rel = Y_excess[sector_cols].sub(Y_excess["SPY"], axis=0)
    return Y_rel, sector_cols


def rolling_ols_forecasts(
    X: pd.DataFrame,
    Y_excess: pd.DataFrame,
    window: int = 84,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling OLS: for each month t, fit on t-window...t-1, forecast t.
    Returns Y_hat (forecasts) and Y_actual on the OOS sample.
    """
    dates = X.index
    Y_pred = pd.DataFrame(index=dates, columns=Y_excess.columns, dtype=float)

    for t_idx in range(window, len(dates)):
        t = dates[t_idx]
        train_idx = dates[t_idx - window : t_idx]

        X_train = X.loc[train_idx]
        Y_train = Y_excess.loc[train_idx]

        mu = X_train.mean()
        sigma = X_train.std(ddof=0).replace(0.0, 1.0)

        X_train_std = (X_train - mu) / sigma
        x_t_std = (X.loc[t] - mu) / sigma
        x_t_vec = np.concatenate(([1.0], x_t_std.values))

        X_design = np.column_stack([np.ones(len(X_train_std)), X_train_std.values])

        for col in Y_excess.columns:
            y_train = Y_train[col].values
            beta, *_ = np.linalg.lstsq(X_design, y_train, rcond=None)
            Y_pred.loc[t, col] = float(x_t_vec @ beta)

    Y_pred_oos = Y_pred.dropna(how="any")
    valid_idx = Y_pred_oos.index

    Y_hat = Y_pred_oos
    Y_actual = Y_excess.reindex(valid_idx)

    return Y_hat, Y_actual


def compute_fit_metrics(Y_actual: pd.DataFrame,
                        Y_hat: pd.DataFrame) -> pd.DataFrame:
    """R² and corr(y, yhat) by ETF."""
    metrics = []
    for col in Y_actual.columns:
        y = Y_actual[col].values
        yhat = Y_hat[col].values

        resid = y - yhat
        sse = np.sum(resid ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        corr = np.corrcoef(y, yhat)[0, 1]

        metrics.append({
            "ETF": col,
            "R2": r2,
            "Corr(y, yhat)": corr,
            "Mean_actual": y.mean(),
            "Mean_forecast": yhat.mean(),
            "Std_actual": y.std(ddof=0),
            "Std_forecast": yhat.std(ddof=0),
        })

    return pd.DataFrame(metrics).set_index("ETF")


def build_long_short_portfolio(
    Y_hat: pd.DataFrame,
    Y_actual: pd.DataFrame,
    top_n: int = 3,
    bottom_n: int = 3,
) -> pd.Series:
    """Cross-sectional long/short on forecasts across ETFs."""
    valid_idx = Y_hat.index
    port_rets = []

    for t in valid_idx:
        yhat_t = Y_hat.loc[t]
        y_t = Y_actual.loc[t]

        valid_cols = yhat_t.dropna().index.intersection(y_t.dropna().index)
        yhat_t = yhat_t[valid_cols]
        y_t = y_t[valid_cols]

        ranked = yhat_t.sort_values(ascending=False)
        longs = ranked.index[:top_n]
        shorts = ranked.index[-bottom_n:]

        w = pd.Series(0.0, index=valid_cols)
        if top_n > 0:
            w[longs] = 1.0 / top_n
        if bottom_n > 0:
            w[shorts] = -1.0 / bottom_n

        port_rets.append(float((w * y_t).sum()))

    return pd.Series(port_rets, index=valid_idx, name="LS_portfolio")


def build_ew_benchmark(Y_actual: pd.DataFrame, index=None) -> pd.Series:
    """Equal-weight long-only benchmark across all ETFs."""
    if index is not None:
        Y_actual = Y_actual.reindex(index)
    n_etfs = Y_actual.shape[1]
    w = np.ones(n_etfs) / n_etfs
    bench = pd.Series(Y_actual.values @ w, index=Y_actual.index, name="EW_long")
    return bench


def build_spy_timing(Y_hat: pd.DataFrame,
                     Y_actual: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Timing strategy: long SPY when forecast > 0, else cash."""
    spy_forecast = Y_hat["SPY"]
    spy_actual = Y_actual["SPY"]
    positions = (spy_forecast > 0).astype(float)
    timing_rets = positions * spy_actual
    timing_rets.name = "SPY_timing"
    return timing_rets, spy_actual


def performance_stats(series: pd.Series) -> Dict[str, float]:
    """CAGR, vol, Sharpe, max drawdown."""
    s = series.dropna()
    if len(s) == 0:
        return {k: np.nan for k in ["Total Return", "CAGR", "AnnVol", "Sharpe", "MaxDD"]}

    cum = (1 + s).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_months = len(s)
    cagr = (1 + total_ret) ** (12 / n_months) - 1
    ann_vol = s.std(ddof=0) * np.sqrt(12)
    sharpe = cagr / ann_vol if ann_vol > 0 else np.nan
    max_dd = (cum / cum.cummax() - 1).min()

    return {
        "Total Return": total_ret,
        "CAGR": cagr,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
    }


def summarize_strategies(
    ls_rets: pd.Series,
    bench_rets: pd.Series,
    spy_timing_rets: pd.Series,
    spy_excess: pd.Series,
) -> pd.DataFrame:
    stats = {
        "Macro LS portfolio": performance_stats(ls_rets),
        "EW long benchmark": performance_stats(bench_rets),
        "SPY timing": performance_stats(spy_timing_rets),
        "SPY buy & hold": performance_stats(spy_excess),
    }
    return pd.DataFrame(stats).T


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)


def compute_rolling_betas(
    X: pd.DataFrame,
    Y_rel: pd.DataFrame,
    factor_name: str = "REAL_FF",
    window: int = 84,
    sectors: List[str] | None = None,
) -> pd.DataFrame:
    """Rolling beta of sector relative returns to a chosen macro factor."""
    if sectors is None:
        sectors = list(Y_rel.columns)

    dates = X.index
    beta_df = pd.DataFrame(index=dates[window:], columns=sectors, dtype=float)

    for i in range(window, len(dates)):
        train_idx = dates[i - window : i]
        t = dates[i]

        X_train = X.loc[train_idx]
        Y_train_rel = Y_rel.loc[train_idx, sectors]

        X_mean = X_train.mean()
        X_std = X_train.std(ddof=0).replace(0.0, 1.0)
        X_train_std = (X_train - X_mean) / X_std

        cols = list(X_train_std.columns)
        if factor_name not in cols:
            raise ValueError(f"{factor_name} not found in factor columns.")
        factor_idx = cols.index(factor_name)

        X_design = np.column_stack([np.ones(len(X_train_std)), X_train_std.values])

        for sec in sectors:
            y = Y_train_rel[sec].values
            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            beta_factor = beta[1:][factor_idx]
            beta_df.at[t, sec] = beta_factor

    return beta_df


# --- Main orchestrator -------------------------------------------------------

def run_full_pipeline(
    fred_api_key: str | None = None,
    start_date: str = "1995-01-01",
    window: int = 84,
) -> Dict[str, Any]:
    """Run the full pipeline and return a dict of DataFrames for the app."""
    fred = get_fred(fred_api_key)

    monthly_prices, monthly_rets = fetch_etf_prices(start_date=start_date)
    macro_monthly = fetch_macro_monthly(fred, start_date=start_date)
    macro_factors, macro_factors_clean = build_macro_factors(macro_monthly)

    (
        monthly_rets_aligned,
        macro_monthly_aligned,
        macro_factors_for_model,
        monthly_rets_for_model,
    ) = align_for_model(monthly_rets, macro_monthly, macro_factors_clean)

    X, Y = build_X_Y(macro_factors_for_model, monthly_rets_for_model)
    Y_excess = compute_excess_returns(Y, macro_monthly_aligned)
    Y_rel, sector_tickers = compute_relative_returns(Y_excess)

    Y_hat, Y_actual = rolling_ols_forecasts(X, Y_excess, window=window)
    metrics_df = compute_fit_metrics(Y_actual, Y_hat)

    ls_portfolio = build_long_short_portfolio(Y_hat, Y_actual, top_n=3, bottom_n=3)
    bench_rets = build_ew_benchmark(Y_actual, index=ls_portfolio.index)
    spy_timing_rets, spy_excess = build_spy_timing(Y_hat, Y_actual)

    perf_table = summarize_strategies(ls_portfolio, bench_rets, spy_timing_rets, spy_excess)

    equity_curves = pd.DataFrame({
        "Macro LS portfolio": (1 + ls_portfolio).cumprod(),
        "EW long benchmark": (1 + bench_rets).cumprod(),
        "SPY timing": (1 + spy_timing_rets).cumprod(),
        "SPY buy & hold": (1 + spy_excess).cumprod(),
    })

    macro_for_plot = macro_factors_clean.reindex(X.index)
    growth_cols = ["IP_YOY", "CPI_YOY"]
    fin_cols = ["T10Y2Y_Z", "REAL_FF"]
    growth_z = zscore(macro_for_plot[growth_cols])
    fin_z = zscore(macro_for_plot[fin_cols])

    rolling_betas = compute_rolling_betas(
        X, Y_rel, factor_name="REAL_FF", window=window, sectors=sector_tickers
    )

    return {
        "etf_tickers": ETF_TICKERS,
        "sector_tickers": sector_tickers,
        "monthly_prices": monthly_prices,
        "monthly_rets": monthly_rets,
        "macro_monthly": macro_monthly,
        "macro_factors": macro_factors,
        "macro_factors_clean": macro_factors_clean,
        "X": X,
        "Y_excess": Y_excess,
        "Y_hat": Y_hat,
        "Y_actual": Y_actual,
        "metrics_df": metrics_df,
        "ls_portfolio": ls_portfolio,
        "bench_rets": bench_rets,
        "spy_timing_rets": spy_timing_rets,
        "spy_buyhold_rets": spy_excess,
        "perf_table": perf_table,
        "equity_curves": equity_curves,
        "growth_z": growth_z,
        "fin_z": fin_z,
        "rolling_betas": rolling_betas,
    }
