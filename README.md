# Macro Factor Model for US Sector Rotation

This repository contains a complete, end-to-end research pipeline for a **macro-driven equity allocation model**. It uses:

- US macroeconomic data from **FRED**
- Monthly returns of **SPY** and **US sector ETFs**
- Rolling **OLS factor models** estimated on macro factors
- Out-of-sample backtests for:
  - A **SPY timing strategy** (risk-on vs cash)
  - A **macro-guided long/short sector rotation strategy (V2)**

The project is framed as a **research notebook**, not as production trading code. The emphasis is on transparent design choices, avoiding look-ahead bias, and understanding why the model behaves the way it does.

---

## 1. Project Overview

The core question:

> Can a small set of macroeconomic indicators systematically improve equity allocation decisions?

To answer this, the project builds a macro factor model that links US macro data to equity returns, then uses that model to construct and backtest simple systematic strategies.

Key elements:

- **Universe**:  
  - SPY (broad US equity market)  
  - 10 US sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY

- **Macro inputs** (from FRED, resampled to month-end):
  - Growth: Industrial Production (INDPRO)
  - Labour: Unemployment Rate (UNRATE)
  - Inflation: CPI (CPIAUCSL)
  - Policy: Fed Funds (FEDFUNDS)
  - Yield curve: 10Y–2Y spread (T10Y2Y)
  - Risk sentiment: VIX (VIXCLS), Baa yield (BAA)
  - Liquidity / USD: M2 (M2SL), Fed balance sheet (WALCL), USD broad index (DTWEXBGS)

- **Factors**:
  - YoY growth and inflation (`IP_YOY`, `CPI_YOY`)
  - 6-month changes in those YoY rates (acceleration signals)
  - Unemployment gap vs 5-year average
  - 10-year z-scores of yield curve and credit spread
  - 2-year z-score of VIX
  - Real policy rate (`REAL_FF = FEDFUNDS – CPI_YOY`)
  - Liquidity and USD YoY changes

- **Models**:
  - Rolling 84-month (7-year) OLS regressions on **lagged** macro factors
  - Excess returns (over Fed Funds) as the base return measure
  - **SPY** model: absolute excess return vs macro
  - **Sector model**: relative excess return vs SPY vs macro

- **Strategies**:
  1. **SPY timing** – long SPY when forecast excess return > 0, otherwise cash  
  2. **Sector L/S (V2)** – macro-driven cross-sectional sector rotation:
     - Forecast sector **relative returns vs SPY**
     - Long top 3 sectors, short bottom 3 (conditional on trend)
     - Trend filter: only short sectors with `P_t < MA_10(P)`
     - Inverse-volatility weighting for both long and short baskets
     - Dollar-neutral portfolio
