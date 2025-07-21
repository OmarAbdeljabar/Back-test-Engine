# HamayelResearchEngine - Equity Strategy Backtesting Platform

This Python script implements a comprehensive equity strategy backtesting platform using Streamlit. It allows users to define multi-factor models, configure universe selection rules, and run historical backtests to evaluate strategy performance against a benchmark (SPY).

## Key Features:

* **Streamlit Web Interface:** User-friendly UI for configuring backtest parameters and viewing results with "HamayelResearchEngine" branding.
* **Data Handling:** Loads financial data (prices, fundamentals, metadata, corporate actions) from Parquet files.
* **Advanced Universe Selection:**
    * Filters by issuer category and average daily dollar volume.
    * Supports multi-stage technical (EMA/SMA criteria) and fundamental filters.
    * Fundamental filters can use various SF1 dimensions (ARQ, MRY, ART, etc.) and evaluate actual values or growth rates (YoY, QoQ).
    * Allows AND/OR logic for combining filter criteria.
* **Multi-Factor Modeling:**
    * Supports fundamental factors, momentum (with optional skip-period) and derived ratios (e.g., `gp/assets`).
    * Users can define factor weights, preferred direction (higher/lower is better), and calculation method (actual value, YoY growth, QoQ growth) for each factor.
    * Computes Z-scores and composite scores for stock ranking.
* **Portfolio Construction:**
    * Selects stocks based on composite factor scores (top N or top X% of universe).
    * Handles scenarios where the filtered universe is smaller than the desired portfolio size.
    * Options for equal or market-cap weighting.
    * Configurable rebalance frequency (e.g., monthly, weekly).
* **Backtesting Engine:**
    * Simulates portfolio performance over the selected period.
    * Handles corporate actions (splits, delistings, spinoffs, ticker changes).
    * Accounts for user-defined slippage costs.
* **Performance Analysis & Reporting:**
    * Calculates a suite of performance metrics: Annualized Return, Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio, Alpha, and Beta against SPY.
    * Displays interactive charts (cumulative returns, drawdowns) in the UI.
    * Generates a detailed PDF report summarizing the backtest, including metrics, charts, holdings log, and corporate actions log.
    * Outputs CSV logs for portfolio holdings and processed corporate actions.
    * Provides a downloadable text summary of the strategy and key results.

## Data Requirements:

The application expects pre-processed financial data in Parquet format located in a `data/parquet/` subdirectory relative to the script. Essential files include:
* `sf1.parquet`: Core US fundamentals data.
* `sep.parquet`: Stock EOD prices and volumes.
* `sfp.parquet`: SPY EOD prices and volumes (for benchmark).
* `tickers.parquet`: Ticker metadata including categories and price dates.
* `actions.parquet`: Corporate actions data.
* `INDICATORS.csv` (in the script's directory): For mapping SF1 indicator codes to human-readable titles.

## How to Run:

1.  Ensure all Python dependencies are installed (e.g., Streamlit, Pandas, NumPy, PyArrow, Matplotlib, ReportLab, SciPy).
2.  Create the `data/parquet/` directory and populate it with the required Parquet files.
3.  Place `INDICATORS.csv` in the same directory as the script.
4.  Execute the Streamlit application from your terminal:
    ```bash
    streamlit run your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of this file).
5.  Configure the backtest parameters in the sidebar of the web application and click "Execute Backtest".

## Outputs:

* Interactive backtest results displayed in the Streamlit application.
* Downloadable PDF backtest report (e.g., `backtest_report_<uuid>.pdf`).
* Downloadable CSV logs for portfolio holdings (e.g., `holdings_log_<uuid>.csv`) and processed corporate actions (e.g., `actions_log_<uuid>.csv`).
* Downloadable text summary of the strategy and key results.
* All output files are saved to the `data/backtest_output/` directory.

*This project is licensed under the MIT License 
