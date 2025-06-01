import streamlit as st
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
# Set matplotlib to use dark theme for professional charts
plt.style.use('dark_background')
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import scipy.stats as stats
import os
from pathlib import Path
import uuid

# ─── Configuration ───────────────────────────────────────────────────────────────
DATA_FOLDER = "data"  # Path to your data folder containing CSV/Parquet files
PARQUET_FOLDER = os.path.join(DATA_FOLDER, "parquet")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "backtest_output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PARQUET_FOLDER, exist_ok=True) # Ensure parquet folder exists

# Risk-free rate for Sharpe/Sortino (annualized, 2%)
RISK_FREE_RATE = 0.02

# Data cushion for loading (1 year before and after)
DATA_CUSHION_DAYS = 365

# SF1 dimension options
DIMENSION_OPTIONS = {
    'ARQ': 'Quarterly (As Reported)',
    'MRQ': 'Quarterly (With Restatements)',
    'ARY': 'Annual (As Reported)',
    'MRY': 'Annual (With Restatements)',
    'ART': 'Trailing Twelve Months (As Reported)',
    'MRT': 'Trailing Twelve Months (With Restatements)'
}

# Create formatted dimension options for display in dropdowns
FORMATTED_DIMENSION_OPTIONS = [f"{code} - {desc}" for code, desc in DIMENSION_OPTIONS.items()]

# Function to extract dimension code from formatted option
def extract_dimension_code(formatted_option):
    return formatted_option.split(' - ')[0]

# ─── Helper Functions ───────────────────────────────────────────────────────────
@st.cache_data
def load_parquet(file_name, start_date=None, end_date=None):
    """Load Parquet file with optional date filtering."""
    file_path = os.path.join(PARQUET_FOLDER, file_name)
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}. Please ensure your Parquet files are in the correct directory.")
        return pd.DataFrame()

    df = pq.read_table(file_path).to_pandas()
    # Ensure date columns are datetime type if they exist and are used for filtering
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'calendardate' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['calendardate']):
        df['calendardate'] = pd.to_datetime(df['calendardate'], errors='coerce')

    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if 'date' in df.columns:
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        elif 'calendardate' in df.columns: # Fallback if 'date' not present but 'calendardate' is
            df = df[(df['calendardate'] >= start_date) & (df['calendardate'] <= end_date)]
    return df

def get_trading_days(df_prices):
    """Get sorted unique trading days."""
    if 'date' not in df_prices.columns or df_prices.empty:
        return []
    return sorted(pd.to_datetime(df_prices['date'].dropna().unique()))

def get_signal_execution_dates(trading_days, frequency='monthly'):
    """Generate signal and execution dates."""
    if not trading_days:
        return pd.Series(dtype='datetime64[ns]'), pd.Series(dtype='datetime64[ns]')

    trading_days_dt = pd.Series(pd.to_datetime(trading_days)).sort_values().unique() # Ensure sorted unique datetimes
    if len(trading_days_dt) == 0:
        return pd.Series(dtype='datetime64[ns]'), pd.Series(dtype='datetime64[ns]')

    signal_dates = []
    execution_dates = []
    df_schedule = pd.DataFrame({'date': trading_days_dt})

    if frequency == 'monthly':
        df_schedule['period_group'] = df_schedule['date'].dt.to_period('M')
    elif frequency == 'weekly':
        df_schedule['period_group'] = df_schedule['date'].dt.to_period('W-FRI') # End of week on Friday
    else: # daily or other unsupported
        st.warning(f"Unsupported rebalance frequency: {frequency}. Defaulting to monthly.")
        df_schedule['period_group'] = df_schedule['date'].dt.to_period('M')


    grouped = df_schedule.groupby('period_group')
    for _, group in grouped:
        current_signal_date = group['date'].iloc[-1]
        signal_dates.append(current_signal_date)
        # Find first trading day STRICTLY AFTER the signal date
        next_day_after_signal_mask = trading_days_dt > current_signal_date
        if next_day_after_signal_mask.any():
            execution_dates.append(trading_days_dt[next_day_after_signal_mask][0])
        else:
            # If no trading day after signal (e.g., end of data), use signal date itself.
            # This might imply trading on close of signal day or that the period cannot be fully executed.
            execution_dates.append(current_signal_date)


    final_signal_dates = []
    final_execution_dates = []
    # Ensure execution date is not before signal date and align lengths
    min_len = min(len(signal_dates), len(execution_dates))
    for i in range(min_len):
        s_date, e_date = signal_dates[i], execution_dates[i]
        if e_date >= s_date: # Standard: execute on or after signal
            final_signal_dates.append(s_date)
            final_execution_dates.append(e_date)

    return pd.Series(final_signal_dates, dtype='datetime64[ns]'), pd.Series(final_execution_dates, dtype='datetime64[ns]')


def calculate_momentum(prices, signal_date, lookback_weeks, skip_4week=False):
    """Calculate momentum."""
    if prices.empty or 'date' not in prices.columns or 'ticker' not in prices.columns or 'closeadj' not in prices.columns:
        return pd.Series(dtype=float)

    signal_date = pd.to_datetime(signal_date)
    # prices['date'] should be datetime from load_parquet

    # Filter relevant data once
    relevant_prices = prices[prices['date'] <= signal_date].copy() # Use .copy()
    if relevant_prices.empty:
        return pd.Series(np.nan, index=prices['ticker'].unique())


    trading_days_before_signal = pd.Series(relevant_prices['date'].unique()).sort_values(ascending=False).reset_index(drop=True)

    if len(trading_days_before_signal) < (lookback_weeks * 5) : # Need enough days for lookback
        return pd.Series(np.nan, index=relevant_prices['ticker'].unique())

    # Determine end_date (which is the signal_date itself if it's a trading day)
    actual_end_date = signal_date
    if signal_date not in trading_days_before_signal.values: # If signal_date is not a trading day, use the closest one before it
        closest_trading_day_before = trading_days_before_signal[trading_days_before_signal <= signal_date]
        if closest_trading_day_before.empty: return pd.Series(np.nan, index=relevant_prices['ticker'].unique())
        actual_end_date = closest_trading_day_before.iloc[0]


    start_date_idx = min(len(trading_days_before_signal) - 1, lookback_weeks * 5 -1) # -1 for 0-based, ensure index valid
    if start_date_idx < 0 : return pd.Series(np.nan, index=relevant_prices['ticker'].unique())
    actual_start_date = trading_days_before_signal.iloc[start_date_idx]


    end_prices_df = relevant_prices[relevant_prices['date'] == actual_end_date][['ticker', 'closeadj']].set_index('ticker')
    start_prices_df = relevant_prices[relevant_prices['date'] == actual_start_date][['ticker', 'closeadj']].set_index('ticker')

    if end_prices_df.empty or start_prices_df.empty:
        return pd.Series(np.nan, index=relevant_prices['ticker'].unique())

    merged = end_prices_df.join(start_prices_df, lsuffix='_end', rsuffix='_start', how='inner')
    if merged.empty or 'closeadj_end' not in merged.columns or 'closeadj_start' not in merged.columns:
        return pd.Series(np.nan, index=relevant_prices['ticker'].unique())

    # Avoid division by zero or NaN in denominator
    valid_start_prices = merged['closeadj_start'].replace(0, np.nan).dropna()
    if valid_start_prices.empty:
        return pd.Series(np.nan, index=merged.index)

    momentum_val = (merged['closeadj_end'].loc[valid_start_prices.index] / valid_start_prices) - 1
    momentum = pd.Series(momentum_val, name='momentum').reindex(relevant_prices['ticker'].unique())


    if skip_4week:
        if len(trading_days_before_signal) < (4 * 5) :
             st.warning(f"Momentum: Not enough data for 4-week skip on {signal_date}. Using full lookback momentum.")
             return momentum.replace([np.inf, -np.inf], np.nan)

        short_lookback_days = 4 * 5
        short_start_date_idx = min(len(trading_days_before_signal) - 1, short_lookback_days - 1)
        if short_start_date_idx < 0: return momentum.replace([np.inf, -np.inf], np.nan)
        actual_short_start_date = trading_days_before_signal.iloc[short_start_date_idx]

        short_start_prices_df = relevant_prices[relevant_prices['date'] == actual_short_start_date][['ticker', 'closeadj']].set_index('ticker')
        if short_start_prices_df.empty: return momentum.replace([np.inf, -np.inf], np.nan)

        merged_short = end_prices_df.join(short_start_prices_df.rename(columns={'closeadj':'closeadj_short_start'}), how='inner')
        if merged_short.empty or 'closeadj' not in merged_short.columns or 'closeadj_short_start' not in merged_short.columns:
            return momentum.replace([np.inf, -np.inf], np.nan)

        valid_short_start_prices = merged_short['closeadj_short_start'].replace(0, np.nan).dropna()
        if valid_short_start_prices.empty: return momentum.replace([np.inf, -np.inf], np.nan)

        short_momentum_val = (merged_short['closeadj'].loc[valid_short_start_prices.index] / valid_short_start_prices) - 1
        short_momentum = pd.Series(short_momentum_val, name='short_momentum').reindex(momentum.index) # Align with main momentum series

        momentum = momentum.subtract(short_momentum, fill_value=np.nan) # Use NaN for fill if one is missing

    return momentum.replace([np.inf, -np.inf], np.nan)


def calculate_dolvol(prices, signal_date, days=30):
    """Calculate X-day dollar volume."""
    if prices.empty or 'date' not in prices.columns or 'ticker' not in prices.columns:
        return pd.Series(dtype=float)

    signal_date = pd.to_datetime(signal_date)
    # prices['date'] should be datetime from load_parquet

    relevant_prices = prices[prices['date'] <= signal_date].copy() # Explicit copy
    if relevant_prices.empty: return pd.Series(dtype=float)

    trading_days_before_signal = pd.Series(relevant_prices['date'].unique()).sort_values(ascending=False).reset_index(drop=True)

    if len(trading_days_before_signal) < days:
        return pd.Series(dtype=float) # Not enough historical data

    start_date_idx = min(len(trading_days_before_signal) - 1, days - 1)
    if start_date_idx < 0: return pd.Series(dtype=float) # Should not happen if len check passed
    actual_start_date = trading_days_before_signal.iloc[start_date_idx]

    # Determine actual end date for the window (signal_date or closest trading day before)
    actual_end_date = signal_date
    if signal_date not in trading_days_before_signal.values:
        closest_trading_day_before = trading_days_before_signal[trading_days_before_signal <= signal_date]
        if closest_trading_day_before.empty: return pd.Series(dtype=float)
        actual_end_date = closest_trading_day_before.iloc[0]

    window_df = relevant_prices[(relevant_prices['date'] >= actual_start_date) & (relevant_prices['date'] <= actual_end_date)].copy()

    if window_df.empty: return pd.Series(dtype=float)

    window_df.loc[:, 'volume'] = pd.to_numeric(window_df['volume'], errors='coerce')
    window_df.loc[:, 'closeadj'] = pd.to_numeric(window_df['closeadj'], errors='coerce')
    window_df.dropna(subset=['volume', 'closeadj'], inplace=True)

    if window_df.empty: return pd.Series(dtype=float)

    window_df.loc[:, 'dollar_volume_daily'] = window_df['volume'] * window_df['closeadj']
    # Ensure groupby operation is on the correct data type and handle potential empty groups
    if 'ticker' in window_df.columns and not window_df.empty:
        dolvol = window_df.groupby('ticker')['dollar_volume_daily'].mean()
    else:
        dolvol = pd.Series(dtype=float)


    return dolvol.replace([np.inf, -np.inf], np.nan).reindex(prices['ticker'].unique()) # Reindex to original universe for consistency


def compute_zscores(series, direction='higher'):
    """Compute z-scores with directionality."""
    if not isinstance(series, pd.Series): series = pd.Series(series) # Ensure it's a Series
    valid_series = series.dropna().astype(float)
    if len(valid_series) < 2:
        return pd.Series(np.nan, index=series.index)

    # ddof=1 for sample standard deviation, which is common.
    # If all values are the same, std will be 0.
    std_dev = valid_series.std(ddof=1)
    if std_dev == 0 or pd.isna(std_dev): # Handle all same values or all NaNs after dropna
        # If all values are same, z-score is 0. If series was all NaN, result is all NaN.
        return pd.Series(0.0 if std_dev == 0 else np.nan, index=series.index)


    z = (valid_series - valid_series.mean()) / std_dev
    z_series = pd.Series(z, index=valid_series.index).reindex(series.index)
    return z_series if direction == 'higher' else -z_series


def get_action_price(prices_df, ticker, action_date, is_delisting=False, is_voluntary_or_ma=False):
    """Get closeadj price for corporate actions."""
    if prices_df.empty or 'date' not in prices_df.columns:
        return 0.0 if is_delisting and not is_voluntary_or_ma else None

    action_date = pd.to_datetime(action_date)
    # prices_df['date'] should be datetime from load_parquet

    if is_delisting and not is_voluntary_or_ma:
        return 0.0 # Assume 0 value for forced delistings/bankruptcy

    # Look for price on action_date or up to 2 days before
    window_prices = prices_df[
        (prices_df['ticker'] == ticker) &
        (prices_df['date'] <= action_date) &
        (prices_df['date'] >= action_date - timedelta(days=2)) # Check a couple of days prior
    ].copy() # Use .copy()

    if not window_prices.empty:
        window_prices.loc[:, 'closeadj'] = pd.to_numeric(window_prices['closeadj'], errors='coerce')
        price_val = window_prices.sort_values('date', ascending=False).iloc[0]['closeadj']
        return price_val if pd.notna(price_val) else None # Return NaN if conversion failed

    # Fallback for voluntary/M&A: look further back if needed
    if is_voluntary_or_ma:
        window_prices_extended = prices_df[
            (prices_df['ticker'] == ticker) &
            (prices_df['date'] <= action_date) &
            (prices_df['date'] >= action_date - timedelta(days=7)) # Wider window
        ].copy()
        if not window_prices_extended.empty:
            window_prices_extended.loc[:, 'closeadj'] = pd.to_numeric(window_prices_extended['closeadj'], errors='coerce')
            price_val = window_prices_extended.sort_values('date', ascending=False).iloc[0]['closeadj']
            return price_val if pd.notna(price_val) else None
        return None # Price not found even with extended window for voluntary/M&A

    # Default for other non-voluntary delistings if no price found earlier (should have been caught by is_delisting)
    return 0.0


def performance_metrics(strategy_returns, benchmark_returns):
    """Calculate backtest metrics for strategy and benchmark."""
    metrics_strategy = {}
    metrics_benchmark = {}
    
    # DEBUG: Log the data being passed to this function
    print(f"\n===== PERFORMANCE METRICS DEBUG =====")
    print(f"Strategy returns data:")
    print(f"  Length: {len(strategy_returns)}")
    if not strategy_returns.empty:
        print(f"  Start date: {strategy_returns.index.min()}")
        print(f"  End date: {strategy_returns.index.max()}")
        print(f"  Time span: {(strategy_returns.index.max() - strategy_returns.index.min()).days} days")
        total_ret_factor = (1 + strategy_returns).cumprod().iloc[-1] if len(strategy_returns) > 0 else 1.0
        time_span_years = (strategy_returns.index.max() - strategy_returns.index.min()).days / 365.25
        expected_ann = (total_ret_factor ** (1 / time_span_years) - 1) * 100 if time_span_years > 0 else 0
        print(f"  Expected annualized return: {expected_ann:.2f}%")
        print(f"  First 5 returns: {strategy_returns.head().tolist()}")
        print(f"  Last 5 returns: {strategy_returns.tail().tolist()}")
    else:
        print(f"  Strategy returns is EMPTY!")
    print(f"=====================================\n")
    
    # Determine annualization factor based on data frequency
    # Check the typical time difference between observations
    if not strategy_returns.empty and len(strategy_returns) > 1:
        time_diffs = strategy_returns.index.to_series().diff().dropna()
        avg_days = time_diffs.dt.days.mean()
        
        if avg_days >= 25 and avg_days <= 35:  # Monthly data (approximately 30 days)
            annual_factor = 12
        elif avg_days >= 5 and avg_days <= 10:  # Weekly data (approximately 7 days)
            annual_factor = 52
        else:  # Daily data or other
            annual_factor = 252
    else:
        annual_factor = 252  # Default to daily

    def calculate_core_metrics(returns_series, annual_factor, risk_free_rate_annual):
        metrics = {}
        empty_index = pd.DatetimeIndex([pd.Timestamp('1900-01-01')])
        cum_returns_with_initial = pd.Series([1.0], index=empty_index)
        drawdowns = pd.Series([0.0], index=empty_index)

        if not isinstance(returns_series, pd.Series) or returns_series.empty or returns_series.isnull().all():
            for k in ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown', 'Calmar Ratio']:
                metrics[k] = np.nan
            return metrics, cum_returns_with_initial, drawdowns

        # Ensure returns are float and drop NaNs that might have been introduced
        returns_series_clean = returns_series.astype(float).dropna()
        if returns_series_clean.empty: # If all were NaN
            for k in ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown', 'Calmar Ratio']:
                metrics[k] = np.nan
            return metrics, cum_returns_with_initial, drawdowns        # Cumulative returns
        initial_idx = returns_series_clean.index.min() - pd.Timedelta(days=1)
        cum_returns = (1 + returns_series_clean).cumprod()
        cum_returns_with_initial = pd.concat([pd.Series([1.0], index=[initial_idx]), cum_returns]).sort_index()

        n_periods = len(returns_series_clean)        # Annualized Return - Use actual time span instead of number of periods
        total_return_factor = cum_returns.iloc[-1] if not cum_returns.empty else 1.0
        if n_periods > 0 and len(returns_series_clean) > 1:
            # Calculate actual time span in years
            time_span_days = (returns_series_clean.index.max() - returns_series_clean.index.min()).days
            time_span_years = max(time_span_days / 365.25, 1/365.25)  # Minimum 1 day to avoid division by zero
            if time_span_years > 0:
                metrics['Annualized Return'] = (total_return_factor ** (1 / time_span_years) - 1) * 100
            else:
                metrics['Annualized Return'] = 0.0
        else:
            metrics['Annualized Return'] = 0.0        # Annualized Volatility - Fix calculation
        if n_periods > 1:
            volatility_decimal = returns_series_clean.std(ddof=1)  # Standard deviation of returns
            metrics['Annualized Volatility'] = volatility_decimal * np.sqrt(annual_factor) * 100
        else:
            metrics['Annualized Volatility'] = 0.0

        # Sharpe Ratio - Fix calculation to use proper risk adjustment
        if n_periods > 1:
            # Calculate excess returns properly
            period_risk_free_rate = risk_free_rate_annual / annual_factor  # Convert annual to period rate
            excess_returns = returns_series_clean - period_risk_free_rate
            
            # Annualized excess return and volatility
            mean_excess_return_annualized = excess_returns.mean() * annual_factor
            volatility_annualized = returns_series_clean.std(ddof=1) * np.sqrt(annual_factor)
            
            metrics['Sharpe Ratio'] = mean_excess_return_annualized / volatility_annualized if volatility_annualized != 0 else np.nan
        else:
            metrics['Sharpe Ratio'] = np.nan        # Sortino Ratio - Fix calculation
        if n_periods > 1:
            # Target return is the risk-free rate
            period_risk_free_rate = risk_free_rate_annual / annual_factor
            downside_returns = returns_series_clean[returns_series_clean < period_risk_free_rate]
            
            if len(downside_returns) > 0:
                # Calculate downside deviation (standard deviation of returns below target)
                downside_variance = np.mean((downside_returns - period_risk_free_rate)**2)
                downside_deviation_annualized = np.sqrt(downside_variance * annual_factor)
                
                # Mean excess return (annualized)
                excess_returns = returns_series_clean - period_risk_free_rate
                mean_excess_return_annualized = excess_returns.mean() * annual_factor
                
                metrics['Sortino Ratio'] = mean_excess_return_annualized / downside_deviation_annualized if downside_deviation_annualized != 0 else np.nan
            else:
                # No negative excess returns
                excess_returns = returns_series_clean - period_risk_free_rate
                mean_excess_return_annualized = excess_returns.mean() * annual_factor
                metrics['Sortino Ratio'] = np.inf if mean_excess_return_annualized > 0 else (0 if mean_excess_return_annualized == 0 else -np.inf)
        else:
            metrics['Sortino Ratio'] = np.nan


        # Maximum Drawdown
        running_max = cum_returns_with_initial.cummax()
        drawdowns_calc = (cum_returns_with_initial - running_max) / running_max
        metrics['Maximum Drawdown'] = drawdowns_calc.min() * 100 if not drawdowns_calc.empty else np.nan # As percentage
        drawdowns = drawdowns_calc # Store the series of drawdowns

        # Calmar Ratio
        annual_ret = metrics.get('Annualized Return', np.nan)
        max_dd = metrics.get('Maximum Drawdown', np.nan)
        metrics['Calmar Ratio'] = annual_ret / -max_dd if pd.notna(annual_ret) and pd.notna(max_dd) and max_dd != 0 else np.nan

        return metrics, cum_returns_with_initial, drawdowns

    metrics_strategy, cum_returns_strategy, drawdowns_strategy = calculate_core_metrics(strategy_returns, annual_factor, RISK_FREE_RATE)
    metrics_benchmark, cum_returns_benchmark, drawdowns_benchmark = calculate_core_metrics(benchmark_returns, annual_factor, RISK_FREE_RATE)

    # Alpha and Beta (Strategy vs Benchmark)
    if not strategy_returns.empty and not benchmark_returns.empty:
        # Align strategy and benchmark returns, dropping NaNs from both for comparison
        aligned_returns = pd.DataFrame({'strategy': strategy_returns, 'benchmark': benchmark_returns}).dropna()
        if len(aligned_returns) > 1: # Need at least 2 data points for covariance/variance
            strat_aligned = aligned_returns['strategy']
            bench_aligned = aligned_returns['benchmark']

            cov_matrix = np.cov(strat_aligned, bench_aligned, ddof=1) # ddof=1 for sample covariance
            beta_val = cov_matrix[0,1] / cov_matrix[1,1] if cov_matrix[1,1] != 0 else np.nan
            metrics_strategy['Beta'] = beta_val            # Jensen's Alpha: Rp - [Rf + Beta * (Rm - Rf)]
            # All inputs should be in decimal form for calculation, then convert alpha to %
            rp_annual_dec = metrics_strategy.get('Annualized Return', np.nan) / 100  # Convert % to decimal
            rf_annual_dec = RISK_FREE_RATE # Already decimal
            rm_annual_dec = metrics_benchmark.get('Annualized Return', np.nan) / 100  # Convert % to decimal

            if pd.notna(beta_val) and pd.notna(rp_annual_dec) and pd.notna(rm_annual_dec):
                alpha_val = (rp_annual_dec - (rf_annual_dec + beta_val * (rm_annual_dec - rf_annual_dec))) * 100 # Convert to %
                metrics_strategy['Alpha'] = alpha_val
            else:
                metrics_strategy['Alpha'] = np.nan
        else:
            metrics_strategy['Beta'] = np.nan
            metrics_strategy['Alpha'] = np.nan
    else:
        metrics_strategy['Beta'] = np.nan
        metrics_strategy['Alpha'] = np.nan

    return metrics_strategy, cum_returns_strategy, drawdowns_strategy, \
           metrics_benchmark, cum_returns_benchmark, drawdowns_benchmark


def generate_pdf_report(strategy_metrics, strategy_cum_returns, strategy_drawdowns,
                        benchmark_metrics, benchmark_cum_returns, benchmark_drawdowns,
                        holdings_log, actions_log, output_path):
    """Generate PDF report."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    current_y = height - 30 # Start a bit lower

    def check_y_add_page(c, current_y, needed_space=30):
        if current_y < needed_space + 30 : # 30 for bottom margin
            c.showPage()
            return height - 30 # Reset Y to top
        return current_y

    current_y = check_y_add_page(c, current_y, 50)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, current_y, "Multifactor Strategy Backtest Report")
    current_y -= 40

    current_y = check_y_add_page(c, current_y, 30)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, current_y, "Performance Summary")
    current_y -= 30

    x_col1, x_col2, x_col3 = 50, 250, 400
    line_height_table = 15

    current_y = check_y_add_page(c, current_y, line_height_table)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x_col1, current_y, "Metric")
    c.drawString(x_col2, current_y, "Strategy")
    c.drawString(x_col3, current_y, "Benchmark (SPY)")
    current_y -= line_height_table

    c.setFont("Helvetica", 9) # Smaller font for table content
    metric_order = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
                    'Sortino Ratio', 'Maximum Drawdown', 'Calmar Ratio', 'Alpha', 'Beta']

    for metric_name in metric_order:
        current_y = check_y_add_page(c, current_y, line_height_table)
        strat_val = strategy_metrics.get(metric_name, np.nan)
        bench_val = benchmark_metrics.get(metric_name, np.nan)

        is_percentage_metric = metric_name in ['Annualized Return', 'Annualized Volatility', 'Maximum Drawdown', 'Alpha']
        strat_str = f"{strat_val:.2f}%" if is_percentage_metric and pd.notna(strat_val) else (f"{strat_val:.2f}" if pd.notna(strat_val) else "N/A")
        bench_str = f"{bench_val:.2f}%" if is_percentage_metric and pd.notna(bench_val) else (f"{bench_val:.2f}" if pd.notna(bench_val) else "N/A")
        # Beta for benchmark is usually not shown or is 1 by definition if benchmark is market
        if metric_name == 'Beta' and benchmark_metrics: bench_str = f"{bench_val:.2f}" if pd.notna(bench_val) else "N/A (or 1.00)"
        if metric_name == 'Alpha' and benchmark_metrics: bench_str = "N/A (or 0.00%)"


        c.drawString(x_col1, current_y, metric_name)
        c.drawString(x_col2, current_y, strat_str)
        c.drawString(x_col3, current_y, bench_str)
        current_y -= line_height_table

    # Plots
    plot_height_on_pdf = 230 # Height of each plot image on PDF
    plot_spacing = 20 # Space between plots or plot and text

    # Cumulative Returns Plot
    current_y = check_y_add_page(c, current_y, plot_height_on_pdf + plot_spacing + 30) # +30 for title
    c.setFont("Helvetica-Bold", 12); c.drawString(50, current_y, "Cumulative Returns"); current_y -= (plot_spacing/2)
    if not strategy_cum_returns.empty and not strategy_cum_returns.isnull().all():
        plt.figure(figsize=(8, 4)) # Adjusted size for PDF
        strategy_cum_returns.plot(label='Strategy', color='blue', lw=1.5)
        if not benchmark_cum_returns.empty and not benchmark_cum_returns.isnull().all():
            benchmark_cum_returns.plot(label='Benchmark (SPY)', color='orange', linestyle='--', lw=1.5)
        plt.title(""); plt.xlabel("Date"); plt.ylabel("Cumulative Return")
        plt.grid(True, linestyle=':', alpha=0.6); plt.legend(fontsize='small')
        plt.tight_layout()
        buf = BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close(); buf.seek(0)
        try: c.drawImage(ImageReader(buf), 40, current_y - plot_height_on_pdf, width=width-80, height=plot_height_on_pdf)
        except: c.drawString(50, current_y - plot_height_on_pdf/2, "Error drawing plot.")
        current_y -= (plot_height_on_pdf + plot_spacing)
    else:
        c.setFont("Helvetica", 9); c.drawString(50, current_y - plot_height_on_pdf/2, "Cumulative Returns data not available.")
        current_y -= (plot_height_on_pdf/2 + plot_spacing)


    # Drawdowns Plot
    current_y = check_y_add_page(c, current_y, plot_height_on_pdf + plot_spacing + 30)
    c.setFont("Helvetica-Bold", 12); c.drawString(50, current_y, "Strategy Drawdowns"); current_y -= (plot_spacing/2)
    if not strategy_drawdowns.empty and not strategy_drawdowns.isnull().all():
        plt.figure(figsize=(8, 4))
        (strategy_drawdowns * 100).plot(label='Strategy Drawdown (%)', color='red', lw=1.5)
        plt.title(""); plt.xlabel("Date"); plt.ylabel("Drawdown (%)")
        plt.grid(True, linestyle=':', alpha=0.6); plt.legend(fontsize='small')
        plt.tight_layout()
        buf = BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close(); buf.seek(0)
        try: c.drawImage(ImageReader(buf), 40, current_y - plot_height_on_pdf, width=width-80, height=plot_height_on_pdf)
        except: c.drawString(50, current_y - plot_height_on_pdf/2, "Error drawing plot.")
        current_y -= (plot_height_on_pdf + plot_spacing)
    else:
        c.setFont("Helvetica", 9); c.drawString(50, current_y - plot_height_on_pdf/2, "Drawdowns data not available.")
        current_y -= (plot_height_on_pdf/2 + plot_spacing)


    # Corporate Actions Log
    c.showPage(); current_y = height - 30
    current_y = check_y_add_page(c, current_y, 30)
    c.setFont("Helvetica-Bold", 14); c.drawString(50, current_y, "Corporate Actions Log"); current_y -= 30
    c.setFont("Helvetica", 7) # Even smaller for dense logs
    line_height_log = 10
    if actions_log:
        for action in actions_log:
            current_y = check_y_add_page(c, current_y, line_height_log)
            action_date_str = action['date'].strftime('%Y-%m-%d') if isinstance(action['date'], (datetime, pd.Timestamp)) else str(action['date'])
            text = f"{action_date_str}: {action.get('action','?')} - {action.get('ticker','?')} ({action.get('outcome','?')})"
            if 'contraticker' in action and pd.notna(action.get('contraticker')) and action.get('contraticker') != "": text += f", Contra: {action.get('contraticker')}"
            if len(text) > 120: text = text[:117] + "..." # Max line length
            c.drawString(50, current_y, text)
            current_y -= line_height_log
    else:
        current_y = check_y_add_page(c, current_y, line_height_log)
        c.drawString(50, current_y, "No corporate actions processed for the portfolio.")
        current_y -= line_height_log


    # Holdings Log
    c.showPage(); current_y = height - 30
    current_y = check_y_add_page(c, current_y, 30)
    c.setFont("Helvetica-Bold", 14); c.drawString(50, current_y, "Holdings & Dropped Tickers Log (Per Rebalance)"); current_y -= 30
    c.setFont("Helvetica", 7)
    if holdings_log:
        for entry in holdings_log:
            current_y = check_y_add_page(c, current_y, line_height_log * 3) # Need more space per entry
            date_str = entry['date'].strftime('%Y-%m-%d') if isinstance(entry['date'], (datetime, pd.Timestamp)) else str(entry['date'])
            total_val_str = f"${entry.get('total_value', 0):,.0f}" if pd.notna(entry.get('total_value')) else "N/A"
            cash_str = f"${entry.get('cash',0):,.0f}" if pd.notna(entry.get('cash')) else "N/A"
            c.drawString(50, current_y, f"Rebal: {date_str}, Val: {total_val_str}, Cash: {cash_str}"); current_y -= line_height_log

            tickers_h = entry.get('tickers',[]); weights_h = entry.get('weights',{}); scores_h = entry.get('scores',{})
            h_list = [f"{t}({weights_h.get(t,0):.0%},sc:{scores_h.get(t,np.nan):.1f})" for t in tickers_h[:4]] # Show 4
            h_text = f"  Held({len(tickers_h)}): {', '.join(h_list)}{'...' if len(tickers_h)>4 else ''}"
            if len(h_text) > 120: h_text = h_text[:117]+"..."
            c.drawString(55, current_y, h_text); current_y -= line_height_log

            if 'dropped_tickers' in entry and entry['dropped_tickers']:
                d_list = [f"{t}({r[:10]})" for t,r in entry['dropped_tickers'].items()][:3] # Show 3
                d_text = f"  Drop({len(entry['dropped_tickers'])}): {', '.join(d_list)}{'...' if len(entry['dropped_tickers'])>3 else ''}"
                if len(d_text) > 120: d_text = d_text[:117]+"..."
                c.drawString(55, current_y, d_text); current_y -= line_height_log
            current_y -= (line_height_log / 2) # Small gap
    else:
        current_y = check_y_add_page(c, current_y, line_height_log)
        c.drawString(50, current_y, "No holdings log entries available."); current_y -= line_height_log

    c.save()


# ─── Additional Helper Functions for Universe Filtering ───────────────────────
@st.cache_data
def load_factor_mappings():
    """Load factor code to title mappings from INDICATORS.csv."""
    try:
        indicators_path = os.path.join(os.path.dirname(__file__), "INDICATORS.csv")
        if not os.path.exists(indicators_path):
            st.warning(f"INDICATORS.csv not found at {indicators_path}. Using default factor names.")
            return {}
        
        df_indicators = pd.read_csv(indicators_path)
        if 'indicator' in df_indicators.columns and 'title' in df_indicators.columns:
            return dict(zip(df_indicators['indicator'], df_indicators['title']))
        else:
            st.warning("INDICATORS.csv missing required 'indicator' or 'title' columns.")
            return {}
    except Exception as e:
        st.warning(f"Error loading factor mappings: {e}")
        return {}

@st.cache_data(ttl=3600)  # Cache for 1 hour to speed up repeated calculations
def calculate_ema_sma(prices, signal_date, ma_type='SMA', period=20, _eligible_tickers=None):
    """Calculate EMA or SMA for each ticker at the signal date - ULTRA OPTIMIZED VERSION."""
    if prices.empty or 'date' not in prices.columns or 'ticker' not in prices.columns or 'closeadj' not in prices.columns:
        return pd.Series(dtype=float)
    
    signal_date = pd.to_datetime(signal_date)
      # OPTIMIZATION 1: Pre-filter by eligible tickers to reduce data size
    if _eligible_tickers is not None:
        prices_filtered = prices[prices['ticker'].isin(_eligible_tickers)].copy()
    else:
        prices_filtered = prices.copy()
    
    # OPTIMIZATION 2: Only get data up to signal date with minimum required history
    min_date = signal_date - pd.Timedelta(days=period * 2)  # Conservative estimate
    relevant_prices = prices_filtered[
        (prices_filtered['date'] <= signal_date) & 
        (prices_filtered['date'] >= min_date)
    ].copy()
    
    if relevant_prices.empty:
        return pd.Series(dtype=float)
    
    # OPTIMIZATION 3: Sort once and set index for faster groupby operations
    relevant_prices = relevant_prices.sort_values(['ticker', 'date']).set_index(['ticker', 'date'])['closeadj']
      # OPTIMIZATION 4: Use the most efficient pandas operations
    if ma_type == 'SMA':
        # Calculate rolling mean using optimized groupby with level numbers
        ma_values = (relevant_prices.groupby(level=0, group_keys=False)
                    .rolling(window=period, min_periods=period)
                    .mean()
                    .groupby(level=0)
                    .last())
    elif ma_type == 'EMA':
        # Calculate EMA using optimized groupby with level numbers
        ma_values = (relevant_prices.groupby(level=0, group_keys=False)
                    .ewm(span=period, min_periods=period)
                    .mean()
                    .groupby(level=0)
                    .last())
    else:
        return pd.Series(dtype=float)
    
    return ma_values.fillna(np.nan)

def apply_multiple_technical_filters(prices, signal_date, technical_filters, logic='AND', eligible_tickers=None):
    """Apply multiple technical filters with AND/OR logic."""
    if not technical_filters:
        return list(eligible_tickers) if eligible_tickers is not None else []
    
    filter_results = []
    
    # Apply each technical filter
    for tech_filter in technical_filters:
        threshold = tech_filter['threshold'] if tech_filter['use_threshold'] else None
        result = apply_technical_filter(
            prices, signal_date, 
            tech_filter['ma_type'], 
            tech_filter['ma_period'],
            tech_filter['direction'], 
            threshold, 
            eligible_tickers
        )
        filter_results.append(set(result))
    
    # Combine results based on logic
    if logic == 'AND':
        # Intersection of all results
        if filter_results:
            combined_result = filter_results[0]
            for result_set in filter_results[1:]:
                combined_result = combined_result.intersection(result_set)
            return list(combined_result)
        else:
            return []
    else:  # OR logic
        # Union of all results
        combined_result = set()
        for result_set in filter_results:
            combined_result = combined_result.union(result_set)
        return list(combined_result)


def apply_multiple_fundamental_filters(sf1_data, signal_date, fundamental_filters, logic='AND'):
    """Apply multiple fundamental filters with AND/OR logic and growth support."""
    if not fundamental_filters:
        return []
    
    filter_results = []
    
    # Apply each fundamental filter with growth support
    for fund_filter in fundamental_filters:
        # Get growth type and convert to function parameter format
        growth_type = fund_filter.get('growth_type', 'Actual')
        if growth_type == 'Year-over-Year (YoY)':
            growth_param = 'yoy'
        elif growth_type == 'Quarter-over-Quarter (QoQ)':
            growth_param = 'qoq'
        else:
            growth_param = 'actual'
        
        # Get dimension to use
        dimension = fund_filter.get('dimension', 'ART')
        
        # Use the growth-aware filter function with dimension
        result = apply_fundamental_filter_with_growth(
            sf1_data, signal_date,
            fund_filter['factor_code'],
            fund_filter['direction'],
            fund_filter['threshold'],
            growth_param,
            dimension
        )
        filter_results.append(set(result))
    
    # Combine results based on logic
    if logic == 'AND':
        # Intersection of all results
        if filter_results:
            combined_result = filter_results[0]
            for result_set in filter_results[1:]:
                combined_result = combined_result.intersection(result_set)
            return list(combined_result)
        else:
            return []
    else:  # OR logic
        # Union of all results
        combined_result = set()
        for result_set in filter_results:
            combined_result = combined_result.union(result_set)
        return list(combined_result)


def apply_technical_filter(prices, signal_date, ma_type='SMA', period=20, direction='above', threshold=None, eligible_tickers=None):
    """Apply technical analysis filter (EMA/SMA) to get eligible tickers - ULTRA OPTIMIZED VERSION."""
    if not isinstance(signal_date, pd.Timestamp):
        signal_date = pd.to_datetime(signal_date)
    
    # OPTIMIZATION 1: Calculate moving averages only for eligible tickers
    ma_values = calculate_ema_sma(prices, signal_date, ma_type, period, _eligible_tickers=eligible_tickers)
    
    if ma_values.empty:
        return []
    
    # OPTIMIZATION 2: Efficient price lookup using vectorized operations
    if eligible_tickers is not None:
        price_subset = prices[prices['ticker'].isin(eligible_tickers)]
    else:
        price_subset = prices
    
    signal_prices = price_subset[price_subset['date'] == signal_date]
    if signal_prices.empty:
        # Use the closest date before signal_date - optimized lookup
        available_dates = price_subset[price_subset['date'] <= signal_date]['date']
        if available_dates.empty:
            return []
        closest_date = available_dates.max()  # More efficient than max()
        signal_prices = price_subset[price_subset['date'] == closest_date]
    
    # OPTIMIZATION 3: Use set_index for faster alignment
    current_prices = signal_prices.set_index('ticker')['closeadj']
    
    # OPTIMIZATION 4: Vectorized alignment and filtering
    common_tickers = ma_values.index.intersection(current_prices.index)
    if len(common_tickers) == 0:
        return []
    
    ma_aligned = ma_values.loc[common_tickers]
    prices_aligned = current_prices.loc[common_tickers]
    
    # OPTIMIZATION 5: Single vectorized comparison with boolean indexing
    valid_mask = ~(ma_aligned.isna() | prices_aligned.isna())
    if not valid_mask.any():
        return []
    
    ma_valid = ma_aligned[valid_mask]
    prices_valid = prices_aligned[valid_mask]
    
    # OPTIMIZATION 6: Efficient threshold calculations
    if direction == 'above':
        if threshold is not None:
            # Vectorized calculation: price/MA - 1 >= threshold/100
            eligible_mask = (prices_valid / ma_valid - 1) >= (threshold / 100)
        else:
            eligible_mask = prices_valid > ma_valid
    else:  # direction == 'below'
        if threshold is not None:
            # Vectorized calculation: MA/price - 1 >= threshold/100
            eligible_mask = (ma_valid / prices_valid - 1) >= (threshold / 100)
        else:
            eligible_mask = prices_valid < ma_valid
    
    return eligible_mask[eligible_mask].index.tolist()

def apply_fundamental_filter(sf1_data, signal_date, factor_code, direction='above', threshold=0, dimension='ART'):
    """Apply fundamental factor filter to get eligible tickers."""
    if sf1_data.empty or factor_code not in sf1_data.columns:
        return []
    
    signal_date = pd.to_datetime(signal_date)
    
    # Get the most recent fundamental data before or on signal date with specified dimension
    relevant_data = sf1_data[
        (sf1_data['calendardate'] <= signal_date) &
        (sf1_data['dimension'] == dimension)  # Use specified dimension
    ].copy()
    
    if relevant_data.empty:
        return []
    
    # Calculate age of financial data (lag in days)
    relevant_data['data_age_days'] = (signal_date - pd.to_datetime(relevant_data['calendardate'])).dt.days
    
    # Apply maximum age limits based on dimension type
    if dimension in ['ARY', 'MRY']:  # Annual dimensions: 185 days maximum age
        relevant_data = relevant_data[relevant_data['data_age_days'] <= 185]
    elif dimension in ['ART', 'MRT', 'ARQ', 'MRQ']:  # TTM and Quarterly dimensions: 140 days maximum age
        relevant_data = relevant_data[relevant_data['data_age_days'] <= 140]
    
    if relevant_data.empty:
        return []
    
    # For each ticker, get the most recent data
    latest_data = relevant_data.groupby('ticker').last().reset_index()
    
    # Filter based on criteria
    eligible_tickers = []
    
    for _, row in latest_data.iterrows():
        ticker = row['ticker']
        factor_value = row[factor_code]
        
        if pd.isna(factor_value):
            continue
        
        if direction == 'above' and factor_value > threshold:
            eligible_tickers.append(ticker)
        elif direction == 'below' and factor_value < threshold:
            eligible_tickers.append(ticker)
    
    return eligible_tickers


# ─── Growth Calculation Functions ───────────────────────────────────────────────
def calculate_factor_growth(sf1_data, signal_date, factor_code, growth_type='actual', dimension='ART'):
    """
    Calculate factor growth (YoY, QoQ, or return actual values) with improved lookback periods.
    
    New approach:
    - TTM: First factor within 140 days gap, second factor max 12.5 months before first factor
    - Quarterly: First factor within 140 days gap, second factor max 95 days before first factor  
    - Yearly: First factor within 185 days gap, second factor max 12.5 months before first factor
    
    Parameters:
    - sf1_data: SF1 fundamental data
    - signal_date: Date for calculation
    - factor_code: The fundamental factor to calculate
    - growth_type: 'actual', 'yoy', or 'qoq'
    - dimension: SF1 dimension to use ('ART', 'MRQ', etc.)
    
    Returns:
    - Series with factor values/growth rates indexed by ticker
    """
    if sf1_data.empty or factor_code not in sf1_data.columns:
        return pd.Series(dtype=float)
    
    signal_date = pd.to_datetime(signal_date)
    
    # Filter data up to signal date with specified dimension
    relevant_data = sf1_data[
        (sf1_data['calendardate'] <= signal_date) &
        (sf1_data['dimension'] == dimension)
    ].copy()
    
    if relevant_data.empty:
        return pd.Series(dtype=float)
    
    # Calculate age of financial data (lag in days)
    relevant_data['data_age_days'] = (signal_date - pd.to_datetime(relevant_data['calendardate'])).dt.days
    
    # Apply maximum age limits for the first (most recent) factor based on dimension type
    if dimension in ['ARY', 'MRY']:  # Annual dimensions: 185 days maximum age
        first_factor_age_limit = 185
    elif dimension in ['ART', 'MRT']:  # TTM dimensions: 140 days maximum age
        first_factor_age_limit = 140
    elif dimension in ['ARQ', 'MRQ']:  # Quarterly dimensions: 140 days maximum age
        first_factor_age_limit = 140
    else:
        first_factor_age_limit = 140  # Default
    
    # Filter data for first factor within age limits
    recent_data = relevant_data[relevant_data['data_age_days'] <= first_factor_age_limit]
    
    if recent_data.empty:
        return pd.Series(dtype=float)
    
    # Sort by ticker and date
    recent_data = recent_data.sort_values(['ticker', 'calendardate'])
    
    if growth_type == 'actual':
        # Return most recent actual values
        latest_data = recent_data.groupby('ticker').last()
        return pd.to_numeric(latest_data[factor_code], errors='coerce')
    
    # For growth calculations, we need multiple periods
    growth_values = {}
    
    # Get all data for historical lookback (not limited by first factor age limit)
    all_data = relevant_data.sort_values(['ticker', 'calendardate'])
    
    for ticker in recent_data['ticker'].unique():
        # Get the most recent valid data point for this ticker (first factor)
        ticker_recent = recent_data[recent_data['ticker'] == ticker].sort_values('calendardate')
        if ticker_recent.empty:
            continue
            
        first_factor_row = ticker_recent.iloc[-1]
        current_value = pd.to_numeric(first_factor_row[factor_code], errors='coerce')
        current_date = first_factor_row['calendardate']
        
        if pd.isna(current_value):
            continue
        
        # Get all historical data for this ticker (for second factor lookback)
        ticker_all_data = all_data[all_data['ticker'] == ticker].copy()
        
        # Define maximum lookback period from the first factor date
        if growth_type == 'yoy':
            if dimension in ['ART', 'MRT']:  # TTM: max 12.5 months before first factor
                max_lookback_days = int(12.5 * 30.44)  # ~380 days
            elif dimension in ['ARY', 'MRY']:  # Yearly: max 12.5 months before first factor
                max_lookback_days = int(12.5 * 30.44)  # ~380 days
            else:
                max_lookback_days = int(12.5 * 30.44)  # Default
                
        elif growth_type == 'qoq':
            if dimension in ['ARQ', 'MRQ']:  # Quarterly: max 95 days before first factor
                max_lookback_days = 95
            else:
                max_lookback_days = 95  # Default for quarterly
        else:
            continue
        
        # Find historical data within the lookback window from first factor date
        earliest_allowed_date = current_date - pd.Timedelta(days=max_lookback_days)
        historical_data = ticker_all_data[
            (ticker_all_data['calendardate'] >= earliest_allowed_date) &
            (ticker_all_data['calendardate'] < current_date)  # Must be before current
        ]
        
        if historical_data.empty:
            continue
        
        if growth_type == 'yoy':
            # Look for data approximately 1 year before the first factor date
            target_date = current_date - pd.DateOffset(years=1)
            date_diff = abs(historical_data['calendardate'] - target_date)
            
            if not date_diff.empty:
                closest_idx = date_diff.idxmin()
                prior_value = pd.to_numeric(historical_data.loc[closest_idx, factor_code], errors='coerce')
                
                if pd.notna(prior_value) and prior_value != 0:
                    growth_rate = (current_value - prior_value) / abs(prior_value)
                    growth_values[ticker] = growth_rate
        
        elif growth_type == 'qoq':
            # Look for data approximately 1 quarter before the first factor date
            target_date = current_date - pd.DateOffset(months=3)
            date_diff = abs(historical_data['calendardate'] - target_date)
            
            if not date_diff.empty:
                closest_idx = date_diff.idxmin()
                prior_value = pd.to_numeric(historical_data.loc[closest_idx, factor_code], errors='coerce')
                
                if pd.notna(prior_value) and prior_value != 0:
                    growth_rate = (current_value - prior_value) / abs(prior_value)
                    growth_values[ticker] = growth_rate
    
    return pd.Series(growth_values)


def apply_fundamental_filter_with_growth(sf1_data, signal_date, factor_code, direction='above', threshold=0, growth_type='actual', dimension='ART'):
    """Apply fundamental factor filter with growth calculation support."""
    if growth_type == 'actual':
        # Use existing function for actual values with dimension parameter
        return apply_fundamental_filter(sf1_data, signal_date, factor_code, direction, threshold, dimension)
    
    # Calculate growth values with dimension parameter
    growth_values = calculate_factor_growth(sf1_data, signal_date, factor_code, growth_type, dimension)
    
    if growth_values.empty:
        return []
    
    # Filter based on criteria
    eligible_tickers = []
    
    for ticker, growth_value in growth_values.items():
        if pd.isna(growth_value):
            continue
        
        if direction == 'above' and growth_value > threshold:
            eligible_tickers.append(ticker)
        elif direction == 'below' and growth_value < threshold:
            eligible_tickers.append(ticker)
    
    return eligible_tickers


# ─── Backtest Function ───────────────────────────────────────────────────────────
def run_backtest(params, start_date_dt, end_date_dt): # Renamed for clarity
    """Run the multifactor backtest."""
    cushion_start = start_date_dt - timedelta(days=DATA_CUSHION_DAYS)
    cushion_end = end_date_dt + timedelta(days=DATA_CUSHION_DAYS)

    # Load all data with cushion
    sf1_all = load_parquet('sf1.parquet', cushion_start, cushion_end)
    sep_all = load_parquet('sep.parquet', cushion_start, cushion_end)
    sfp_all = load_parquet('sfp.parquet', cushion_start, cushion_end) # For SPY
    tickers_meta = load_parquet('tickers.parquet') # No date filter needed for metadata
    actions_all = load_parquet('actions.parquet', cushion_start, cushion_end)

    # Critical data checks
    if any(df.empty for df in [sf1_all, sep_all, sfp_all, tickers_meta, actions_all]):
        st.error("One or more essential data files (SF1, SEP, SFP, TICKERS, ACTIONS) could not be loaded or are empty within the cushioned date range. Backtest cannot proceed.")
        return {}, pd.Series(dtype=float), pd.Series(dtype=float), [], [], "", {}, pd.Series(dtype=float), pd.Series(dtype=float)
      # Extract parameters
    selected_categories = params['categories']
    dolvol_threshold = params['dolvol_threshold']
    factors_selected = params['factors'] # Renamed to avoid conflict
    weights_map = params['weights'] # Renamed
    directions_map = params['directions'] # Renamed
    growth_types_map = params.get('growth_types', {})  # Add growth types
    momentum_lookback_weeks = params['momentum_lookback'] # Renamed
    use_skip_momentum_flag = params['use_skip_momentum'] # Renamed
    portfolio_size_config = params['portfolio_size'] # Renamed
    portfolio_size_type = params.get('portfolio_size_type', 'Number of Stocks') # Added
    weighting_method_selected = params['weighting'] # Renamed
    rebalance_freq_selected = params['rebalance_freq'] # Renamed
    slippage_percentage = params['slippage'] / 100 # Convert to decimal
    initial_capital_amount = params['initial_capital'] # Renamed    small_universe_handling = params.get('small_universe_handling', 'Invest in all available stocks') # Added
    
    # Extract multi-filter parameters
    technical_filters_config = params.get('technical_filters', [])
    fundamental_filters_config = params.get('fundamental_filters', [])
    technical_filter_logic = params.get('technical_filter_logic', 'AND')
    fundamental_filter_logic = params.get('fundamental_filter_logic', 'AND')
    overall_filter_logic = params.get('overall_filter_logic', 'AND')    # Prepare fundamental data for default view (no longer limited to just ART dimension)
    sf1_art_filtered = sf1_all.copy()
    if sf1_art_filtered.empty:
        st.warning("No fundamental data found in SF1 within the date range. Factor calculations might fail.")

    # Get all trading days within the cushioned period
    all_trading_days_list = get_trading_days(sep_all)
    if not all_trading_days_list:
        st.error("No trading days found in SEP price data. Backtest cannot proceed.")
        return {}, pd.Series(dtype=float), pd.Series(dtype=float)

    # Generate rebalance signal and execution dates
    raw_signal_dates_series, raw_execution_dates_series = get_signal_execution_dates(all_trading_days_list, rebalance_freq_selected)

    # Filter signal and execution dates to be within the *actual* backtest period [start_date_dt, end_date_dt]
    # And ensure execution date is not after signal date, and signal is not after end_date_dt
    valid_schedule_pairs = []
    temp_exec_map_lookup = {s: e for s, e in zip(raw_signal_dates_series, raw_execution_dates_series)}

    for s_date_candidate in raw_signal_dates_series:
        if s_date_candidate >= start_date_dt and s_date_candidate <= end_date_dt:
            e_date_candidate = temp_exec_map_lookup.get(s_date_candidate)
            # Ensure execution date is valid, not after end_date_dt, and not before signal_date
            if e_date_candidate and e_date_candidate >= s_date_candidate and e_date_candidate <= end_date_dt:
                valid_schedule_pairs.append((s_date_candidate, e_date_candidate))

    if not valid_schedule_pairs:
        st.error(f"No valid rebalancing periods found for the selected date range [{start_date_dt.date()} to {end_date_dt.date()}] and frequency '{rebalance_freq_selected}'. Please adjust.")
        return {}, pd.Series(dtype=float), pd.Series(dtype=float)

    actual_signal_dates, actual_execution_dates = zip(*valid_schedule_pairs)

    # Initialize backtest state
    current_portfolio_shares = {}  # ticker: shares
    current_cash = initial_capital_amount
    strategy_period_returns = pd.Series(dtype=float)
    benchmark_period_returns = pd.Series(dtype=float)
    portfolio_holdings_log = [] # Detailed log for PDF/CSV
    processed_actions_log = [] # Log of CAs that affected the portfolio

    # SPY benchmark prices
    spy_prices_indexed = sfp_all[sfp_all['ticker'] == 'SPY'][['date', 'closeadj']].set_index('date')['closeadj'].sort_index()
    if spy_prices_indexed.empty:
        st.warning("SPY (benchmark) price data not found. Benchmark metrics will be unavailable.")
    # Variables to track state between rebalances for return calculation
    portfolio_value_at_last_rebalance = initial_capital_amount
    date_of_last_rebalance_execution = None


    # --- Main Backtest Loop ---
    for i, (signal_date, execution_date) in enumerate(zip(actual_signal_dates, actual_execution_dates)):
        st.text(f"Period {i+1}: Signal {signal_date.strftime('%Y-%m-%d')}, Execution {execution_date.strftime('%Y-%m-%d')}")

        # --- 1. Calculate Returns for the PREVIOUS Investment Period ---
        # For the first period (i==0), we calculate from initial capital to first portfolio value
        # For subsequent periods (i>0), we calculate from previous portfolio value to current portfolio value
        if date_of_last_rebalance_execution is not None:
            # Value of portfolio *before* current rebalance trades, based on prices at `execution_date`
            market_value_of_old_holdings_now = 0
            prices_at_current_execution_for_old_portfolio = sep_all[sep_all['date'] == execution_date].set_index('ticker')['closeadj']

            for ticker, shares in current_portfolio_shares.items(): # `current_portfolio_shares` is from *previous* rebalance
                price_now = prices_at_current_execution_for_old_portfolio.get(ticker)
                if pd.notna(price_now) and price_now > 0:
                    market_value_of_old_holdings_now += shares * price_now
                # If price is NaN/0, stock became worthless or delisted; its value is 0 here.

            total_equity_before_current_period_trades = current_cash + market_value_of_old_holdings_now
            # `current_cash` is also from the state *after* the previous rebalance's trades.

            if portfolio_value_at_last_rebalance != 0:
                period_return_val = (total_equity_before_current_period_trades / portfolio_value_at_last_rebalance) - 1
            else: # Avoid division by zero if portfolio was $0 (e.g., full liquidation)
                period_return_val = 0.0 if total_equity_before_current_period_trades == 0 else np.nan # Or handle as extreme return
            strategy_period_returns.loc[execution_date] = period_return_val

            # Benchmark return for the same period
            if not spy_prices_indexed.empty:
                try:
                    spy_start_val = spy_prices_indexed.loc[date_of_last_rebalance_execution]
                    spy_end_val = spy_prices_indexed.loc[execution_date]
                    if pd.notna(spy_start_val) and pd.notna(spy_end_val) and spy_start_val != 0:
                        benchmark_period_returns.loc[execution_date] = (spy_end_val / spy_start_val) - 1
                    else: benchmark_period_returns.loc[execution_date] = np.nan
                except KeyError: # Date not found in SPY prices
                    benchmark_period_returns.loc[execution_date] = np.nan


        # --- 2. Process Corporate Actions Affecting Current Holdings ---
        # Window for CAs: from day after last exec date up to current exec date
        ca_processing_window_start = (date_of_last_rebalance_execution + timedelta(days=1)) if date_of_last_rebalance_execution else cushion_start
        actions_in_period = actions_all[
            (actions_all['date'] >= ca_processing_window_start) & (actions_all['date'] <= execution_date)
        ].sort_values('date')

        for _, action_details in actions_in_period.iterrows():
            act_ticker_involved = action_details['ticker']
            act_date_occurred = action_details['date'] # Already datetime
            act_type_code = action_details['action']
            act_contraticker_val = action_details.get('contraticker')
            act_value_data = action_details.get('value') # e.g., split ratio, spinoff shares

            # Only process if the action involves a ticker we currently hold, or creates a new holding from one we hold
            shares_currently_held = current_portfolio_shares.get(act_ticker_involved, 0)
            action_outcome_description = ""

            if act_type_code in ['delisted', 'regulatorydelisting', 'bankruptcyliquidation', 'voluntarydelisting', 'acquisitionby', 'mergerfrom']:
                if shares_currently_held > 0: # Only if we held it
                    liquidation_price = get_action_price(sep_all, act_ticker_involved, act_date_occurred,
                                                         is_delisting=(act_type_code in ['delisted', 'regulatorydelisting', 'bankruptcyliquidation']),
                                                         is_voluntary_or_ma=(act_type_code in ['voluntarydelisting', 'acquisitionby', 'mergerfrom']))
                    if liquidation_price is not None: # Can be 0.0
                        cash_from_liquidation = shares_currently_held * liquidation_price
                        current_cash += cash_from_liquidation
                        action_outcome_description = f"Liquidated {shares_currently_held:.2f} sh @ ${liquidation_price:.2f}. Cash +${cash_from_liquidation:.2f}"
                    else: # Price not found, assume $0 for safety if it was a delisting we should have priced
                        action_outcome_description = f"Liquidated {shares_currently_held:.2f} sh @ $0 (price N/A)."
                    del current_portfolio_shares[act_ticker_involved] # Remove from portfolio

            elif act_type_code in ['spinoff', 'spunofffrom'] and act_contraticker_val: # ticker is parent, contraticker is spun-off
                if shares_currently_held > 0 and pd.notna(act_value_data) and act_value_data > 0:
                    new_spunoff_shares = shares_currently_held * act_value_data
                    current_portfolio_shares[act_contraticker_val] = current_portfolio_shares.get(act_contraticker_val, 0) + new_spunoff_shares
                    action_outcome_description = f"Received {new_spunoff_shares:.2f} sh of {act_contraticker_val} from {act_ticker_involved}"

            elif act_type_code in ['split', 'adrratiosplit']:
                if shares_currently_held > 0 and pd.notna(act_value_data) and act_value_data > 0:
                    current_portfolio_shares[act_ticker_involved] = shares_currently_held * act_value_data
                    action_outcome_description = f"Shares of {act_ticker_involved} adjusted by factor {act_value_data:.2f}"

            elif act_type_code == 'tickerchangeto' and act_contraticker_val: # `act_ticker_involved` is OLD, `act_contraticker_val` is NEW
                if shares_currently_held > 0 : # If we held the old ticker
                    current_portfolio_shares[act_contraticker_val] = current_portfolio_shares.pop(act_ticker_involved) # Remove old, add new with same shares
                    action_outcome_description = f"Ticker {act_ticker_involved} changed to {act_contraticker_val}"

            if action_outcome_description: # Log if the action had an effect
                processed_actions_log.append({
                    'date': act_date_occurred, 'action': act_type_code, 'ticker': act_ticker_involved,
                    'contraticker': act_contraticker_val, 'outcome': action_outcome_description
                })

        # --- 3. Determine Total Equity Available for Rebalancing ---
        # This is current_cash (after CAs) + market value of *remaining* holdings at `execution_date` prices
        total_equity_for_rebalance = current_cash
        prices_at_execution_for_rebal_val = sep_all[sep_all['date'] == execution_date].set_index('ticker')['closeadj']

        for ticker, shares in current_portfolio_shares.items(): # Portfolio after CAs
            price_at_exec = prices_at_execution_for_rebal_val.get(ticker)
            if pd.notna(price_at_exec) and price_at_exec > 0:
                total_equity_for_rebalance += shares * price_at_exec
            # If price is NaN/0, it means it's worthless for rebalancing.        # --- 4. Rebalance Decision (using `signal_date` data) ---
        # Universe Construction
        tickers_from_selected_categories = tickers_meta[tickers_meta['category'].isin(selected_categories)]['ticker'].unique()
        dollar_volume_at_signal = calculate_dolvol(sep_all, signal_date) # Uses corrected function
        liquid_tickers_universe = dollar_volume_at_signal[dollar_volume_at_signal >= dolvol_threshold].index.unique()
        eligible_universe_tickers = pd.Index(list(set(tickers_from_selected_categories) & set(liquid_tickers_universe)))
        
        # Apply multiple technical filters if any are configured
        technical_filters = technical_filters_config
        technical_logic = technical_filter_logic
        
        # Apply multiple fundamental filters if any are configured  
        fundamental_filters = fundamental_filters_config
        fundamental_logic = fundamental_filter_logic
        
        # Apply multi-filter logic
        technical_eligible = apply_multiple_technical_filters(
            sep_all, signal_date, technical_filters, technical_logic, eligible_universe_tickers
        )
        
        fundamental_eligible = apply_multiple_fundamental_filters(
            sf1_all, signal_date, fundamental_filters, fundamental_logic
        )
        
        # Combine technical and fundamental results based on overall logic
        if len(technical_filters) > 0 and len(fundamental_filters) > 0:
            # Both filter types have configurations
            if overall_filter_logic == "AND":
                eligible_universe_tickers = pd.Index(list(set(technical_eligible) & set(fundamental_eligible)))
            else:  # OR
                eligible_universe_tickers = pd.Index(list(set(technical_eligible) | set(fundamental_eligible)))
        elif len(technical_filters) > 0:
            # Only technical filters configured
            eligible_universe_tickers = pd.Index(list(set(eligible_universe_tickers) & set(technical_eligible)))
        elif len(fundamental_filters) > 0:
            # Only fundamental filters configured  
            eligible_universe_tickers = pd.Index(list(set(eligible_universe_tickers) & set(fundamental_eligible)))
        # If no filters configured, eligible_universe_tickers remains unchanged

        # Fundamental Data Snapshot (most recent ART report as of signal_date)
        sf1_art_snapshot = sf1_art_filtered[
            (sf1_art_filtered['ticker'].isin(eligible_universe_tickers)) &
            (sf1_art_filtered['calendardate'] <= signal_date) # Ensure report date is on or before signal date
        ].sort_values('calendardate', ascending=False).groupby('ticker').first() # Get the latest report for each ticker

        if sf1_art_snapshot.empty and eligible_universe_tickers.size > 0 :
            st.warning(f"No fundamental data (SF1 ART) found for any eligible tickers at signal date {signal_date.date()}. Portfolio will likely be empty.")        # Factor Calculation with Growth Support
        factor_scores_for_universe = pd.DataFrame(index=sf1_art_snapshot.index) # Start with tickers that have fundamentals
        tickers_dropped_due_to_missing_factors = {}

        for factor_name_iter in factors_selected:
            if factor_name_iter == 'momentum':
                # Momentum always uses actual values (growth not applicable)
                momentum_scores_series = calculate_momentum(sep_all, signal_date, momentum_lookback_weeks, use_skip_momentum_flag)
                factor_scores_for_universe[factor_name_iter] = momentum_scores_series # Aligns by index (ticker)
            else:
                # Get growth type for this factor (default to 'Actual' if not specified)
                growth_type = growth_types_map.get(factor_name_iter, 'Actual')
                
                # Get dimension for this factor (default to 'ART' if not specified)
                dimension = params.get('dimensions', {}).get(factor_name_iter, 'ART')
                
                # Convert UI growth type to function parameter format
                if growth_type == 'Year-over-Year (YoY)':
                    growth_param = 'yoy'
                elif growth_type == 'Quarter-over-Quarter (QoQ)':
                    growth_param = 'qoq'
                else:
                    growth_param = 'actual'
                
                if '/' in factor_name_iter: # Derived factor like 'gp/assets'
                    # For derived factors, we calculate growth on the individual components if growth is requested
                    if growth_param == 'actual':
                        # Need to get sf1 data with the correct dimension for actual values
                        # Filter sf1 data for the specified dimension
                        sf1_dimension_snapshot = sf1_all[
                            (sf1_all['ticker'].isin(eligible_universe_tickers)) &
                            (sf1_all['calendardate'] <= signal_date) &  # Ensure report date is on or before signal date
                            (sf1_all['dimension'] == dimension)
                        ].sort_values('calendardate', ascending=False).groupby('ticker').first() 

                        numerator_col, denominator_col = factor_name_iter.split('/')
                        if numerator_col in sf1_dimension_snapshot.columns and denominator_col in sf1_dimension_snapshot.columns:
                            num_values = pd.to_numeric(sf1_dimension_snapshot[numerator_col], errors='coerce')
                            den_values = pd.to_numeric(sf1_dimension_snapshot[denominator_col], errors='coerce')
                            # Avoid division by zero or NaN denominator; result will be NaN
                            factor_scores_for_universe[factor_name_iter] = num_values / den_values.replace(0, np.nan)
                        else:
                            factor_scores_for_universe[factor_name_iter] = np.nan # Cannot compute
                    else:
                        # For growth calculations on derived factors, calculate growth of the ratio itself
                        # This requires implementing ratio growth calculation
                        numerator_col, denominator_col = factor_name_iter.split('/')
                        if numerator_col in sf1_all.columns and denominator_col in sf1_all.columns:
                            # Create temporary ratio column in sf1_all for growth calculation
                            sf1_temp = sf1_all.copy()
                            num_vals = pd.to_numeric(sf1_temp[numerator_col], errors='coerce')
                            den_vals = pd.to_numeric(sf1_temp[denominator_col], errors='coerce')
                            sf1_temp[factor_name_iter] = num_vals / den_vals.replace(0, np.nan)
                            
                            # Calculate growth of the derived factor using the specified dimension
                            growth_scores = calculate_factor_growth(sf1_temp, signal_date, factor_name_iter, growth_param, dimension)
                            factor_scores_for_universe[factor_name_iter] = growth_scores
                        else:
                            factor_scores_for_universe[factor_name_iter] = np.nan
                            
                elif factor_name_iter in sf1_all.columns: # Direct fundamental factor
                    # Use growth calculation function with specified dimension
                    factor_scores = calculate_factor_growth(sf1_all, signal_date, factor_name_iter, growth_param, dimension)
                    factor_scores_for_universe[factor_name_iter] = factor_scores
                else: # Factor not found
                    factor_scores_for_universe[factor_name_iter] = np.nan
                    st.warning(f"Factor '{factor_name_iter}' definition not found for signal date {signal_date.date()}. Will be NaN.")

        # Filter out tickers that don't have all required factors AFTER calculation
        initial_factor_candidates = set(factor_scores_for_universe.index)
        factor_scores_for_universe.dropna(subset=factors_selected, how='any', inplace=True) # Drop if ANY selected factor is NaN
        final_factor_candidates = set(factor_scores_for_universe.index)
        dropped_now = initial_factor_candidates - final_factor_candidates
        for dropped_ticker_name in dropped_now:
            tickers_dropped_due_to_missing_factors[dropped_ticker_name] = "Missing value for one or more selected factors"


        # Z-scores and Composite Score
        final_composite_scores = pd.Series(dtype=float) # Default to empty
        if not factor_scores_for_universe.empty:
            final_composite_scores = pd.Series(0.0, index=factor_scores_for_universe.index) # Initialize with 0 for tickers with all factors
            for factor_name_iter in factors_selected:
                if factor_name_iter in factor_scores_for_universe.columns: # Should always be true due to earlier dropna
                    z_scores_for_factor = compute_zscores(factor_scores_for_universe[factor_name_iter], directions_map.get(factor_name_iter, 'higher'))
                    # Add weighted z-score; fill_value=0 ensures NaNs in z_scores_for_factor don't propagate if composite_scores has a value
                    final_composite_scores = final_composite_scores.add(z_scores_for_factor * weights_map.get(factor_name_iter, 0), fill_value=0)
            final_composite_scores.dropna(inplace=True) # Drop tickers if composite score ended up NaN (e.g. all z-scores were NaN)
        else:
             st.info(f"No tickers remained after factor calculation and filtering at signal date {signal_date.date()}.")        # Portfolio Selection based on composite scores
        desired_portfolio_size = None
        if isinstance(portfolio_size_config, int): # Fixed number of stocks
            desired_portfolio_size = portfolio_size_config
            num_stocks_to_select = min(portfolio_size_config, len(final_composite_scores))
        else: # Percentage of universe
            desired_portfolio_size = int(len(final_composite_scores) / portfolio_size_config) if portfolio_size_config > 0 else len(final_composite_scores)
            num_stocks_to_select = int(len(final_composite_scores) * portfolio_size_config)

        # Handle small universe scenario
        universe_smaller_than_desired = len(final_composite_scores) < desired_portfolio_size
        
        if universe_smaller_than_desired and len(final_composite_scores) > 0:
            if small_universe_handling == 'Go completely cash':
                st.info(f"Filtered universe ({len(final_composite_scores)} stocks) is smaller than desired portfolio size ({desired_portfolio_size}). Going completely cash as configured.")
                selected_target_tickers = []  # Empty portfolio - go to cash
            elif small_universe_handling == 'Use 1/N allocation + cash':
                st.info(f"Filtered universe ({len(final_composite_scores)} stocks) is smaller than desired portfolio size ({desired_portfolio_size}). Using 1/N allocation with extra cash as configured.")
                selected_target_tickers = final_composite_scores.index.tolist()  # Use all available stocks
            else: # 'Invest in all available stocks' (default)
                st.info(f"Filtered universe ({len(final_composite_scores)} stocks) is smaller than desired portfolio size ({desired_portfolio_size}). Investing equally in all available stocks as configured.")
                selected_target_tickers = final_composite_scores.index.tolist()  # Use all available stocks
        else:
            # Normal case: universe is large enough or empty
            selected_target_tickers = final_composite_scores.nlargest(num_stocks_to_select).index.tolist()


        # --- 5. Execute Trades to Form New Portfolio (at `execution_date` prices) ---
        new_target_portfolio_shares = {} # ticker: shares for the new portfolio        # Filter `selected_target_tickers` to only those with valid prices at `execution_date`
        prices_for_selected_targets = prices_at_execution_for_rebal_val[prices_at_execution_for_rebal_val.index.isin(selected_target_tickers)]
        tradable_target_tickers = [
            ticker for ticker in selected_target_tickers
            if pd.notna(prices_for_selected_targets.get(ticker)) and prices_for_selected_targets.get(ticker, 0) > 0
        ]
        if len(tradable_target_tickers) < len(selected_target_tickers):
            st.warning(f"{len(selected_target_tickers) - len(tradable_target_tickers)} target tickers had no valid price at execution on {execution_date.date()} and were excluded.")        # Calculate target dollar allocations for TRADABLE target tickers
        target_dollar_allocations = {}
        if tradable_target_tickers: # Only if there are tickers we can actually trade
            
            # Handle special allocation for small universe scenarios
            if universe_smaller_than_desired and small_universe_handling == 'Use 1/N allocation + cash':
                # Use 1/N allocation based on DESIRED portfolio size, not actual available stocks
                # This leaves extra cash when universe is smaller than desired
                dollar_amount_per_stock = total_equity_for_rebalance / desired_portfolio_size
                target_dollar_allocations = {ticker: dollar_amount_per_stock for ticker in tradable_target_tickers}
                st.info(f"Using 1/{desired_portfolio_size} allocation per stock. Remaining cash will be held as cash.")
            
            elif weighting_method_selected == 'equal':
                dollar_amount_per_stock = total_equity_for_rebalance / len(tradable_target_tickers)
                target_dollar_allocations = {ticker: dollar_amount_per_stock for ticker in tradable_target_tickers}
            else: # Market Cap weighting
                # Use `sf1_art_snapshot` for `sharesbas` (as of signal date)
                # And `prices_for_selected_targets` for current prices (as of execution date)
                mcap_calc_data = sf1_art_snapshot.loc[sf1_art_snapshot.index.isin(tradable_target_tickers), ['sharesbas']].copy() # Ensure it's a df
                mcap_calc_data = mcap_calc_data.join(prices_for_selected_targets.rename('current_price'), how='inner') # Join with execution prices
                mcap_calc_data['market_cap_value'] = pd.to_numeric(mcap_calc_data['sharesbas'], errors='coerce') * pd.to_numeric(mcap_calc_data['current_price'], errors='coerce')
                mcap_calc_data.dropna(subset=['market_cap_value'], inplace=True)

                if not mcap_calc_data.empty and mcap_calc_data['market_cap_value'].sum() > 0:
                    total_market_cap_of_targets = mcap_calc_data['market_cap_value'].sum()
                    target_dollar_allocations = {
                        ticker: (mcap_calc_data.loc[ticker, 'market_cap_value'] / total_market_cap_of_targets) * total_equity_for_rebalance
                        for ticker in mcap_calc_data.index # Iterate over tickers present in mcap_calc_data
                    }
                else: # Fallback to equal weight if mcap calculation fails
                    st.warning(f"Market cap weighting failed for {execution_date.date()} (e.g. no valid mcap data). Falling back to equal weighting for tradable targets.")
                    if tradable_target_tickers: # Check again, as mcap_calc_data might be empty
                        dollar_amount_per_stock = total_equity_for_rebalance / len(tradable_target_tickers)
                        target_dollar_allocations = {ticker: dollar_amount_per_stock for ticker in tradable_target_tickers}        # Calculate target shares for the new portfolio
        # First pass: calculate initial shares using floor division (realistic for most brokers)
        fractional_remainders = {}
        total_cost_initial = 0
        
        for ticker, dollar_allocation in target_dollar_allocations.items():
            price_at_trade = prices_for_selected_targets.get(ticker) # Should be valid from `tradable_target_tickers` filter
            if price_at_trade: # Redundant check, but safe
                shares_whole = int(dollar_allocation / price_at_trade)  # Floor division for whole shares
                new_target_portfolio_shares[ticker] = shares_whole
                
                cost_for_whole_shares = shares_whole * price_at_trade
                total_cost_initial += cost_for_whole_shares
                
                # Store fractional remainder for potential additional share allocation
                remainder = dollar_allocation - cost_for_whole_shares
                fractional_remainders[ticker] = remainder
        
        # Second pass: allocate remaining cash to stocks with highest fractional remainders
        remaining_cash_to_allocate = total_equity_for_rebalance - total_cost_initial
        
        # Sort tickers by fractional remainder (descending) to prioritize allocation
        sorted_by_remainder = sorted(fractional_remainders.items(), key=lambda x: x[1], reverse=True)
        
        for ticker, remainder in sorted_by_remainder:
            price_at_trade = prices_for_selected_targets.get(ticker)
            if price_at_trade and remaining_cash_to_allocate >= price_at_trade:
                # Add one more share if we have enough cash
                new_target_portfolio_shares[ticker] += 1
                remaining_cash_to_allocate -= price_at_trade


        # Calculate turnover value (sum of absolute value of trades)
        turnover_monetary_value = 0
        # Sells (or reduction in shares)
        for ticker, old_shares_count in current_portfolio_shares.items():
            price_at_trade_time = prices_at_execution_for_rebal_val.get(ticker) # Price of currently held stock
            if pd.notna(price_at_trade_time) and price_at_trade_time > 0:
                new_target_shares_count = new_target_portfolio_shares.get(ticker, 0) # 0 if ticker is to be sold completely
                if new_target_shares_count < old_shares_count: # If shares reduced or sold
                    turnover_monetary_value += (old_shares_count - new_target_shares_count) * price_at_trade_time
        # Buys (or increase in shares)        for ticker, new_target_shares_count in new_target_portfolio_shares.items():
            price_at_trade_time = prices_for_selected_targets.get(ticker) # Price of target stock (already validated)
            if price_at_trade_time : # Should be true
                old_shares_count = current_portfolio_shares.get(ticker, 0)
                if new_target_shares_count > old_shares_count: # If shares increased or new position
                    turnover_monetary_value += (new_target_shares_count - old_shares_count) * price_at_trade_time

        total_slippage_cost = turnover_monetary_value * slippage_percentage

        # Calculate actual cost of the new target portfolio (with whole shares + optimally allocated remainders)
        cost_of_buying_new_portfolio = 0
        for ticker, shares_to_buy in new_target_portfolio_shares.items():
            price_at_trade_final = prices_for_selected_targets.get(ticker) # Should be valid
            if price_at_trade_final:
                cost_of_buying_new_portfolio += shares_to_buy * price_at_trade_final        # Update portfolio and cash
        # Note: remaining_cash_to_allocate from above represents truly leftover cash 
        # (after optimal allocation considering whole share constraints)
        current_cash = total_equity_for_rebalance - total_slippage_cost - cost_of_buying_new_portfolio
        
        # Debug: Track cash allocation efficiency
        total_invested = cost_of_buying_new_portfolio
        total_available_after_slippage = total_equity_for_rebalance - total_slippage_cost
        investment_efficiency = (total_invested / total_available_after_slippage) * 100 if total_available_after_slippage > 0 else 0
        
        if i == 0:  # Log for first rebalance as example
            st.info(f"Rebalance {i+1} Investment Efficiency: {investment_efficiency:.2f}% "
                   f"(${total_invested:,.0f} invested of ${total_available_after_slippage:,.0f} available, "
                   f"${current_cash:,.0f} remaining cash)")
        
        current_portfolio_shares = new_target_portfolio_shares # This is the new state of the portfolio

        # --- 6. Log Holdings for the period STARTING from `execution_date` ---
        # Calculate actual portfolio value and weights AFTER rebalancing trades and slippage
        final_portfolio_value_this_period = current_cash
        for ticker, shares in current_portfolio_shares.items():
            price_final_val = prices_for_selected_targets.get(ticker, 0) # Use prices of selected targets
            final_portfolio_value_this_period += shares * price_final_val

        actual_weights_post_rebalance = {}
        if final_portfolio_value_this_period > 0:
            for ticker, shares in current_portfolio_shares.items():
                price_final_val = prices_for_selected_targets.get(ticker, 0)
                actual_weights_post_rebalance[ticker] = (shares * price_final_val) / final_portfolio_value_this_period

        portfolio_holdings_log.append({
            'date': execution_date,
            'tickers': list(current_portfolio_shares.keys()),
            'weights': actual_weights_post_rebalance,
            'scores': final_composite_scores.reindex(list(current_portfolio_shares.keys())).to_dict(), # Scores for tickers actually held
            'cash': current_cash,
            'total_value': final_portfolio_value_this_period,
            'dropped_tickers': tickers_dropped_due_to_missing_factors # Log tickers dropped at this rebalance due to factor data
        })        # --- Prepare for Next Iteration ---
        portfolio_value_at_last_rebalance = final_portfolio_value_this_period # This is the new base for next period's return
        date_of_last_rebalance_execution = execution_date
    # --- End of Backtest Loop ---

    # --- Calculate Final Period Return (from last rebalance to end date) ---
    if date_of_last_rebalance_execution is not None and current_portfolio_shares:
        # Calculate portfolio value at the end date using final prices
        final_prices = sep_all[sep_all['date'] == end_date_dt].set_index('ticker')['closeadj']
        if not final_prices.empty:
            final_market_value = 0
            for ticker, shares in current_portfolio_shares.items():
                final_price = final_prices.get(ticker)
                if pd.notna(final_price) and final_price > 0:
                    final_market_value += shares * final_price
            
            final_total_equity = current_cash + final_market_value
            
            if portfolio_value_at_last_rebalance != 0:
                final_period_return = (final_total_equity / portfolio_value_at_last_rebalance) - 1
                strategy_period_returns.loc[end_date_dt] = final_period_return
                
                # Final benchmark return
                if not spy_prices_indexed.empty:
                    try:
                        spy_start_final = spy_prices_indexed.loc[date_of_last_rebalance_execution]
                        spy_end_final = spy_prices_indexed.loc[end_date_dt]
                        if pd.notna(spy_start_final) and pd.notna(spy_end_final) and spy_start_final != 0:
                            benchmark_period_returns.loc[end_date_dt] = (spy_end_final / spy_start_final) - 1
                    except KeyError:
                        pass  # End date not found in SPY prices
    # --- Fix Strategy Period Returns from Holdings Log ---
    # Recalculate period returns from the actual portfolio values in holdings log
    # This ensures consistency between displayed portfolio growth and calculated returns
    if portfolio_holdings_log and len(portfolio_holdings_log) > 1:
        print("DEBUG: Recalculating strategy period returns from holdings log...")
        holdings_df = pd.DataFrame(portfolio_holdings_log)
        holdings_df['date'] = pd.to_datetime(holdings_df['date'])
        holdings_df = holdings_df.set_index('date').sort_index()
        
        # Calculate period returns from total_value column
        corrected_period_returns = holdings_df['total_value'].pct_change().dropna()
        
        print(f"DEBUG: Original strategy_period_returns length: {len(strategy_period_returns)}")
        print(f"DEBUG: Corrected period returns length: {len(corrected_period_returns)}")
        print(f"DEBUG: Original expected annualized: {((1 + strategy_period_returns).cumprod().iloc[-1] ** (365.25 / (strategy_period_returns.index.max() - strategy_period_returns.index.min()).days) - 1) * 100:.2f}%" if not strategy_period_returns.empty else "N/A")
        print(f"DEBUG: Corrected expected annualized: {((1 + corrected_period_returns).cumprod().iloc[-1] ** (365.25 / (corrected_period_returns.index.max() - corrected_period_returns.index.min()).days) - 1) * 100:.2f}%" if not corrected_period_returns.empty else "N/A")
        
        # Replace the strategy period returns with corrected ones
        strategy_period_returns = corrected_period_returns

    # --- Final Metrics Calculation ---
    metrics_strategy_final, cum_returns_strategy_final, drawdowns_strategy_final, \
    metrics_benchmark_final, cum_returns_benchmark_final, drawdowns_benchmark_final = \
        performance_metrics(strategy_period_returns, benchmark_period_returns)

    # Save outputs
    unique_run_id = str(uuid.uuid4())[:8] # Short unique ID for filenames
    if portfolio_holdings_log:
        pd.DataFrame(portfolio_holdings_log).to_csv(os.path.join(OUTPUT_FOLDER, f'holdings_log_{unique_run_id}.csv'), index=False)

    # Flatten dropped_tickers_log from portfolio_holdings_log for a separate CSV
    all_dropped_tickers_for_csv = []
    if portfolio_holdings_log:
        for entry in portfolio_holdings_log:
            if entry.get('dropped_tickers'): # Check if the key exists and has content
                for ticker_name, reason_text in entry['dropped_tickers'].items():
                    all_dropped_tickers_for_csv.append({
                        'rebalance_date': entry['date'], # Date of rebalance when drop occurred
                        'ticker': ticker_name,
                        'reason': reason_text
                    })
    if all_dropped_tickers_for_csv:
        pd.DataFrame(all_dropped_tickers_for_csv).to_csv(os.path.join(OUTPUT_FOLDER, f'dropped_tickers_log_{unique_run_id}.csv'), index=False)

    if processed_actions_log:
        pd.DataFrame(processed_actions_log).to_csv(os.path.join(OUTPUT_FOLDER, f'actions_log_{unique_run_id}.csv'), index=False)

    generated_pdf_path = os.path.join(OUTPUT_FOLDER, f'backtest_report_{unique_run_id}.pdf')
    generate_pdf_report(metrics_strategy_final, cum_returns_strategy_final, drawdowns_strategy_final,
                        metrics_benchmark_final, cum_returns_benchmark_final, drawdowns_benchmark_final,
                        portfolio_holdings_log, processed_actions_log, generated_pdf_path)

    return metrics_strategy_final, cum_returns_strategy_final, drawdowns_strategy_final, \
           portfolio_holdings_log, processed_actions_log, generated_pdf_path, \
           metrics_benchmark_final, cum_returns_benchmark_final, drawdowns_benchmark_final


# ─── Streamlit App ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HamayelResearchEngine - Equity Strategy Backtesting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bloomberg-style Dark Theme CSS with HamayelResearchEngine Branding
st.markdown("""
<style>
    /* Main App Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #40e0d0;
    }
    
    /* Sidebar Dark Theme */
    .css-1d391kg {
        background: linear-gradient(180deg, #000000 0%, #1a1a1a 100%);
        border-right: 2px solid #40e0d0;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #000000 0%, #1a1a1a 50%, #000000 100%);
        padding: 2rem 1rem;
        border-radius: 12px;
        border: 2px solid #40e0d0;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(64, 224, 208, 0.3);
    }
    
    .brand-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #40e0d0;
        text-align: center;
        text-shadow: 0 0 20px rgba(64, 224, 208, 0.5);
        margin-bottom: 0.5rem;
    }
    
    .brand-subtitle {
        font-size: 1.2rem;
        color: #87ceeb;
        text-align: center;
        font-weight: 300;
        margin-bottom: 0.8rem;
    }
    
    .brand-tagline {
        font-size: 0.9rem;
        color: #696969;
        text-align: center;
        font-style: italic;
    }
    
    /* Professional Metrics Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid #40e0d0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(64, 224, 208, 0.2);
    }
    
    .metric-title {
        color: #40e0d0;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value-positive {
        color: #00ff87;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-value-negative {
        color: #ff4757;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-value-neutral {
        color: #40e0d0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Sidebar Styling */
    .css-1lcbmhc {
        background: #000000;
        color: #40e0d0;
    }
    
    .sidebar-header {
        color: #40e0d0;
        font-size: 1.3rem;
        font-weight: 700;
        text-transform: uppercase;
        border-bottom: 2px solid #40e0d0;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Input Styling */
    .stNumberInput > div > div > input {
        background: #1a1a1a;
        color: #40e0d0;
        border: 1px solid #40e0d0;
        border-radius: 4px;
    }
    
    .stSelectbox > div > div > div {
        background: #1a1a1a;
        color: #40e0d0;
        border: 1px solid #40e0d0;
    }
    
    .stMultiSelect > div > div > div {
        background: #1a1a1a;
        color: #40e0d0;
        border: 1px solid #40e0d0;
    }
    
    .stSlider > div > div > div {
        color: #40e0d0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #40e0d0 0%, #20b2aa 100%);
        color: #000000;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 16px rgba(64, 224, 208, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(64, 224, 208, 0.6);
    }
    
    /* Chart Container */
    .chart-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid #40e0d0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(64, 224, 208, 0.3);
    }
    
    .chart-title {
        color: #40e0d0;
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Data Tables */
    .dataframe {
        background: #1a1a1a;
        border: 1px solid #40e0d0;
        border-radius: 8px;
    }
    
    .dataframe th {
        background: #40e0d0;
        color: #000000;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .dataframe td {
        background: #1a1a1a;
        color: #87ceeb;
        border-bottom: 1px solid #2a2a2a;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #1a3d1a 0%, #2a5d2a 100%);
        border: 1px solid #00ff87;
        color: #00ff87;
    }
    
    .stError {
        background: linear-gradient(135deg, #3d1a1a 0%, #5d2a2a 100%);
        border: 1px solid #ff4757;
        color: #ff4757;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #3d3d1a 0%, #5d5d2a 100%);
        border: 1px solid #ffa726;
        color: #ffa726;
    }
    
    /* Professional Section Headers */
    .section-header {
        background: linear-gradient(90deg, #1a1a1a 0%, #40e0d0 50%, #1a1a1a 100%);
        color: #000000;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 4px 16px rgba(64, 224, 208, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Professional Header with HamayelResearchEngine Branding
st.markdown("""
<div class="main-header">
    <div class="brand-title">HamayelResearchEngine</div>
    <div class="brand-subtitle">Institutional-Grade Equity Strategy Backtesting Platform</div>
    <div class="brand-tagline">Powered by Advanced Multi-Factor Models & Quantitative Analytics</div>
</div>
""", unsafe_allow_html=True)

# Professional Sidebar Configuration
st.sidebar.markdown('<div class="sidebar-header">⚙️ Strategy Configuration</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Capital & Timeline Configuration
st.sidebar.markdown("### 💰 Capital & Timeline")
initial_capital_input = st.sidebar.number_input("Initial Capital (USD)", min_value=1000.0, value=1_000_000.0, step=100000.0, format="%f")

# Attempt to get date ranges from tickers.parquet, with fallbacks
min_allowable_date = datetime(1980, 1, 1) # Absolute earliest
max_allowable_date = datetime.now()
try:
    # Only read necessary columns for speed if tickers_meta is large
    tickers_meta_for_dates = load_parquet('tickers.parquet') # Keep it simple, load_parquet handles path
    if not tickers_meta_for_dates.empty and 'firstpricedate' in tickers_meta_for_dates and 'lastpricedate' in tickers_meta_for_dates:
        # Ensure dates are datetime before min/max
        first_prices = pd.to_datetime(tickers_meta_for_dates['firstpricedate'], errors='coerce').dropna()
        last_prices = pd.to_datetime(tickers_meta_for_dates['lastpricedate'], errors='coerce').dropna()
        if not first_prices.empty: min_allowable_date = first_prices.min()
        if not last_prices.empty: max_allowable_date = last_prices.max()
except Exception as e:
    st.sidebar.warning(f"Could not auto-set date range from tickers.parquet: {e}. Using defaults.")

# Ensure default start/end dates are within the determined min/max allowable dates
default_start_date = max(min_allowable_date, datetime(2010,1,1)) # Sensible default start
default_end_date = max_allowable_date

start_date_selected = st.sidebar.date_input("Start Date", default_start_date, min_value=min_allowable_date, max_value=max_allowable_date)
end_date_selected = st.sidebar.date_input("End Date", default_end_date, min_value=min_allowable_date, max_value=max_allowable_date)
st.sidebar.markdown("---")


# Universe Selection
st.sidebar.markdown("### 🎯 Investment Universe")
tickers_metadata_full = load_parquet('tickers.parquet') # Load once for categories
available_categories_list = []
if not tickers_metadata_full.empty and 'category' in tickers_metadata_full.columns:
    available_categories_list = sorted(list(tickers_metadata_full['category'].dropna().unique()))

default_categories = ['Domestic'] if 'Domestic' in available_categories_list else (available_categories_list[:1] if available_categories_list else [])
selected_categories_list = st.sidebar.multiselect("Issuer Categories", options=available_categories_list, default=default_categories)

dollar_volume_threshold_input = st.sidebar.number_input("Min. Avg Daily Dollar Volume (USD)", min_value=0.0, value=1000000.0, step=100000.0, format="%f", help="e.g., 1000000 for $1M")

# Professional Universe Filters Section
st.sidebar.markdown("### 🔍 Advanced Universe Filters")

# Initialize session state for filters if not exists
if 'technical_filters' not in st.session_state:
    st.session_state.technical_filters = []
if 'fundamental_filters' not in st.session_state:
    st.session_state.fundamental_filters = []

# Multi-Filter Configuration
st.sidebar.markdown("**📊 Technical Filters (EMA/SMA)**")

# Technical Filters Management
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    st.sidebar.markdown(f"Active Technical Filters: {len(st.session_state.technical_filters)}")
with col2:
    if st.sidebar.button("➕", key="add_tech", help="Add Technical Filter"):
        st.session_state.technical_filters.append({
            'ma_type': 'SMA',
            'ma_period': 50,
            'direction': 'above',
            'use_threshold': False,
            'threshold': 0.0
        })

# Display and configure each technical filter
technical_filters_to_remove = []
for i, tech_filter in enumerate(st.session_state.technical_filters):
    with st.sidebar.expander(f"Technical Filter {i+1}: {tech_filter['ma_type']}{tech_filter['ma_period']}", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            tech_filter['ma_type'] = st.selectbox("MA Type", ["SMA", "EMA"], 
                                                 index=0 if tech_filter['ma_type'] == 'SMA' else 1, 
                                                 key=f"tech_type_{i}")
        with col2:
            tech_filter['ma_period'] = st.number_input("Period", min_value=5, max_value=200, 
                                                      value=tech_filter['ma_period'], step=1, 
                                                      key=f"tech_period_{i}")
        
        tech_filter['direction'] = st.selectbox("Price Direction", ["above", "below"], 
                                               index=0 if tech_filter['direction'] == 'above' else 1,
                                               key=f"tech_dir_{i}")
        
        tech_filter['use_threshold'] = st.checkbox("Use Threshold", value=tech_filter['use_threshold'], 
                                                  key=f"tech_thresh_enable_{i}")
        
        if tech_filter['use_threshold']:
            tech_filter['threshold'] = st.number_input("Threshold (%)", min_value=0.0, 
                                                      value=tech_filter['threshold'], step=0.01, 
                                                      key=f"tech_thresh_val_{i}")
        
        if st.button("🗑️ Remove", key=f"remove_tech_{i}"):
            technical_filters_to_remove.append(i)

# Remove marked technical filters
for i in reversed(technical_filters_to_remove):
    st.session_state.technical_filters.pop(i)

# Technical Filter Logic
if len(st.session_state.technical_filters) > 1:
    technical_filter_logic = st.sidebar.selectbox("Technical Filter Logic", ["AND", "OR"], 
                                                 index=0, key="tech_logic", 
                                                 help="AND: Stock must pass ALL technical filters, OR: Stock must pass ANY technical filter")
else:
    technical_filter_logic = "AND"

st.sidebar.markdown("---")
st.sidebar.markdown("**🏢 Fundamental Filters**")

# Load factor mappings for user-friendly names (moved outside conditional)
factor_mappings = load_factor_mappings()

# Get available factors from SF1 schema (moved outside conditional)
sf1_factor_options = []
try:
    sf1_schema = pq.read_schema(os.path.join(PARQUET_FOLDER, 'sf1.parquet'))
    sf1_cols = sf1_schema.names
    # Exclude non-numeric/non-factor columns
    excluded_sf1_cols = ['ticker', 'dimension', 'calendardate', 'datekey', 'reportperiod', 'lastupdated', 'fiscalperiod', 'compname', 'siccode', 'currency']
    sf1_factor_options = sorted([col for col in sf1_cols if col not in excluded_sf1_cols])
except Exception as e:
    st.sidebar.warning(f"Could not read SF1 schema for fundamental filter: {e}")
    sf1_factor_options = ['roe', 'roa', 'debt', 'revenue', 'netinc', 'assets']

# Create options with user-friendly names (moved outside conditional)
factor_display_options = []
factor_code_mapping = {}
display_to_code = {}
factor_code_to_display = {}

# Define custom/derived factors
custom_factors = {
    'momentum': 'Price Momentum',
    'gp/assets': 'Gross Profit to Assets'
}

for factor_code in sf1_factor_options:
    if factor_code in factor_mappings:
        display_name = f"{factor_mappings[factor_code]} ({factor_code})"
    else:
        display_name = factor_code
    factor_display_options.append(display_name)
    factor_code_mapping[display_name] = factor_code
    display_to_code[display_name] = factor_code
    factor_code_to_display[factor_code] = display_name

# Add custom/derived factors
for factor_code, description in custom_factors.items():
    display_name = f"{description} ({factor_code})"
    factor_display_options.append(display_name)
    factor_code_mapping[factor_code] = display_name
    display_to_code[display_name] = factor_code
    factor_code_to_display[factor_code] = display_name

# Fundamental Filters Management
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    st.sidebar.markdown(f"Active Fundamental Filters: {len(st.session_state.fundamental_filters)}")
with col2:
    if st.sidebar.button("➕", key="add_fund", help="Add Fundamental Filter"):
        st.session_state.fundamental_filters.append({
            'factor_code': 'roe',
            'direction': 'above',
            'threshold': 0.0,
            'growth_type': 'Actual',
            'dimension': 'ART'  # Default to Trailing Twelve Months (As Reported)
        })

# Show threshold format guidance
if st.session_state.fundamental_filters:
    st.sidebar.info("💡 **Threshold Format Guide:**\n\n• **Actual**: Use factor's native units\n• **Growth**: Use decimal format (0.20 = 20% growth)")

# Display and configure each fundamental filter
fundamental_filters_to_remove = []
for i, fund_filter in enumerate(st.session_state.fundamental_filters):
    factor_display = factor_code_to_display.get(fund_filter['factor_code'], fund_filter['factor_code'])
    with st.sidebar.expander(f"Fundamental Filter {i+1}: {factor_display}", expanded=False):
        # Factor selection
        if fund_filter['factor_code'] in display_to_code.values():
            # Find the display name for current factor
            current_display = next((k for k, v in display_to_code.items() if v == fund_filter['factor_code']), fund_filter['factor_code'])
        else:
            current_display = fund_filter['factor_code']
        
        selected_factor_display = st.selectbox("Factor", 
                                             options=sorted(list(display_to_code.keys())), 
                                             index=sorted(list(display_to_code.keys())).index(current_display) if current_display in display_to_code.keys() else 0,
                                             key=f"fund_factor_{i}")
        fund_filter['factor_code'] = display_to_code[selected_factor_display]
        
        # Direction and threshold
        fund_filter['direction'] = st.selectbox("Direction", ["above", "below"], 
                                               index=0 if fund_filter['direction'] == 'above' else 1,
                                               key=f"fund_dir_{i}")
          # Dynamic help text based on growth type
        if fund_filter['growth_type'] in ['Year-over-Year (YoY)', 'Quarter-over-Quarter (QoQ)']:
            threshold_help = "Growth threshold in decimal format. Examples: 0.20 = 20% growth, -0.10 = -10% decline, 0.05 = 5% growth"
            threshold_step = 0.01
        else:
            threshold_help = "Actual value threshold. Use the same units as the fundamental factor."
            threshold_step = 0.1
            
        fund_filter['threshold'] = st.number_input("Threshold", 
                                                  value=fund_filter['threshold'], 
                                                  step=threshold_step, 
                                                  key=f"fund_thresh_{i}",
                                                  help=threshold_help)
          # Growth type selection
        growth_options = ['Actual', 'Year-over-Year (YoY)', 'Quarter-over-Year (QoQ)']
        current_growth_idx = 0
        if fund_filter['growth_type'] in growth_options:
            current_growth_idx = growth_options.index(fund_filter['growth_type'])
        
        fund_filter['growth_type'] = st.selectbox("Growth Type", 
                                                 growth_options,
                                                 index=current_growth_idx,
                                                 key=f"fund_growth_{i}",
                                                 help=f"Choose calculation method for {factor_display}: Actual values, YoY growth, or QoQ growth")
        
        # Dimension selection
        current_dimension_idx = 4  # Default to ART (index 4)
        if 'dimension' in fund_filter:
            # Find the formatted option that contains this dimension code
            for idx, formatted_option in enumerate(FORMATTED_DIMENSION_OPTIONS):
                if formatted_option.startswith(fund_filter['dimension'] + ' '):
                    current_dimension_idx = idx
                    break
        
        formatted_dimension = st.selectbox(
            "SF1 Dimension", 
            FORMATTED_DIMENSION_OPTIONS,
            index=current_dimension_idx, 
            key=f"fund_dimension_{i}",
            help="Select which financial statement dimension to use"
        )
        fund_filter['dimension'] = extract_dimension_code(formatted_dimension)
        
        if st.button("🗑️ Remove", key=f"remove_fund_{i}"):
            fundamental_filters_to_remove.append(i)

# Remove marked fundamental filters
for i in reversed(fundamental_filters_to_remove):
    st.session_state.fundamental_filters.pop(i)

# Fundamental Filter Logic
if len(st.session_state.fundamental_filters) > 1:
    fundamental_filter_logic = st.sidebar.selectbox("Fundamental Filter Logic", ["AND", "OR"], 
                                                   index=0, key="fund_logic", 
                                                   help="AND: Stock must pass ALL fundamental filters, OR: Stock must pass ANY fundamental filter")
else:
    fundamental_filter_logic = "AND"

# Overall Filter Logic (when both technical and fundamental filters exist)
if len(st.session_state.technical_filters) > 0 and len(st.session_state.fundamental_filters) > 0:
    st.sidebar.markdown("**🔗 Overall Filter Logic**")
    overall_filter_logic = st.sidebar.selectbox("Technical + Fundamental Logic", ["AND", "OR"], 
                                               index=0, key="overall_logic", 
                                               help="AND: Stock must pass BOTH technical AND fundamental filters, OR: Stock must pass EITHER technical OR fundamental filters")
else:
    overall_filter_logic = "AND"

st.sidebar.markdown("---")

# Factor Configuration Section
st.sidebar.markdown("### 📊 Multi-Factor Model Configuration")

# Sensible default factors (convert to display names)
default_factor_codes = ['momentum', 'roe', 'gp/assets']
default_factor_selection = [factor_code_to_display.get(code, code) for code in default_factor_codes if code in factor_code_to_display]

selected_factors_display = st.sidebar.multiselect("Select Factors for Model", options=factor_display_options, default=default_factor_selection)
selected_factors_list = [display_to_code[display_name] for display_name in selected_factors_display]

input_weights = {}
input_directions = {}
input_growth_types = {}  # Add growth types storage
input_dimensions = {}    # Add dimensions storage
if selected_factors_list:
    st.sidebar.markdown("**📊 Factor Weights, Directions & Growth:**")
    # Equal default weight initially
    equal_default_weight = round(1.0 / len(selected_factors_list), 2) if selected_factors_list else 0.0
    
    for factor_code in selected_factors_list:
        factor_display_name = factor_code_to_display.get(factor_code, factor_code)
        
        # Use a shorter version for the slider label
        short_display = factor_display_name.split('(')[0].strip()[:25] + "..." if len(factor_display_name) > 25 else factor_display_name
        
        input_weights[factor_code] = st.sidebar.slider(
            f"Weight: {short_display}", 
            0.0, 1.0, equal_default_weight, 0.01, 
            key=f"weight_{factor_code}",
            help=f"Weight for {factor_display_name}"
        )
        
        # Default direction (heuristic)
        is_lower_better = any(keyword in factor_code.lower() for keyword in ['debt', 'pe', 'pb', 'ps', 'evebit', 'price', 'payables']) and \
                          not any(keyword in factor_code.lower() for keyword in ['margin', 'return', 'profit', 'growth', 'fcf', 'eps', 'dps', 'assets', 'equity', 'revenue', 'income'])
        default_dir_idx = 1 if is_lower_better else 0
        
        input_directions[factor_code] = st.sidebar.selectbox(
            f"Direction: {short_display}", 
            ['Higher', 'Lower'], 
            index=default_dir_idx, 
            key=f"direction_{factor_code}",
            help=f"Whether higher or lower values of {factor_display_name} are preferred"
        ).lower()
          # Add Growth Type Selection for fundamental factors (skip momentum)
        if factor_code != 'momentum':
            input_growth_types[factor_code] = st.sidebar.selectbox(
                f"Growth Type: {short_display}",
                ['Actual', 'Year-over-Year (YoY)', 'Quarter-over-Year (QoQ)'],
                index=0,
                key=f"growth_{factor_code}",
                help=f"Choose calculation method for {factor_display_name}: Actual values, YoY growth, or QoQ growth"
            )            # Add dimension selection for fundamental factors with improved labels
            formatted_dimension = st.sidebar.selectbox(
                f"Dimension: {short_display}",
                FORMATTED_DIMENSION_OPTIONS,
                index=4,  # Default to 'ART'
                key=f"dimension_{factor_code}",
                help=f"Select financial statement dimension for {factor_display_name}"
            )
            input_dimensions[factor_code] = extract_dimension_code(formatted_dimension)
        else:
            input_growth_types[factor_code] = 'Actual'  # Momentum always uses actual
            input_dimensions[factor_code] = 'ART'  # Default for momentum

    # Professional weight sum validation
    weight_sum = sum(input_weights.values())
    if selected_factors_list and abs(weight_sum - 1.0) > 0.01:
        if weight_sum < 0.99:
            st.sidebar.warning(f"⚠️ Factor weights sum to {weight_sum:.2f} (under-weighted)")
        elif weight_sum > 1.01:
            st.sidebar.error(f"🚨 Factor weights sum to {weight_sum:.2f} (over-weighted)")
        else:
            st.sidebar.info(f"✅ Factor weights sum to {weight_sum:.2f}")
    elif selected_factors_list:
        st.sidebar.success(f"✅ Factor weights properly balanced ({weight_sum:.2f})")
else:
    st.sidebar.info("📊 Select factors above to configure weights and directions.")
    # Provide empty defaults when no factors selected
    input_growth_types = {}

# Professional Momentum Configuration
if 'momentum' in selected_factors_list:
    st.sidebar.markdown("### ⚡ Advanced Momentum Settings")
    
    col_mom1, col_mom2 = st.sidebar.columns(2)
    with col_mom1:
        momentum_lookback_input_val = st.sidebar.slider("Lookback Period (Weeks)", 1, 52, 26, key="mom_lookback", help="Primary momentum calculation period")
    with col_mom2:
        use_skip_momentum_input_val = st.sidebar.checkbox("Skip Momentum", value=True, key="mom_skip", help="Subtract recent 4-week performance to reduce noise")
    
    if use_skip_momentum_input_val:
        st.sidebar.info("💡 Skip momentum removes recent 4-week performance to avoid reversal effects")
else:
    # Provide defaults even if not shown, so params object is complete
    momentum_lookback_input_val = 26
    use_skip_momentum_input_val = False
st.sidebar.markdown("---")


# Professional Portfolio Construction
st.sidebar.markdown("### 📈 Portfolio Construction & Risk Management")
portfolio_size_type_selected = st.sidebar.radio("Portfolio Size By:", ['Number of Stocks', 'Percentage of Universe'], index=0, key="size_type")
if portfolio_size_type_selected == 'Number of Stocks':
    portfolio_size_input_val = st.sidebar.number_input("Number of Stocks in Portfolio", min_value=1, value=30, step=1, key="num_stocks")
else: # Percentage
    portfolio_size_input_val = st.sidebar.slider("Top % of Universe to Select", 0.01, 0.50, 0.10, 0.01, format="%.2f", key="pct_stocks", help="e.g., 0.10 for top 10%")

weighting_method_input_val = st.sidebar.selectbox("Stock Weighting Method", ['Equal', 'Market Cap'], index=0, key="weight_method").lower().replace(" ", "") # 'marketcap'
rebalance_frequency_input_val = st.sidebar.selectbox("Rebalance Frequency", ['Monthly', 'Weekly'], index=0, key="rebal_freq").lower()
slippage_input_val = st.sidebar.number_input("Slippage (% per trade, one-way)", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.3f", key="slippage_pct", help="e.g., 0.05% means 0.0005 cost factor")

# Small Universe Handling
st.sidebar.markdown("**Small Universe Handling**")
small_universe_handling = st.sidebar.selectbox(
    "When filtered universe < desired portfolio size:",
    ['Invest in all available stocks', 'Use 1/N allocation + cash', 'Go completely cash'],
    index=0,
    key="small_universe_handling",
    help="Choose how to handle periods when fewer stocks pass filters than desired portfolio size"
)
st.sidebar.markdown("---")


# Professional Strategy Execution
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Strategy Execution")

if st.sidebar.button("🎯 Execute Backtest", type="primary", use_container_width=True, 
                    help="Launch comprehensive strategy backtesting with current parameters"):    # Validations before running
    form_valid = True
    if not selected_factors_list:
        st.sidebar.error("Please select at least one factor.")
        form_valid = False
    if selected_factors_list and abs(sum(input_weights.values()) - 1.0) > 0.01:
        st.sidebar.warning(f"Factor weights sum to {sum(input_weights.values()):.2f}. This might not be intended. Continuing anyway...")
        # form_valid = False # Optional: make it a hard stop    if pd.to_datetime(start_date_selected) >= pd.to_datetime(end_date_selected):
        st.sidebar.error("Start Date must be before End Date.")
        form_valid = False
    if not selected_categories_list:
        st.sidebar.error("Please select at least one Issuer Category.")
        form_valid = False
    
    if form_valid:
        # Consolidate params for backtest function
        backtest_params = {
            'categories': selected_categories_list,
            'dolvol_threshold': dollar_volume_threshold_input,
            'factors': selected_factors_list,
            'weights': input_weights,
            'directions': input_directions,
            'growth_types': input_growth_types,  # Add growth types
            'dimensions': input_dimensions,  # Add dimensions for each factor
            'momentum_lookback': momentum_lookback_input_val,
            'use_skip_momentum': use_skip_momentum_input_val,
            'portfolio_size': portfolio_size_input_val,
            'portfolio_size_type': portfolio_size_type_selected,
            'weighting': weighting_method_input_val,
            'rebalance_freq': rebalance_frequency_input_val,            'slippage': slippage_input_val, # Pass as %, will be /100 in backtest
            'initial_capital': initial_capital_input,
            'small_universe_handling': small_universe_handling,
            # Multi-Filter System
            'technical_filters': st.session_state.technical_filters,
            'fundamental_filters': st.session_state.fundamental_filters,
            'technical_filter_logic': technical_filter_logic,
            'fundamental_filter_logic': fundamental_filter_logic,
            'overall_filter_logic': overall_filter_logic
        }

        with st.spinner("⚡ HamayelResearchEngine Computing... \n\n🔄 Loading market data and fundamentals \n📊 Applying multi-factor model \n💼 Optimizing portfolio allocation \n📈 Calculating risk metrics"):
            try:
                # Professional progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Initializing backtesting engine...")
                progress_bar.progress(10)
                
                status_text.text("📊 Loading fundamental and price data...")
                progress_bar.progress(30)
                
                # Call the backtest function
                results_tuple = run_backtest(
                    backtest_params,
                    pd.to_datetime(start_date_selected),
                    pd.to_datetime(end_date_selected)
                )
                
                status_text.text("💼 Processing portfolio allocations...")
                progress_bar.progress(70)
                
                status_text.text("📈 Computing performance metrics...")
                progress_bar.progress(90)
                
                # Unpack results
                (metrics_strat_res, cum_ret_strat_res, dd_strat_res,
                 holdings_log_list, actions_log_list, pdf_file_path,
                 metrics_bench_res, cum_ret_bench_res, dd_bench_res) = results_tuple

                status_text.text("✅ Backtest completed successfully!")
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

            except Exception as e:
                st.error(f"""
                🛑 **BACKTESTING ENGINE ERROR**
                
                **Error Type**: {type(e).__name__}
                **Description**: {str(e)}
                
                **Troubleshooting Steps**:
                1. Verify all required data files are present
                2. Check date range has sufficient data coverage
                3. Ensure selected factors are available in SF1 data
                4. Review parameter combinations for validity
                
                **Support**: Contact HamayelResearchEngine technical support for assistance
                """)
                st.exception(e)
                # Initialize result variables to prevent errors in the display section below
                metrics_strat_res, cum_ret_strat_res, dd_strat_res = {}, pd.Series(dtype=float), pd.Series(dtype=float)
                holdings_log_list, actions_log_list, pdf_file_path = [], [], ""
                metrics_bench_res, cum_ret_bench_res, dd_bench_res = {}, pd.Series(dtype=float), pd.Series(dtype=float)# --- Professional Bloomberg-Style Results Display ---
        st.markdown('<div class="section-header">📊 STRATEGY PERFORMANCE ANALYSIS</div>', unsafe_allow_html=True)

        if not metrics_strat_res: # Check if backtest produced any metrics
            st.error("🛑 Backtest execution failed. Please review parameters and data availability.")
        else:
            # === Professional Performance Summary Cards ===
            st.markdown("#### 🎯 Executive Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ann_ret = metrics_strat_res.get('Annualized Return', 0)
                color_class = "positive" if ann_ret > 0 else ("negative" if ann_ret < 0 else "neutral")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Annual Return</div>
                    <div class="metric-value-{color_class}">{ann_ret:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                sharpe = metrics_strat_res.get('Sharpe Ratio', 0)
                color_class = "positive" if sharpe > 1 else ("neutral" if sharpe > 0 else "negative")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Sharpe Ratio</div>
                    <div class="metric-value-{color_class}">{sharpe:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                max_dd = metrics_strat_res.get('Maximum Drawdown', 0)
                color_class = "positive" if max_dd > -5 else ("neutral" if max_dd > -15 else "negative")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Max Drawdown</div>
                    <div class="metric-value-{color_class}">{max_dd:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                alpha = metrics_strat_res.get('Alpha', 0)
                color_class = "positive" if alpha > 0 else ("negative" if alpha < 0 else "neutral")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Alpha vs SPY</div>
                    <div class="metric-value-{color_class}">{alpha:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # === Professional Performance Chart ===
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Strategy vs Benchmark Performance</div>', unsafe_allow_html=True)
            
            fig_performance, ax_performance = plt.subplots(figsize=(14, 8))
            ax_performance.set_facecolor('#1a1a1a')
            fig_performance.patch.set_facecolor('#0a0a0a')
            
            plot_drawn = False
            if cum_ret_strat_res is not None and not cum_ret_strat_res.empty:
                # Convert to percentage returns for better readability
                strategy_pct = (cum_ret_strat_res - 1) * 100
                strategy_pct.plot(ax=ax_performance, label='HamayelResearchEngine Strategy', 
                                color='#40e0d0', linewidth=3, alpha=0.9)
                plot_drawn = True
                
            if cum_ret_bench_res is not None and not cum_ret_bench_res.empty:
                benchmark_pct = (cum_ret_bench_res - 1) * 100
                benchmark_pct.plot(ax=ax_performance, label='SPY Benchmark', 
                                 color='#ff6b6b', linewidth=2, linestyle='--', alpha=0.8)
                plot_drawn = True
                
            if plot_drawn:
                ax_performance.set_title("Cumulative Performance Comparison", 
                                       fontsize=16, color='#40e0d0', fontweight='bold', pad=20)
                ax_performance.set_xlabel("Date", fontsize=12, color='#87ceeb')
                ax_performance.set_ylabel("Cumulative Return (%)", fontsize=12, color='#87ceeb')
                ax_performance.grid(True, linestyle=':', alpha=0.3, color='#40e0d0')
                ax_performance.legend(fontsize='11', loc='upper left', 
                                    fancybox=True, shadow=True, facecolor='#1a1a1a', 
                                    edgecolor='#40e0d0', labelcolor='#87ceeb')
                
                # Style the axes
                ax_performance.tick_params(colors='#87ceeb', labelsize=10)
                ax_performance.spines['bottom'].set_color('#40e0d0')
                ax_performance.spines['top'].set_color('#40e0d0')
                ax_performance.spines['right'].set_color('#40e0d0')
                ax_performance.spines['left'].set_color('#40e0d0')
                
                plt.tight_layout()
                st.pyplot(fig_performance)
            else:
                st.warning("Performance data not available for charting.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # === Comprehensive Performance Metrics Table ===
            st.markdown('<div class="section-header">📈 DETAILED PERFORMANCE METRICS</div>', unsafe_allow_html=True)
            
            col_strat, col_bench = st.columns(2)
            
            with col_strat:
                st.markdown("#### 🎯 Strategy Metrics")
                strategy_metrics_df = pd.DataFrame([
                    {"Metric": "Annual Return", "Value": f"{metrics_strat_res.get('Annualized Return', 0):.2f}%"},
                    {"Metric": "Annual Volatility", "Value": f"{metrics_strat_res.get('Annualized Volatility', 0):.2f}%"},
                    {"Metric": "Sharpe Ratio", "Value": f"{metrics_strat_res.get('Sharpe Ratio', 0):.3f}"},
                    {"Metric": "Sortino Ratio", "Value": f"{metrics_strat_res.get('Sortino Ratio', 0):.3f}"},
                    {"Metric": "Maximum Drawdown", "Value": f"{metrics_strat_res.get('Maximum Drawdown', 0):.2f}%"},
                    {"Metric": "Calmar Ratio", "Value": f"{metrics_strat_res.get('Calmar Ratio', 0):.3f}"},
                    {"Metric": "Alpha vs SPY", "Value": f"{metrics_strat_res.get('Alpha', 0):.2f}%"},
                    {"Metric": "Beta vs SPY", "Value": f"{metrics_strat_res.get('Beta', 0):.3f}"},
                ])
                st.dataframe(strategy_metrics_df.set_index('Metric'), use_container_width=True)

            with col_bench:
                st.markdown("#### 📊 Benchmark (SPY) Metrics")
                benchmark_metrics_df = pd.DataFrame([
                    {"Metric": "Annual Return", "Value": f"{metrics_bench_res.get('Annualized Return', 0):.2f}%"},
                    {"Metric": "Annual Volatility", "Value": f"{metrics_bench_res.get('Annualized Volatility', 0):.2f}%"},
                    {"Metric": "Sharpe Ratio", "Value": f"{metrics_bench_res.get('Sharpe Ratio', 0):.3f}"},
                    {"Metric": "Sortino Ratio", "Value": f"{metrics_bench_res.get('Sortino Ratio', 0):.3f}"},
                    {"Metric": "Maximum Drawdown", "Value": f"{metrics_bench_res.get('Maximum Drawdown', 0):.2f}%"},
                    {"Metric": "Calmar Ratio", "Value": f"{metrics_bench_res.get('Calmar Ratio', 0):.3f}"},
                    {"Metric": "Alpha (N/A)", "Value": "0.00%"},
                    {"Metric": "Beta (By Definition)", "Value": "1.000"},
                ])
                st.dataframe(benchmark_metrics_df.set_index('Metric'), use_container_width=True)

            # === Risk Analysis Section ===
            st.markdown('<div class="section-header">⚠️ RISK ANALYSIS & DRAWDOWN PROFILE</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Drawdown Analysis</div>', unsafe_allow_html=True)
            
            fig_dd, ax_dd = plt.subplots(figsize=(14, 6))
            ax_dd.set_facecolor('#1a1a1a')
            fig_dd.patch.set_facecolor('#0a0a0a')
            
            if dd_strat_res is not None and not dd_strat_res.empty:
                drawdown_pct = dd_strat_res * 100
                drawdown_pct.plot(ax=ax_dd, color='#ff4757', linewidth=2, alpha=0.8)
                ax_dd.fill_between(drawdown_pct.index, drawdown_pct.values, 0, 
                                 color='#ff4757', alpha=0.3)
                
                ax_dd.set_title("Portfolio Drawdown Profile", fontsize=16, color='#40e0d0', 
                              fontweight='bold', pad=20)
                ax_dd.set_xlabel("Date", fontsize=12, color='#87ceeb')
                ax_dd.set_ylabel("Drawdown (%)", fontsize=12, color='#87ceeb')
                ax_dd.grid(True, linestyle=':', alpha=0.3, color='#40e0d0')
                ax_dd.axhline(y=0, color='#40e0d0', linestyle='-', alpha=0.5)
                
                # Style the axes
                ax_dd.tick_params(colors='#87ceeb', labelsize=10)
                ax_dd.spines['bottom'].set_color('#40e0d0')
                ax_dd.spines['top'].set_color('#40e0d0')
                ax_dd.spines['right'].set_color('#40e0d0')
                ax_dd.spines['left'].set_color('#40e0d0')
                
                plt.tight_layout()
                st.pyplot(fig_dd)
            else:
                st.warning("Drawdown data not available for analysis.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # === Portfolio Holdings Analysis ===
            st.markdown('<div class="section-header">💼 PORTFOLIO COMPOSITION & REBALANCE HISTORY</div>', unsafe_allow_html=True)
            
            if holdings_log_list:
                # Enhanced Holdings Display with Professional Formatting
                st.markdown("#### 📋 Recent Portfolio Rebalances (Last 10)")
                
                holdings_df_display = pd.DataFrame(holdings_log_list).tail(10).copy()
                
                # Enhanced formatting function
                def format_professional_holdings(row):
                    tickers_list = row.get('tickers', [])
                    weights_dict = row.get('weights', {})
                    scores_dict = row.get('scores', {})
                    
                    if not tickers_list:
                        return "No holdings"
                    
                    # Create detailed holdings summary
                    holdings_details = []
                    for i, ticker in enumerate(tickers_list[:5]):  # Show top 5
                        weight = weights_dict.get(ticker, 0)
                        score = scores_dict.get(ticker, np.nan)
                        score_str = f"{score:.2f}" if not pd.isna(score) else "N/A"
                        holdings_details.append(f"{ticker}({weight:.1%},sc:{score_str})")
                    
                    result = " | ".join(holdings_details)
                    if len(tickers_list) > 5:
                        result += f" | +{len(tickers_list) - 5} more"
                    
                    return result
                
                # Enhanced dataframe with professional metrics
                holdings_df_display['Date'] = pd.to_datetime(holdings_df_display['date']).dt.strftime('%Y-%m-%d')
                holdings_df_display['Portfolio Value'] = holdings_df_display['total_value'].apply(lambda x: f"${x:,.0f}")
                holdings_df_display['Cash Position'] = holdings_df_display['cash'].apply(lambda x: f"${x:,.0f}")
                holdings_df_display['# Positions'] = holdings_df_display['tickers'].apply(len)
                holdings_df_display['Top Holdings (Weight, Z-Score)'] = holdings_df_display.apply(format_professional_holdings, axis=1)
                
                display_cols = ['Date', 'Portfolio Value', 'Cash Position', '# Positions', 'Top Holdings (Weight, Z-Score)']
                st.dataframe(holdings_df_display[display_cols], use_container_width=True, height=400)
                
                # Portfolio Statistics Summary
                st.markdown("#### 📊 Portfolio Statistics Summary")
                total_rebalances = len(holdings_log_list)
                avg_portfolio_size = np.mean([len(h.get('tickers', [])) for h in holdings_log_list])
                avg_cash_pct = np.mean([h.get('cash', 0) / h.get('total_value', 1) for h in holdings_log_list if h.get('total_value', 0) > 0]) * 100
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Rebalances</div>
                        <div class="metric-value-neutral">{total_rebalances}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stats_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Avg Portfolio Size</div>
                        <div class="metric-value-neutral">{avg_portfolio_size:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stats_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Avg Cash %</div>
                        <div class="metric-value-neutral">{avg_cash_pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No portfolio holdings data available for analysis.")

            # === Corporate Actions Impact Analysis ===
            st.markdown('<div class="section-header">🏢 CORPORATE ACTIONS IMPACT</div>', unsafe_allow_html=True)
            
            if actions_log_list:
                st.markdown("#### 📋 Recent Corporate Actions (Last 10)")
                actions_df_display = pd.DataFrame(actions_log_list).tail(10).copy()
                actions_df_display['Date'] = pd.to_datetime(actions_df_display['date']).dt.strftime('%Y-%m-%d')
                actions_df_display['Action Type'] = actions_df_display['action'].str.title()
                actions_df_display['Ticker'] = actions_df_display['ticker']
                actions_df_display['Impact Description'] = actions_df_display['outcome']
                
                display_cols = ['Date', 'Action Type', 'Ticker', 'Impact Description'
                ]
                st.dataframe(actions_df_display[display_cols], use_container_width=True, height=300)
                
                # Corporate Actions Summary
                action_counts = pd.Series([a['action'] for a in actions_log_list]).value_counts()
                st.markdown("#### 📊 Corporate Actions Summary")
                
                if len(action_counts) > 0:
                    action_summary_cols = st.columns(min(4, len(action_counts)))
                    for i, (action_type, count) in enumerate(action_counts.head(4).items()):
                        with action_summary_cols[i]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">{action_type.title()}</div>
                                <div class="metric-value-neutral">{count}</div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No corporate actions affected the portfolio during the backtest period.")

            # === Professional Download Section ===
            st.markdown('<div class="section-header">📥 EXPORT & DOCUMENTATION</div>', unsafe_allow_html=True)
            
            if pdf_file_path and os.path.exists(pdf_file_path):
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    with open(pdf_file_path, "rb") as f_pdf_download:
                        st.download_button(
                            label="📄 Download Comprehensive PDF Report",
                            data=f_pdf_download,
                            file_name=f"HamayelResearchEngine_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                
                with col_download2:
                    # Create a strategy summary text for download
                    strategy_summary = f"""
HamayelResearchEngine Strategy Backtest Summary
{'='*50}
Backtest Period: {start_date_selected} to {end_date_selected}
Initial Capital: ${initial_capital_input:,.0f}

Key Performance Metrics:
- Annual Return: {metrics_strat_res.get('Annualized Return', 0):.2f}%
- Annual Volatility: {metrics_strat_res.get('Annualized Volatility', 0):.2f}%
- Sharpe Ratio: {metrics_strat_res.get('Sharpe Ratio', 0):.3f}
- Maximum Drawdown: {metrics_strat_res.get('Maximum Drawdown', 0):.2f}%
- Alpha vs SPY: {metrics_strat_res.get('Alpha', 0):.2f}%

Strategy Configuration:
- Factors: {', '.join(selected_factors_list)}
- Universe: {', '.join(selected_categories_list)}
- Portfolio Size: {portfolio_size_input_val}
- Rebalance Frequency: {rebalance_frequency_input_val}
- Dollar Volume Threshold: ${dollar_volume_threshold_input:,.0f}

Generated by HamayelResearchEngine on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    
                    st.download_button(
                        label="📊 Download Strategy Summary",
                        data=strategy_summary,
                        file_name=f"HamayelResearchEngine_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            else:
                st.error("Report generation failed. Please check system logs for detailed error information.")

# ─── Professional Application Launch ───────────────────────────────────────────
if __name__ == "__main__":
    # Professional application status and data validation
    st.sidebar.markdown("### 🔧 System Status")
    
    # Data validation with professional status indicators
    if not os.path.exists(PARQUET_FOLDER) or not os.listdir(PARQUET_FOLDER):
        st.sidebar.error("🔴 DATA UNAVAILABLE")
        st.error(f"""
        **CRITICAL SYSTEM ERROR**: Data repository not found
        
        Expected location: `{PARQUET_FOLDER}`
        
        **Resolution Required**: 
        1. Verify data folder exists
        2. Ensure Parquet files are properly converted
        3. Check file permissions
        """)
    elif not os.path.exists(os.path.join(PARQUET_FOLDER, 'tickers.parquet')):
        st.sidebar.error("🔴 TICKERS DATA MISSING")
        st.error("**CRITICAL**: `tickers.parquet` not found. This file is essential for universe construction.")
    elif not os.path.exists(os.path.join(PARQUET_FOLDER, 'sep.parquet')):
        st.sidebar.error("🔴 PRICE DATA MISSING")
        st.error("**CRITICAL**: `sep.parquet` (equity prices) not found. Essential for backtesting engine.")
    elif not os.path.exists(os.path.join(PARQUET_FOLDER, 'sf1.parquet')):
        st.sidebar.error("🔴 FUNDAMENTALS MISSING")
        st.error("**CRITICAL**: `sf1.parquet` (fundamentals) not found. Essential for factor calculations.")
    else:
        st.sidebar.success("🟢 ALL SYSTEMS OPERATIONAL")
        st.sidebar.info("""
        **Ready for Strategy Backtesting**
        
        ✅ Data repository validated
        ✅ Core files verified
        ✅ HamayelResearchEngine ready
        
        Configure parameters above and launch backtest.
        """)
        
        # Additional system information
        try:
            total_files = len([f for f in os.listdir(PARQUET_FOLDER) if f.endswith('.parquet')])
            st.sidebar.metric("Data Files Available", total_files)
        except:
            pass


