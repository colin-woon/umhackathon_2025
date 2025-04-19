import pandas as pd
import numpy as np
import yaml
import os
import argparse
from sklearn.preprocessing import StandardScaler
import joblib
from hmmlearn import hmm
import matplotlib.pyplot as plt # For plotting
import seaborn as sns          # For heatmap
from datetime import datetime  # For timestamped output directory
import shutil                 # For copying config file

# --- Configuration Loading ---
def load_config(config_path='config.yaml'):
    """Loads configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

# --- Data Loading and Preprocessing (REVISED RENAMING) ---
def load_and_preprocess_data(config, mode='backtest'):
    """Loads data from CSVs, merges, and preprocesses with explicit renaming."""
    data_dir = config['data_directory']
    data_files = config['data_files']
    suffix = f"_{mode}"

    all_data = {}
    base_candle_key = 'candles'
    final_column_names = {} # Keep track of final names for engineer_features

    # --- Load Base Candle Data ---
    try:
        candle_info = data_files.get(base_candle_key, None) # Get the dictionary for candles
        if not isinstance(candle_info, dict) or 'file_base' not in candle_info:
             # Handle cases where 'candles' entry is missing, not a dict, or lacks 'file_base'
             raise ValueError(f"'{base_candle_key}' entry in config['data_files'] is missing or improperly configured (must be a dict with 'file_base').")

        candle_file_base_str = candle_info['file_base'] # Extract the file_base string

        candle_path = os.path.join(data_dir, f"{candle_file_base_str}{suffix}.csv")
        print(f"Loading candles: {candle_path}")
        df_candles = pd.read_csv(candle_path, index_col='timestamp', parse_dates=True)

        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_candles.columns for col in ohlcv_cols):
            ohlc_map = {'t':'start_time', 'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'}
            if all(col in df_candles.columns for col in ohlc_map.keys()):
                df_candles.rename(columns=ohlc_map, inplace=True)
                print("  Renamed candle columns from t,o,h,l,c,v")
            else:
                 # Check if required columns exist even if not named perfectly
                 if not ('close' in df_candles.columns and 'volume' in df_candles.columns):
                    raise ValueError(f"Candle file missing essential 'close' or 'volume' columns")
        # Ensure numeric types for essential columns
        df_candles['close'] = pd.to_numeric(df_candles['close'], errors='coerce')
        df_candles['volume'] = pd.to_numeric(df_candles['volume'], errors='coerce')

        # Store only essential columns with standard names
        merged_df = df_candles[['close', 'volume']].copy()
        final_column_names['close'] = 'close'
        final_column_names['volume'] = 'volume'
        print(f"  Loaded {len(df_candles)} candle records.")

    except FileNotFoundError:
        print(f"Error: Base candle file not found: {candle_path}")
        return None, None
    except Exception as e:
        print(f"Error loading candle data: {e}")
        return None, None

    # --- Load and Merge Other Data Files ---
    for key, file_info in data_files.items():
        if key == base_candle_key:
            continue

        # Ensure file_info is a dictionary (handles the simple 'candles' case if needed)
        if not isinstance(file_info, dict):
            print(f"Warning: Skipping entry '{key}' in data_files as it's not a dictionary.")
            continue

        file_base = file_info.get('file_base')
        original_columns = file_info.get('original_columns', []) # Get list of potential original cols
        target_col_name = file_info.get('target_column')       # Get the desired target name

        if not file_base or not target_col_name:
            print(f"Warning: Skipping entry '{key}' due to missing 'file_base' or 'target_column' in config.")
            continue

        try:
            file_path = os.path.join(data_dir, f"{file_base}{suffix}.csv")
            print(f"Loading {key}: {file_path}")
            df_temp = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

            col_to_use = None
            # Find the first available column from the original_columns list
            for potential_col in original_columns:
                if potential_col in df_temp.columns:
                    col_to_use = potential_col
                    print(f"  Found specified column '{col_to_use}' for key '{key}'.")
                    break # Stop searching once found


            # Fallback if no specific logic or column found
            if col_to_use is None:
                data_cols = [col for col in df_temp.columns if col.lower() not in ['timestamp', 'start_time', 'end_time', 'time', 'date']] # Exclude 'date' too
                if data_cols:
                    col_to_use = data_cols[0]
                    # Use a generic fallback name if target_col_name wasn't set
                    if target_col_name is None: target_col_name = f"{key}_generic"
                    print(f"  Using fallback column '{col_to_use}' for key '{key}', renaming to '{target_col_name}'")
                else:
                    print(f"  Warning: Could not find suitable data column in {key}. Skipping merge.")
                    continue

            # Perform rename and merge
            df_to_merge = df_temp[[col_to_use]].rename(columns={col_to_use: target_col_name})
            # Ensure numeric before merge
            df_to_merge[target_col_name] = pd.to_numeric(df_to_merge[target_col_name], errors='coerce')

            merged_df = pd.merge(merged_df, df_to_merge, left_index=True, right_index=True, how='outer')
            final_column_names[key] = target_col_name # Store the final name used
            print(f"  Loaded and merged {len(df_temp)} records for {key} as '{target_col_name}'.")

        except FileNotFoundError:
            print(f"Warning: Data file not found for {key}: {file_path}. Skipping.")
        except Exception as e:
            print(f"Warning: Error loading or merging data for {key}: {e}. Skipping.")

    # --- Handle Missing Data ---
    print(f"\nColumns before ffill: {merged_df.columns.tolist()}")
    print(f"NaN counts before ffill:\n{merged_df.isna().sum()}")
    # Forward fill NaNs first (handles gaps within series)
    merged_df.ffill(inplace=True)
    # Backward fill NaNs next (handles leading NaNs if first value was missing)
    merged_df.bfill(inplace=True)
    # Drop any rows where essential data (like price) might still be missing
    merged_df.dropna(subset=['close'], inplace=True)
    # Optionally fill remaining NaNs in feature columns with 0 (if bfill wasn't enough)
    merged_df.fillna(0, inplace=True)


    print(f"\nTotal records after merging and NaN handling: {len(merged_df)}")
    print(f"Columns after merge/fill: {merged_df.columns.tolist()}")
    if merged_df.empty:
        print("Error: No data available after loading and merging.")
        return None, None

    # Return both the dataframe and the mapping of config keys to final column names
    return merged_df, final_column_names

# --- Feature Engineering (Revised for Core Robust Features) ---
def engineer_features(df, final_column_names, config_features): # config_features now unused, we use the fixed list
    """Creates a core set of features for HMM, prioritizing stationarity."""
    print("Engineering core features...")
    if df is None or df.empty:
        print("Error: Cannot engineer features on empty DataFrame.")
        return None
    df_feat = df.copy()
    print(f"\nDEBUG: Columns available at start of engineer_features: {df_feat.columns.tolist()}\n")

    # --- Essential Base Columns ---
    close_col = 'close'
    volume_col = 'volume'
    funding_col = final_column_names.get('cryptoquant_funding')
    active_col = final_column_names.get('glassnode_active')
    tx_col = final_column_names.get('glassnode_tx') # Needed for active_addr_tx_ratio calc if used
    inflow_col = final_column_names.get('cryptoquant_inflow')
    oi_col = final_column_names.get('coinglass_oi')
    lsr_col = final_column_names.get('coinglass_lsr') # Assuming this holds the raw L/S ratio

    # --- 1. Price Return ---
    # Hypothesis: Basic market movement.
    if close_col in df_feat.columns:
        df_feat['price_return'] = df_feat[close_col].pct_change()
    else:
        print(f"Warning: Column '{close_col}' not found for price_return.")
        df_feat['price_return'] = 0 # Assign default or handle error

    # --- 2. Price Volatility ---
    # Hypothesis: Market risk regime.
    df_feat['volatility_10d'] = df_feat['price_return'].rolling(window=10).std()

    # --- 3. Funding Rate Change ---
    # Hypothesis: Momentum in derivatives sentiment/cost.
    if funding_col and funding_col in df_feat.columns:
        df_feat['funding_rate_change_1d'] = df_feat[funding_col].diff()
    else:
        print(f"Warning: Column '{funding_col}' not found for funding_rate_change_1d.")
        df_feat['funding_rate_change_1d'] = 0

    # --- 4. Funding Rate Extreme Flag ---
    # Hypothesis: Funding extremes signal potential reversals.
    if funding_col and funding_col in df_feat.columns:
        rolling_window_90d = 90
        # Calculate 5th and 95th percentiles over rolling window
        lower_bound = df_feat[funding_col].rolling(window=rolling_window_90d, min_periods=int(rolling_window_90d*0.8)).quantile(0.05)
        upper_bound = df_feat[funding_col].rolling(window=rolling_window_90d, min_periods=int(rolling_window_90d*0.8)).quantile(0.95)
        df_feat['funding_rate_extreme_flag'] = ((df_feat[funding_col] < lower_bound) | (df_feat[funding_col] > upper_bound)).astype(int)
    else:
        print(f"Warning: Column '{funding_col}' not found for funding_rate_extreme_flag.")
        df_feat['funding_rate_extreme_flag'] = 0

    # --- 5. Active Address Rate of Change (14d) ---
    # Hypothesis: Network adoption/activity momentum.
    if active_col and active_col in df_feat.columns:
        df_feat['active_addr_roc_14d'] = df_feat[active_col].pct_change(periods=14)
    else:
        print(f"Warning: Column '{active_col}' not found for active_addr_roc_14d.")
        df_feat['active_addr_roc_14d'] = 0

    # --- 6. Inflow / Volume Ratio ---
    # Hypothesis: Significance of exchange flow pressure relative to market activity.
    if inflow_col and inflow_col in df_feat.columns and volume_col in df_feat.columns:
         denominator = pd.to_numeric(df_feat[volume_col], errors='coerce').replace(0, np.nan)
         numerator = pd.to_numeric(df_feat[inflow_col], errors='coerce')
         df_feat['inflow_vol_ratio'] = (numerator / denominator)
    else:
         print(f"Warning: Columns needed for inflow_vol_ratio not found ('{inflow_col}', '{volume_col}').")
         df_feat['inflow_vol_ratio'] = 0

    # --- 7. Open Interest Rate of Change (7d) ---
    # Hypothesis: Momentum in leverage and market participation.
    if oi_col and oi_col in df_feat.columns:
        df_feat['oi_roc_7d'] = df_feat[oi_col].pct_change(periods=7)
    else:
        print(f"Warning: Column '{oi_col}' not found for oi_roc_7d.")
        df_feat['oi_roc_7d'] = 0

    # --- 8. & 9. L/S Ratio Z-Score & Extreme Flag ---
    # Hypothesis: Statistically extreme retail sentiment signals potential reversals.
    if lsr_col and lsr_col in df_feat.columns:
        rolling_window_90d_lsr = 90
        lsr_series = pd.to_numeric(df_feat[lsr_col], errors='coerce')
        rolling_mean = lsr_series.rolling(window=rolling_window_90d_lsr, min_periods=int(rolling_window_90d_lsr*0.8)).mean()
        rolling_std = lsr_series.rolling(window=rolling_window_90d_lsr, min_periods=int(rolling_window_90d_lsr*0.8)).std()
        # Calculate Z-score, handle potential division by zero if std is 0
        df_feat['lsr_zscore_90d'] = (lsr_series - rolling_mean) / rolling_std.replace(0, np.nan)
        # Extreme Flag based on Z-score
        z_threshold = 2.0
        df_feat['lsr_extreme_flag'] = ((df_feat['lsr_zscore_90d'] > z_threshold) | (df_feat['lsr_zscore_90d'] < -z_threshold)).astype(int)
    else:
        print(f"Warning: Column '{lsr_col}' not found for lsr_zscore_90d/lsr_extreme_flag.")
        df_feat['lsr_zscore_90d'] = 0
        df_feat['lsr_extreme_flag'] = 0

    # --- 10. OI Change x Price Change Interaction ---
    # Hypothesis: Combined momentum confirms trends or signals divergences.
    # Using 1-day OI change for closer interaction timing, assuming oi_change_1d exists or can be calc'd easily
    if oi_col and oi_col in df_feat.columns: # Recalculate 1d OI change if not present
         if 'oi_change_1d' not in df_feat.columns:
              df_feat['oi_change_1d'] = df_feat[oi_col].pct_change(periods=1)
         df_feat['oi_change_x_price_change'] = df_feat['oi_change_1d'] * df_feat['price_return']
    else:
         print(f"Warning: Columns needed for oi_change_x_price_change not found ('{oi_col}', 'price_return').")
         df_feat['oi_change_x_price_change'] = 0


    # --- Select only the core features for the final DataFrame ---
    core_features_list = [
        "price_return",
        "volatility_10d",
        "funding_rate_change_1d",
        "funding_rate_extreme_flag",
        "active_addr_roc_14d",
        "inflow_vol_ratio",
        "oi_roc_7d",
        "lsr_zscore_90d", # Keep Z-score, it's more informative than just the flag
        # "lsr_extreme_flag", # Can be excluded if Z-score is used, or keep both
        "oi_change_x_price_change"
    ]
    # Also add back 'close' column needed for backtesting calculations downstream
    if close_col in df.columns:
         core_features_list.append(close_col)

    df_final_feat = df_feat[[col for col in core_features_list if col in df_feat.columns]].copy()


    # --- Clean up NaNs/Infs introduced by calculations in the final selection ---
    df_final_feat.fillna(method='ffill', inplace=True)
    df_final_feat.fillna(method='bfill', inplace=True)
    df_final_feat.fillna(0, inplace=True) # Fill any remaining NaNs with 0
    df_final_feat.replace([np.inf, -np.inf], 0, inplace=True)

    print(f"Core features engineered. Data shape: {df_final_feat.shape}")
    print(f"Engineered columns available: {df_final_feat.columns.tolist()}")

    # Ensure no NaNs/Infs remain in the final DataFrame before returning
    if df_final_feat.isnull().values.any() or np.isinf(df_final_feat.select_dtypes(include=np.number)).values.any():
         print("Warning: NaNs or Infs still present after cleaning!")
         # Optional: Implement stricter cleaning or return None
         # df_final_feat = df_final_feat.fillna(0).replace([np.inf, -np.inf], 0) # Re-apply just in case

    return df_final_feat

# --- HMM Model (Modified for Scaled Data & Reduced Verbosity) ---
def train_hmm(X_scaled, n_states, covariance_type, n_iter): # Takes X_scaled directly
    """Trains the Gaussian HMM model on pre-scaled data with reduced output."""
    # Note: Changed verbose=False to hide detailed iteration logs
    print(f"Training HMM with {n_states} states on scaled data...")

    if X_scaled is None or len(X_scaled) == 0:
        print("Error: No scaled data available for HMM training.")
        return None

    try:
        model = hmm.GaussianHMM(n_components=n_states,
                                covariance_type=covariance_type,
                                n_iter=n_iter,
                                random_state=42,
                                verbose=False) # <<< Set verbose=False to hide iteration details

        # Although verbose is False, we still need to capture convergence info from the monitor
        # We can temporarily redirect stdout/stderr during fit if needed, but often
        # checking monitor_.converged after fit is sufficient. HMMlearn might still print
        # a small amount, but the iteration table will be gone. Let's rely on the monitor check.

        model.fit(X_scaled) # Fit the model

        # Check convergence status AFTER fitting
        if not model.monitor_.converged:
             print(f"Warning: HMM n_states={n_states} did not converge in {n_iter} iterations!")
        else:
             print("HMM training complete (converged).") # Still useful to know it finished and converged
        return model
    except Exception as e:
        print(f"Error during HMM training: {e}")
        return None

def predict_states(model, X_scaled): # Takes X_scaled directly
    """Predicts hidden states using the trained HMM on pre-scaled data."""
    print("Predicting hidden states on scaled data...")
    # X = df[features].values # REMOVED

    # Removed NaN checks here

    if X_scaled is None or len(X_scaled) == 0:
        print("Error: No scaled data available for state prediction.")
        return None

    try:
        states = model.predict(X_scaled)
        return states
    except Exception as e:
        print(f"Error during state prediction: {e}")
        return None

# --- Signal Generation (Modified) ---
def generate_signals(df_with_state, signal_map): # Takes DataFrame with 'state' column
    """Generates trading signals based on predicted states in the DataFrame."""
    print("Generating trading signals...")
    # if states is None: # Removed - check happens before calling
    #     print("Error: Cannot generate signals, states are None.")
    #     return df
    if 'state' not in df_with_state.columns:
         print("Error: 'state' column not found in DataFrame for signal generation.")
         # Return original df or handle error
         return df_with_state.copy() # Avoid modifying original if error occurs

    df = df_with_state.copy() # Work on a copy
    df['signal'] = df['state'].map(signal_map).fillna(0) # Map states to signals

    df['signal'] = df['signal'].astype(int)

    print("Signals generated.")
    return df

# --- Backtesting ---
def run_backtest(df, fee_percent):
    """Runs the backtest simulation and calculates PnL."""
    print("Running backtest...")
    df_backtest = df.copy()
    cash = 1.0 # Start with $1 unit of cash
    position = 0.0 # Start with 0 units of the asset
    portfolio_value = [cash]

    # Use vectorized approach for speed where possible
    df_backtest['position'] = df_backtest['signal'].shift(1).fillna(0) # Trade on next bar's open based on previous signal
    df_backtest['price_change'] = df_backtest['close'].diff().fillna(0)

    # Calculate returns based on holding the position
    df_backtest['strategy_return_gross'] = df_backtest['position'] * df_backtest['close'].pct_change().fillna(0)

    # Calculate trade occurrences
    df_backtest['trade'] = df_backtest['position'].diff().fillna(0)
    trades = df_backtest[df_backtest['trade'] != 0]

    # Apply trading fees
    fees = abs(df_backtest['trade']) * fee_percent
    df_backtest['strategy_return_net'] = df_backtest['strategy_return_gross'] - fees

    # Calculate cumulative returns
    df_backtest['cumulative_strategy_return_net'] = (1 + df_backtest['strategy_return_net']).cumprod()

    print("Backtest finished.")
    return df_backtest


# --- Performance Metrics ---
def calculate_performance(df_backtest, fee_percent):
    """Calculates performance metrics."""
    print("Calculating performance metrics...")
    results = {}
    net_returns = df_backtest['strategy_return_net']

    # Total Return
    total_return = df_backtest['cumulative_strategy_return_net'].iloc[-1] - 1
    results['Total Return (%)'] = total_return * 100

    # Sharpe Ratio (Annualized, assuming daily returns if freq is daily, adjust risk-free rate and periods per year)
    # Infer frequency for annualization factor (basic example)
    time_diff = df_backtest.index.to_series().diff().median()
    periods_per_year = pd.Timedelta(days=365) / time_diff if time_diff else 252 # Default to 252 trading days if freq unknown
    mean_return = net_returns.mean()
    std_dev = net_returns.std()
    # Avoid division by zero if std_dev is 0
    sharpe_ratio = (mean_return / std_dev) * np.sqrt(periods_per_year) if std_dev != 0 else 0
    results['Annualized Sharpe Ratio'] = sharpe_ratio

    # Maximum Drawdown
    cumulative_returns = df_backtest['cumulative_strategy_return_net']
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    results['Maximum Drawdown (%)'] = max_drawdown * 100

    # Trade Frequency
    num_trades = (df_backtest['trade'] != 0).sum()
    total_rows = len(df_backtest)
    trade_frequency = (num_trades / total_rows) * 100 if total_rows > 0 else 0
    results['Trade Frequency (%)'] = trade_frequency
    results['Number of Trades'] = num_trades


    print("\n--- Performance Summary ---")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    print("---------------------------\n")

    return results

# --- Function to Find Best HMM based on AIC/BIC ---
def find_best_hmm_model(X_scaled, config, feature_list):
    """
    Loops through HMM states, trains models, and selects the best based on BIC/AIC.

    Args:
        X_scaled (np.ndarray): The scaled feature data for training.
        config (dict): The configuration dictionary.
        feature_list (list): List of feature names used (for context in logs).

    Returns:
        tuple: (best_hmm_model, best_n_states, best_bic, best_aic)
               Returns (None, -1, np.inf, np.inf) if no model converges.
    """
    best_n_states = -1
    best_bic = np.inf
    best_aic = np.inf
    best_hmm_model = None

    # Get parameters from config
    hmm_iterations = config['hmm_iterations']
    covariance_type = config['hmm_covariance_type']
    min_states = config['min_hmm_states']
    max_states = config['max_hmm_states']

    print(f"\n--- Searching for best n_states ({min_states} to {max_states}) using BIC/AIC ---")
    print(f"Features being used: {feature_list}")

    for n_states_test in range(min_states, max_states + 1):
        print(f"\nTesting HMM with {n_states_test} states...")
        # Train HMM directly on scaled data
        current_hmm_model = train_hmm(X_scaled,
                                      n_states_test,
                                      covariance_type,
                                      hmm_iterations) # Pass existing train_hmm function

        if current_hmm_model:
            # Check for convergence explicitly if train_hmm doesn't guarantee it
            if hasattr(current_hmm_model, 'monitor_') and not current_hmm_model.monitor_.converged:
                 print(f"  HMM n_states={n_states_test} did not converge. Skipping.")
                 continue # Skip BIC/AIC if not converged

            try:
                current_bic = current_hmm_model.bic(X_scaled)
                current_aic = current_hmm_model.aic(X_scaled)
                print(f"  n_states={n_states_test} -> BIC: {current_bic:.2f}, AIC: {current_aic:.2f}")

                # Check if this model is better (lower BIC is primary criteria)
                if current_bic < best_bic:
                    best_bic = current_bic
                    best_aic = current_aic
                    best_n_states = n_states_test
                    best_hmm_model = current_hmm_model
                    print(f"  *** New best model found with n_states={best_n_states} (BIC: {best_bic:.2f}) ***")
                elif current_bic == best_bic and current_aic < best_aic: # Tie-break with AIC
                    best_aic = current_aic
                    best_n_states = n_states_test
                    best_hmm_model = current_hmm_model
                    print(f"  *** New best model found with n_states={best_n_states} (AIC tie-breaker: {best_aic:.2f}) ***")

            except Exception as e:
                print(f"  Error calculating BIC/AIC for n_states={n_states_test}: {e}")
        else:
             print(f"  HMM training failed for n_states={n_states_test}. Skipping.")

    # --- End of Loop ---
    return best_hmm_model, best_n_states, best_bic, best_aic

# --- Feature Selection (Modified to return correlation matrix) ---
def select_features_by_correlation(df_features_backtest, features_available, corr_threshold):
    """
    Performs feature selection by removing highly correlated features.
    Also returns the correlation matrix and corresponding feature names for plotting.

    Args:
        df_features_backtest (pd.DataFrame): DataFrame containing engineered features for backtesting.
        features_available (list): List of feature names initially available for selection.
        corr_threshold (float): The correlation threshold above which features will be dropped.

    Returns:
        tuple: (final_feature_list, correlation_matrix, numeric_feature_names)
               - final_feature_list (list): Selected feature names.
               - correlation_matrix (pd.DataFrame or None): The calculated correlation matrix.
               - numeric_feature_names (list or None): Names of features used in the matrix.
               Returns (features_available, None, None) if selection cannot be performed.
    """
    print("\n--- Performing Basic Feature Selection (using Backtest data) ---")
    print(f"Features available for selection: {features_available}")
    final_feature_list = features_available[:] # Start with all available features
    correlation_matrix = None
    numeric_feature_names = None

    if len(features_available) <= 1:
        print("Not enough features available for correlation check.")
        return final_feature_list, correlation_matrix, numeric_feature_names

    # Select only numeric features for correlation calculation
    numeric_features_df = df_features_backtest[features_available].select_dtypes(include=np.number)
    numeric_feature_names = numeric_features_df.columns.tolist() # Get names used
    print(f"Numeric features considered for correlation: {numeric_feature_names}")

    if len(numeric_features_df.columns) <= 1:
        print("Not enough numeric features for correlation check.")
        return final_feature_list, correlation_matrix, numeric_feature_names # Return names even if no matrix

    # Calculate correlation matrix
    correlation_matrix = numeric_features_df.corr() # Keep original signs for heatmap
    corr_matrix_abs = correlation_matrix.abs() # Use absolute for dropping
    upper_tri = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))

    # Find features to drop
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]

    if to_drop:
        print(f"Features to drop due to high correlation (>{corr_threshold}): {to_drop}")
        final_feature_list = [f for f in features_available if f not in to_drop]
    else:
        print("No features dropped based on correlation threshold.")

    if not final_feature_list:
        print("Warning: Feature selection removed all features! Returning original list.")
        # Return original list, but still return matrix and names for potential plotting
        return features_available, correlation_matrix, numeric_feature_names

    print(f"\nFeatures selected for HMM: {final_feature_list}")
    # Return list, the calculated matrix (not abs), and the names used for the matrix
    return final_feature_list, correlation_matrix, numeric_feature_names

def perform_state_stability_check(states_backtest, states_forwardtest, stability_threshold):
    """
    Compares state distributions between backtest and forward test periods.

    Args:
        states_backtest (np.ndarray or pd.Series): Predicted states for the backtest period.
        states_forwardtest (np.ndarray or pd.Series): Predicted states for the forward test period.
        stability_threshold (float): Max allowed percentage point difference for stability.

    Returns:
        tuple: (comparison_df, max_diff, model_is_stable)
               - comparison_df (pd.DataFrame): DataFrame showing percentage distribution.
               - max_diff (float): Maximum absolute difference found.
               - model_is_stable (bool): True if max_diff <= threshold, False otherwise.
               Returns (None, np.inf, False) if check cannot be performed.
    """
    print("\n--- State Distribution Comparison ---")
    model_is_stable = False # Default to not stable
    max_diff = np.inf
    comparison_df = None

    if states_backtest is None or states_forwardtest is None or len(states_backtest) == 0 or len(states_forwardtest) == 0:
        print("Could not perform state stability check due to missing state predictions.")
        return comparison_df, max_diff, model_is_stable

    try:
        backtest_state_dist = pd.Series(states_backtest).value_counts(normalize=True).sort_index()
        forwardtest_state_dist = pd.Series(states_forwardtest).value_counts(normalize=True).sort_index()

        # Combine distributions for easy comparison
        comparison_df = pd.DataFrame({
            'Backtest %': backtest_state_dist * 100,
            'Forward Test %': forwardtest_state_dist * 100
        }).fillna(0).sort_index() # Fill missing states with 0% and sort

        print(comparison_df)

        # Simple stability check: Max absolute difference in proportions
        max_diff = (comparison_df['Backtest %'] - comparison_df['Forward Test %']).abs().max()
        print(f"\nMax difference in state proportions: {max_diff:.2f}%")

        # ANSI escape codes for colors
        COLOR_GREEN = "\033[92m" # Bright Green
        COLOR_RED = "\033[91m"   # Bright Red
        COLOR_RESET = "\033[0m"  # Reset color

        if max_diff <= stability_threshold:
            print(f"{COLOR_GREEN}State distributions appear reasonably stable (Max Diff <= {stability_threshold}%).{COLOR_RESET}")
            model_is_stable = True
        else:
            print(f"{COLOR_RED}Warning: State distributions differ significantly (Max Diff > {stability_threshold}%). Model/features may lack robustness.{COLOR_RESET}")
            model_is_stable = False
        print("------------------------------------")

    except Exception as e:
        print(f"Error during state stability check: {e}")
        # Ensure default return values in case of error during calculation
        max_diff = np.inf
        model_is_stable = False
        # comparison_df might be partially created, leave it as is or set to None
        comparison_df = None


    return comparison_df, max_diff, model_is_stable

def prepare_feature_arrays(df_features_backtest, df_features_forwardtest, feature_list):
    """
    Selects features, converts to NumPy arrays, and handles NaNs/Infs.

    Args:
        df_features_backtest (pd.DataFrame): Engineered features for backtest.
        df_features_forwardtest (pd.DataFrame): Engineered features for forward test.
        feature_list (list): List of selected feature names to use.

    Returns:
        tuple: (X_backtest, X_forwardtest) NumPy arrays, or (None, None) if errors occur.
    """
    print("\n--- Preparing Data for Model ---")
    # Ensure selected features exist in both dataframes
    if not all(f in df_features_backtest.columns for f in feature_list):
        print("Error: Selected features missing from backtest feature dataframe.")
        return None, None
    if not all(f in df_features_forwardtest.columns for f in feature_list):
        print("Error: Selected features missing from forward test feature dataframe.")
        return None, None

    X_backtest = df_features_backtest[feature_list].values
    X_forwardtest = df_features_forwardtest[feature_list].values

    # Handle NaNs/Infs that might remain or be introduced
    if np.any(np.isnan(X_backtest)) or np.any(np.isinf(X_backtest)):
        print("Warning: Handling NaNs/Infs in backtest feature array.")
        X_backtest = np.nan_to_num(X_backtest)
    if np.any(np.isnan(X_forwardtest)) or np.any(np.isinf(X_forwardtest)):
        print("Warning: Handling NaNs/Infs in forward test feature array.")
        X_forwardtest = np.nan_to_num(X_forwardtest)

    if len(X_backtest) == 0:
        print("Error: No backtest data after NaN handling.")
        return None, None
    if len(X_forwardtest) == 0:
        print("Error: No forward test data after NaN handling.")
        return None, None

    print("Feature arrays prepared.")
    return X_backtest, X_forwardtest

def save_model_and_scaler(model, scaler, directory='models'):
    """Saves the HMM model and scaler to disk."""
    print(f"\n--- Saving Model and Scaler to '{directory}/' ---")
    try:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(scaler, os.path.join(directory, 'scaler.pkl'))
        joblib.dump(model, os.path.join(directory, 'hmm_model.pkl'))
        print("Saved scaler and best HMM model successfully.")
    except Exception as e:
        print(f"Warning: Could not save model/scaler: {e}")

def analyze_hmm_states(hmm_model, feature_list, df_features_backtest, states_backtest):
    """
    Analyzes and prints characteristics of the HMM states based on backtest data.

    Args:
        hmm_model: The trained hmmlearn model instance.
        feature_list (list): List of features used in the model.
        df_features_backtest (pd.DataFrame): Backtest feature dataframe (without 'state').
        states_backtest (np.ndarray): Predicted states for the backtest period.
    """
    if hmm_model is None or states_backtest is None:
        print("Skipping HMM state analysis due to missing model or states.")
        return

    print("\n--- HMM State Analysis (Best Model - Backtest Period) ---")
    best_n_states = hmm_model.n_components
    print(f"Number of States: {best_n_states}")
    print(f"Features used for analysis: {feature_list}")

    # Print Mean Scaled Features
    print("Mean of Scaled Features for each State:")
    try:
        # Check if means_ attribute exists (it should for GaussianHMM)
        if hasattr(hmm_model, 'means_'):
            for i in range(best_n_states):
                print(f"  State {i}:")
                # Ensure feature_list length matches dimensions of means_
                if len(feature_list) == hmm_model.means_.shape[1]:
                    for feature_name, mean_val in zip(feature_list, hmm_model.means_[i]):
                        print(f"    {feature_name}: {mean_val:.4f}")
                else:
                    print(f"    Error: Feature list length ({len(feature_list)}) mismatch with model means dimension ({hmm_model.means_.shape[1]}).")
        else:
             print("    Model does not have 'means_' attribute for analysis.")
    except Exception as e:
        print(f"  Could not print HMM means: {e}")

    # Print Average Raw Return per State
    # Ensure alignment before adding state column for analysis
    if len(states_backtest) == len(df_features_backtest.index):
        # Use assign to create a temporary DataFrame with states without modifying original
        df_analysis = df_features_backtest.assign(state=states_backtest)
        print("\nAverage Raw Price Return per State (Backtest):")
        try:
            if 'price_return' in df_analysis.columns:
                avg_returns = df_analysis.groupby('state')['price_return'].mean()
                print(avg_returns)
            else:
                print("  'price_return' column not found in features for analysis.")
        except Exception as e:
            print(f"  Could not calculate average return per state: {e}")
    else:
        print("  Skipping average return per state analysis due to length mismatch between states and features index.")
    print("--------------------------\n")

def run_and_evaluate_test_period(df_features, states, config, period_label="Test"):
    """
    Adds states, runs signals, backtest simulation, and calculates performance.

    Args:
        df_features (pd.DataFrame): Feature DataFrame for the period.
        states (np.ndarray): Predicted states for the period.
        config (dict): Configuration dictionary.
        period_label (str): Label for printing ('Backtest' or 'Forward Test').

    Returns:
        tuple: (df_results, performance_summary)
               - df_results (pd.DataFrame): DataFrame with backtest results.
               - performance_summary (dict): Dictionary of performance metrics.
               Returns (None, None) if execution fails.
    """
    print(f"\n--- Running Final {period_label} ---")
    if states is None:
        print(f"Skipping final {period_label} due to state prediction error.")
        return None, None
    if len(states) != len(df_features.index):
        print(f"Skipping final {period_label} due to state/index length mismatch.")
        return None, None

    # Safely add 'state' column
    if 'state' in df_features.columns:
        df_features = df_features.drop(columns=['state']) # Drop if re-running cell
    df_features_with_state = df_features.assign(state=states)

    # Run the pipeline
    signal_map_from_config = config.get('signal_map', {})
    # Optional: Add signal map size warning here too if desired, or rely on backtest warning
    print(f"Using signal map from config: {signal_map_from_config}") # Moved print here

    df_signals = generate_signals(df_features_with_state, signal_map_from_config)
    df_results = run_backtest(df_signals, config['trading_fee_percent'])

    print(f"\n--- FINAL {period_label.upper()} PERFORMANCE ---")
    performance_summary = calculate_performance(df_results, config['trading_fee_percent'])

    # Optional: Save results
    # file_name = f'final_{period_label.lower()}_results.csv'
    # df_results.to_csv(os.path.join(config.get('data_directory', 'data'), file_name))
    # print(f"Saved {period_label} results to {file_name}")

    return df_results, performance_summary

# --- Plotting Functions ---
def plot_correlation_heatmap(corr_matrix, feature_names, output_path):
    """Generates and saves a heatmap of the feature correlation matrix."""
    if corr_matrix is None or corr_matrix.empty:
        print("Skipping correlation heatmap: No correlation matrix provided.")
        return
    print(f"Generating Correlation Heatmap to: {output_path}")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print("Correlation heatmap saved.")
    except Exception as e:
        print(f"Error saving correlation heatmap: {e}")
    plt.close() # Close the plot to free memory

def plot_equity_curves(results_bt, results_ft, output_path):
    """Generates and saves equity curves for backtest and forward test."""
    if results_bt is None or results_ft is None:
        print("Skipping equity curve plot: Missing results data.")
        return
    if 'cumulative_strategy_return_net' not in results_bt.columns or \
       'cumulative_strategy_return_net' not in results_ft.columns:
           print("Skipping equity curve plot: 'cumulative_strategy_return_net' column missing.")
           return

    print(f"Generating Equity Curves to: {output_path}")
    plt.figure(figsize=(12, 6))

    # Plot Backtest Equity (Portfolio Value starting from 1)
    plt.plot(results_bt.index, results_bt['cumulative_strategy_return_net'], label='Backtest Equity')

    # Plot Forward Test Equity (Portfolio Value starting from 1)
    # Ensure index alignment if needed, but usually separate plots or just concat works if dates are contiguous
    # For simplicity, plotting on same axes assuming date index handles it.
    plt.plot(results_ft.index, results_ft['cumulative_strategy_return_net'], label='Forward Test Equity')

    plt.title('Strategy Equity Curve (Backtest vs Forward Test)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Starting from 1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print("Equity curve plot saved.")
    except Exception as e:
        print(f"Error saving equity curve plot: {e}")
    plt.close()

# --- Saving Function ---
def save_run_summary(config, performance_bt, performance_ft, stability_results, output_dir):
    """Saves config and performance summaries to a YAML file in the output dir."""
    print(f"\n--- Saving Run Summary to '{output_dir}/' ---")
    summary_data = {
        'run_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'configuration': config,
        'performance_backtest': performance_bt,
        'performance_forwardtest': performance_ft,
        'state_stability': stability_results # Include stability results dict
    }
    summary_path = os.path.join(output_dir, 'run_summary.yaml')
    try:
        with open(summary_path, 'w') as f:
            yaml.dump(summary_data, f, default_flow_style=False, sort_keys=False)
        print(f"Run summary saved to {summary_path}")

        # Also copy the original config file for exact reference
        config_source_path = config.get('__source_path__', 'config.yaml') # Get source if stored, else default
        if os.path.exists(config_source_path):
             shutil.copy2(config_source_path, os.path.join(output_dir, 'config_used.yaml'))
             print(f"Copied original config '{config_source_path}' to output directory.")
        else:
             print(f"Warning: Could not find original config at '{config_source_path}' to copy.")

    except Exception as e:
        print(f"Error saving run summary: {e}")

# Helper function to create output directory
def create_output_directory(base_dir="outputs"):
    """Creates a timestamped directory for the current run's outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(base_dir, f"run_{timestamp}")
    try:
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Created output directory: {run_output_dir}")
        return run_output_dir
    except Exception as e:
        print(f"Error creating output directory '{run_output_dir}': {e}")
        return None

# --- Main Execution (Including Plotting and Saving) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM Trading Strategy Backtest.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # --- Create Output Directory for this Run ---
    run_output_dir = create_output_directory(base_dir="outputs")
    if not run_output_dir: exit("Error: Could not create output directory.")

    # --- Configuration ---
    config_path = args.config
    config = load_config(config_path)
    if not config: exit(1)
    config['__source_path__'] = config_path # Store path for later copying

    # --- Data Loading ---
    # ... (keep existing data loading logic) ...
    df_raw_backtest, final_names_bt = load_and_preprocess_data(config, mode='backtest')
    df_raw_forwardtest, final_names_ft = load_and_preprocess_data(config, mode='forwardtest')
    final_column_names = final_names_bt if final_names_bt else {}
    if df_raw_backtest is None or df_raw_backtest.empty: exit("Error: Could not load backtest data.")
    if df_raw_forwardtest is None or df_raw_forwardtest.empty: exit("Error: Could not load forward test data.")


    # --- Feature Engineering ---
    # ... (keep existing feature engineering logic) ...
    feature_list_from_config = config.get('features', [])
    if not feature_list_from_config: exit("Error: 'features' list in config is empty.")
    print("\n--- Engineering Features ---")
    df_features_backtest = engineer_features(df_raw_backtest, final_column_names, feature_list_from_config)
    df_features_forwardtest = engineer_features(df_raw_forwardtest, final_column_names, feature_list_from_config)
    if df_features_backtest is None or df_features_backtest.empty: exit("Error: Backtest feature engineering failed.")
    if df_features_forwardtest is None or df_features_forwardtest.empty: exit("Error: Forward test feature engineering failed.")


    # --- Feature Selection & Correlation Plot ---
    features_available = [f for f in feature_list_from_config if f in df_features_backtest.columns]
    corr_threshold = config.get('feature_corr_threshold', 0.95)
    # ---- MODIFIED CALL ----
    feature_list, corr_matrix, corr_feature_names = select_features_by_correlation(
        df_features_backtest, features_available, corr_threshold
    )
    if not feature_list: exit("Error: Feature selection removed all features.")
    # ---- PLOT CORRELATION ----
    if corr_matrix is not None:
        heatmap_path = os.path.join(run_output_dir, 'correlation_heatmap.png')
        plot_correlation_heatmap(corr_matrix, corr_feature_names, heatmap_path)


    # --- Prepare Feature Arrays ---
    X_backtest, X_forwardtest = prepare_feature_arrays(df_features_backtest, df_features_forwardtest, feature_list)
    if X_backtest is None or X_forwardtest is None: exit("Error: Failed to prepare feature arrays.")


    # --- Scale Data ---
    # ... (keep existing scaling logic) ...
    print("\n--- Scaling Data ---")
    scaler = StandardScaler()
    X_scaled_backtest = scaler.fit_transform(X_backtest)
    X_scaled_forwardtest = scaler.transform(X_forwardtest)
    print("Features scaled.")


    # --- Find Best HMM Model ---
    # ... (keep existing HMM finding logic) ...
    hmm_model, best_n_states, _, _ = find_best_hmm_model(X_scaled_backtest, config, feature_list)
    if hmm_model is None: exit("Error: Failed to find suitable HMM model.")
    print(f"\n--- Best model selected: n_states = {best_n_states} ---")


    # --- Save Model Artifacts ---
    # <<<< CALL NEW FUNCTION >>>>
    save_model_and_scaler(hmm_model, scaler, directory=os.path.join(run_output_dir, 'models'))


    # --- Predict States ---
    # ... (keep existing state prediction logic) ...
    print("\n--- Predicting States ---")
    states_backtest = predict_states(hmm_model, X_scaled_backtest)
    states_forwardtest = predict_states(hmm_model, X_scaled_forwardtest)


    # --- State Stability Check ---
    stability_threshold = config.get('state_stability_threshold', 20)
    stability_comparison_df, max_diff, model_is_stable = perform_state_stability_check(
        states_backtest, states_forwardtest, stability_threshold
    )
    # Store stability results for summary
    stability_results = {
        'max_difference_pct': max_diff,
        'is_stable': model_is_stable,
        'stability_threshold_pct': stability_threshold,
        'distribution_comparison': stability_comparison_df.to_dict() if stability_comparison_df is not None else None
    }


    # --- Analyze HMM States (Backtest) ---
    # <<<< CALL NEW FUNCTION >>>>
    analyze_hmm_states(hmm_model, feature_list, df_features_backtest, states_backtest)


    # --- Run Backtest Period ---
    # <<<< CALL NEW FUNCTION >>>>
    results_bt, performance_bt = run_and_evaluate_test_period(
        df_features_backtest, states_backtest, config, period_label="Backtest"
    )
    # Handle potential failure
    if performance_bt is None: performance_bt = {"Error": "Backtest execution failed"}


    # --- Run Forward Test Period ---
    # <<<< CALL NEW FUNCTION >>>>
    results_ft, performance_ft = run_and_evaluate_test_period(
        df_features_forwardtest, states_forwardtest, config, period_label="Forward Test"
    )
    # Handle potential failure
    if performance_ft is None: performance_ft = {"Error": "Forward test execution failed"}


    # --- Plot Equity Curves ---
    # <<<< CALL NEW FUNCTION >>>>
    equity_curve_path = os.path.join(run_output_dir, 'equity_curve.png')
    # Pass results_bt and results_ft which are returned by run_and_evaluate_test_period
    plot_equity_curves(results_bt, results_ft, equity_curve_path)


    # --- Save Run Summary ---
    # <<<< CALL NEW FUNCTION >>>>
    save_run_summary(config, performance_bt, performance_ft, stability_results, run_output_dir)


    print(f"\n--- Strategy execution finished ---")
    print(f"Outputs saved in: {run_output_dir}")
