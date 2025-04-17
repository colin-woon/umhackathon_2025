import pandas as pd
import numpy as np
import yaml
import os
import argparse
from sklearn.preprocessing import StandardScaler
import joblib # For saving the scaler later if needed for forward testing
from hmmlearn import hmm # Requires hmmlearn
# from sklearn.preprocessing import StandardScaler # Optional: If scaling features
# from sklearn.metrics import accuracy_score # Optional: If evaluating HMM directly

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

# --- Feature Engineering (Enhanced with Hypotheses) ---
def engineer_features(df, final_column_names, config_features):
    """Creates features for the HMM model with justifications."""
    print("Engineering features...")
    if df is None or df.empty:
        print("Error: Cannot engineer features on empty DataFrame.")
        return None
    df_feat = df.copy()
    print(f"\nDEBUG: Columns available at start of engineer_features: {df_feat.columns.tolist()}\n")

    # --- Base Feature Calculations ---
    # Hypothesis: Past returns influence future behaviour.
    df_feat['price_return'] = df_feat['close'].pct_change()
    # Hypothesis: Recent volatility impacts market regime.
    df_feat['volatility_10d'] = df_feat['price_return'].rolling(window=10).std() # Use 10-day window

    # --- Engineer Features from Loaded Data (with Hypotheses) ---

    # Funding Rate Features
    funding_col = final_column_names.get('cryptoquant_funding')
    if funding_col and funding_col in df_feat.columns:
        # Hypothesis: The rate of change in funding indicates shifting sentiment momentum.
        df_feat['funding_rate_change_1d'] = df_feat[funding_col].diff()
        # Hypothesis: The short-term trend in funding rate reflects prevailing leverage bias.
        df_feat['funding_rate_MA_5d'] = df_feat[funding_col].rolling(window=5).mean()
    else:
        if 'cryptoquant_funding_rate' in config_features: print(f"Warning: Column '{funding_col}' needed for funding features not found.")

    # Active Addresses Features
    active_col = final_column_names.get('glassnode_active')
    if active_col and active_col in df_feat.columns:
        # Hypothesis: The percentage change in active addresses reflects network growth/decline.
        df_feat['active_addresses_change_1d'] = df_feat[active_col].pct_change()
        # Hypothesis: Active address trend relative to overall activity matters. (Example below uses tx_count)
    else:
        if 'glassnode_active_value' in config_features or 'active_addresses_change' in config_features: print(f"Warning: Column '{active_col}' needed for active address features not found.")

    # Transaction Count Features
    tx_col = final_column_names.get('glassnode_tx')
    if tx_col and tx_col in df_feat.columns:
         # Hypothesis: Transaction count reflects overall network usage.
         df_feat['tx_count_MA_5d'] = df_feat[tx_col].rolling(window=5).mean()
         # Hypothesis: The ratio of active addresses to transactions might indicate type of usage (e.g., many small vs few large tx).
         if active_col and active_col in df_feat.columns:
              # Use pd.to_numeric to handle potential non-numeric data robustly, replace 0 denominator
              denominator = pd.to_numeric(df_feat[tx_col], errors='coerce').replace(0, np.nan)
              numerator = pd.to_numeric(df_feat[active_col], errors='coerce')
              df_feat['active_addr_tx_ratio'] = (numerator / denominator)

    # Inflow Features
    inflow_col = final_column_names.get('cryptoquant_inflow')
    if inflow_col and inflow_col in df_feat.columns:
         # Hypothesis: Inflow relative to volume indicates significance of exchange flow pressure.
         if 'volume' in df_feat.columns:
             # Use pd.to_numeric and handle division by zero
             denominator = pd.to_numeric(df_feat['volume'], errors='coerce').replace(0, np.nan)
             numerator = pd.to_numeric(df_feat[inflow_col], errors='coerce')
             df_feat['inflow_vol_ratio'] = (numerator / denominator)

    # Open Interest Features
    oi_col = final_column_names.get('coinglass_oi')
    if oi_col and oi_col in df_feat.columns:
         # Hypothesis: Change in open interest signals change in market participation/conviction.
         df_feat['oi_change_1d'] = df_feat[oi_col].pct_change()

    # Long/Short Ratio Features (Using the generic column name from your load log)
    lsr_col = final_column_names.get('coinglass_lsr') # This resolved to 'coinglass_lsr_generic'
    if lsr_col and lsr_col in df_feat.columns:
         # Hypothesis: The trend in L/S ratio reflects retail sentiment shifts.
         # Note: Log showed this loaded 'longAccount'. If ratio needs calculating: longAccount / shortAccount
         # For now, just use the loaded column if it represents the ratio, or calculate MA.
         df_feat['lsr_MA_5d'] = df_feat[lsr_col].rolling(window=5).mean()


    # --- Clean up NaNs introduced by calculations ---
    # Use forward fill first, then backward fill for robustness
    df_feat.fillna(method='ffill', inplace=True)
    df_feat.fillna(method='bfill', inplace=True)
    # Fill any remaining NaNs (e.g., if entire column was NaN initially) with 0
    df_feat.fillna(0, inplace=True)
    # Replace any potential infinities resulting from division by zero etc.
    df_feat.replace([np.inf, -np.inf], 0, inplace=True)


    print(f"Features engineered. Data shape: {df_feat.shape}")
    print(f"Engineered columns available: {df_feat.columns.tolist()}")

    # --- Final Check: Removed FATAL error, handle missing features in main block ---
    # Check if features expected by config still exist after engineering
    # missing_check = [f for f in config_features if f not in df_feat.columns]
    # if missing_check:
    #      print(f"Warning: Some features listed in config might be missing post-engineering: {missing_check}")

    return df_feat

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

# --- Main Execution (Refactored with State Stability Check) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM Trading Strategy Backtest.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)

    # --- Always Load Both Backtest and Forward Test Data ---
    print("\n--- Loading Data ---")
    df_raw_backtest, final_column_names_backtest = load_and_preprocess_data(config, mode='backtest')
    df_raw_forwardtest, final_column_names_forwardtest = load_and_preprocess_data(config, mode='forwardtest')

    # Use backtest names map, assuming columns are consistent
    final_column_names = final_column_names_backtest if final_column_names_backtest else {}

    # Exit if essential data is missing
    if df_raw_backtest is None or df_raw_backtest.empty: exit("Error: Could not load backtest data.")
    if df_raw_forwardtest is None or df_raw_forwardtest.empty: exit("Error: Could not load forward test data.")

    # --- Feature Engineering (on both datasets) ---
    feature_list_from_config = config.get('features', [])
    if not feature_list_from_config: exit("Error: 'features' list empty.")

    print("\n--- Engineering Features for Backtest Data ---")
    df_features_backtest = engineer_features(df_raw_backtest, final_column_names, feature_list_from_config)
    if df_features_backtest is None or df_features_backtest.empty: exit("Error: Backtest feature engineering failed.")

    print("\n--- Engineering Features for Forward Test Data ---")
    df_features_forwardtest = engineer_features(df_raw_forwardtest, final_column_names, feature_list_from_config)
    if df_features_forwardtest is None or df_features_forwardtest.empty: exit("Error: Forward test feature engineering failed.")

    # --- Feature Selection (Based on Backtest Data) ---
    print("\n--- Performing Basic Feature Selection (using Backtest data) ---")
    features_available = [f for f in feature_list_from_config if f in df_features_backtest.columns]
    print(f"Features available for selection: {features_available}")
    feature_list = features_available # Default if selection fails below

    if len(features_available) > 1:
        numeric_features_df = df_features_backtest[features_available].select_dtypes(include=np.number)
        print(f"Numeric features considered for correlation: {numeric_features_df.columns.tolist()}")
        if len(numeric_features_df.columns) > 1:
            corr_matrix = numeric_features_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            corr_threshold = config.get('feature_corr_threshold', 0.95) # Make threshold configurable
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
            if to_drop:
                 print(f"Features to drop due to high correlation (>{corr_threshold}): {to_drop}")
                 feature_list = [f for f in features_available if f not in to_drop]
            else:
                 print("No features dropped based on correlation threshold.")
        else: print("Not enough numeric features for correlation check.")
    else: print("Not enough features for correlation check.")

    if not feature_list: exit("Error: Feature selection removed all features.")
    print(f"\nFeatures selected for HMM: {feature_list}")

    # --- Prepare Data for Model ---
    # Ensure selected features exist in both dataframes
    if not all(f in df_features_backtest.columns for f in feature_list):
         exit("Error: Selected features missing from backtest feature dataframe.")
    if not all(f in df_features_forwardtest.columns for f in feature_list):
         exit("Error: Selected features missing from forward test feature dataframe.")

    X_backtest = df_features_backtest[feature_list].values
    X_forwardtest = df_features_forwardtest[feature_list].values

    # Handle NaNs/Infs
    if np.any(np.isnan(X_backtest)) or np.any(np.isinf(X_backtest)): X_backtest = np.nan_to_num(X_backtest)
    if np.any(np.isnan(X_forwardtest)) or np.any(np.isinf(X_forwardtest)): X_forwardtest = np.nan_to_num(X_forwardtest)

    if len(X_backtest) == 0: exit("Error: No backtest data after NaN handling for scaling.")
    if len(X_forwardtest) == 0: exit("Error: No forward test data after NaN handling for scaling.")

    # --- Scale Data (Fit on Backtest, Transform Both) ---
    print("\nFitting scaler on backtest data and scaling features...")
    scaler = StandardScaler()
    X_scaled_backtest = scaler.fit_transform(X_backtest)
    X_scaled_forwardtest = scaler.transform(X_forwardtest) # Use transform only!
    print("Features scaled for both periods.")

    # --- Find Best HMM Model (using Backtest Data) ---
    hmm_model, best_n_states, _, _ = find_best_hmm_model(X_scaled_backtest, config, feature_list)
    if hmm_model is None: exit("Error: Failed to find suitable HMM model during backtest.")
    print(f"\n--- Best model selected: n_states = {best_n_states} ---")

    # --- Save Scaler & Best Model ---
    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(hmm_model, 'models/hmm_model.pkl')
        print("Saved scaler and best HMM model to 'models/' directory.")
    except Exception as e: print(f"Warning: Could not save model/scaler: {e}")

    # --- Predict States for Both Periods ---
    print("\nPredicting states for Backtest period...")
    states_backtest = predict_states(hmm_model, X_scaled_backtest)
    print("Predicting states for Forward Test period...")
    states_forwardtest = predict_states(hmm_model, X_scaled_forwardtest)

    # --- State Stability Check ---
    model_is_stable = False # Flag
    if states_backtest is not None and states_forwardtest is not None:
        print("\n--- State Distribution Comparison ---")
        backtest_state_dist = pd.Series(states_backtest).value_counts(normalize=True).sort_index()
        forwardtest_state_dist = pd.Series(states_forwardtest).value_counts(normalize=True).sort_index()

        # Combine distributions for easy comparison
        comparison_df = pd.DataFrame({
            'Backtest %': backtest_state_dist * 100,
            'Forward Test %': forwardtest_state_dist * 100
        }).fillna(0) # Fill missing states with 0%

        print(comparison_df)

        # Simple stability check: Max absolute difference in proportions
        max_diff = (comparison_df['Backtest %'] - comparison_df['Forward Test %']).abs().max()
        stability_threshold = config['state_stability_threshold'] # Configurable threshold %
        print(f"\nMax difference in state proportions: {max_diff:.2f}%")

        COLOR_GREEN = "\033[92m" # Bright Green
        COLOR_RED = "\033[91m"   # Bright Red
        COLOR_RESET = "\033[0m"  # Reset color

        if max_diff <= stability_threshold:
            print(f"{COLOR_GREEN}State distributions appear reasonably stable (Max Diff <= {stability_threshold}%).{COLOR_RESET}")
            model_is_stable = True
        else:
            # Use red for the warning
            print(f"{COLOR_RED}Warning: State distributions differ significantly (Max Diff > {stability_threshold}%). Model/features may lack robustness.{COLOR_RESET}")
            model_is_stable = False # Explicitly set flag
        print("------------------------------------")
    else:
        print("Could not perform state stability check due to prediction errors.")

    # --- HMM State Analysis (on Backtest Results for Signal Mapping) ---
    # Run analysis regardless of stability, but interpret with caution if unstable
    if states_backtest is not None:
        print("\n--- HMM State Analysis (Best Model - Backtest Period) ---")
        # ... (Your existing state analysis print logic here, using hmm_model, best_n_states, feature_list, and df_features_backtest / states_backtest) ...
        print(f"Number of States: {best_n_states}")
        print(f"Features used for analysis: {feature_list}")
        print("Mean of Scaled Features for each State:")
        try:
             for i in range(hmm_model.n_components):
                  print(f"  State {i}:")
                  for feature_name, mean_val in zip(feature_list, hmm_model.means_[i]):
                       print(f"    {feature_name}: {mean_val:.4f}")
        except Exception as e: print(f"  Could not print HMM means: {e}")

        # Ensure alignment before adding state column for analysis
        if len(states_backtest) == len(df_features_backtest.index):
             df_features_backtest['state'] = states_backtest # Add state column for analysis
             print("\nAverage Raw Price Return per State (Backtest):")
             try:
                  if 'price_return' in df_features_backtest.columns: print(df_features_backtest.groupby('state')['price_return'].mean())
                  else: print("  'price_return' column not found.")
             except Exception as e: print(f"  Could not calculate average return per state: {e}")
        else: print("  Skipping average return per state analysis due to length mismatch.")
        print("--------------------------\n")


    # --- Final Backtest Execution ---
    print("\n--- Running Final Backtest ---")
    if states_backtest is not None:
         # Add states column if lengths match (check again)
         if len(states_backtest) == len(df_features_backtest.index):
              df_features_backtest['state'] = states_backtest
              signal_map_from_config = config.get('signal_map', {})
              print(f"Using signal map from config: {signal_map_from_config}")
              if best_n_states != -1 and len(signal_map_from_config) != best_n_states:
                   print(f"Warning: Signal map size mismatch! Config: {len(signal_map_from_config)}, Model: {best_n_states}. Defaulting undefined states to 0.")

              df_signals_backtest = generate_signals(df_features_backtest, signal_map_from_config)
              df_results_backtest = run_backtest(df_signals_backtest, config['trading_fee_percent'])
              print("\n--- FINAL BACKTEST PERFORMANCE ---")
              performance_summary_backtest = calculate_performance(df_results_backtest, config['trading_fee_percent'])
              # Optional: Save backtest results
              # df_results_backtest.to_csv(os.path.join(config['data_directory'], 'final_backtest_results.csv'))
         else:
              print("Skipping final backtest due to state/index length mismatch.")
    else:
        print("Skipping final backtest due to state prediction error.")


    # --- Final Forward Test Execution ---
    print("\n--- Running Final Forward Test ---")
    if states_forwardtest is not None:
        # Add states column if lengths match (check again)
        if len(states_forwardtest) == len(df_features_forwardtest.index):
             df_features_forwardtest['state'] = states_forwardtest
             signal_map_from_config = config.get('signal_map', {}) # Use same signal map
             # Note: No warning print needed here as it was shown before generate_signals in backtest
             df_signals_forwardtest = generate_signals(df_features_forwardtest, signal_map_from_config)
             df_results_forwardtest = run_backtest(df_signals_forwardtest, config['trading_fee_percent'])
             print("\n--- FINAL FORWARD TEST PERFORMANCE ---")
             performance_summary_forwardtest = calculate_performance(df_results_forwardtest, config['trading_fee_percent'])
             # Optional: Save forward test results
             # df_results_forwardtest.to_csv(os.path.join(config['data_directory'], 'final_forwardtest_results.csv'))
        else:
             print("Skipping final forward test due to state/index length mismatch.")
    else:
       print("Skipping final forward test due to state prediction error.")


    print(f"--- Strategy execution finished ---")
