import pandas as pd
import numpy as np
import yaml
import os
import argparse

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
        candle_file_base = data_files.get(base_candle_key, None)
        if not candle_file_base:
            raise ValueError(f"'{base_candle_key}' missing in config['data_files']")
        candle_path = os.path.join(data_dir, f"{candle_file_base}{suffix}.csv")
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
    for key, file_base in data_files.items():
        if key == base_candle_key:
            continue
        try:
            file_path = os.path.join(data_dir, f"{file_base}{suffix}.csv")
            print(f"Loading {key}: {file_path}")
            df_temp = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

            col_to_use = None
            # --- Define TARGET column name explicitly based on key ---
            target_col_name = None
            if key == 'cryptoquant_funding':
                target_col_name = 'cryptoquant_funding_rate' # Desired final name
                if 'fundingRate' in df_temp.columns: col_to_use = 'fundingRate'
                elif 'value' in df_temp.columns: col_to_use = 'value'
            elif key == 'glassnode_active':
                target_col_name = 'glassnode_active_value' # Desired final name
                if 'value' in df_temp.columns: col_to_use = 'value'
                elif 'v' in df_temp.columns: col_to_use = 'v' # Use 'v' if 'value' not present
            elif key == 'cryptoquant_inflow': # Add specific cases as needed
                 target_col_name = 'cryptoquant_inflow_value'
                 if 'value' in df_temp.columns: col_to_use = 'value'
            elif key == 'glassnode_tx':
                 target_col_name = 'glassnode_tx_value'
                 if 'value' in df_temp.columns: col_to_use = 'value'
                 elif 'v' in df_temp.columns: col_to_use = 'v'
            elif key == 'coinglass_oi':
                 target_col_name = 'coinglass_oi_value'
                 if 'openInterest' in df_temp.columns: col_to_use = 'openInterest'
                 elif 'c' in df_temp.columns: col_to_use = 'c' # Close OI value


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

# --- Feature Engineering (Simplified) ---
def engineer_features(df, final_column_names, config_features):
    """Creates features for the HMM model using known column names."""
    print("Engineering features...")
    if df is None or df.empty:
         print("Error: Cannot engineer features on empty DataFrame.")
         return None
    df_feat = df.copy()

	 # --- ADD THIS LINE ---
    print(f"\nDEBUG: Columns in df_feat at start of engineer_features: {df_feat.columns.tolist()}\n")
    # --- END OF ADDED LINE ---

    # --- Calculate Price-Based Features ---
    df_feat['price_return'] = df_feat['close'].pct_change()
    df_feat['volatility'] = df_feat['price_return'].rolling(window=10).std()

    # --- Add other derived features if desired ---
    # Example: Use the final column name for funding rate if available
    funding_col = final_column_names.get('cryptoquant_funding') # Get actual column name used
    if funding_col and funding_col in df_feat.columns:
         df_feat['funding_rate_change'] = df_feat[funding_col].diff()
         # You might want this change INSTEAD of the raw rate in config features
    else:
         # Handle case where funding rate wasn't successfully loaded/named
         if 'cryptoquant_funding_rate' in config_features: # Check if config expected it
              print("Warning: Funding rate column expected but not found for feature engineering.")


    active_col = final_column_names.get('glassnode_active') # Get actual column name used
    if active_col and active_col in df_feat.columns:
         df_feat['active_addresses_change'] = df_feat[active_col].pct_change()
    else:
         if 'glassnode_active_value' in config_features: # Check if config expected it
              print("Warning: Active addresses column expected but not found for feature engineering.")


    # --- Clean up NaNs introduced by NEW calculations ---
    df_feat.fillna(method='ffill', inplace=True) # Fill NaNs from diff/pct_change/rolling
    df_feat.fillna(0, inplace=True)             # Fill any remaining NaNs (e.g., at start)
    # Consider removing initial rows if rolling/diff creates leading NaNs that aren't filled
    # df_feat.dropna(inplace=True) # Alternative: drop rows with any NaNs


    print(f"Features engineered. Data shape: {df_feat.shape}")
    print(f"Engineered columns available: {df_feat.columns.tolist()}")


    # --- Final Check: Ensure features listed in config EXIST ---
    missing_features = [f for f in config_features if f not in df_feat.columns]
    if missing_features:
        print(f"FATAL Error: Features listed in config are missing from DataFrame after engineering: {missing_features}")
        print(f"Check config 'features' list and engineered columns above.")
        return None # Stop execution if required features aren't there

    return df_feat

# --- HMM Model ---
def train_hmm(df, features, n_states, covariance_type, n_iter):
    """Trains the Gaussian HMM model."""
    print(f"Training HMM with {n_states} states...")
    X = df[features].values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
         print("Warning: NaNs or Infs found in feature data. Replacing with 0.")
         X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) == 0:
        print("Error: No data available for HMM training.")
        return None

    try:
        model = hmm.GaussianHMM(n_components=n_states,
                                covariance_type=covariance_type,
                                n_iter=n_iter,
                                random_state=42, # for reproducibility
                                verbose=False) # Set to True for convergence details
        model.fit(X)
        print("HMM training complete.")
        return model
    except Exception as e:
        print(f"Error during HMM training: {e}")
        # Consider falling back to fewer states or different covariance if convergence fails
        return None


def predict_states(model, df, features):
    """Predicts hidden states using the trained HMM."""
    print("Predicting hidden states...")
    X = df[features].values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
         print("Warning: NaNs or Infs found in feature data for prediction. Replacing with 0.")
         X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) == 0:
        print("Error: No data available for state prediction.")
        return None

    try:
        states = model.predict(X)
        return states
    except Exception as e:
        print(f"Error during state prediction: {e}")
        return None


# --- Signal Generation ---
def generate_signals(df, states, signal_map):
    """Generates trading signals based on predicted states."""
    print("Generating trading signals...")
    if states is None:
        print("Error: Cannot generate signals, states are None.")
        return df
    df['state'] = states
    df['signal'] = df['state'].map(signal_map).fillna(0) # Map states to signals (Buy=1, Sell=-1, Hold=0)

    # Ensure signal is integer
    df['signal'] = df['signal'].astype(int)

     # Optional: Prevent consecutive Buy/Sell signals (force hold after entry)
    # df['signal'] = df['signal'].diff().fillna(0).clip(-1, 1).astype(int) # Basic diff logic

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

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM Trading Strategy Backtest.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)

    run_mode = config.get('run_mode', 'backtest')
    print(f"\n--- Running in {run_mode.upper()} mode ---")

    # 1. Load Data -> Now returns df_raw AND final_column_names map
    df_raw, final_column_names = load_and_preprocess_data(config, mode=run_mode)

	# --- ADD THIS LINE ---
    if df_raw is not None: print(f"\nDEBUG: Columns in df_raw after loading: {df_raw.columns.tolist()}\n")
    # --- END OF ADDED LINE ---

    # Make sure final_column_names is a dict even if loading fails partially
    if final_column_names is None: final_column_names = {}

    if df_raw is not None and not df_raw.empty:
        # Get feature list from config
        feature_list_from_config = config.get('features', [])
        if not feature_list_from_config:
             print("Error: 'features' list is empty in config.yaml")
             exit()

        # 2. Engineer Features -> Pass names map and config list
        df_features = engineer_features(df_raw, final_column_names, feature_list_from_config)

        # --- Check if df_features is valid before proceeding ---
        if df_features is None or df_features.empty:
             print("Error: Feature engineering failed or produced empty DataFrame. Exiting.")
             exit()

        # Use only the features specified in the config that actually exist now
        feature_list = [f for f in feature_list_from_config if f in df_features.columns]
        if not feature_list:
             print("Error: None of the features specified in config exist after engineering. Exiting.")
             exit()
        print(f"Using features for HMM: {feature_list}")

        # --- HMM Training and Backtesting (rest is the same) ---
        hmm_model = None
        if run_mode == 'backtest':
            hmm_model = train_hmm(df_features, feature_list,
                                  config['hmm_states'],
                                  config['hmm_covariance_type'],
                                  config['hmm_iterations'])
            # Optional: Save model
        elif run_mode == 'forwardtest':
             # Implement model loading here
             print("Error: Model loading for forward testing not implemented.")
             exit(1)

        if hmm_model:
            states = predict_states(hmm_model, df_features, feature_list)
            if states is not None:
                df_signals = generate_signals(df_features, states, config['signal_map'])
                df_results = run_backtest(df_signals, config['trading_fee_percent'])
                performance_summary = calculate_performance(df_results, config['trading_fee_percent'])
                # Optional: Save results
            else:
                print("Skipping strategy simulation due to state prediction error.")
        else:
            print("Skipping strategy simulation due to HMM training/loading error.")
    else:
        print("Could not load or process data. Exiting.")

    print(f"--- Strategy execution finished for {run_mode.upper()} mode. ---")
