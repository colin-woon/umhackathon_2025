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

# --- Feature Engineering (Simplified) ---
def engineer_features(df, final_column_names, config_features):
    """Creates features for the HMM model using known column names."""
    print("Engineering features...")
    if df is None or df.empty:
         print("Error: Cannot engineer features on empty DataFrame.")
         return None
    df_feat = df.copy()

	 # --- ADD THIS LINE ---
    # print(f"\nDEBUG: Columns in df_feat at start of engineer_features: {df_feat.columns.tolist()}\n")
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

# --- HMM Model (Modified for Scaled Data) ---
def train_hmm(X_scaled, n_states, covariance_type, n_iter): # Takes X_scaled directly
    """Trains the Gaussian HMM model on pre-scaled data."""
    print(f"Training HMM with {n_states} states on scaled data...")
    # X = df[features].values # REMOVED - data is already scaled numpy array X_scaled

    # Removed NaN checks here, assuming scaling handled them or they were checked before scaling

    if X_scaled is None or len(X_scaled) == 0:
        print("Error: No scaled data available for HMM training.")
        return None

    try:
        model = hmm.GaussianHMM(n_components=n_states,
                                covariance_type=covariance_type,
                                n_iter=n_iter,
                                random_state=42,
                                verbose=True) # Set verbose=True to see convergence progress
        model.fit(X_scaled)
        if not model.monitor_.converged:
             print(f"Warning: HMM did not converge in {n_iter} iterations!")
        else:
             print("HMM training complete (converged).")
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

# --- Main Execution (Refactored) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM Trading Strategy Backtest.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)

    run_mode = config.get('run_mode', 'backtest')
    print(f"\n--- Running in {run_mode.upper()} mode ---")

    # 1. Load Data
    df_raw, final_column_names = load_and_preprocess_data(config, mode=run_mode)
    if final_column_names is None: final_column_names = {}

    if df_raw is not None and not df_raw.empty:
        # Get feature list from config
        feature_list_from_config = config.get('features', [])
        if not feature_list_from_config: exit("Error: 'features' list empty.")

        # 2. Engineer Features
        df_features = engineer_features(df_raw, final_column_names, feature_list_from_config)
        if df_features is None or df_features.empty: exit("Error: Feature engineering failed.")

        # Filter to features actually available
        feature_list = [f for f in feature_list_from_config if f in df_features.columns]
        if not feature_list: exit("Error: None of config features exist after engineering.")
        print(f"\nUsing features: {feature_list}")

        # --- Prepare Data (Scaling depends on mode) ---
        X_scaled = None
        scaler = None
        hmm_model = None
        best_n_states = -1 # Keep track of the number of states used

        if run_mode == 'backtest':
            # --- Scale Data ---
            X_train = df_features[feature_list].values
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                 X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            if len(X_train) == 0: exit("Error: No data for scaling.")

            print("\nFitting scaler and scaling features...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            print("Features scaled.")

            # --- Find Best HMM Model using the new function ---
            hmm_model, best_n_states, best_bic, best_aic = find_best_hmm_model(X_scaled, config, feature_list)

            if hmm_model is None:
                exit("Error: Failed to find a suitable HMM model.")

            print(f"\n--- Proceeding with selected model: n_states = {best_n_states} ---")

            # --- Save Scaler & Best Model ---
            try:
                os.makedirs('models', exist_ok=True)
                joblib.dump(scaler, 'models/scaler.pkl')
                joblib.dump(hmm_model, 'models/hmm_model.pkl')
                print("Saved scaler and best HMM model to 'models/' directory.")
            except Exception as e:
                print(f"Warning: Could not save model/scaler: {e}")

        elif run_mode == 'forwardtest':
            # --- Load Model & Scaler ---
            print("\nForward test mode: Loading pre-trained HMM model and scaler...")
            try:
                hmm_model = joblib.load('models/hmm_model.pkl')
                scaler = joblib.load('models/scaler.pkl')
                print("Loaded HMM model and scaler from 'models/' directory.")
                best_n_states = hmm_model.n_components # Get n_states from loaded model
            except FileNotFoundError:
                 exit("Error: hmm_model.pkl or scaler.pkl not found. Run backtest first.")
            except Exception as e:
                 exit(f"Error loading model or scaler: {e}")

            # --- Scale Forward Data ---
            X_forward = df_features[feature_list].values
            if np.any(np.isnan(X_forward)) or np.any(np.isinf(X_forward)):
                 X_forward = np.nan_to_num(X_forward, nan=0.0, posinf=0.0, neginf=0.0)
            if len(X_forward) == 0: exit("Error: No data for forward test scaling.")

            print("\nScaling forward test features using loaded scaler...")
            try:
                 X_scaled = scaler.transform(X_forward) # Use transform()
                 print("Forward test features scaled.")
            except ValueError as ve:
                 # Catch the feature number mismatch error explicitly
                 print(f"Error: Feature mismatch during forward test scaling: {ve}")
                 print("Ensure the 'features' list in config.yaml matches the one used during backtest.")
                 exit()
            except Exception as e:
                 exit(f"Error scaling forward test data: {e}")


        # --- Prediction and Backtesting ---
        if hmm_model and X_scaled is not None and len(X_scaled) > 0:
            # 4. Predict States
            states = predict_states(hmm_model, X_scaled)

            # --- HMM State Analysis (Runs on the best model) ---
            if states is not None:
                print("\n--- HMM State Analysis (Best Model) ---")
                print(f"Number of States: {best_n_states}")
                # (Your existing state analysis print logic here - using hmm_model / best_n_states)
                print(f"Features used for analysis: {feature_list}")
                print("Mean of Scaled Features for each State:")
                try:
                      for i in range(hmm_model.n_components):
                           print(f"  State {i}:")
                           for feature_name, mean_val in zip(feature_list, hmm_model.means_[i]):
                                print(f"    {feature_name}: {mean_val:.4f}")
                except Exception as e:
                      print(f"  Could not print HMM means: {e}")

                if len(states) == len(df_features.index):
                      df_features['state'] = states
                      print("\nAverage Raw Price Return per State:")
                      try:
                           if 'price_return' in df_features.columns:
                                print(df_features.groupby('state')['price_return'].mean())
                           else: print("  'price_return' column not found.")
                      except Exception as e: print(f"  Could not calculate average return per state: {e}")
                else: print("  Skipping average return per state due to length mismatch.")
                print("--------------------------\n")


                # Alignment Logic (Keep as before)
                if len(states) == len(df_features.index):
                     df_features['state'] = states
                else:
                     print(f"Warning: Length mismatch ...") # Abridged
                     # Basic padding/trimming
                     if len(states) < len(df_features.index):
                           padding = np.full(len(df_features.index) - len(states), states[-1] if len(states)>0 else 0)
                           states = np.concatenate((states, padding))
                           df_features['state'] = states
                     elif len(states) > len(df_features.index):
                           states = states[:len(df_features.index)]
                           df_features['state'] = states
                     else: exit("Error aligning states.")

                # 5. Generate Signals
                # !!! REMINDER: Update signal_map in config based on analysis of BEST n_states !!!
                signal_map_from_config = config.get('signal_map', {})
                print(f"Using signal map from config: {signal_map_from_config}")
                if best_n_states != -1 and len(signal_map_from_config) != best_n_states:
                     print(f"Warning: Signal map in config has {len(signal_map_from_config)} entries, but best model has {best_n_states} states. Defaulting undefined states to 0 (Hold).")

                df_signals = generate_signals(df_features, signal_map_from_config)

                # 6. Run Backtest
                df_results = run_backtest(df_signals, config['trading_fee_percent'])

                # 7. Calculate Performance
                performance_summary = calculate_performance(df_results, config['trading_fee_percent'])
            else:
                print("Skipping strategy simulation due to state prediction error.")
        else:
             print("Skipping prediction/backtesting as HMM model or scaled data is invalid.")
    else:
        print("Could not load or process data. Exiting.")

    print(f"--- Strategy execution finished for {run_mode.upper()} mode. ---")
