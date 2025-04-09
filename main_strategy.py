import pandas as pd
import numpy as np
import yaml
import os
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

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(config, mode='backtest'):
    """Loads data from CSVs, merges, and preprocesses."""
    data_dir = config['data_directory']
    data_files = config['data_files']
    suffix = f"_{mode}"

    all_data = {}
    base_candle_key = 'candles'

    # Load base candle data first
    try:
        candle_file_base = data_files[base_candle_key]
        candle_path = os.path.join(data_dir, f"{candle_file_base}{suffix}.csv")
        print(f"Loading candles: {candle_path}")
        df_candles = pd.read_csv(candle_path, index_col='timestamp', parse_dates=True)
        # Ensure OHLCV columns exist (adjust names if needed from API/CSV)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_candles.columns for col in ohlcv_cols):
             # Attempt mapping if 'o', 'h', 'l', 'c', 'v' exist from Coinglass OHLC format
            ohlc_map = {'t':'start_time', 'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'}
            if all(col in df_candles.columns for col in ohlc_map.keys()):
                 df_candles.rename(columns=ohlc_map, inplace=True)
                 df_candles['close'] = pd.to_numeric(df_candles['close']) # Ensure numeric
                 print("  Renamed candle columns from t,o,h,l,c,v")
            else:
                raise ValueError(f"Candle file missing required columns (e.g., {', '.join(ohlcv_cols)})")
        all_data['candles'] = df_candles[['close', 'volume']].copy() # Keep only close and volume for now
        print(f"  Loaded {len(df_candles)} candle records.")
    except FileNotFoundError:
        print(f"Error: Base candle file not found: {candle_path}")
        return None
    except Exception as e:
        print(f"Error loading candle data: {e}")
        return None

    # Load other data files and merge
    merged_df = all_data['candles']
    for key, file_base in data_files.items():
        if key == base_candle_key:
            continue
        try:
            file_path = os.path.join(data_dir, f"{file_base}{suffix}.csv")
            print(f"Loading {key}: {file_path}")
            df_temp = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

            # Select and rename columns to avoid clashes (customize as needed)
            # Example: keep 'value' column and rename based on key
            if 'value' in df_temp.columns: # Common in Glassnode/CryptoQuant
                 df_to_merge = df_temp[['value']].rename(columns={'value': f"{key}_value"})
            elif 'fundingRate' in df_temp.columns: # Example for funding rates
                 df_to_merge = df_temp[['fundingRate']].rename(columns={'fundingRate': f"{key}_rate"})
            elif 'longShortRatio' in df_temp.columns: # Example for Coinglass LSR
                 df_to_merge = df_temp[['longShortRatio']].rename(columns={'longShortRatio': f"{key}_lsr"})
            elif 'openInterest' in df_temp.columns: # Example for Coinglass OI
                 df_to_merge = df_temp[['openInterest']].rename(columns={'openInterest': f"{key}_oi"})
            # Add more specific column handling based on your downloaded metrics
            else:
                 # Attempt to take the first non-timestamp column if 'value' etc. not found
                 data_cols = [col for col in df_temp.columns if col.lower() not in ['timestamp', 'start_time', 'end_time', 'time']]
                 if data_cols:
                      col_to_use = data_cols[0]
                      df_to_merge = df_temp[[col_to_use]].rename(columns={col_to_use: f"{key}_{col_to_use}"})
                      print(f"  Using column '{col_to_use}' for key '{key}'")
                 else:
                      print(f"  Warning: Could not find suitable data column in {key}. Skipping merge.")
                      continue

            # Merge using outer join and forward fill NaNs
            merged_df = pd.merge(merged_df, df_to_merge, left_index=True, right_index=True, how='outer')
            print(f"  Loaded and merged {len(df_temp)} records for {key}.")

        except FileNotFoundError:
            print(f"Warning: Data file not found for {key}: {file_path}. Skipping.")
        except Exception as e:
            print(f"Warning: Error loading or merging data for {key}: {e}. Skipping.")

    # --- Handle Missing Data ---
    # Forward fill NaNs (common for time series)
    merged_df.ffill(inplace=True)
    # Drop any remaining NaNs (e.g., at the start before first value)
    merged_df.dropna(inplace=True)

    print(f"\nTotal records after merging and initial NaN handling: {len(merged_df)}")
    if merged_df.empty:
        print("Error: No data available after loading and merging.")
        return None

    return merged_df

# --- Feature Engineering ---
def engineer_features(df):
    """Creates features for the HMM model."""
    print("Engineering features...")
    df_feat = df.copy()
    # Example features (adjust and add more based on data and strategy)
    df_feat['price_return'] = df_feat['close'].pct_change().fillna(0)
    df_feat['volatility'] = df_feat['price_return'].rolling(window=10).std().fillna(0) # Example: 10-period volatility

    # Example using other loaded data (replace with actual column names)
    # Check if columns exist before creating features
    # if 'cryptoquant_funding_rate' in df_feat.columns:
    #    df_feat['funding_rate_change'] = df_feat['cryptoquant_funding_rate'].diff().fillna(0)
    # if 'glassnode_active_value' in df_feat.columns:
    #    df_feat['active_addresses_change'] = df_feat['glassnode_active_value'].pct_change().fillna(0)

    # Drop initial rows with NaNs from rolling calculations etc.
    df_feat.dropna(inplace=True)
    print(f"Features engineered. Data shape: {df_feat.shape}")
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

    run_mode = config.get('run_mode', 'backtest') # Default to backtest
    print(f"\n--- Running in {run_mode.upper()} mode ---")

    # 1. Load Data
    df_raw = load_and_preprocess_data(config, mode=run_mode)

    if df_raw is not None and not df_raw.empty:
        # 2. Engineer Features
        df_features = engineer_features(df_raw)
        feature_list = config.get('features', [])
        # Ensure all selected features exist
        feature_list = [f for f in feature_list if f in df_features.columns]
        if not feature_list:
             print("Error: No valid features selected or generated. Check config and feature engineering.")
             exit()
        print(f"Using features for HMM: {feature_list}")


        hmm_model = None
        if run_mode == 'backtest':
            # 3. Train HMM (only in backtest mode)
            hmm_model = train_hmm(df_features, feature_list,
                                  config['hmm_states'],
                                  config['hmm_covariance_type'],
                                  config['hmm_iterations'])
            # Optional: Save the trained model
            # import joblib
            # joblib.dump(hmm_model, 'hmm_model.pkl')

        elif run_mode == 'forwardtest':
             # In forward test mode, load the previously trained model
             print("Forward test mode: Loading pre-trained HMM model...")
             # import joblib
             # try:
             #    hmm_model = joblib.load('hmm_model.pkl')
             #    print("Loaded HMM model from hmm_model.pkl")
             # except FileNotFoundError:
             #    print("Error: hmm_model.pkl not found. Train the model in backtest mode first.")
             #    exit(1)
             # except Exception as e:
             #    print(f"Error loading HMM model: {e}")
             #    exit(1)
             # --- Placeholder: Need to implement saving/loading for forward test ---
             print("Error: Model loading for forward testing is not fully implemented in this example.")
             print("Please train in backtest mode first and add model saving/loading logic (e.g., using joblib).")
             exit(1) # Exit until loading is implemented


        if hmm_model:
            # 4. Predict States
            states = predict_states(hmm_model, df_features, feature_list)

            if states is not None:
                # 5. Generate Signals
                df_signals = generate_signals(df_features, states, config['signal_map'])

                # 6. Run Backtest/Forward Test Simulation
                df_results = run_backtest(df_signals, config['trading_fee_percent'])

                # 7. Calculate Performance
                performance_summary = calculate_performance(df_results, config['trading_fee_percent'])

                # Optional: Save results to CSV
                # results_filename = f"strategy_results_{run_mode}.csv"
                # results_path = os.path.join(config['data_directory'], results_filename)
                # df_results.to_csv(results_path)
                # print(f"Full results saved to {results_path}")

            else:
                print("Skipping strategy simulation due to state prediction error.")
        else:
            print("Skipping strategy simulation due to HMM training/loading error.")
    else:
        print("Could not load or process data. Exiting.")

    print(f"--- Strategy execution finished for {run_mode.upper()} mode. ---")
