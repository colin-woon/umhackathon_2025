<div align="center">
  <h1 style="font-size: 3em;">🤖 libQT 📈</h1>
  <p><em>Hidden Markov Model Trading Strategy Framework</em></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/UMHackathon-2025-purple.svg" alt="UMHackathon 2025">
  </p>
</div>

<div align="center"><h1>Overview</h1></div>

**libQT** is a user-friendly Python framework for developing, backtesting, and evaluating quantitative trading strategies using Hidden Markov Models (HMMs). Designed for UMHackathon 2025 (Domain 2 by Balaena Quant), it streamlines the process of using on-chain cryptocurrency data (from CryptoQuant, Glassnode, Coinglass via the cybotrade.rs API) to generate and test alpha trading strategies.

The workflow automates:
1. **Data Acquisition**: Fetches metrics using the cybotrade.rs API (`download_data.py`).
2. **Feature Engineering**: Calculates features based on `config.yaml`, focusing on on-chain and market data.
3. **Feature Selection**: Applies correlation analysis to remove redundant features.
4. **Model Training**: Selects the optimal number of HMM states (AIC/BIC) and trains the model.
5. **Signal Generation**: Maps HMM states to trading signals (Buy/Sell/Hold), with optional Bayesian optimization.
6. **Backtesting & Forward Testing**: Simulates strategy performance on historical and unseen data, including trading fees.
7. **Statistical Validation**: Runs T-tests, permutation tests, and state stability checks to assess robustness.
8. **Evaluation & Reporting**: Outputs performance metrics, plots, and saves artifacts for analysis.

<hr>

# Presentation Slide Link
[https://drive.google.com/file/d/1UlPR5IOaHXM3yFykAmw-tSijgYSNkQ9l/view?usp=sharing]

## Project Structure

```
config.yaml
README.md
requirements.txt
download_data.py
generate_signal_maps.py
main_strategy.py
data/
outputs/
```

## Requirements

- Python 3.8+
- Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup

1. **API Key**: Set your cybotrade.rs API key as an environment variable named `CYBOTRADE_API_KEY`.
   ```bash
   export CYBOTRADE_API_KEY='YOUR_API_KEY_HERE'
   ```
2. **Data Directory**: The `data/` and `outputs/` folders are created automatically.

## Configuration (`config.yaml`)

- `data_directory / output_base_dir`: Paths for data and results.
- `page_limit`: pagination limit, min 1000, max 4500, refer API docs
- `data_timeframe`: Options: "daily", "hourly", used for calculate_performance for Sharpe Ratio
- `metrics_to_download`: API endpoints for data fetching.
- `backtest_start_time_ms / forward_test_end_time_ms`: Time ranges for data in Unix timestamp
- `data_files`: Mapping keys to downloaded filenames.
- `permutation_test_runs`: Number of runs for the permutation test.
- `feature_corr_threshold`: Threshold for dropping highly correlated features.
- `features`: List of features to consider for engineering and selection.
- `hmm_settings`: Parameters for HMM training, including state selection.
- `trading_fee_percent`: Trading fees.
- `signal_map`: Maps HMM states (0, 1, ...) to signals (1=Buy, -1=Sell, 0=Hold). Update after analyzing HMM state characteristics.

## Running the Workflow

### 1. Download Data

```bash
python download_data.py --config config.yaml
```

### 2. Run Strategy Development & Evaluation

```bash
python main_strategy.py --config config.yaml
```

This will:
- Load and preprocess data
- Engineer and select features
- Train and select HMM model
- Predict states and generate signals
- Run backtest and forward test
- Perform statistical validation
- Save results to `outputs/`

**Iterate:**
- Analyze HMM state characteristics in the output
- Update `signal_map` in `config.yaml` based on your interpretation
- Re-run `main_strategy.py`
- Optionally, enable Bayesian optimization for automated signal map search

## Statistical Validation Methods

- **AIC/BIC for Model Selection**: Using a range of states, calculates HMM complexity and picks the most suitable one
- **Correlation Analysis**: Removes redundant features.
- **State Stability Check**: Ensures HMM states are stable during backtest periods when splitted
- **T-Test on Returns**: Checks if strategy returns are statistically significant.
- **Permutation Testing**: Compares strategy performance to random signal shuffles.

## Signal Mapping & Optimization

- **Manual Mapping**: Assign 1 (Buy), -1 (Sell), or 0 (Hold) to each HMM state after reviewing state characteristics in the `config.yaml` file.

- **Bayesian Optimization**: Use the `generate_signal_maps.py` script to automatically search for optimal signal mappings:
  ```bash
  python generate_signal_maps.py \
    --run_dir outputs/run_YYYYMMDD_HHMMSS \  # Path to the run output directory
    --states N \                             # Number of HMM states (e.g., 8)
    --config config.yaml \                   # Base config (for fee)
    --n_calls 1000 \                        # Number of optimization iterations
    --patience 100 \                        # Early stopping patience
    --optimize_metric total_return          # Metric to optimize (total_return or sharpe)
  ```

  The script will:
  - Load model artifacts from the specified run directory
  - Use Bayesian optimization to search for the best signal map
  - Display both backtest and forward test performance for the best map found
  - Early stop if no improvement is seen after `patience` iterations

  Note: Run `main_strategy.py` first to generate the necessary artifacts.

## Challenges & Solutions

1. **Data Quality & Availability**
   - **Issue**: Missing data points and inconsistent timestamps across different API endpoints
   - **Solution**: Implemented robust data alignment and forward/backward filling strategies

2. **HMM State Interpretation**
   - **Issue**: HMM states can be abstract and difficult to interpret
   - **Solution**: Added detailed state characteristic analysis in outputs, showing mean feature values per state

3. **Performance Validation**
   - **Issue**: Risk of overfitting when manually creating signal maps
   - **Solution**: Implemented multiple validation methods (permutation tests, state stability checks)

4. **Known Limitations**
   - Permutation tests currently only support daily timeframes
   - Bayesian optimization can be time-consuming for large numbers of states
   - State stability might vary significantly in highly volatile market conditions

5. **Technical Challenges**
   - **Issue**: Memory usage with large datasets and multiple permutation runs
   - **Solution**: Implemented efficient data structures and optional early stopping

6. **API Rate Limits**
   - **Issue**: cybotrade.rs API has request limits
   - **Solution**: Added pagination handling and request delays in `download_data.py`

## Improvements & Extensions

- Add advanced features or alternative models
- Adapt for different timeframes (Permutation tests still doesnt do that, only works for daily)
- Implement risk management (position sizing, stop-loss)
- Expand Bayesian optimization to tune more parameters

## References
- [cybotrade.rs API documentation](https://cybotrade.rs)
- Hidden Markov Models in Quantitative Finance
- On-chain analytics resources

---

*libQT is developed for educational and research purposes. Use at your own risk.*
