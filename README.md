# UMHackathon 2025 Quant Trading HMM Strategy

## Overview

This project implements a Minimum Viable Product (MVP) for the UMHackathon 2025 Quantitative Trading challenge (Domain 2 by Balaena Quant). It aims to create a Machine Learning model, specifically utilizing Hidden Markov Models (HMMs), to generate an alpha trading strategy based on on-chain cryptocurrency data (from CryptoQuant, Glassnode, Coinglass via the cybotrade.rs API).

The project consists of two main Python scripts:
1.  `download_data.py`: Fetches required data using the cybotrade.rs API and saves it locally.
2.  `main_strategy.py`: Loads the downloaded data, trains an HMM model, simulates a spot trading strategy, and evaluates its performance.

## Project Structure

```
.
├── config.yaml           # Configuration file for all settings
├── download_data.py      # Script to download data from API
├── main_strategy.py      # Script for ML model, backtesting, and evaluation
├── data/                 # Directory to store downloaded data (created automatically if needed)
└── README.md             # This file
```

## Requirements

* Python 3.7+
* Required Python libraries:
    * `requests`
    * `pandas`
    * `PyYAML`
    * `NumPy`
    * `hmmlearn`
    * `scikit-learn` (used implicitly by hmmlearn, explicitly for potential future enhancements)

### Package Installation

1. Create and activate a virtual environment:

   **Linux/macOS**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   **Windows**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

> Note: To deactivate the virtual environment when you're done, simply run `deactivate`

## Setup

1. **Clone/Download**: Get the project files (download_data.py, main_strategy.py, config.yaml, README.md).
2. **Install Dependencies**: Run the pip install command above in your terminal.
3. **API Key**: This project requires an API key from cybotrade.rs to download data.
   - Obtain your API key from the provider.
   - Set Environment Variable: You must set your API key as an environment variable named `CYBOTRADE_API_KEY`. Do not hardcode the key in the scripts.

   **Linux/macOS**:
   ```bash
   export CYBOTRADE_API_KEY='YOUR_API_KEY_HERE'
   # Add this line to your .bashrc or .zshrc for persistence
   ```

   **Windows (Command Prompt)**:
   ```cmd
   set CYBOTRADE_API_KEY=YOUR_API_KEY_HERE
   ```

   **Windows (PowerShell)**:
   ```powershell
   $env:CYBOTRADE_API_KEY = 'YOUR_API_KEY_HERE'
   ```
   > Note: You might need to set this in system environment variables for persistence.

4. **Create Data Directory**: Although the script attempts to create it, you can manually create the data directory:
   ```bash
   mkdir data
   ```

## Configuration (config.yaml)

All settings for both data downloading and the ML strategy are controlled via the `config.yaml` file. Before running the scripts, review and customize this file:

- **data_directory**: Specifies the folder for data storage (default: `data`).
- **metrics_to_download**: Define which metrics and candle data to fetch from the API. Use the format specified in the comments (e.g., `cryptoquant: ["btc/market-data/..."]`). Refer to the cybotrade.rs documentation for available endpoints/topics.
- **\*_start_time_ms / \*_end_time_ms**: Set the desired time ranges (in Unix milliseconds) for backtesting and forward testing data downloads.
- **data_files**: Maps internal keys (used in main_strategy.py) to the base filenames generated by download_data.py.
- **features**: List of column names to be used for training the HMM model.
- **hmm_settings**: Configure the number of HMM states, covariance type, and iterations.
- **trading_strategy_settings**: Set the trading fee and map HMM states to trading signals (0=Hold, 1=Buy, -1=Sell).
- **run_mode**: Set to `backtest` to train the model and test on the backtest period, or `forwardtest` to run a pre-trained model.

## Running the Scripts

### Step 1: Download Data

Run the data downloader script first:

```bash
python download_data.py --config config.yaml
```

> Note: Depending on the amount of data requested, this step can take a significant amount of time and API calls.

### Step 2: Run Backtest

Ensure `run_mode: backtest` is set in your config.yaml. Then run:

```bash
python main_strategy.py --config config.yaml
```

This script will:
- Load data from the CSV files
- Preprocess the data and engineer features
- Train the HMM model on the backtest period data
- Predict hidden states
- Generate trading signals
- Run the backtest simulation applying fees
- Calculate and print performance metrics

### Step 3: Run Forward Test

1. Ensure forward test data exists from Step 1
2. Implement model saving/loading in main_strategy.py
3. Change `run_mode: forwardtest` in config.yaml
4. Run:
   ```bash
   python main_strategy.py --config config.yaml
   ```

## Customization & Potential Improvements

- **Feature Engineering**: Experiment with different features derived from the base metrics in main_strategy.py.
- **HMM Tuning**: Adjust the number of states, covariance type, and iterations in config.yaml. Explore different HMM initialization parameters.
- **Signal Logic**: Refine the mapping between HMM states and trading signals in config.yaml. Implement more sophisticated signal logic (e.g., avoid consecutive signals, use state probabilities).
- **Error Handling**: Add more robust error handling, especially in the data download script (e.g., retries for API calls).
- **Model Persistence**: Fully implement model saving and loading using joblib or pickle for proper forward testing.
- **Data Validation**: Add checks for data quality and consistency after loading.
- **Visualization**: Add code to plot equity curves, drawdowns, or feature distributions.

## Sources and Related Content

Refer to the cybotrade.rs documentation for API details and additional resources.

Presentation slide: https://gamma.app/docs/DATABAES-5049dexzc2ktecg 
