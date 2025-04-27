"""
main.py

Entry point for the hybrid crypto trading model pipeline.
- Loads configuration and API keys
- Fetches data using the data_fetcher module (with caching)
- Cleans and feature-engineers the data
- Merges datasets on common datetime range
- Handles Coinglass 2-year data limitation
- Trains HMM and LSTM models
- Runs backtesting and evaluation
"""

from data import DataFetcher, clean_onchain_data, feature_engineering, align_and_merge_datasets, create_synthetic_features
from utils.config import load_api_keys
from models.model_training import (
    train_improved_hmm_regime_detector,
    train_enhanced_lstm_signal_predictor,
    predict_with_ensemble
)
from backtest import backtest_and_evaluate
import time
from datetime import datetime, timedelta
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Caching Utilities
# ---------------------------
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.pkl")

def save_to_cache(data_dict):
    for k, df in data_dict.items():
        df.to_pickle(cache_path(k))

def load_from_cache(keys):
    data = {}
    for k in keys:
        path = cache_path(k)
        if os.path.exists(path):
            data[k] = pd.read_pickle(path)
        else:
            return None  # If any file is missing, don't use cache
    return data

def plot_equity_curve(df_backtest, filepath='equity_curve.png'):
    """Plot equity curve and save to file"""
    plt.figure(figsize=(12, 6))
    plt.plot(df_backtest['equity'])
    plt.title('Strategy Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    print(f"Equity curve saved to {filepath}")

def plot_regime_breakdown(df, filepath='regime_breakdown.png'):
    """Plot regime breakdown and save to file"""
    if 'regime_type' not in df.columns:
        return
    
    plt.figure(figsize=(12, 6))
    regime_counts = df['regime_type'].value_counts()
    plt.bar(regime_counts.index, regime_counts.values)
    plt.title('Market Regime Distribution')
    plt.xlabel('Regime Type')
    plt.ylabel('Count')
    plt.savefig(filepath)
    plt.close()
    print(f"Regime breakdown saved to {filepath}")

def main():
    # Load API keys and print status
    api_keys = load_api_keys()
    print("Loaded API keys:", {k: 'SET' if v else 'NOT SET' for k, v in api_keys.items()})

    # Define time range (last 4 years)
    end_time = int(time.time() * 1000)
    start_time = end_time - 4 * 365 * 24 * 3600 * 1000

    # Coinglass only has up to 2 years of data
    coinglass_start_dt = datetime.now() - timedelta(days=2*365)
    coinglass_start_time = int(coinglass_start_dt.timestamp() * 1000)

    fetcher = DataFetcher()
    data_keys = ["cryptoquant", "glassnode", "coinglass", "bybit_linear_candle", "binance_spot_candle"]

    USE_CACHE = True  # Set to False to force re-fetch

    if USE_CACHE:
        data = load_from_cache(data_keys)
        if data:
            print("Loaded data from cache.")
        else:
            print("Cache not found or incomplete. Fetching data from API...")
            # Fetch all sources except Coinglass for full 4 years
            data = {
                "cryptoquant": fetcher.fetch_cryptoquant(
                    "btc/exchange-flows/inflow?exchange=okx&window=hour", start_time, end_time),
                "glassnode": fetcher.fetch_glassnode(
                    "blockchain/utxo_created_value_median?a=BTC&c=usd&i=1h", start_time, end_time),
                "bybit_linear_candle": fetcher.fetch_bybit_candle(
                    "BTCUSDT", "1h", start_time, end_time, market_type="linear"),
                "binance_spot_candle": fetcher.fetch_binance_candle(
                    "BTCUSDT", "1h", start_time, end_time, market_type="spot"),
            }
            # Fetch Coinglass only for last 2 years
            data["coinglass"] = fetcher.fetch_coinglass(
                "futures/openInterest/ohlc-history?exchange=Binance&symbol=BTCUSDT&interval=1h",
                coinglass_start_time, end_time
            )
            save_to_cache(data)
    else:
        print("Fetching data from API (cache disabled)...")
        data = {
            "cryptoquant": fetcher.fetch_cryptoquant(
                "btc/exchange-flows/inflow?exchange=okx&window=hour", start_time, end_time),
            "glassnode": fetcher.fetch_glassnode(
                "blockchain/utxo_created_value_median?a=BTC&c=usd&i=1h", start_time, end_time),
            "bybit_linear_candle": fetcher.fetch_bybit_candle(
                "BTCUSDT", "1h", start_time, end_time, market_type="linear"),
            "binance_spot_candle": fetcher.fetch_binance_candle(
                "BTCUSDT", "1h", start_time, end_time, market_type="spot"),
        }
        data["coinglass"] = fetcher.fetch_coinglass(
            "futures/openInterest/ohlc-history?exchange=Binance&symbol=BTCUSDT&interval=1h",
            coinglass_start_time, end_time
        )
        save_to_cache(data)

    for k, df in data.items():
        print(f"{k} data shape: {df.shape}")

    # Clean and feature engineer each dataset
    cleaned = {}
    features = {}
    for k, df in data.items():
        try:
            print(f"Cleaning {k} data...")
            cleaned_df = clean_onchain_data(df, timestamp_col="start_time")
            print(f"Feature engineering {k} data...")
            features_df = feature_engineering(cleaned_df)
            cleaned[k] = cleaned_df
            features[k] = features_df
        except Exception as e:
            print(f"Error processing {k}: {e}")

    # Merge datasets on common datetime range
    if features:
        print("Merging datasets on common datetime range...")
        merged = align_and_merge_datasets(features)
        # Add a coinglass_available indicator (1 if coinglass data present, 0 otherwise)
        merged["coinglass_available"] = merged["start_time_coinglass"].notna().astype(int) if "start_time_coinglass" in merged.columns else 0
        print(f"Merged dataset shape: {merged.shape}")
        print(merged.head())
        # Warn about missing Coinglass data for first 2 years
        missing_coinglass = merged["coinglass_available"].value_counts().to_dict()
        print(f"Coinglass data availability (1=present, 0=missing): {missing_coinglass}")
        print("Note: For the first 2 years, Coinglass data will be missing (NaN). Handle this in modeling and backtesting.")

        # Apply synthetic features to compensate for missing data
        print("Creating synthetic features...")
        merged = create_synthetic_features(merged)

        # ---------------------------
        # Model Training
        # ---------------------------
        print("Training improved HMM regime detector...")
        hmm, hmm_scaler, hmm_feature_cols, merged, regime_analysis = train_improved_hmm_regime_detector(merged, n_states=3)
        
        # Print regime analysis
        print("\n==== Market Regime Analysis ====")
        for regime_id, analysis in regime_analysis.items():
            print(f"Regime {regime_id} ({analysis['type']}): Mean Return: {analysis['mean_return']:.4f}, "
                  f"Volatility: {analysis['volatility']:.4f}, Sharpe: {analysis['sharpe']:.2f}, "
                  f"Count: {analysis['count']}")
        
        # Plot regime breakdown
        plot_regime_breakdown(merged)
        
        print("\nTraining enhanced LSTM signal predictor (last 2 years only)...")
        # Use 'close' as the target for LSTM (can be changed to another target)
        lstm_model, lstm_scaler, lstm_features = train_enhanced_lstm_signal_predictor(
            merged, target_col='close', sequence_length=24, epochs=50)

        # ---------------------------
        # Ensemble Prediction
        # ---------------------------
        print("Generating ensemble predictions with regime-adaptive signals...")
        merged = predict_with_ensemble(
            merged, hmm, hmm_scaler, hmm_feature_cols, 
            lstm_model, lstm_scaler, lstm_features, 
            sequence_length=24, regime_analysis=regime_analysis
        )

        # ---------------------------
        # Backtesting & Evaluation
        # ---------------------------
        print("Running backtest with improved position sizing and risk management...")
        # Use regime_adjusted_signal for backtesting
        df_bt, metrics = backtest_and_evaluate(merged, signal_col='regime_adjusted_signal', price_col='close')
        
        # Plot equity curve
        plot_equity_curve(df_bt)

        # Print summary of key performance indicators
        print("\n==== Summary ====")
        for k, v in metrics.items():
            if isinstance(v, float):
                if k in ['avg_win', 'avg_loss']:
                    print(f"{k}: {v:.2%}")
                else:
                    print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    else:
        print("No features to merge.")

if __name__ == "__main__":
    main()