"""
data_fetcher.py

Module for fetching on-chain, derivatives, and exchange data using the Cybotrade REST API.
Supports CryptoQuant, Glassnode, Coinglass, Bybit, Binance, and more.

Usage:
    from data_fetcher import DataFetcher
    df = DataFetcher().fetch_all(start_time, end_time)
"""

import os
import pandas as pd
from dotenv import load_dotenv
import time
import requests
from typing import Optional, Dict

# Load API keys from .env
load_dotenv()
CYBOTRADE_API_KEY = os.getenv("CYBOTRADE_API_KEY") or os.getenv("CRYPTOQUANT_API_KEY")

class DataFetcher:
    def __init__(self):
        self.api_key = CYBOTRADE_API_KEY
        if not self.api_key:
            raise ValueError("Cybotrade API key not found in environment variables.")

    def fetch(self, provider: str, endpoint: str, params: dict) -> pd.DataFrame:
        """
        Generic fetch method for any provider supported by Cybotrade.
        """
        base_url = f"https://api.datasource.cybotrade.rs/{provider}/{endpoint}"
        headers = {"X-API-KEY": self.api_key}
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code == 429:
            # Handle rate limit: sleep until reset
            reset_ts = int(response.headers.get("X-Api-Limit-Reset-Timestamp", 0)) // 1000
            now = int(time.time())
            wait = max(reset_ts - now, 1)
            print(f"Rate limit hit. Sleeping for {wait} seconds...")
            time.sleep(wait)
            response = requests.get(base_url, params=params, headers=headers)
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} {response.text}")
        data = response.json()
        # Some endpoints return data under "data", others as a list
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return pd.DataFrame(data)

    def fetch_cryptoquant(self, endpoint: str, start_time: int, end_time: int, flatten: bool = True) -> pd.DataFrame:
        params = {"start_time": start_time, "end_time": end_time, "flatten": flatten}
        return self.fetch("cryptoquant", endpoint, params)

    def fetch_glassnode(self, endpoint: str, start_time: int, end_time: int, flatten: bool = True) -> pd.DataFrame:
        params = {"start_time": start_time, "end_time": end_time, "flatten": flatten}
        return self.fetch("glassnode", endpoint, params)

    def fetch_coinglass(self, endpoint: str, start_time: int, end_time: int) -> pd.DataFrame:
        params = {"start_time": start_time, "end_time": end_time}
        return self.fetch("coinglass", endpoint, params)

    def fetch_bybit_candle(self, symbol: str, interval: str, start_time: int, end_time: int, market_type: str = "linear") -> pd.DataFrame:
        # market_type: "spot", "linear", "inverse"
        provider = f"bybit-{market_type}"
        endpoint = "candle"
        params = {"symbol": symbol, "interval": interval, "start_time": start_time, "end_time": end_time}
        return self.fetch(provider, endpoint, params)

    def fetch_binance_candle(self, symbol: str, interval: str, start_time: int, end_time: int, market_type: str = "spot") -> pd.DataFrame:
        provider = f"binance-{market_type}"
        endpoint = "candle"
        params = {"symbol": symbol, "interval": interval, "start_time": start_time, "end_time": end_time}
        return self.fetch(provider, endpoint, params)

    def fetch_all(self, start_time: int, end_time: int) -> Dict[str, pd.DataFrame]:
        """
        Fetches a sample of all available data sources for the given time range.
        Returns a dictionary of DataFrames.
        """
        # Example endpoints (customize as needed)
        cryptoquant_endpoint = "btc/exchange-flows/inflow?exchange=okx&window=hour"
        glassnode_endpoint = "blockchain/utxo_created_value_median?a=BTC&c=usd&i=1h"
        coinglass_endpoint = "futures/openInterest/ohlc-history?exchange=Binance&symbol=BTCUSDT&interval=1h"

        data = {
            "cryptoquant": self.fetch_cryptoquant(cryptoquant_endpoint, start_time, end_time),
            "glassnode": self.fetch_glassnode(glassnode_endpoint, start_time, end_time),
            "coinglass": self.fetch_coinglass(coinglass_endpoint, start_time, end_time),
            "bybit_linear_candle": self.fetch_bybit_candle("BTCUSDT", "1h", start_time, end_time, market_type="linear"),
            "binance_spot_candle": self.fetch_binance_candle("BTCUSDT", "1h", start_time, end_time, market_type="spot"),
        }
        return data

if __name__ == "__main__":
    # Example usage
    end_time = int(time.time() * 1000)
    start_time = end_time - 7 * 24 * 3600 * 1000  # last 7 days
    fetcher = DataFetcher()
    data = fetcher.fetch_all(start_time, end_time)
    for k, df in data.items():
        print(f"{k} data shape: {df.shape}")