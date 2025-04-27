"""
data_cleaning.py

Utility functions for cleaning, transforming, and feature engineering on raw crypto data.
"""

import pandas as pd
import numpy as np

def clean_onchain_data(df: pd.DataFrame, timestamp_col: str = "start_time") -> pd.DataFrame:
    """
    Clean the data:
      - Convert UNIX timestamps (in ms) to datetime objects.
      - Sort data by the timestamp.
      - Remove missing values and reset the index.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df[timestamp_col], unit="ms")
    
    # Convert numeric columns from object to float type if necessary
    for col in df.columns:
        if col not in [timestamp_col, "datetime"]:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                print(f"Could not convert column '{col}' to numeric")
    
    df.sort_values("datetime", inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def feature_engineering(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Generate features for appropriate numeric columns:
      - Percentage change of each column
      - Moving average over a specified window
      - Multiple timeframe momentum indicators
      - Volatility-based features
      - Market microstructure indicators
    """
    df = df.copy()
    
    # Identify columns to exclude from feature engineering
    exclude_cols = ['start_time', 'end_time', 't']
    # Find all numeric columns excluding the ones in exclude_cols
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols and not col.endswith('_pct_change') 
                   and not col.endswith(f'_ma_{window}')]
    
    # Original features
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_pct_change"] = df[col].pct_change() * 100  # Calculate percentage change
            df[f"{col}_ma_{window}"] = df[col].rolling(window=window).mean()
    
    # Enhanced features - Multiple timeframe indicators
    price_cols = [col for col in numeric_cols if 'close' in col or 'price' in col]
    volume_cols = [col for col in numeric_cols if 'volume' in col]
    
    # Apply enhanced features only to price and volume columns
    for col in price_cols:
        # Multiple timeframe momentum
        for window_size in [12, 24, 48, 96]:  # Multiple horizons (hours)
            # Rate of Change (ROC)
            df[f"{col}_roc_{window_size}h"] = df[col].pct_change(window_size) * 100
            
            # RSI with multiple lookbacks
            delta = df[col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=window_size).mean()
            avg_loss = loss.rolling(window=window_size).mean()
            rs = avg_gain / avg_loss
            df[f"{col}_rsi_{window_size}h"] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            std = df[col].rolling(window=window_size).std()
            middle_band = df[col].rolling(window=window_size).mean()
            df[f"{col}_bb_upper_{window_size}h"] = middle_band + 2 * std
            df[f"{col}_bb_lower_{window_size}h"] = middle_band - 2 * std
            df[f"{col}_bb_width_{window_size}h"] = (df[f"{col}_bb_upper_{window_size}h"] - df[f"{col}_bb_lower_{window_size}h"]) / middle_band
            
        # Volatility metrics
        for window_size in [24, 48, 96]:
            df[f"{col}_volatility_{window_size}h"] = df[col].rolling(window_size).std() / df[col].rolling(window_size).mean() * 100
        
        # Price momentum divergence
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns:
            # Average True Range (ATR)
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_24h'] = df['tr'].rolling(24).mean()
            
            # Oscillator metrics
            df['close_to_high_ratio'] = df['close'] / df['high'].rolling(48).max()
            df['close_to_low_ratio'] = df['close'] / df['low'].rolling(48).min()
    
    # Volume and liquidity features
    for col in volume_cols:
        for window_size in [24, 48, 96]:
            # Volume momentum
            df[f"{col}_momentum_{window_size}h"] = df[col] / df[col].rolling(window_size).mean()
            
            # Volume z-score (normalized)
            df[f"{col}_zscore_{window_size}h"] = (df[col] - df[col].rolling(window_size).mean()) / df[col].rolling(window_size).std()
    
    # Create synthetic features for missing data
    if 'close' in df.columns and 'volume' in df.columns:
        # Synthetic funding rate proxy
        df['synthetic_funding'] = df['close'].pct_change(8) * df['volume'].pct_change(8).clip(-1, 1)
        df['synthetic_funding_ma'] = df['synthetic_funding'].rolling(24).mean()
        
        # Synthetic liquidity metric
        if 'high' in df.columns and 'low' in df.columns:
            # High-Low/Volume as liquidity proxy
            df['illiquidity'] = (df['high'] - df['low']) / df['volume']
            df['illiquidity_zscore'] = (df['illiquidity'] - df['illiquidity'].rolling(168).mean()) / df['illiquidity'].rolling(168).std()
    
    # Create futures basis features if available
    basis_cols = [col for col in numeric_cols if 'futures' in col or 'perp' in col]
    spot_cols = [col for col in numeric_cols if 'spot' in col]
    
    if basis_cols and spot_cols:
        for basis_col in basis_cols:
            for spot_col in spot_cols:
                if 'close' in basis_col and 'close' in spot_col:
                    # Calculate basis (premium/discount)
                    df['futures_basis'] = (df[basis_col] / df[spot_col] - 1) * 100
                    df['basis_zscore'] = (df['futures_basis'] - df['futures_basis'].rolling(168).mean()) / df['futures_basis'].rolling(168).std()
    
    return df

def align_and_merge_datasets(dfs: dict, datetime_col: str = "datetime") -> pd.DataFrame:
    """
    Align and merge multiple DataFrames on a common datetime range using asof merge.
    Expects a dict of {name: DataFrame}.
    """
    # Find common date range
    min_dates = [df[datetime_col].min() for df in dfs.values()]
    max_dates = [df[datetime_col].max() for df in dfs.values()]
    common_min = max(min_dates)
    common_max = min(max_dates)
    
    # Filter each DataFrame to the common range
    dfs_common = {
        name: df[(df[datetime_col] >= common_min) & (df[datetime_col] <= common_max)].copy()
        for name, df in dfs.items()
    }
    
    # Merge using asof (nearest datetime)
    merged = None
    for name, df in dfs_common.items():
        if merged is None:
            merged = df
        else:
            merged = pd.merge_asof(
                merged.sort_values(datetime_col),
                df.sort_values(datetime_col),
                on=datetime_col,
                direction="nearest",
                suffixes=('', f'_{name}')
            )
    return merged

def create_synthetic_features(merged_df):
    """
    Create synthetic features when actual data isn't available
    """
    # Synthetic exchange flow indicators
    if 'close' in merged_df.columns and 'volume' in merged_df.columns:
        # Volume spikes as proxy for exchange flows
        merged_df['volume_zscore'] = (merged_df['volume'] - merged_df['volume'].rolling(168).mean()) / merged_df['volume'].rolling(168).std()
        merged_df['synthetic_exchange_flow'] = merged_df['volume_zscore'].clip(-3, 3)
        
        # Price-volume divergence as whale activity proxy
        price_change = merged_df['close'].pct_change(24)
        volume_change = merged_df['volume'].pct_change(24)
        merged_df['pv_divergence'] = price_change - volume_change
        merged_df['synthetic_whale_activity'] = merged_df['pv_divergence'].rolling(48).mean()
    
    return merged_df