"""
model_training.py

Implements selective training for HMM (regime detection) and LSTM (signal prediction)
to handle missing Coinglass data in the merged dataset.

- HMM: Trains on all data, excluding Coinglass features.
- LSTM: Trains only on rows where coinglass_available == 1 (last 2 years).
- Includes enhanced ensemble/fallback logic with regime-specific signal thresholds.
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------------------------
# HMM Regime Detection
# ---------------------------

def train_improved_hmm_regime_detector(df: pd.DataFrame, n_states: int = 3):
    """
    Enhanced HMM with regime transition analysis and duration features.
    Returns: hmm, scaler, feature_cols, df_with_regime and transition details
    """
    # First get basic HMM results
    hmm, scaler, hmm_feature_cols, df = train_hmm_regime_detector(df, n_states)
    
    # Add transition probability analysis
    transition_matrix = hmm.transmat_
    
    # Store regime transition probabilities for use in decision-making
    df['regime_stability'] = 0
    for i in range(n_states):
        # Probability of staying in the same state
        mask = df['regime'] == i
        df.loc[mask, 'regime_stability'] = transition_matrix[i, i]
    
    # Add regime duration features
    df['regime_duration'] = 0
    current_regime = -1
    duration = 0
    
    for i, regime in enumerate(df['regime']):
        if pd.isna(regime):
            continue
        
        if regime == current_regime:
            duration += 1
        else:
            current_regime = regime
            duration = 1
        
        df.iloc[i, df.columns.get_loc('regime_duration')] = duration
    
    # Analyze regimes to determine market conditions
    regime_analysis = {}
    for i in range(n_states):
        mask = df['regime'] == i
        if mask.sum() > 0:
            regime_data = df[mask]
            if 'close' in df.columns:
                # Calculate returns within this regime
                returns = regime_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(24)  # Scaled to daily
                mean_return = returns.mean() * 24  # Scaled to daily
                sharpe = mean_return / volatility if volatility > 0 else 0
                
                # Determine regime type based on stats
                if mean_return > 0 and sharpe > 1:
                    regime_type = "bull"
                elif mean_return < 0 and abs(mean_return) > volatility:
                    regime_type = "bear"
                else:
                    regime_type = "choppy"
                
                regime_analysis[i] = {
                    'type': regime_type,
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'count': mask.sum()
                }
    
    # Add regime type to dataframe
    df['regime_type'] = np.nan
    for regime_id, analysis in regime_analysis.items():
        df.loc[df['regime'] == regime_id, 'regime_type'] = analysis['type']
    
    return hmm, scaler, hmm_feature_cols, df, regime_analysis


def train_hmm_regime_detector(df: pd.DataFrame, n_states: int = 3):
    """
    Train HMM on all available data, excluding Coinglass features.
    Handles NaN, inf, and -inf by dropping rows with any such values in the selected features.
    If no valid rows remain, raises a clear error.
    Returns: hmm, scaler, feature_cols, df_with_regime
    """
    # Select only features always available (e.g., on-chain, spot, etc.)
    exclude_cols = [col for col in df.columns if "coinglass" in col or "available" in col or "datetime" in col]
    hmm_features = df.drop(columns=exclude_cols, errors='ignore').select_dtypes(include=[np.number])

    # Remove rows with NaN, inf, or -inf
    hmm_features_clean = hmm_features.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    valid_idx = hmm_features_clean.index

    if hmm_features_clean.empty:
        raise ValueError("No valid rows remain for HMM training after removing NaN/inf values. "
                         "Check your data for extreme values or consider additional cleaning.")

    hmm_feature_cols = hmm_features_clean.columns.tolist()

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(hmm_features_clean.values)

    # Fit HMM
    hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
    hmm.fit(X)
    regimes = hmm.predict(X)

    # Assign regimes back to the original DataFrame (NaN for dropped rows)
    df = df.copy()
    df['regime'] = np.nan
    df.loc[valid_idx, 'regime'] = regimes
    return hmm, scaler, hmm_feature_cols, df

# ---------------------------
# Enhanced LSTM Signal Prediction
# ---------------------------

def train_enhanced_lstm_signal_predictor(df: pd.DataFrame, target_col: str, sequence_length: int = 24, epochs: int = 50):
    """
    Enhanced LSTM with dropout, batch normalization, and a more complex architecture.
    """
    # Filter for available Coinglass data
    df_lstm = df[df['coinglass_available'] == 1].copy()
    df_lstm = df_lstm.reset_index(drop=True)

    # Select features (exclude non-numeric, regime, target, datetime)
    exclude_cols = [col for col in df_lstm.columns if col in ['datetime', 'regime', target_col]]
    feature_cols = [col for col in df_lstm.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

    # Prepare sequences
    X, y = [], []
    for i in range(len(df_lstm) - sequence_length):
        X.append(df_lstm[feature_cols].iloc[i:i+sequence_length].values)
        y.append(df_lstm[target_col].iloc[i+sequence_length])
    X, y = np.array(X), np.array(y)

    # Remove sequences with NaN, inf, or -inf
    mask = ~np.isnan(X).any(axis=(1,2)) & ~np.isinf(X).any(axis=(1,2)) & ~np.isnan(y) & ~np.isinf(y)
    X, y = X[mask], y[mask]

    # Standardize features
    scaler = StandardScaler()
    X_shape = X.shape
    X_reshaped = X.reshape(-1, X_shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X_shape)

    # Build enhanced LSTM model with more complex architecture
    model = Sequential([
        # First LSTM layer with batch normalization
        LSTM(128, input_shape=(sequence_length, len(feature_cols)), return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second LSTM layer
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Use Adam optimizer with custom learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    # Early stopping with patience
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train with smaller batch size
    model.fit(
        X_scaled, y, 
        epochs=epochs, 
        batch_size=32, 
        validation_split=0.2, 
        callbacks=[early_stop], 
        verbose=1
    )
    
    return model, scaler, feature_cols


def train_lstm_signal_predictor(df: pd.DataFrame, target_col: str, sequence_length: int = 24, epochs: int = 20):
    """
    Train LSTM only on rows where coinglass_available == 1 (last 2 years).
    """
    # Filter for available Coinglass data
    df_lstm = df[df['coinglass_available'] == 1].copy()
    df_lstm = df_lstm.reset_index(drop=True)

    # Select features (all numeric except regime, target, datetime)
    exclude_cols = [col for col in df_lstm.columns if col in ['datetime', 'regime', target_col]]
    feature_cols = [col for col in df_lstm.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

    # Prepare sequences
    X, y = [], []
    for i in range(len(df_lstm) - sequence_length):
        X.append(df_lstm[feature_cols].iloc[i:i+sequence_length].values)
        y.append(df_lstm[target_col].iloc[i+sequence_length])
    X, y = np.array(X), np.array(y)

    # Remove sequences with NaN, inf, or -inf
    mask = ~np.isnan(X).any(axis=(1,2)) & ~np.isinf(X).any(axis=(1,2)) & ~np.isnan(y) & ~np.isinf(y)
    X, y = X[mask], y[mask]

    # Standardize features
    scaler = StandardScaler()
    X_shape = X.shape
    X_reshaped = X.reshape(-1, X_shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X_shape)

    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, len(feature_cols)), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_scaled, y, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)
    return model, scaler, feature_cols

# ---------------------------
# Enhanced Fallback/Ensemble Logic
# ---------------------------

def predict_with_ensemble(df: pd.DataFrame, hmm, hmm_scaler, hmm_feature_cols, lstm_model, lstm_scaler, lstm_features, sequence_length: int = 24, regime_analysis=None):
    """
    Predict using HMM for all periods, and LSTM only where coinglass_available == 1.
    Implements regime-adaptive thresholds for more frequent and higher-quality signals.
    """
    # HMM regime prediction (all data) - use the exact columns as during training
    hmm_X = df[hmm_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    hmm_X_scaled = hmm_scaler.transform(hmm_X.values)
    df['regime_pred'] = hmm.predict(hmm_X_scaled)

    # LSTM prediction (only where coinglass_available == 1)
    lstm_preds = np.full(len(df), np.nan)
    idxs = df[df['coinglass_available'] == 1].index
    for idx in idxs:
        if idx < sequence_length:
            continue
        seq = df.loc[idx-sequence_length:idx-1, lstm_features].values
        if seq.shape[0] == sequence_length and not np.isnan(seq).any() and not np.isinf(seq).any():
            seq_scaled = lstm_scaler.transform(seq)
            seq_scaled = seq_scaled.reshape(1, sequence_length, len(lstm_features))
            lstm_preds[idx] = lstm_model.predict(seq_scaled, verbose=0)[0, 0]
    df['lstm_pred'] = lstm_preds
    
    # Generate trading signals with regime-adaptive thresholds
    df['regime_adjusted_signal'] = 0
    
    # If regime_analysis is provided (from improved HMM), use it for sophisticated thresholds
    if regime_analysis:
        for regime_id, analysis in regime_analysis.items():
            regime_type = analysis['type']
            regime_mask = df['regime_pred'] == regime_id
            
            # Only generate signals for rows with valid LSTM predictions
            valid_mask = ~df['lstm_pred'].isna() & regime_mask
            
            if valid_mask.sum() > 0:
                close_prices = df.loc[valid_mask, 'close']
                lstm_preds = df.loc[valid_mask, 'lstm_pred']
                price_diff_pct = (lstm_preds / close_prices - 1) * 100
                
                # Bull market - more aggressive signals with lower thresholds
                if regime_type == "bull":
                    # Long signal: Price expected to rise by at least 0.5%
                    long_mask = price_diff_pct > 0.5
                    # Short signal: Price expected to fall by at least 0.7% (more conservative with shorts in bull market)
                    short_mask = price_diff_pct < -0.7
                    
                # Bear market - more aggressive short signals
                elif regime_type == "bear":
                    # Long signal: Price expected to rise by at least 1.0% (more conservative with longs in bear market)
                    long_mask = price_diff_pct > 1.0
                    # Short signal: Price expected to fall by at least 0.4%
                    short_mask = price_diff_pct < -0.4
                    
                # Choppy/range-bound market - focus on mean reversion, larger thresholds
                else:
                    # Long signal: Price expected to rise by at least 0.8%
                    long_mask = price_diff_pct > 0.8
                    # Short signal: Price expected to fall by at least 0.8%
                    short_mask = price_diff_pct < -0.8
                
                # Apply signals to the dataframe
                df.loc[valid_mask & long_mask, 'regime_adjusted_signal'] = 1
                df.loc[valid_mask & short_mask, 'regime_adjusted_signal'] = -1
    else:
        # Simplified approach without regime analysis - use general thresholds
        # Only generate signals for rows with valid LSTM predictions
        valid_mask = ~df['lstm_pred'].isna()
        
        if valid_mask.sum() > 0:
            # Calculate percentage difference between prediction and current price
            close_prices = df.loc[valid_mask, 'close']
            lstm_preds = df.loc[valid_mask, 'lstm_pred']
            price_diff_pct = (lstm_preds / close_prices - 1) * 100
            
            # Generate signals with fixed thresholds
            df.loc[valid_mask & (price_diff_pct > 0.7), 'regime_adjusted_signal'] = 1
            df.loc[valid_mask & (price_diff_pct < -0.7), 'regime_adjusted_signal'] = -1
    
    # Add signal strength metric (confidence score)
    df['signal_strength'] = 0
    valid_mask = ~df['lstm_pred'].isna()
    if valid_mask.sum() > 0:
        close_prices = df.loc[valid_mask, 'close']
        lstm_preds = df.loc[valid_mask, 'lstm_pred']
        price_diff_pct = (lstm_preds / close_prices - 1) * 100
        
        # Scale signal strength based on prediction deviation (clipped to reasonable range)
        df.loc[valid_mask, 'signal_strength'] = price_diff_pct.abs().clip(0, 5) / 5.0
    
    return df