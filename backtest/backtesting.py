"""
backtesting.py

Enhanced backtesting with advanced visualizations and additional metrics.
- Interactive equity curve and drawdown charts (using plotly)
- Prints advanced trade statistics

Assumes the merged DataFrame contains:
- 'datetime'
- 'lstm_pred' (LSTM signal, may be NaN for periods without Coinglass)
- 'regime_pred' (HMM regime label: 0=Sideways, 1=Bull, 2=Bear - adjust if needed)
- 'close' (actual price, e.g., from bybit_linear_candle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio

def compute_returns(df, signal_col='lstm_pred', price_col='close', regime_col='regime_pred', signal_threshold=0.0):
    df = df.copy()
    df['signal'] = 0
    bull_mask = (df[regime_col] == 1) & (df[signal_col] > signal_threshold)
    bear_mask = (df[regime_col] == 2) & (df[signal_col] < -signal_threshold)
    df.loc[bull_mask, 'signal'] = 1
    df.loc[bear_mask, 'signal'] = -1
    df['signal'] = df['signal'].fillna(0)
    df['price_return'] = df[price_col].pct_change().shift(-1)
    df['strategy_return'] = df['signal'] * df['price_return']
    return df

def sharpe_ratio(returns, risk_free_rate=0.0):
    mean = np.nanmean(returns)
    std = np.nanstd(returns)
    if std == 0:
        return 0
    return (mean - risk_free_rate) / std * np.sqrt(252 * 24)

def max_drawdown(equity_curve):
    roll_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - roll_max) / roll_max
    return np.min(drawdown)

def trade_frequency(df):
    n_trades = np.nansum(np.abs(np.diff(df['signal'].fillna(0))))
    return n_trades / len(df)

def win_loss_ratio(df):
    wins = np.nansum(df['strategy_return'] > 0)
    losses = np.nansum(df['strategy_return'] < 0)
    return (wins / losses) if losses > 0 else np.inf

def advanced_trade_stats(df):
    trades = df[df['signal'] != 0]
    total_trades = len(trades)
    win_rate = np.mean(trades['strategy_return'] > 0) if total_trades > 0 else 0
    avg_win = trades[trades['strategy_return'] > 0]['strategy_return'].mean() if win_rate > 0 else 0
    avg_loss = trades[trades['strategy_return'] < 0]['strategy_return'].mean() if win_rate < 1 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor
    }

def plot_equity_curve(df, equity_col='equity', drawdown_col='drawdown', datetime_col='datetime', filename='equity_curve.png'):
    plt.figure(figsize=(14, 7))
    plt.plot(df[datetime_col], df[equity_col], label='Equity Curve')
    plt.plot(df[datetime_col], df[drawdown_col], label='Drawdown', color='red', alpha=0.5)
    plt.title('Equity Curve and Drawdown')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Equity curve saved to {filename}")

def plot_interactive_equity_curve(df, equity_col='equity', drawdown_col='drawdown', datetime_col='datetime', filename='equity_curve_interactive.html'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[datetime_col], y=df[equity_col], mode='lines', name='Equity Curve'))
    fig.add_trace(go.Scatter(x=df[datetime_col], y=df[drawdown_col], mode='lines', name='Drawdown', line=dict(color='red', dash='dot')))
    fig.update_layout(title='Interactive Equity Curve and Drawdown', xaxis_title='Time', yaxis_title='Equity')
    pio.write_html(fig, file=filename, auto_open=False)
    print(f"Interactive equity curve saved to {filename}")

def backtest_and_evaluate(df, signal_col='lstm_pred', price_col='close', regime_col='regime_pred', signal_threshold=0.0):
    df_bt = compute_returns(df, signal_col=signal_col, price_col=price_col, regime_col=regime_col, signal_threshold=signal_threshold)
    df_bt['equity'] = (1 + df_bt['strategy_return'].fillna(0)).cumprod()
    roll_max = np.maximum.accumulate(df_bt['equity'])
    df_bt['drawdown'] = (df_bt['equity'] - roll_max) / roll_max

    sharpe = sharpe_ratio(df_bt['strategy_return'])
    mdd = max_drawdown(df_bt['equity'])
    freq = trade_frequency(df_bt)
    wl = win_loss_ratio(df_bt)
    stats = advanced_trade_stats(df_bt)

    print("==== Backtest Results ====")
    print(f"Signal Threshold: {signal_threshold}")
    print(f"Sharpe Ratio: {sharpe:.2f} (Target: ≥ 1.8)")
    print(f"Max Drawdown: {mdd:.2%} (Target: ≥ -40%)")
    print(f"Trade Frequency: {freq:.2%} (Target: ≥ 3%)")
    print(f"Win/Loss Ratio: {wl:.2f} (Target: > 1.0)")
    print("\n==== Additional Metrics ====")
    print(f"Total Trades: {stats['total_trades']:,}")
    print(f"Win Rate: {stats['win_rate']:.2%}")
    print(f"Average Win: {stats['avg_win']:.2%}")
    print(f"Average Loss: {stats['avg_loss']:.2%}")
    print(f"Profit Factor: {stats['profit_factor']:.4f}")

    meets_criteria = (
        sharpe >= 1.8 and
        mdd >= -0.40 and
        freq >= 0.03 and
        wl > 1.0
    )
    print("\nMeets all criteria:", "YES" if meets_criteria else "NO")

    # Save static and interactive equity curve
    plot_equity_curve(df_bt, equity_col='equity', drawdown_col='drawdown', datetime_col='datetime', filename='equity_curve.png')
    plot_interactive_equity_curve(df_bt, equity_col='equity', drawdown_col='drawdown', datetime_col='datetime', filename='equity_curve_interactive.html')

    return df_bt, {
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "trade_frequency": freq,
        "win_loss_ratio": wl,
        "total_trades": stats['total_trades'],
        "win_rate": stats['win_rate'],
        "avg_win": stats['avg_win'],
        "avg_loss": stats['avg_loss'],
        "profit_factor": stats['profit_factor'],
        "meets_criteria": meets_criteria
    }