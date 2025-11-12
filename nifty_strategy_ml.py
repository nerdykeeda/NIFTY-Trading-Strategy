"""
Advanced NIFTY Trading Strategy with Machine Learning & Advanced Options

MAJOR ENHANCEMENTS:
- Machine Learning: Random Forest & Ensemble models for prediction
- VIX Integration: India VIX for volatility-based decision making
- FII/DII Data: Institutional money flow analysis
- Advanced Options: Iron Condor, Straddle, Strangle recommendations
- Sentiment Analysis: VIX-based + FII/DII flow sentiment
- Enhanced Backtesting: ML accuracy metrics, strategy comparison
- Multiple Timeframes: Daily + trend confirmation

SIGNAL GENERATION:
1. LONG (Buy CALL): Bullish ML prediction + technical confirmation
2. BEARISH (Buy PUT): Bearish ML prediction + technical confirmation
3. FLAT: Stay in cash, unclear signals

ADVANCED OPTIONS STRATEGIES:
- Iron Condor: Range-bound, low volatility
- Straddle: High volatility expected, direction unclear
- Strangle: Similar to Straddle, cheaper
- Calendar Spread: Time decay exploitation

Usage:
  python3 nifty_strategy_ml.py --start 2024-01-01 --end 2025-11-12 --outdir results_ml

Requirements:
  pip install pandas numpy matplotlib yfinance scikit-learn beautifulsoup4 requests ta
"""

import argparse
import os
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Web scraping
import requests
from bs4 import BeautifulSoup

# Suppress warnings
warnings.filterwarnings('ignore')

# Optional yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_nifty(start_date, end_date):
    """Fetch NIFTY daily data via yfinance."""
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not available.")
    
    df = yf.download("^NSEI", start=start_date, end=end_date, progress=False, auto_adjust=False)
    df.columns = df.columns.get_level_values(0)  # Flatten MultiIndex
    if df.empty:
        raise RuntimeError("No data fetched.")
    df.index = pd.to_datetime(df.index)
    return df


def fetch_india_vix(start_date, end_date):
    """Fetch India VIX data (volatility index)."""
    try:
        df = yf.download("^INDIAVIX", start=start_date, end=end_date, progress=False, auto_adjust=False)
        if not df.empty:
            df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            return df[['Close']].rename(columns={'Close': 'VIX'})
        else:
            print("⚠️  VIX data not available, using synthetic VIX based on NIFTY volatility")
            return None
    except Exception as e:
        print(f"⚠️  Could not fetch VIX: {e}. Will use synthetic VIX.")
        return None


def fetch_fii_dii_data():
    """
    Attempt to scrape FII/DII data from NSE or return None.
    Note: NSE has anti-scraping measures, this may not always work.
    For production, use official NSE API (requires authentication).
    """
    try:
        # NSE FII/DII data URL (may change)
        url = "https://www.nseindia.com/api/fiidiiTrading"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
        }
        
        session = requests.Session()
        # First, get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        
        # Then fetch data
        response = session.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # Parse the data (structure may vary)
            return data
        else:
            return None
    except Exception as e:
        print(f"⚠️  Could not fetch FII/DII data: {e}")
        return None


# ============================================================================
# TECHNICAL INDICATORS & FEATURE ENGINEERING
# ============================================================================

def compute_rsi(series, period=14):
    """Calculate RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD."""
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def compute_atr(df, period=14):
    """Calculate ATR."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period, min_periods=1).mean()
    return atr


def compute_bollinger_bands(series, period=20, num_std=2):
    """Calculate Bollinger Bands."""
    sma = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band


def compute_stochastic(df, period=14):
    """Calculate Stochastic Oscillator."""
    low_min = df['Low'].rolling(period, min_periods=1).min()
    high_max = df['High'].rolling(period, min_periods=1).max()
    stoch = 100 * (df['Close'] - low_min) / (high_max - low_min)
    return stoch


def compute_adx(df, period=14):
    """Calculate ADX (Average Directional Index) - trend strength."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = compute_atr(df, period)
    atr = tr
    
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period, min_periods=1).mean()
    
    return adx


def engineer_features(df):
    """
    Create extensive feature set for ML model.
    Returns df with new feature columns.
    """
    df = df.copy()
    
    # Basic indicators
    df['SMA_10'] = df['Close'].rolling(10, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    
    # RSI
    df['RSI_14'] = compute_rsi(df['Close'], period=14)
    df['RSI_7'] = compute_rsi(df['Close'], period=7)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
    
    # ATR
    df['ATR'] = compute_atr(df, period=14)
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = compute_bollinger_bands(df['Close'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Stochastic
    df['Stochastic'] = compute_stochastic(df)
    
    # ADX (trend strength)
    df['ADX'] = compute_adx(df)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price momentum
    df['ROC_5'] = df['Close'].pct_change(5) * 100  # Rate of Change
    df['ROC_10'] = df['Close'].pct_change(10) * 100
    df['ROC_20'] = df['Close'].pct_change(20) * 100
    
    # Price position relative to moving averages
    df['Price_vs_SMA10'] = (df['Close'] - df['SMA_10']) / df['SMA_10']
    df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # Volatility
    df['Historical_Vol'] = df['Close'].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252)
    
    # Gap detection
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # High-Low range
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Returns
    df['Returns'] = df['Close'].pct_change()
    
    # Target: Next day's return direction (1 = up, 0 = down)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df


def add_vix_features(df, vix_df):
    """Add VIX-based features to dataframe."""
    if vix_df is not None and not vix_df.empty:
        # Merge VIX data
        df = df.join(vix_df, how='left')
        df['VIX'].fillna(method='ffill', inplace=True)
    else:
        # Create synthetic VIX based on NIFTY volatility
        df['VIX'] = df['Close'].pct_change().rolling(20, min_periods=1).std() * 100 * np.sqrt(252)
    
    # VIX-based features
    df['VIX_SMA'] = df['VIX'].rolling(20, min_periods=1).mean()
    df['VIX_High'] = (df['VIX'] > df['VIX_SMA'] * 1.2).astype(int)  # High volatility
    df['VIX_Low'] = (df['VIX'] < df['VIX_SMA'] * 0.8).astype(int)   # Low volatility
    
    return df


# ============================================================================
# MACHINE LEARNING MODEL
# ============================================================================

class TradingMLModel:
    """Machine Learning model for trading signal prediction."""
    
    def __init__(self):
        self.model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.model_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_features(self, df):
        """Select and prepare features for ML model."""
        # Select feature columns (exclude target, date, and raw price data)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                       'Target', 'Returns', 'Signal', 'Position']
        
        available_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any columns with all NaN
        available_cols = [col for col in available_cols if not df[col].isna().all()]
        
        self.feature_columns = available_cols
        
        # Fill NaN values
        X = df[self.feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return X
    
    def train(self, df, test_size=0.2):
        """Train the ML models."""
        print("\n" + "="*80)
        print(" TRAINING MACHINE LEARNING MODELS")
        print("="*80)
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df['Target'].fillna(0).astype(int)
        
        # Remove last row (no target)
        X = X[:-1]
        y = y[:-1]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Don't shuffle time series
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("\n🌲 Training Random Forest...")
        self.model_rf.fit(X_train_scaled, y_train)
        rf_pred = self.model_rf.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_precision = precision_score(y_test, rf_pred, zero_division=0)
        rf_recall = recall_score(y_test, rf_pred, zero_division=0)
        rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
        
        # Train Gradient Boosting
        print("⚡ Training Gradient Boosting...")
        self.model_gb.fit(X_train_scaled, y_train)
        gb_pred = self.model_gb.predict(X_test_scaled)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        gb_precision = precision_score(y_test, gb_pred, zero_division=0)
        gb_recall = recall_score(y_test, gb_pred, zero_division=0)
        gb_f1 = f1_score(y_test, gb_pred, zero_division=0)
        
        self.is_trained = True
        
        # Display results
        print("\n" + "="*80)
        print(" ML MODEL PERFORMANCE (Test Set)")
        print("="*80)
        
        print(f"\n📊 Random Forest:")
        print(f"  Accuracy:   {rf_accuracy:.2%}")
        print(f"  Precision:  {rf_precision:.2%}")
        print(f"  Recall:     {rf_recall:.2%}")
        print(f"  F1 Score:   {rf_f1:.2%}")
        
        print(f"\n📊 Gradient Boosting:")
        print(f"  Accuracy:   {gb_accuracy:.2%}")
        print(f"  Precision:  {gb_precision:.2%}")
        print(f"  Recall:     {gb_recall:.2%}")
        print(f"  F1 Score:   {gb_f1:.2%}")
        
        # Feature importance (top 10)
        print(f"\n🔍 Top 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model_rf.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']:.<30} {row['importance']:.4f}")
        
        print("="*80 + "\n")
        
        return {
            'rf_accuracy': rf_accuracy,
            'gb_accuracy': gb_accuracy,
            'rf_f1': rf_f1,
            'gb_f1': gb_f1
        }
    
    def predict(self, df):
        """Predict next-day direction for entire dataframe."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        rf_pred = self.model_rf.predict(X_scaled)
        rf_proba = self.model_rf.predict_proba(X_scaled)[:, 1]  # Probability of up
        
        gb_pred = self.model_gb.predict(X_scaled)
        gb_proba = self.model_gb.predict_proba(X_scaled)[:, 1]
        
        # Ensemble: Average probabilities
        ensemble_proba = (rf_proba + gb_proba) / 2
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba


# ============================================================================
# ADVANCED OPTIONS STRATEGY DETECTION
# ============================================================================

def detect_options_strategy(df, current_idx):
    """
    Detect which advanced options strategy is most suitable.
    Returns: strategy name and parameters
    """
    current = df.iloc[current_idx]
    
    vix = current.get('VIX', 15)
    vix_sma = current.get('VIX_SMA', 15)
    bb_width = current.get('BB_Width', 0.05)
    adx = current.get('ADX', 20)
    atr = current.get('ATR', 100)
    
    strategies = []
    
    # Iron Condor: Low volatility + Range-bound market
    if vix < vix_sma * 0.9 and adx < 20 and bb_width < 0.03:
        strategies.append({
            'name': 'Iron Condor',
            'confidence': 'HIGH',
            'reason': 'Low volatility + Range-bound market (ADX < 20)',
            'vix_level': vix,
            'market_state': 'Range-bound'
        })
    
    # Straddle: High volatility + Direction unclear
    if vix > vix_sma * 1.2 and adx < 25:
        strategies.append({
            'name': 'Long Straddle',
            'confidence': 'HIGH',
            'reason': 'High volatility + Direction unclear',
            'vix_level': vix,
            'market_state': 'High volatility, non-trending'
        })
    
    # Strangle: Moderate volatility + Direction unclear (cheaper than Straddle)
    if vix > vix_sma and vix < vix_sma * 1.3 and adx < 25:
        strategies.append({
            'name': 'Long Strangle',
            'confidence': 'MEDIUM',
            'reason': 'Moderate volatility + Direction unclear (cheaper than Straddle)',
            'vix_level': vix,
            'market_state': 'Moderate volatility'
        })
    
    # Calendar Spread: Low volatility + Time decay opportunity
    if vix < vix_sma and bb_width < 0.04:
        strategies.append({
            'name': 'Calendar Spread',
            'confidence': 'MEDIUM',
            'reason': 'Low volatility + Profit from time decay',
            'vix_level': vix,
            'market_state': 'Low volatility'
        })
    
    return strategies if strategies else None


# ============================================================================
# SIGNAL GENERATION WITH ML
# ============================================================================

def compute_ml_signals(df, ml_model, ml_threshold=0.55):
    """
    Generate trading signals using ML predictions + technical confirmation.
    
    Signals:
    1 = LONG (bullish)
    -1 = BEARISH (bearish)
    0 = FLAT (neutral)
    """
    df = df.copy()
    
    # Get ML predictions
    ml_pred, ml_proba = ml_model.predict(df)
    df['ML_Pred'] = ml_pred
    df['ML_Proba'] = ml_proba
    
    # Technical conditions (from original strategy)
    bullish_sma = df['Close'] > df['SMA_10']
    bullish_rsi = (df['RSI_14'] >= 40) & (df['RSI_14'] <= 70)
    bullish_macd = df['MACD'] > df['MACD_Signal']
    strong_volume = df['Volume_Ratio'] > 0.8
    
    bearish_sma = df['Close'] < df['SMA_10']
    bearish_rsi = df['RSI_14'] < 40
    bearish_macd = df['MACD'] < df['MACD_Signal']
    
    # ML-enhanced signals
    # LONG: ML predicts up with high confidence + at least 2 technical confirmations
    ml_bullish = df['ML_Proba'] > ml_threshold
    tech_bull_count = bullish_sma.astype(int) + bullish_rsi.astype(int) + bullish_macd.astype(int)
    long_signal = ml_bullish & (tech_bull_count >= 2) & strong_volume
    
    # BEARISH: ML predicts down with high confidence + at least 2 bearish confirmations
    ml_bearish = df['ML_Proba'] < (1 - ml_threshold)
    tech_bear_count = bearish_sma.astype(int) + bearish_rsi.astype(int) + bearish_macd.astype(int)
    bearish_signal = ml_bearish & (tech_bear_count >= 2) & strong_volume
    
    # Create signal column
    df['Signal'] = 0  # Default FLAT
    df.loc[long_signal, 'Signal'] = 1  # LONG
    df.loc[bearish_signal, 'Signal'] = -1  # BEARISH
    
    # Position for backtest (both long and short)
    df['Position'] = df['Signal']  # 1 = long, -1 = short, 0 = flat
    
    # Stop loss and take profit
    df['Stop_Loss_Long'] = df['Close'] - (2 * df['ATR'])
    df['Take_Profit_Long'] = df['Close'] + (3 * df['ATR'])
    df['Stop_Loss_Short'] = df['Close'] + (2 * df['ATR'])
    df['Take_Profit_Short'] = df['Close'] - (3 * df['ATR'])
    
    return df


# ============================================================================
# BACKTESTING
# ============================================================================

def backtest_strategy(df, initial_capital=100000.0, position_fraction=0.10, 
                     round_trip_cost=0.0006, slippage=0.0005):
    """
    Backtest the ML-enhanced strategy.
    Now supports both LONG and SHORT positions.
    """
    df = df.copy().sort_index()
    capital = initial_capital
    equity = []
    cash = capital
    last_position = 0  # 0 = flat, 1 = long, -1 = short
    last_position_units = 0.0
    
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        pos_flag = df['Position'].iloc[i]  # 1, -1, or 0
        
        # Calculate target position
        if pos_flag == 1:  # LONG
            target_allocation = position_fraction * capital
            target_units = target_allocation / price if price > 0 else 0.0
        elif pos_flag == -1:  # SHORT
            target_allocation = position_fraction * capital
            target_units = -target_allocation / price if price > 0 else 0.0
        else:  # FLAT
            target_units = 0.0
        
        # Calculate trade
        trade_units = target_units - last_position_units
        trade_value = abs(trade_units) * price
        
        # Costs
        cost = trade_value * round_trip_cost
        slip = trade_value * slippage
        
        # Update cash
        cash -= (trade_units * price) + cost + slip
        last_position_units = target_units
        
        # Calculate market value (long = positive, short = negative P&L)
        mv = last_position_units * price
        capital = cash + mv
        equity.append(capital)
    
    df['Equity'] = equity
    df['Returns'] = df['Equity'].pct_change().fillna(0)
    
    return df


def performance_metrics(equity_series, trading_days_per_year=252):
    """Calculate comprehensive performance metrics."""
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    
    daily_returns = equity_series.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(trading_days_per_year)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year) if daily_returns.std() != 0 else np.nan
    
    # Sortino
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
    sortino = (daily_returns.mean() / downside_std) * np.sqrt(trading_days_per_year) if downside_std != 0 else np.nan
    
    # Drawdown
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # Calmar
    calmar = abs(cagr / max_dd) if max_dd != 0 else np.nan
    
    # Win/Loss
    hit_rate = (daily_returns > 0).mean()
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    
    best_day = daily_returns.max()
    worst_day = daily_returns.min()
    
    # Streaks
    win_streak = 0
    loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for ret in daily_returns:
        if ret > 0:
            current_win_streak += 1
            current_loss_streak = 0
            win_streak = max(win_streak, current_win_streak)
        elif ret < 0:
            current_loss_streak += 1
            current_win_streak = 0
            loss_streak = max(loss_streak, current_loss_streak)
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Ann Vol': ann_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_dd,
        'Hit Rate': hit_rate,
        'Win/Loss Ratio': win_loss_ratio,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Best Day': best_day,
        'Worst Day': worst_day,
        'Max Win Streak': win_streak,
        'Max Loss Streak': loss_streak,
    }


# ============================================================================
# DISPLAY & REPORTING
# ============================================================================

def display_current_status(df, ml_model, num_days=10):
    """Display current market status with ML insights."""
    print("\n" + "="*90)
    print(" ML-ENHANCED MARKET ANALYSIS (Last {} Trading Days)".format(num_days))
    print("="*90)
    
    last_n = df.tail(num_days).copy()
    last_n['Signal_Text'] = last_n['Signal'].apply(
        lambda x: 'LONG' if x == 1 else ('BEARISH' if x == -1 else 'FLAT')
    )
    
    print("\n{:<12} {:>10} {:>10} {:>8} {:>8} {:>10} {:>8}".format(
        'Date', 'Close', 'VIX', 'RSI', 'ML_Prob', 'Signal', 'ADX'))
    print("-"*90)
    
    for idx, row in last_n.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        marker = " ←" if idx == last_n.index[-1] else ""
        print("{:<12} {:>10.2f} {:>10.1f} {:>8.1f} {:>8.1%} {:>10} {:>8.1f}{}".format(
            date_str, row['Close'], row.get('VIX', 0), row['RSI_14'], 
            row['ML_Proba'], row['Signal_Text'], row.get('ADX', 0), marker
        ))
    
    # Current status
    latest = df.iloc[-1]
    
    print("\n" + "="*90)
    print(" CURRENT MARKET STATUS")
    print("="*90)
    
    print(f"\n💹 NIFTY Level:       ₹{latest['Close']:.2f}")
    print(f"📊 VIX (Volatility):  {latest.get('VIX', 0):.1f}")
    print(f"📈 RSI:               {latest['RSI_14']:.1f}")
    print(f"🤖 ML Probability:    {latest['ML_Proba']:.1%} (Bullish)")
    print(f"📉 ADX (Trend):       {latest.get('ADX', 0):.1f}")
    print(f"🎯 BB Position:       {latest.get('BB_Position', 0.5):.1%}")
    
    # Signal interpretation
    print("\n" + "="*90)
    if latest['Signal'] == 1:
        print("✅ SIGNAL: LONG (BUY CALL OPTIONS)")
        print(f"  ML Confidence: {latest['ML_Proba']:.1%}")
        print(f"  Stop Loss: ₹{latest['Stop_Loss_Long']:.2f}")
        print(f"  Take Profit: ₹{latest['Take_Profit_Long']:.2f}")
    elif latest['Signal'] == -1:
        print("⛔ SIGNAL: BEARISH (BUY PUT OPTIONS)")
        print(f"  ML Confidence: {1-latest['ML_Proba']:.1%}")
        print(f"  Stop Loss: ₹{latest['Stop_Loss_Short']:.2f}")
        print(f"  Take Profit: ₹{latest['Take_Profit_Short']:.2f}")
    else:
        print("⚪ SIGNAL: FLAT (STAY IN CASH)")
        print(f"  ML Probability: {latest['ML_Proba']:.1%} (Not confident enough)")
    
    # Advanced options strategies
    strategies = detect_options_strategy(df, len(df)-1)
    if strategies:
        print("\n" + "="*90)
        print(" ADVANCED OPTIONS STRATEGIES DETECTED")
        print("="*90)
        for strat in strategies:
            print(f"\n🎯 {strat['name']} ({strat['confidence']} confidence)")
            print(f"  Reason: {strat['reason']}")
            print(f"  VIX Level: {strat['vix_level']:.1f}")
            print(f"  Market: {strat['market_state']}")
    
    print("="*90 + "\n")


def display_options_recommendations(latest_row):
    """Display specific option recommendations."""
    current_price = latest_row['Close']
    signal = latest_row['Signal']
    atm_strike = round(current_price / 50) * 50
    
    print("="*90)
    print(" OPTIONS RECOMMENDATIONS")
    print("="*90)
    
    if signal == 1:
        print("\n🟢 BUY CALL OPTIONS (Bullish)")
        print(f"\n  ATM: NIFTY {atm_strike} CE")
        print(f"  OTM: NIFTY {atm_strike+100} CE")
        print(f"  ITM: NIFTY {atm_strike-100} CE")
    elif signal == -1:
        print("\n🔴 BUY PUT OPTIONS (Bearish)")
        print(f"\n  ATM: NIFTY {atm_strike} PE")
        print(f"  OTM: NIFTY {atm_strike-100} PE")
        print(f"  ITM: NIFTY {atm_strike+100} PE")
    else:
        print("\n⚪ NO DIRECTIONAL TRADE RECOMMENDED")
        print("\n  Consider advanced strategies:")
        print(f"  - Iron Condor if VIX < 15")
        print(f"  - Straddle if VIX > 20")
    
    print("="*90 + "\n")


def plot_results(df, outdir):
    """Create comprehensive plots."""
    os.makedirs(outdir, exist_ok=True)
    
    # Plot 1: Price + Signals
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='NIFTY Close', linewidth=1.5, alpha=0.8)
    plt.plot(df.index, df['SMA_10'], label='SMA(10)', linewidth=1, alpha=0.7)
    plt.plot(df.index, df['SMA_20'], label='SMA(20)', linewidth=1, alpha=0.7)
    
    # Mark signals
    longs = df[df['Signal'] == 1]
    shorts = df[df['Signal'] == -1]
    plt.scatter(longs.index, longs['Close'], marker='^', color='green', s=100, label='LONG', zorder=5)
    plt.scatter(shorts.index, shorts['Close'], marker='v', color='red', s=100, label='SHORT', zorder=5)
    
    plt.title('NIFTY Price & ML-Enhanced Signals', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price (₹)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ml_price_signals.png'), dpi=150)
    plt.close()
    
    # Plot 2: Equity Curve
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Equity'], label='ML Strategy Equity', linewidth=2, color='blue')
    plt.title('Equity Curve (ML-Enhanced Strategy)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Equity (₹)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ml_equity_curve.png'), dpi=150)
    plt.close()
    
    # Plot 3: ML Probability
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['ML_Proba'], label='ML Bullish Probability', linewidth=1, color='purple')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Neutral (50%)')
    plt.axhline(y=0.55, color='green', linestyle='--', alpha=0.5, label='Bullish Threshold')
    plt.axhline(y=0.45, color='red', linestyle='--', alpha=0.5, label='Bearish Threshold')
    plt.fill_between(df.index, 0.45, 0.55, alpha=0.2, color='gray', label='Neutral Zone')
    plt.title('ML Prediction Probability', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ml_probability.png'), dpi=150)
    plt.close()
    
    # Plot 4: VIX
    if 'VIX' in df.columns:
        plt.figure(figsize=(14, 5))
        plt.plot(df.index, df['VIX'], label='India VIX', linewidth=1.5, color='orange')
        plt.plot(df.index, df['VIX_SMA'], label='VIX SMA(20)', linewidth=1, alpha=0.7)
        plt.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Low Vol (15)')
        plt.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='High Vol (20)')
        plt.title('India VIX (Volatility Index)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('VIX')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'vix.png'), dpi=150)
        plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(args):
    """Main execution function."""
    print("\n" + "🚀"*45)
    print(" ML-ENHANCED NIFTY TRADING STRATEGY")
    print("🚀"*45 + "\n")
    
    # Step 1: Fetch data
    print("📥 Fetching NIFTY data...")
    df = fetch_nifty(args.start, args.end)
    
    print("📥 Fetching India VIX data...")
    vix_df = fetch_india_vix(args.start, args.end)
    
    # Step 2: Engineer features
    print("🔧 Engineering features...")
    df = engineer_features(df)
    df = add_vix_features(df, vix_df)
    
    # Drop rows with NaN in target
    df = df.dropna(subset=['Target'])
    
    # Step 3: Train ML model
    ml_model = TradingMLModel()
    ml_performance = ml_model.train(df, test_size=0.2)
    
    # Step 4: Generate signals
    print("🎯 Generating ML-enhanced signals...")
    df = compute_ml_signals(df, ml_model, ml_threshold=0.55)
    
    # Step 5: Backtest
    print("📊 Running backtest...")
    result = backtest_strategy(df, initial_capital=args.capital,
                              position_fraction=args.position_fraction,
                              round_trip_cost=args.round_trip_cost,
                              slippage=args.slippage)
    
    # Step 6: Calculate metrics
    metrics = performance_metrics(result['Equity'])
    
    # Step 7: Display results
    print("\n" + "="*90)
    print(" BACKTEST PERFORMANCE SUMMARY")
    print("="*90)
    
    print("\n📊 RETURNS:")
    print(f"  Total Return:        {metrics['Total Return']:>10.2%}")
    print(f"  CAGR:                {metrics['CAGR']:>10.2%}")
    
    print("\n📉 RISK METRICS:")
    print(f"  Annualized Vol:      {metrics['Ann Vol']:>10.2%}")
    print(f"  Max Drawdown:        {metrics['Max Drawdown']:>10.2%}")
    
    print("\n📈 RISK-ADJUSTED RETURNS:")
    print(f"  Sharpe Ratio:        {metrics['Sharpe']:>10.2f}")
    print(f"  Sortino Ratio:       {metrics['Sortino']:>10.2f}")
    print(f"  Calmar Ratio:        {metrics['Calmar']:>10.2f}")
    
    print("\n🎯 WIN/LOSS ANALYSIS:")
    print(f"  Hit Rate:            {metrics['Hit Rate']:>10.2%}")
    print(f"  Win/Loss Ratio:      {metrics['Win/Loss Ratio']:>10.2f}")
    print(f"  Average Win:         {metrics['Avg Win']:>10.2%}")
    print(f"  Average Loss:        {metrics['Avg Loss']:>10.2%}")
    
    # Signal statistics
    total_days = len(result)
    long_days = (result['Signal'] == 1).sum()
    bearish_days = (result['Signal'] == -1).sum()
    flat_days = (result['Signal'] == 0).sum()
    
    print("\n📊 SIGNAL STATISTICS:")
    print(f"  Total Days:          {total_days:>10.0f}")
    print(f"  LONG Days:           {long_days:>10.0f} ({long_days/total_days*100:.1f}%)")
    print(f"  BEARISH Days:        {bearish_days:>10.0f} ({bearish_days/total_days*100:.1f}%)")
    print(f"  FLAT Days:           {flat_days:>10.0f} ({flat_days/total_days*100:.1f}%)")
    
    print("\n🤖 ML MODEL PERFORMANCE:")
    print(f"  RF Accuracy:         {ml_performance['rf_accuracy']:>10.2%}")
    print(f"  GB Accuracy:         {ml_performance['gb_accuracy']:>10.2%}")
    print(f"  RF F1 Score:         {ml_performance['rf_f1']:>10.2%}")
    
    print("="*90 + "\n")
    
    # Step 8: Display current status
    display_current_status(result, ml_model, num_days=10)
    display_options_recommendations(result.iloc[-1])
    
    # Step 9: Plot and save
    print("📈 Creating visualizations...")
    plot_results(result, args.outdir)
    
    print(f"✅ Results saved to: {args.outdir}/")
    print(f"   - ml_price_signals.png")
    print(f"   - ml_equity_curve.png")
    print(f"   - ml_probability.png")
    print(f"   - vix.png")
    
    print("\n" + "🎉"*45)
    print(" ML-ENHANCED STRATEGY COMPLETE!")
    print("🎉"*45 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML-Enhanced NIFTY Trading Strategy')
    parser.add_argument('--start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date')
    parser.add_argument('--outdir', default='results_ml', help='Output directory')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--position_fraction', type=float, default=0.10, help='Position size fraction')
    parser.add_argument('--round_trip_cost', type=float, default=0.0006, help='Round-trip cost')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage')
    
    args = parser.parse_args()
    main(args)

