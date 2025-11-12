"""
Enhanced NIFTY Trading Strategy with Multiple Technical Indicators

ENHANCED FEATURES:
- Multiple Technical Indicators: SMA, RSI, MACD, ATR, Volume Analysis
- Smart Signal Logic: ALL conditions must align (reduces false signals)
- Risk Management: Auto-calculated Stop Loss & Take Profit levels
- CALL & PUT Options: Recommendations for both bullish and bearish trades
- Comprehensive Metrics: Sharpe, Sortino, Calmar, Win/Loss ratios
- Real Data: Fetches live NIFTY data via yfinance

THREE SIGNAL STATES:

1. LONG (Signal = 1) - Buy CALL Options when ALL bullish conditions met:
   - Price > SMA(10) - Uptrend confirmation
   - RSI between 40-70 - Not overbought/oversold
   - MACD Positive - Bullish momentum
   - Volume > 80% of average - Strong participation

2. BEARISH (Signal = -1) - Buy PUT Options when ALL bearish conditions met:
   - Price < SMA(10) - Downtrend confirmation
   - RSI < 40 - Oversold/weakness
   - MACD Negative - Bearish momentum
   - Volume > 80% of average - Strong selling

3. FLAT (Signal = 0) - Stay in CASH - Mixed/unclear signals

RISK MANAGEMENT:
- Stop Loss: 2x ATR (above/below entry based on direction)
- Take Profit: 3x ATR (1:1.5 Risk:Reward ratio)
- Position sizing with capital allocation control

OPTIONS RECOMMENDATIONS:
- CALL options (ATM, OTM, ITM) for LONG signals
- PUT options (ATM, OTM, ITM) for BEARISH signals
- Weekly/Monthly expiry suggestions
- Risk-reward calculations for each trade
- Alternative: NIFTY Futures (Long/Short)

Usage example:
  python3 nifty_strategy.py --start 2024-10-12 --end 2025-11-12 --outdir results

Requirements:
  pip install pandas numpy matplotlib yfinance
"""

import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False


def fetch_nifty(start_date, end_date):
    """Fetch NIFTY daily data via yfinance (^NSEI)."""
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not available. Install it or use --csv with historic data.")

    # Explicitly disable auto_adjust to get both Close and Adj Close
    df = yf.download("^NSEI", start=start_date, end=end_date, progress=False, auto_adjust=False)
    df.columns = df.columns.get_level_values(0)  # Flatten MultiIndex columns
    if df.empty:
        raise RuntimeError("No data fetched. Check dates or network.")
    df.index = pd.to_datetime(df.index)
    return df


def read_csv(path):
    df = pd.read_csv(path, parse_dates=[0])
    df = df.rename(columns={df.columns[0]: 'Date'}).set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col not in df.columns:
            if col == 'Adj Close' and 'Adj_Close' in df.columns:
                df['Adj Close'] = df['Adj_Close']
            elif col == 'Close' and 'close' in df.columns:
                df['Close'] = df['close']
    return df


def compute_rsi(series, period=14):
    """Calculate RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def compute_atr(df, period=14):
    """Calculate ATR (Average True Range) for volatility measure."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period, min_periods=1).mean()
    return atr


def compute_signals(df, sma_window=10, rsi_period=14, rsi_lower=40, rsi_upper=70, rsi_oversold=30):
    """
    Enhanced signal generation with multiple indicators.
    
    THREE SIGNAL STATES:
    
    1. LONG (Signal = 1) when ALL bullish conditions met:
       - Close > SMA(10)
       - RSI between 40-70 (not overbought/oversold)
       - MACD positive (bullish momentum)
       - Volume > 80% of average (strong participation)
    
    2. BEARISH (Signal = -1) when ALL bearish conditions met:
       - Close < SMA(10)
       - RSI < 40 (oversold/weakness)
       - MACD negative (bearish momentum)
       - Volume > 80% of average (strong selling)
    
    3. FLAT (Signal = 0) - No clear direction, stay in cash
    """
    df = df.copy()
    
    # Basic indicators
    df['SMA'] = df['Close'].rolling(sma_window, min_periods=1).mean()
    df['RSI'] = compute_rsi(df['Close'], period=rsi_period)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
    
    # ATR for stop loss calculation
    df['ATR'] = compute_atr(df, period=14)
    
    # Volume analysis
    df['Volume_MA'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Bullish conditions
    bullish_sma = df['Close'] > df['SMA']
    bullish_rsi = (df['RSI'] >= rsi_lower) & (df['RSI'] <= rsi_upper)
    bullish_macd = df['MACD'] > df['MACD_Signal']
    strong_volume = df['Volume_Ratio'] > 0.8
    
    # Bearish conditions
    bearish_sma = df['Close'] < df['SMA']
    bearish_rsi = df['RSI'] < rsi_lower  # RSI shows weakness
    bearish_macd = df['MACD'] < df['MACD_Signal']
    
    # Combined signal logic
    # LONG when all bullish conditions met
    long_signal = bullish_sma & bullish_rsi & bullish_macd & strong_volume
    
    # BEARISH when all bearish conditions met
    bearish_signal = bearish_sma & bearish_rsi & bearish_macd & strong_volume
    
    # Create signal: 1 = LONG, -1 = BEARISH, 0 = FLAT
    df['Signal'] = 0  # Default FLAT
    df.loc[long_signal, 'Signal'] = 1  # LONG
    df.loc[bearish_signal, 'Signal'] = -1  # BEARISH
    
    # Position for backtest (only LONG or FLAT, ignore BEARISH for now)
    df['Position'] = (df['Signal'] == 1).astype(int)
    
    # Calculate Stop Loss and Take Profit levels for LONG
    df['Stop_Loss_Long'] = df['Close'] - (2 * df['ATR'])  # 2x ATR below entry
    df['Take_Profit_Long'] = df['Close'] + (3 * df['ATR'])  # 3x ATR above entry
    
    # Calculate Stop Loss and Take Profit levels for BEARISH/SHORT
    df['Stop_Loss_Short'] = df['Close'] + (2 * df['ATR'])  # 2x ATR above entry
    df['Take_Profit_Short'] = df['Close'] - (3 * df['ATR'])  # 3x ATR below entry
    
    # Store individual condition flags for display
    df['SMA_Signal'] = bullish_sma.astype(int)
    df['RSI_Signal'] = bullish_rsi.astype(int)
    df['MACD_Signal_Flag'] = bullish_macd.astype(int)
    df['Volume_Signal'] = strong_volume.astype(int)
    df['Bearish_SMA'] = bearish_sma.astype(int)
    df['Bearish_RSI'] = bearish_rsi.astype(int)
    df['Bearish_MACD'] = bearish_macd.astype(int)
    
    return df


def get_option_recommendations(current_price, signal, capital=100000, position_fraction=0.10, 
                               stop_loss_long=None, take_profit_long=None,
                               stop_loss_short=None, take_profit_short=None, atr=None):
    """Generate NIFTY option trading recommendations for both CALL and PUT options."""
    from datetime import datetime, timedelta
    
    # Round to nearest 50 for strike price (NIFTY strikes are in 50 multiples)
    atm_strike = round(current_price / 50) * 50
    
    # Calculate CALL strikes (for bullish trades)
    otm_call_strike = atm_strike + 100  # 100 points OTM
    itm_call_strike = atm_strike - 100  # 100 points ITM
    
    # Calculate PUT strikes (for bearish trades)
    otm_put_strike = atm_strike - 100  # 100 points OTM (below current price)
    itm_put_strike = atm_strike + 100  # 100 points ITM (above current price)
    
    # Find next Thursday (weekly expiry) and last Thursday (monthly expiry)
    today = datetime.now()
    days_ahead = 3 - today.weekday()  # Thursday is 3
    if days_ahead <= 0:
        days_ahead += 7
    next_expiry = today + timedelta(days=days_ahead)
    
    # NIFTY lot size
    lot_size = 25  # Standard NIFTY options lot size
    
    # Capital allocation
    allocated_capital = capital * position_fraction
    
    # Risk-reward calculation for LONG
    risk_long = current_price - stop_loss_long if stop_loss_long else 0
    reward_long = take_profit_long - current_price if take_profit_long else 0
    risk_reward_ratio_long = reward_long / risk_long if risk_long > 0 else 0
    
    # Risk-reward calculation for SHORT
    risk_short = stop_loss_short - current_price if stop_loss_short else 0
    reward_short = current_price - take_profit_short if take_profit_short else 0
    risk_reward_ratio_short = reward_short / risk_short if risk_short > 0 else 0
    
    recommendations = {
        'signal': signal,
        'current_price': current_price,
        # CALL option strikes
        'atm_strike': atm_strike,
        'otm_call_strike': otm_call_strike,
        'itm_call_strike': itm_call_strike,
        # PUT option strikes
        'atm_put_strike': atm_strike,
        'otm_put_strike': otm_put_strike,
        'itm_put_strike': itm_put_strike,
        # Common fields
        'expiry': next_expiry.strftime('%d-%b-%Y'),
        'lot_size': lot_size,
        'allocated_capital': allocated_capital,
        # LONG risk management
        'stop_loss_long': stop_loss_long,
        'take_profit_long': take_profit_long,
        'risk_long': risk_long,
        'reward_long': reward_long,
        'risk_reward_ratio_long': risk_reward_ratio_long,
        # SHORT risk management
        'stop_loss_short': stop_loss_short,
        'take_profit_short': take_profit_short,
        'risk_short': risk_short,
        'reward_short': reward_short,
        'risk_reward_ratio_short': risk_reward_ratio_short,
        # Volatility
        'atr': atr,
    }
    
    return recommendations


def display_current_signal(df, num_days=10, capital=100000, position_fraction=0.10):
    """Display the last N days with current signal and technical indicators."""
    print("\n" + "="*80)
    print(" LATEST MARKET SIGNALS (Last {} Trading Days)".format(num_days))
    print("="*80)
    
    # Select columns to display
    display_cols = ['Close', 'SMA', 'RSI', 'MACD', 'Volume_Ratio', 'Signal']
    last_n = df.tail(num_days)[display_cols].copy()
    last_n['Signal_Text'] = last_n['Signal'].apply(lambda x: 'LONG' if x == 1 else ('BEARISH' if x == -1 else 'FLAT'))
    
    print("\n{:<12} {:>10} {:>10} {:>8} {:>10} {:>8} {:>8}".format(
        'Date', 'Close', 'SMA(10)', 'RSI', 'MACD', 'Vol', 'Signal'))
    print("-"*80)
    
    for idx, row in last_n.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        signal_marker = " ←" if idx == last_n.index[-1] else ""
        print("{:<12} {:>10.2f} {:>10.2f} {:>8.1f} {:>10.1f} {:>8.2f} {:>8}{}".format(
            date_str, row['Close'], row['SMA'], row['RSI'], row['MACD'], 
            row['Volume_Ratio'], row['Signal_Text'], signal_marker
        ))
    
    # Get latest data with all columns
    latest = df.iloc[-1]
    
    # Display detailed indicator status
    print("\n" + "="*80)
    print(" INDICATOR STATUS (Current)")
    print("="*80)
    
    # Check each condition
    sma_status = "✓ PASS" if latest['Close'] > latest['SMA'] else "✗ FAIL"
    rsi_status = "✓ PASS" if 40 <= latest['RSI'] <= 70 else "✗ FAIL"
    macd_status = "✓ PASS" if latest['MACD'] > latest['MACD_Signal'] else "✗ FAIL"
    volume_status = "✓ PASS" if latest['Volume_Ratio'] > 0.8 else "✗ FAIL"
    
    print(f"\n1. Price vs SMA:     Close ({latest['Close']:.2f}) > SMA ({latest['SMA']:.2f})    {sma_status}")
    print(f"2. RSI Range:        RSI = {latest['RSI']:.1f} (Target: 40-70)              {rsi_status}")
    print(f"3. MACD Momentum:    MACD ({latest['MACD']:.1f}) vs Signal ({latest['MACD_Signal']:.1f})   {macd_status}")
    print(f"4. Volume:           Vol Ratio = {latest['Volume_Ratio']:.2f}x (Target: >0.8)      {volume_status}")
    
    print("\n" + "="*80)
    if latest['Signal'] == 1:
        # LONG signal
        print("✓ FINAL SIGNAL: LONG - All bullish conditions met!")
        print(f"  Stop Loss:   ₹{latest['Stop_Loss_Long']:.2f} (Risk: ₹{latest['Close'] - latest['Stop_Loss_Long']:.2f})")
        print(f"  Take Profit: ₹{latest['Take_Profit_Long']:.2f} (Reward: ₹{latest['Take_Profit_Long'] - latest['Close']:.2f})")
        risk_reward = (latest['Take_Profit_Long'] - latest['Close']) / (latest['Close'] - latest['Stop_Loss_Long'])
        print(f"  Risk:Reward Ratio: 1:{risk_reward:.2f}")
    elif latest['Signal'] == -1:
        # BEARISH signal
        print("✓ FINAL SIGNAL: BEARISH - All bearish conditions met!")
        print(f"  Stop Loss:   ₹{latest['Stop_Loss_Short']:.2f} (Risk: ₹{latest['Stop_Loss_Short'] - latest['Close']:.2f})")
        print(f"  Take Profit: ₹{latest['Take_Profit_Short']:.2f} (Reward: ₹{latest['Close'] - latest['Take_Profit_Short']:.2f})")
        risk_reward = (latest['Close'] - latest['Take_Profit_Short']) / (latest['Stop_Loss_Short'] - latest['Close'])
        print(f"  Risk:Reward Ratio: 1:{risk_reward:.2f}")
    else:
        # FLAT signal
        failed_conditions = []
        if latest['Close'] <= latest['SMA']:
            failed_conditions.append("Price below SMA")
        if not (40 <= latest['RSI'] <= 70):
            if latest['RSI'] < 40:
                failed_conditions.append("RSI too low (oversold)")
            else:
                failed_conditions.append("RSI too high (overbought)")
        if latest['MACD'] <= latest['MACD_Signal']:
            failed_conditions.append("MACD bearish")
        if latest['Volume_Ratio'] <= 0.8:
            failed_conditions.append("Low volume")
        
        print("✗ FINAL SIGNAL: FLAT/CASH - Conditions not met")
        print(f"  Failed: {', '.join(failed_conditions)}")
    print("="*80 + "\n")
    
    # Generate and display option recommendations with stop loss info
    rec = get_option_recommendations(
        latest['Close'], latest['Signal'], capital, position_fraction,
        stop_loss_long=latest['Stop_Loss_Long'], take_profit_long=latest['Take_Profit_Long'],
        stop_loss_short=latest['Stop_Loss_Short'], take_profit_short=latest['Take_Profit_Short'],
        atr=latest['ATR']
    )
    display_option_recommendations(rec)


def display_option_recommendations(rec):
    """Display NIFTY INDEX OPTIONS recommendations for CALL and PUT options."""
    print("="*80)
    print(" NIFTY INDEX OPTIONS RECOMMENDATION")
    print("="*80)
    print("\nCurrent NIFTY Level: ₹{:.2f}".format(rec['current_price']))
    print("Capital Allocated: ₹{:,.0f} ({:.0%} of total)".format(
        rec['allocated_capital'], rec['allocated_capital'] / (rec['allocated_capital'] / 0.10)))
    print("Lot Size: {} units".format(rec['lot_size']))
    
    if rec['signal'] == 1:
        # LONG signal - recommend Call options
        print("\n" + "▶"*40)
        print("🟢 ACTION: GO LONG - BUY CALL OPTIONS")
        print("▶"*40)
        
        # Display risk management first
        print("\n🛡️  RISK MANAGEMENT:")
        print("-"*80)
        print(f"Entry Price:     ₹{rec['current_price']:.2f}")
        print(f"Stop Loss:       ₹{rec['stop_loss_long']:.2f} (Exit if NIFTY falls below this)")
        print(f"Take Profit:     ₹{rec['take_profit_long']:.2f} (Target exit level)")
        print(f"Risk per trade:  ₹{rec['risk_long']:.2f} ({(rec['risk_long']/rec['current_price']*100):.2f}%)")
        print(f"Reward target:   ₹{rec['reward_long']:.2f} ({(rec['reward_long']/rec['current_price']*100):.2f}%)")
        print(f"Risk:Reward:     1:{rec['risk_reward_ratio_long']:.2f}")
        print(f"ATR (Volatility): ₹{rec['atr']:.2f}")
        
        print("\n📋 RECOMMENDED OPTION STRATEGIES:")
        print("-"*80)
        
        print("\n1️⃣  CONSERVATIVE (ATM Call) - Best for beginners:")
        print("   → Buy NIFTY {} CE".format(rec['atm_strike']))
        print("   → Strike: {} (At-The-Money)".format(rec['atm_strike']))
        print("   → Expiry: {} (Weekly/Monthly)".format(rec['expiry']))
        print("   → Quantity: 1 Lot ({} units)".format(rec['lot_size']))
        print("   → Expected Premium: ₹150-250 per unit (approx)")
        print("   → Total Cost: ₹3,750 - ₹6,250 for 1 lot")
        print("   → Exit if NIFTY closes below ₹{:.2f}".format(rec['stop_loss_long']))
        print("   → Book profit if NIFTY reaches ₹{:.2f}".format(rec['take_profit_long']))
        
        print("\n2️⃣  MODERATE (Slightly OTM Call) - Higher leverage:")
        print("   → Buy NIFTY {} CE".format(rec['otm_call_strike']))
        print("   → Strike: {} (Out-of-The-Money)".format(rec['otm_call_strike']))
        print("   → Expiry: {} (Weekly/Monthly)".format(rec['expiry']))
        print("   → Expected Premium: ₹80-150 per unit (cheaper)")
        print("   → Total Cost: ₹2,000 - ₹3,750 for 1 lot")
        print("   → Higher risk, but lower capital required")
        print("   → Quantity: Can buy 1-2 lots with allocated capital")
        
        print("\n3️⃣  AGGRESSIVE (ITM Call) - Safer but expensive:")
        print("   → Buy NIFTY {} CE".format(rec['itm_call_strike']))
        print("   → Strike: {} (In-The-Money)".format(rec['itm_call_strike']))
        print("   → Expected Premium: ₹250-400 per unit")
        print("   → Total Cost: ₹6,250 - ₹10,000 for 1 lot")
        print("   → Higher delta, moves more with NIFTY")
        print("   → Better for directional trades")
        
        print("\n💡 ALTERNATIVE: NIFTY FUTURES (If you have margin):")
        print("   → Buy 1 Lot NIFTY FUT (Current Month)")
        print("   → Lot Size: 25 units")
        print("   → Contract Value: ₹{:,.0f}".format(rec['current_price'] * 25))
        print("   → Margin Required: ~₹1,20,000 - ₹1,50,000")
        print("   → Stop Loss: Exit if NIFTY < ₹{:.2f}".format(rec['stop_loss_long']))
        print("   → Take Profit: Exit if NIFTY > ₹{:.2f}".format(rec['take_profit_long']))
        
        print("\n⚡ TRADE EXECUTION TIPS:")
        print("   • Place stop-loss orders immediately after entry")
        print("   • Don't trade without stops - options can go to zero!")
        print("   • Monitor position daily, especially near expiry")
        print("   • Exit if overall signal turns FLAT or BEARISH")
        print("   • Book partial profits at 50% of target")
        
    elif rec['signal'] == -1:
        # BEARISH signal - recommend PUT options
        print("\n" + "▼"*40)
        print("🔴 ACTION: GO SHORT - BUY PUT OPTIONS")
        print("▼"*40)
        
        # Display risk management first
        print("\n🛡️  RISK MANAGEMENT:")
        print("-"*80)
        print(f"Entry Price:     ₹{rec['current_price']:.2f}")
        print(f"Stop Loss:       ₹{rec['stop_loss_short']:.2f} (Exit if NIFTY rises above this)")
        print(f"Take Profit:     ₹{rec['take_profit_short']:.2f} (Target exit level)")
        print(f"Risk per trade:  ₹{rec['risk_short']:.2f} ({(rec['risk_short']/rec['current_price']*100):.2f}%)")
        print(f"Reward target:   ₹{rec['reward_short']:.2f} ({(rec['reward_short']/rec['current_price']*100):.2f}%)")
        print(f"Risk:Reward:     1:{rec['risk_reward_ratio_short']:.2f}")
        print(f"ATR (Volatility): ₹{rec['atr']:.2f}")
        
        print("\n📋 RECOMMENDED PUT OPTION STRATEGIES:")
        print("-"*80)
        
        print("\n1️⃣  CONSERVATIVE (ATM Put) - Best for beginners:")
        print("   → Buy NIFTY {} PE".format(rec['atm_put_strike']))
        print("   → Strike: {} (At-The-Money)".format(rec['atm_put_strike']))
        print("   → Expiry: {} (Weekly/Monthly)".format(rec['expiry']))
        print("   → Quantity: 1 Lot ({} units)".format(rec['lot_size']))
        print("   → Expected Premium: ₹150-250 per unit (approx)")
        print("   → Total Cost: ₹3,750 - ₹6,250 for 1 lot")
        print("   → Exit if NIFTY closes above ₹{:.2f}".format(rec['stop_loss_short']))
        print("   → Book profit if NIFTY falls to ₹{:.2f}".format(rec['take_profit_short']))
        
        print("\n2️⃣  MODERATE (OTM Put) - Higher leverage:")
        print("   → Buy NIFTY {} PE".format(rec['otm_put_strike']))
        print("   → Strike: {} (Out-of-The-Money, below current price)".format(rec['otm_put_strike']))
        print("   → Expiry: {} (Weekly/Monthly)".format(rec['expiry']))
        print("   → Expected Premium: ₹80-150 per unit (cheaper)")
        print("   → Total Cost: ₹2,000 - ₹3,750 for 1 lot")
        print("   → Higher risk, but lower capital required")
        print("   → Profits more if NIFTY falls sharply")
        
        print("\n3️⃣  AGGRESSIVE (ITM Put) - Safer but expensive:")
        print("   → Buy NIFTY {} PE".format(rec['itm_put_strike']))
        print("   → Strike: {} (In-The-Money, above current price)".format(rec['itm_put_strike']))
        print("   → Expected Premium: ₹250-400 per unit")
        print("   → Total Cost: ₹6,250 - ₹10,000 for 1 lot")
        print("   → Higher delta, moves more with NIFTY decline")
        print("   → Better for directional bearish trades")
        
        print("\n💡 ALTERNATIVE: NIFTY FUTURES SHORT (If you have margin):")
        print("   → Sell 1 Lot NIFTY FUT (Current Month)")
        print("   → Lot Size: 25 units")
        print("   → Contract Value: ₹{:,.0f}".format(rec['current_price'] * 25))
        print("   → Margin Required: ~₹1,20,000 - ₹1,50,000")
        print("   → Stop Loss: Exit if NIFTY > ₹{:.2f}".format(rec['stop_loss_short']))
        print("   → Take Profit: Exit if NIFTY < ₹{:.2f}".format(rec['take_profit_short']))
        
        print("\n⚡ TRADE EXECUTION TIPS (Bearish Trades):")
        print("   • PUT options profit when NIFTY falls")
        print("   • Place stop-loss orders immediately (exit if NIFTY rises)")
        print("   • Bearish moves can be sharp - monitor closely")
        print("   • Exit if overall signal turns LONG or FLAT")
        print("   • Book profits quickly in bearish trades (moves are faster)")
        
    else:
        # FLAT signal - recommend staying out BUT show strikes for reference
        print("\n" + "■"*40)
        print("🔴 ACTION: STAY IN CASH / EXIT POSITIONS")
        print("■"*40)
        
        print("\n📋 RECOMMENDED ACTION:")
        print("-"*80)
        print("\n⚠️  NO LONG POSITION RECOMMENDED")
        print("   → Current technical conditions do not support going long")
        print("   → Stay in CASH or LIQUIDBEES")
        print("   → If holding Call options: Consider exiting or tightening stops")
        print("   → Wait for ALL indicators to align (Close > SMA, RSI 40-70, MACD +ve, Volume)")
        
        print("\n💡 WHAT TO WATCH FOR (Signal may turn LONG when):")
        if rec['stop_loss_long']:
            print(f"   → NIFTY closes above ₹{rec['current_price'] + 50:.2f}")
        print("   → RSI enters 40-70 range")
        print("   → MACD turns positive")
        print("   → Volume increases above average")
        
        # Show option strikes for reference even when FLAT
        print("\n" + "="*80)
        print("📋 OPTION STRIKES (FOR REFERENCE ONLY - NOT RECOMMENDED NOW)")
        print("="*80)
        print("\n⚠️  THESE ARE NOT BUY RECOMMENDATIONS - Signal is FLAT!")
        print("   These strikes are shown for planning purposes only.")
        print("   Wait for signal to turn LONG or BEARISH before trading.")
        
        print("\n📍 IF Signal Turns LONG (Bullish), Consider These CALL Strikes:")
        print("-"*80)
        
        print("\n1️⃣  CONSERVATIVE Option (ATM Call):")
        print("   → NIFTY {} CE".format(rec['atm_strike']))
        print("   → Strike: {} (At-The-Money)".format(rec['atm_strike']))
        print("   → Expiry: {} (Weekly/Monthly)".format(rec['expiry']))
        print("   → Lot Size: {} units".format(rec['lot_size']))
        print("   → Current Premium: Check market (approx ₹150-250/unit)")
        
        print("\n2️⃣  MODERATE Option (OTM Call):")
        print("   → NIFTY {} CE".format(rec['otm_call_strike']))
        print("   → Strike: {} (Out-of-The-Money)".format(rec['otm_call_strike']))
        
        print("\n3️⃣  AGGRESSIVE Option (ITM Call):")
        print("   → NIFTY {} CE".format(rec['itm_call_strike']))
        print("   → Strike: {} (In-The-Money)".format(rec['itm_call_strike']))
        
        print("\n📍 IF Signal Turns BEARISH, Consider These PUT Strikes:")
        print("-"*80)
        
        print("\n1️⃣  CONSERVATIVE Option (ATM Put):")
        print("   → NIFTY {} PE".format(rec['atm_put_strike']))
        print("   → Strike: {} (At-The-Money)".format(rec['atm_put_strike']))
        print("   → Expiry: {} (Weekly/Monthly)".format(rec['expiry']))
        print("   → Lot Size: {} units".format(rec['lot_size']))
        print("   → Current Premium: Check market (approx ₹150-250/unit)")
        
        print("\n2️⃣  MODERATE Option (OTM Put):")
        print("   → NIFTY {} PE".format(rec['otm_put_strike']))
        print("   → Strike: {} (Out-of-The-Money, below price)".format(rec['otm_put_strike']))
        
        print("\n3️⃣  AGGRESSIVE Option (ITM Put):")
        print("   → NIFTY {} PE".format(rec['itm_put_strike']))
        print("   → Strike: {} (In-The-Money, above price)".format(rec['itm_put_strike']))
        
        print("\n💡 ALTERNATIVE: NIFTY FUTURES")
        print("   → LONG: Buy NIFTY FUT if signal turns bullish")
        print("   → SHORT: Sell NIFTY FUT if signal turns bearish")
        print("   → Contract Value: ₹{:,.0f}".format(rec['current_price'] * 25))
        
        print("\n🚫 REMINDER: DO NOT TRADE NOW - Signal is FLAT!")
        print("   Wait for clear LONG or BEARISH signal before entering positions.")
        
        print("\n💡 OPTIONAL (For Advanced Traders ONLY):")
        print("   → You could sell OTM Calls (bearish view)")
        print("   → Or use Bear Put Spread (limited risk bearish strategy)")
        print("   → Or stay completely out of market (Recommended for most)")
    
    print("\n" + "="*80)
    print("⚠️  DISCLAIMER:")
    print("    This is an enhanced technical strategy with multiple indicators.")
    print("    Options involve significant risk of loss. Past performance ≠ future results.")
    print("    Always use stop losses. Never risk more than 1-2% of capital per trade.")
    print("    Consult a SEBI registered financial advisor before trading.")
    print("="*80 + "\n")


def backtest(df, initial_capital=100000.0, position_fraction=0.10, round_trip_cost=0.0006, slippage=0.0005):
    df = df.copy().sort_index()
    capital = initial_capital
    equity = []
    cash = capital
    last_position_units = 0.0

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        pos_flag = df['Position'].iloc[i]

        target_allocation = position_fraction * capital if pos_flag == 1 else 0.0
        target_units = target_allocation / price if price > 0 else 0.0

        trade_units = target_units - last_position_units
        trade_value = abs(trade_units) * price

        cost = trade_value * round_trip_cost
        slip = trade_value * slippage

        cash -= (trade_units * price) + cost + slip
        last_position_units = target_units

        mv = last_position_units * price
        capital = cash + mv
        equity.append(capital)

    res = df.copy()
    res['Equity'] = equity
    res['Returns'] = res['Equity'].pct_change().fillna(0)
    return res


def performance_metrics(equity_series, trading_days_per_year=252):
    """Calculate comprehensive performance metrics."""
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    
    daily_returns = equity_series.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(trading_days_per_year)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year) if daily_returns.std() != 0 else np.nan
    
    # Sortino Ratio (uses only downside deviation)
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
    sortino = (daily_returns.mean() / downside_std) * np.sqrt(trading_days_per_year) if downside_std != 0 else np.nan
    
    # Drawdown analysis
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # Calmar Ratio (CAGR / Max Drawdown)
    calmar = abs(cagr / max_dd) if max_dd != 0 else np.nan
    
    # Win/Loss metrics
    hit_rate = (daily_returns > 0).mean()
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    
    # Best and worst day
    best_day = daily_returns.max()
    worst_day = daily_returns.min()
    
    # Consecutive wins/losses
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


def plot_and_save(res, outdir, prefix='nifty'):
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(res.index, res['Close'], label='Close')
    plt.plot(res.index, res['SMA'], label='SMA(10)')
    plt.scatter(res.index[res['Position'] == 1],
                res.loc[res['Position'] == 1, 'Close'], marker='^', color='g', label='Long')
    plt.title('NIFTY Close & SMA(10)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_price_signals.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(res.index, res['Equity'], label='Equity')
    plt.title('Equity Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_equity_curve.png'))
    plt.close()

    roll_max = res['Equity'].cummax()
    drawdown = (res['Equity'] - roll_max) / roll_max
    plt.figure(figsize=(12, 4))
    plt.plot(res.index, drawdown)
    plt.title('Drawdown')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_drawdown.png'))
    plt.close()


def main(args):
    df = read_csv(args.csv) if args.csv else fetch_nifty(args.start, args.end)

    # Fallback handling for missing Adj Close
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'Adj Close' in df.columns:
        required_cols.append('Adj Close')
    else:
        df['Adj Close'] = df['Close']
    df = df[required_cols].dropna()

    df = compute_signals(df, sma_window=args.sma)
    
    # Display current market signal with option recommendations
    display_current_signal(df, num_days=10, capital=args.capital, position_fraction=args.position_fraction)

    res = backtest(df, initial_capital=args.capital,
                   position_fraction=args.position_fraction,
                   round_trip_cost=args.round_trip_cost,
                   slippage=args.slippage)

    metrics = performance_metrics(res['Equity'])
    
    # Display enhanced metrics
    print("\n" + "="*80)
    print(" BACKTEST PERFORMANCE SUMMARY")
    print("="*80)
    
    print("\n📊 RETURNS:")
    print(f"  Total Return:        {metrics['Total Return']:>10.2%}")
    print(f"  CAGR:                {metrics['CAGR']:>10.2%}")
    
    print("\n📉 RISK METRICS:")
    print(f"  Annualized Vol:      {metrics['Ann Vol']:>10.2%}")
    print(f"  Max Drawdown:        {metrics['Max Drawdown']:>10.2%}")
    
    print("\n📈 RISK-ADJUSTED RETURNS:")
    print(f"  Sharpe Ratio:        {metrics['Sharpe']:>10.2f}")
    print(f"  Sortino Ratio:       {metrics['Sortino']:>10.2f}  (Better than Sharpe)")
    print(f"  Calmar Ratio:        {metrics['Calmar']:>10.2f}")
    
    print("\n🎯 WIN/LOSS ANALYSIS:")
    print(f"  Hit Rate:            {metrics['Hit Rate']:>10.2%}")
    print(f"  Win/Loss Ratio:      {metrics['Win/Loss Ratio']:>10.2f}")
    print(f"  Average Win:         {metrics['Avg Win']:>10.2%}")
    print(f"  Average Loss:        {metrics['Avg Loss']:>10.2%}")
    
    print("\n🔥 STREAKS:")
    print(f"  Best Day:            {metrics['Best Day']:>10.2%}")
    print(f"  Worst Day:           {metrics['Worst Day']:>10.2%}")
    print(f"  Max Win Streak:      {metrics['Max Win Streak']:>10.0f} days")
    print(f"  Max Loss Streak:     {metrics['Max Loss Streak']:>10.0f} days")
    
    print("\n" + "="*80)
    
    # Calculate signal statistics
    total_days = len(res)
    long_days = (res['Signal'] == 1).sum()
    bearish_days = (res['Signal'] == -1).sum()
    flat_days = (res['Signal'] == 0).sum()
    
    print("\n📊 SIGNAL STATISTICS:")
    print(f"  Total Days:          {total_days:>10.0f}")
    print(f"  LONG Days:           {long_days:>10.0f} ({long_days/total_days*100:.1f}%)")
    print(f"  BEARISH Days:        {bearish_days:>10.0f} ({bearish_days/total_days*100:.1f}%)")
    print(f"  FLAT Days:           {flat_days:>10.0f} ({flat_days/total_days*100:.1f}%)")
    print("\n  Strategy trades {long_days} bullish signals in the backtest period")
    print(f"  ({bearish_days} bearish signals were identified but not traded in backtest)")
    print("="*80)

    plot_and_save(res, args.outdir)
    print(f"\nSaved results and plots in: {args.outdir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2019-11-12')
    p.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'))
    p.add_argument('--csv', default=None)
    p.add_argument('--outdir', default='results')
    p.add_argument('--sma', type=int, default=10)
    p.add_argument('--capital', type=float, default=100000.0)
    p.add_argument('--position_fraction', type=float, default=0.10)
    p.add_argument('--round_trip_cost', type=float, default=0.0006)
    p.add_argument('--slippage', type=float, default=0.0005)
    args = p.parse_args()
    main(args)
