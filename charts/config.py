# config.py - Configuration settings
class TradingConfig:
    # Risk Management
    MAX_PORTFOLIO_RISK = 0.05  # 5% maximum portfolio risk
    MAX_POSITION_RISK = 0.02   # 2% risk per position
    MAX_POSITIONS = 10         # Maximum number of positions
    
    # Pattern Detection
    MIN_PATTERN_CONFIDENCE = 60  # Minimum confidence score
    MIN_STOCK_PRICE = 10        # Minimum stock price to trade
    
    # Moving Averages
    SHORT_MA = 20
    LONG_MA = 50
    SIGNAL_MA = 9
    
    # Trading Parameters
    PROFIT_TARGET_MULTIPLIER = 2.0  # Risk:Reward ratio
    TRAILING_STOP_THRESHOLD = 0.10  # Start trailing at 10% profit
    MAX_HOLD_DAYS = 30             # Maximum days to hold position
    
    # Market Data
    LOOKBACK_PERIODS = 100
    MIN_VOLUME = 100000  # Minimum daily volume