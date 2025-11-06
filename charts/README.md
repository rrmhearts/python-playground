# Algorithmic Chart Pattern Trading System

A sophisticated Python-based algorithmic trading system that automatically detects chart patterns, analyzes moving averages, and executes trades using the Alpaca API. The system implements advanced pattern recognition algorithms, comprehensive risk management, and automated position management for systematic trading.

## 🚀 Features

### Pattern Recognition
- **Head and Shoulders**: Detects classic reversal patterns with confidence scoring
- **Triangle Patterns**: Identifies ascending, descending, and symmetrical triangles
- **Flag and Pennant**: Recognizes continuation patterns with volume confirmation
- **Cup and Handle**: Detects accumulation patterns with proper validation

### Technical Analysis
- **Moving Average Systems**: Multiple timeframe analysis (20/50 SMA, 12/26 EMA)
- **MACD Integration**: Momentum confirmation for trade signals
- **Volume Analysis**: Volume surge detection and pattern confirmation
- **Trend Quality Assessment**: Multi-factor trend strength evaluation

### Risk Management
- **Portfolio Risk Control**: Maximum 5% portfolio risk exposure
- **Position Sizing**: Kelly Criterion-based sizing with pattern confidence weighting
- **Stop Loss Management**: Automatic stop placement and trailing stops
- **Position Limits**: Maximum 10 concurrent positions with correlation checks

### Automated Trading
- **Real-time Scanning**: Continuous monitoring of 40+ liquid stocks and ETFs
- **Order Execution**: Automated market orders with stop loss and profit targets
- **Position Management**: Dynamic trailing stops and exit signal detection
- **Performance Tracking**: Comprehensive logging and P&L monitoring

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Stable internet connection
- Windows, macOS, or Linux

### Python Dependencies
```
alpaca-trade-api==3.0.2
pandas==2.0.3
numpy==1.24.3
yfinance==0.2.18
TA-Lib==0.4.26
scipy==1.11.1
python-dateutil==2.8.2
requests==2.31.0
```

### Trading Account
- Alpaca Markets account (paper or live trading)
- API credentials (Key ID and Secret Key)
- Minimum $1,000 account balance recommended

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rrmhearts/algorithmic-pattern-trading.git
cd algorithmic-pattern-trading
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv trading_env

# Activate virtual environment
# On Windows:
trading_env\Scripts\activate
# On macOS/Linux:
source trading_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install TA-Lib (Technical Analysis Library)

#### Windows:
```bash
# Download appropriate wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.26-cp39-cp39-win_amd64.whl
```

#### macOS:
```bash
brew install ta-lib
pip install TA-Lib
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

## 📊 Alpaca Markets Setup

### 1. Create Alpaca Account
1. Visit [Alpaca Markets](https://alpaca.markets/)
2. Sign up for a free account
3. Complete account verification
4. Choose between paper trading (recommended for testing) or live trading

### 2. Generate API Credentials
1. Log into your Alpaca dashboard
2. Navigate to "Account" → "API Keys"
3. Generate new API key pair
4. Save your Key ID and Secret Key securely

### 3. API Endpoints
- **Paper Trading**: `https://paper-api.alpaca.markets`
- **Live Trading**: `https://api.alpaca.markets`

### 4. Trading Permissions
Ensure your account has the following permissions enabled:
- ✅ Equity Trading
- ✅ Extended Hours Trading (optional)
- ✅ Pattern Day Trading (if applicable)

### 5. Account Requirements
- **Paper Trading**: No minimum balance
- **Live Trading**: $500 minimum (varies by account type)
- **Pattern Day Trading**: $25,000 minimum for unlimited day trades

## ⚙️ Configuration

### 1. API Credentials
Edit the main trading file and replace the placeholder credentials:

```python
# In main() function
API_KEY = "YOUR_ALPACA_API_KEY"        # Replace with your Key ID
SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"  # Replace with your Secret Key
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading URL
```

### 2. Risk Parameters
Modify risk settings in the `RiskManager` class:

```python
class RiskManager:
    def __init__(self, max_portfolio_risk: float = 0.05):  # 5% max portfolio risk
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = 0.02    # 2% risk per position
        self.max_positions = 10          # Maximum positions
        self.correlation_limit = 0.7     # Position correlation limit
```

### 3. Trading Universe
Customize the stock universe in the `TradingSystem` class:

```python
self.universe = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Add/remove symbols
    # ... more symbols
]
```

### 4. Pattern Sensitivity
Adjust pattern detection parameters:

```python
# In PatternDetector class
self.min_pattern_length = 20      # Minimum bars for pattern
self.lookback_period = 100        # Historical data period
MIN_PATTERN_CONFIDENCE = 60       # Minimum confidence threshold
```

## 🚀 Usage

### Basic Execution
```bash
python trading_system.py
```

### Continuous Trading Mode
Uncomment the continuous trading loop in `main()` for 24/7 operation:

```python
# Continuous trading loop
while True:
    try:
        clock = trading_system.api.get_clock()
        if clock.is_open:
            trading_system.run_trading_cycle()
            time.sleep(300)  # 5-minute intervals
        else:
            logger.info("Market closed - waiting...")
            time.sleep(3600)  # Check hourly when closed
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
        break
```

### Command Line Options
```bash
# Run with custom configuration
python trading_system.py --config config.json

# Run in backtest mode
python trading_system.py --backtest --start-date 2023-01-01 --end-date 2023-12-31

# Run with specific symbols
python trading_system.py --symbols "AAPL,MSFT,GOOGL"
```

## 📈 Pattern Detection Details

### Head and Shoulders
- **Detection**: Identifies three peaks with center peak higher
- **Validation**: Checks shoulder symmetry and neckline support
- **Entry**: Break below neckline with 2% confirmation
- **Target**: Pattern height projected from neckline

### Triangle Patterns
- **Ascending**: Horizontal resistance, rising support
- **Descending**: Horizontal support, declining resistance
- **Symmetrical**: Converging support and resistance
- **Breakout**: Volume-confirmed break beyond triangle boundary

### Flag and Pennant
- **Flagpole**: Minimum 8% price move in 5-15 bars
- **Flag**: Tight consolidation with declining volume
- **Entry**: Breakout above flag high with volume surge
- **Target**: Flagpole height added to breakout point

### Cup and Handle
- **Cup**: U-shaped pattern over 4-12 weeks
- **Handle**: Small pullback after cup formation
- **Entry**: Break above handle high
- **Target**: Cup depth added to breakout point

## 🛡️ Risk Management Features

### Position Sizing
```python
# Kelly Criterion with confidence adjustment
position_size = (confidence_score / 100) * kelly_fraction * account_value
```

### Stop Loss Management
- **Initial Stop**: Pattern-based stop placement
- **Trailing Stop**: Activated at 10% profit
- **Time Stop**: Maximum 30-day hold period
- **Technical Stop**: MA breakdown exit signals

### Portfolio Protection
- **Maximum Risk**: 5% of total portfolio
- **Position Correlation**: Maximum 70% correlation between positions
- **Drawdown Limits**: Trading halt at 15% account drawdown
- **Daily Loss Limit**: Maximum 3% daily loss

## 📊 Performance Monitoring

### Logging System
The system provides comprehensive logging:

```
2024-01-15 09:30:00 - INFO - Trading system initialized successfully
2024-01-15 09:30:05 - INFO - Found bull_flag in AAPL - Confidence: 78.5%
2024-01-15 09:30:10 - INFO - Placed order for 50 shares of AAPL
2024-01-15 09:35:00 - INFO - Updated trailing stop for AAPL to $185.50
```

### Performance Metrics
- **Win Rate**: Percentage of profitable trades
- **Average Return**: Mean return per trade
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Ratio of gross profits to gross losses

### Portfolio Summary
```
=== Portfolio Summary ===
Equity: $10,450.25
Cash: $2,150.75
Positions: 7
AAPL: 50 shares @ $180.00 Current: $185.50 P&L: $275.00
MSFT: 30 shares @ $350.00 Current: $345.00 P&L: -$150.00
Total Unrealized P&L: $1,250.75
========================
```

## ⚠️ Important Warnings

### Financial Risk
- **Past Performance**: Historical results don't guarantee future performance
- **Market Risk**: All trading involves risk of financial loss
- **System Risk**: Technical failures can result in unintended positions
- **Regulatory Risk**: Trading rules and regulations may change

### Testing Requirements
- **Paper Trading**: Always test thoroughly with paper trading first
- **Backtesting**: Validate strategies on historical data
- **Small Position Sizes**: Start with minimal position sizes
- **Gradual Scaling**: Increase position sizes only after proven performance

### Technical Considerations
- **Internet Connection**: Stable connection required for real-time trading
- **System Uptime**: Computer must remain running during market hours
- **API Limits**: Alpaca has rate limits on API calls
- **Data Quality**: Market data delays can affect performance

## 🔧 Troubleshooting

### Common Issues

#### API Connection Errors
```python
# Error: "Invalid API credentials"
# Solution: Verify API keys are correct and active
```

#### Pattern Detection Issues
```python
# Error: "Insufficient data for pattern detection"
# Solution: Ensure minimum 100 bars of historical data
```

#### TA-Lib Installation Problems
```bash
# Windows: Download pre-compiled wheel
# macOS: Use Homebrew installation
# Linux: Compile from source
```

### Debug Mode
Enable debug logging for troubleshooting:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Issues
- **Slow Pattern Detection**: Reduce universe size or increase scan intervals
- **Memory Usage**: Implement data cleanup and garbage collection
- **API Rate Limits**: Add delays between API calls

## 📚 Additional Resources

### Educational Materials
- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [Technical Analysis Patterns](https://www.investopedia.com/articles/technical/112601.asp)
- [Risk Management Strategies](https://www.investopedia.com/articles/trading/09/risk-management.asp)

### Community
- [Alpaca Community Forum](https://forum.alpaca.markets/)
- [Algorithmic Trading Subreddit](https://www.reddit.com/r/algotrading/)
- [QuantConnect Community](https://www.quantconnect.com/forum/)

### Books
- "Algorithmic Trading" by Ernie Chan
- "Technical Analysis of Financial Markets" by John Murphy
- "Quantitative Trading" by Ernie Chan

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Disclaimer

This software is for educational and research purposes only. The authors and contributors:

- Do not provide investment advice
- Are not responsible for any financial losses
- Do not guarantee system performance
- Recommend consulting with financial professionals

**Trading involves substantial risk of loss and is not suitable for all investors.**

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black trading_system.py

# Type checking
mypy trading_system.py
```

## 📞 Support

For support and questions:

1. Check the [Issues](https://github.com/rrmhearts/algorithmic-pattern-trading/issues) page
2. Review the documentation
3. Contact the maintainers

**Remember: Start with paper trading and never risk more than you can afford to lose!**