import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
import talib
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PatternSignal:
    """Data class for pattern signals"""
    symbol: str
    pattern_type: str
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    signal_time: datetime
    timeframe: str

@dataclass
class Position:
    """Data class for tracking positions"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float
    target_price: float
    pattern_type: str
    entry_time: datetime

class PatternDetector:
    """Advanced pattern detection with multiple algorithms"""
    
    def __init__(self):
        self.min_pattern_length = 20
        self.lookback_period = 100
    
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Head and Shoulders pattern"""
        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Find peaks and troughs
            peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
            troughs, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
            
            if len(peaks) < 3 or len(troughs) < 2:
                return None
            
            # Get last 3 peaks for potential H&S
            recent_peaks = peaks[-3:]
            left_shoulder, head, right_shoulder = recent_peaks
            
            # Validate H&S geometry
            left_high = highs[left_shoulder]
            head_high = highs[head]
            right_high = highs[right_shoulder]
            
            # Head should be highest, shoulders roughly equal
            if (head_high > left_high and head_high > right_high and
                abs(left_high - right_high) / left_high < 0.05):
                
                # Find neckline
                neckline_left = lows[left_shoulder:head].min()
                neckline_right = lows[head:right_shoulder].min()
                neckline = max(neckline_left, neckline_right)
                
                # Calculate target
                pattern_height = head_high - neckline
                target = neckline - pattern_height
                
                return {
                    'type': 'head_and_shoulders',
                    'confidence': self._calculate_hs_confidence(left_high, head_high, right_high, neckline),
                    'entry': neckline * 0.98,  # Break below neckline
                    'stop_loss': head_high * 1.02,
                    'target': target,
                    'neckline': neckline
                }
        except Exception as e:
            logger.error(f"Error in H&S detection: {e}")
        
        return None
    
    def detect_triangle_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        try:
            highs = df['high'].values[-50:]  # Last 50 bars
            lows = df['low'].values[-50:]
            closes = df['close'].values
            
            if len(highs) < 20:
                return None
            
            # Find recent peaks and troughs
            peaks, _ = find_peaks(highs, distance=3)
            troughs, _ = find_peaks(-lows, distance=3)
            
            if len(peaks) < 3 or len(troughs) < 3:
                return None
            
            # Get trend lines
            peak_slope = self._calculate_trendline_slope(peaks[-3:], highs[peaks[-3:]])
            trough_slope = self._calculate_trendline_slope(troughs[-3:], lows[troughs[-3:]])
            
            # Determine triangle type
            triangle_type = self._classify_triangle(peak_slope, trough_slope)
            
            if triangle_type:
                # Calculate breakout levels
                recent_high = highs[-5:].max()
                recent_low = lows[-5:].min()
                
                if triangle_type == 'ascending':
                    entry = recent_high * 1.02
                    stop_loss = recent_low * 0.98
                    target = entry + (recent_high - recent_low) * 1.5
                elif triangle_type == 'descending':
                    entry = recent_low * 0.98
                    stop_loss = recent_high * 1.02
                    target = entry - (recent_high - recent_low) * 1.5
                else:  # symmetrical
                    # Wait for breakout direction
                    entry = recent_high * 1.02  # Assume bullish breakout
                    stop_loss = recent_low * 0.98
                    target = entry + (recent_high - recent_low) * 1.2
                
                return {
                    'type': f'{triangle_type}_triangle',
                    'confidence': self._calculate_triangle_confidence(peaks, troughs, highs, lows),
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'target': target,
                    'pattern_range': recent_high - recent_low
                }
        except Exception as e:
            logger.error(f"Error in triangle detection: {e}")
        
        return None
    
    def detect_flag_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect flag/pennant patterns"""
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            if len(closes) < 30:
                return None
            
            # Look for strong move (flagpole)
            recent_moves = []
            for i in range(10, 25):
                move_pct = (closes[-i] - closes[-i-10]) / closes[-i-10]
                recent_moves.append(move_pct)
            
            max_move = max(recent_moves)
            
            # Require significant move for flagpole
            if max_move < 0.08:  # 8% minimum move
                return None
            
            # Check for consolidation after move (flag)
            flagpole_end = recent_moves.index(max_move)
            consolidation_period = closes[-flagpole_end:]
            
            # Flag should be sideways with declining volume
            consolidation_range = (max(consolidation_period) - min(consolidation_period)) / max(consolidation_period)
            
            if consolidation_range < 0.06:  # Tight consolidation
                entry = max(consolidation_period) * 1.01
                stop_loss = min(consolidation_period) * 0.98
                target = entry + (entry - stop_loss) * 2  # 1:2 risk/reward
                
                return {
                    'type': 'bull_flag',
                    'confidence': self._calculate_flag_confidence(max_move, consolidation_range, volumes),
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'target': target,
                    'flagpole_move': max_move
                }
        except Exception as e:
            logger.error(f"Error in flag detection: {e}")
        
        return None
    
    def detect_cup_and_handle(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect cup and handle pattern"""
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            if len(closes) < 60:  # Need sufficient data for cup formation
                return None
            
            # Look for cup formation (U-shape)
            cup_start_idx = -60
            cup_data = closes[cup_start_idx:]
            
            # Find the left rim, bottom, and right rim
            left_third = cup_data[:20]
            middle_third = cup_data[20:40]
            right_third = cup_data[40:]
            
            left_high = max(left_third)
            cup_bottom = min(middle_third)
            right_high = max(right_third)
            
            # Validate cup shape
            if (abs(left_high - right_high) / left_high < 0.05 and  # Rims roughly equal
                (left_high - cup_bottom) / left_high > 0.12):  # Sufficient depth
                
                # Look for handle formation
                handle_data = closes[-15:]
                handle_high = max(handle_data)
                handle_low = min(handle_data)
                
                # Handle should be smaller pullback
                if (right_high - handle_low) / right_high < 0.08:
                    entry = handle_high * 1.01
                    stop_loss = handle_low * 0.97
                    target = entry + (left_high - cup_bottom)
                    
                    return {
                        'type': 'cup_and_handle',
                        'confidence': self._calculate_cup_confidence(left_high, cup_bottom, right_high, handle_low),
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'target': target,
                        'cup_depth': (left_high - cup_bottom) / left_high
                    }
        except Exception as e:
            logger.error(f"Error in cup and handle detection: {e}")
        
        return None
    
    def _calculate_hs_confidence(self, left: float, head: float, right: float, neckline: float) -> float:
        """Calculate confidence score for H&S pattern"""
        # Symmetry of shoulders
        shoulder_symmetry = 1 - abs(left - right) / max(left, right)
        
        # Head prominence
        head_prominence = (head - max(left, right)) / head
        
        # Distance from neckline
        neckline_distance = (min(left, right) - neckline) / min(left, right)
        
        confidence = (shoulder_symmetry * 0.4 + head_prominence * 0.4 + neckline_distance * 0.2) * 100
        return min(100, max(0, confidence))
    
    def _calculate_triangle_confidence(self, peaks: np.ndarray, troughs: np.ndarray, 
                                     highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate confidence for triangle pattern"""
        # Convergence quality
        peak_trend = np.polyfit(peaks[-3:], highs[peaks[-3:]], 1)[0]
        trough_trend = np.polyfit(troughs[-3:], lows[troughs[-3:]], 1)[0]
        
        convergence = abs(peak_trend - trough_trend)
        confidence = min(100, convergence * 1000)  # Scale appropriately
        
        return max(50, confidence)  # Minimum 50% confidence
    
    def _calculate_flag_confidence(self, flagpole_move: float, consolidation_range: float, 
                                  volumes: np.ndarray) -> float:
        """Calculate confidence for flag pattern"""
        # Strong flagpole
        flagpole_score = min(100, flagpole_move * 500)  # Scale 8% move to 40 points
        
        # Tight consolidation
        consolidation_score = max(0, 100 - consolidation_range * 1000)
        
        # Volume pattern (declining during flag)
        volume_trend = np.polyfit(range(len(volumes[-10:])), volumes[-10:], 1)[0]
        volume_score = 100 if volume_trend < 0 else 50
        
        confidence = (flagpole_score * 0.4 + consolidation_score * 0.4 + volume_score * 0.2)
        return min(100, max(60, confidence))
    
    def _calculate_cup_confidence(self, left_high: float, bottom: float, 
                                 right_high: float, handle_low: float) -> float:
        """Calculate confidence for cup and handle"""
        # Rim symmetry
        rim_symmetry = 1 - abs(left_high - right_high) / max(left_high, right_high)
        
        # Cup depth appropriateness
        cup_depth = (left_high - bottom) / left_high
        depth_score = 1 if 0.12 <= cup_depth <= 0.35 else 0.5
        
        # Handle quality
        handle_depth = (right_high - handle_low) / right_high
        handle_score = 1 if handle_depth <= 0.08 else 0.5
        
        confidence = (rim_symmetry * 40 + depth_score * 30 + handle_score * 30)
        return min(100, max(65, confidence))
    
    def _calculate_trendline_slope(self, x_points: np.ndarray, y_points: np.ndarray) -> float:
        """Calculate slope of trendline"""
        if len(x_points) < 2:
            return 0
        return np.polyfit(x_points, y_points, 1)[0]
    
    def _classify_triangle(self, peak_slope: float, trough_slope: float) -> Optional[str]:
        """Classify triangle type based on slopes"""
        slope_threshold = 0.1
        
        if abs(peak_slope) < slope_threshold and trough_slope > slope_threshold:
            return 'ascending'
        elif peak_slope < -slope_threshold and abs(trough_slope) < slope_threshold:
            return 'descending'
        elif peak_slope < -slope_threshold and trough_slope > slope_threshold:
            return 'symmetrical'
        
        return None

class MovingAverageAnalyzer:
    """Moving average analysis and signals"""
    
    def __init__(self):
        self.short_ma = 20
        self.long_ma = 50
        self.signal_ma = 9
    
    def calculate_ma_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate moving average signals"""
        try:
            # Calculate different MAs
            df['ma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['ma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
            df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            
            # Current values
            current_price = df['close'].iloc[-1]
            ma_20 = df['ma_20'].iloc[-1]
            ma_50 = df['ma_50'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            
            # Generate signals
            # Generate signals
            signals = {
                'ma_trend': 'bullish' if ma_20 > ma_50 else 'bearish',
                'price_vs_ma20': 'above' if current_price > ma_20 else 'below',
                'price_vs_ma50': 'above' if current_price > ma_50 else 'below',
                'macd_signal': 'bullish' if macd > macd_signal else 'bearish',
                'ma_strength': self._calculate_ma_strength(df),
                'trend_quality': self._assess_trend_quality(df)
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating MA signals: {e}")
            return {}
    
    def _calculate_ma_strength(self, df: pd.DataFrame) -> float:
        """Calculate moving average trend strength"""
        try:
            ma_20_slope = (df['ma_20'].iloc[-1] - df['ma_20'].iloc[-5]) / df['ma_20'].iloc[-5]
            ma_50_slope = (df['ma_50'].iloc[-1] - df['ma_50'].iloc[-10]) / df['ma_50'].iloc[-10]
            
            # Normalize slopes
            strength = (abs(ma_20_slope) + abs(ma_50_slope)) * 100
            return min(100, strength)
        except:
            return 50
    
    def _assess_trend_quality(self, df: pd.DataFrame) -> str:
        """Assess overall trend quality"""
        try:
            recent_closes = df['close'].tail(10)
            recent_ma20 = df['ma_20'].tail(10)
            
            # Count bars above/below MA
            above_ma = sum(recent_closes > recent_ma20)
            
            if above_ma >= 8:
                return 'strong_bullish'
            elif above_ma >= 6:
                return 'weak_bullish'
            elif above_ma <= 2:
                return 'strong_bearish'
            else:
                return 'weak_bearish'
        except:
            return 'neutral'

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, max_portfolio_risk: float = 0.05):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = 0.02  # 2% per position
        self.max_positions = 10
        self.correlation_limit = 0.7
    
    def calculate_position_size(self, account_value: float, entry_price: float, 
                              stop_loss: float, pattern_confidence: float) -> int:
        """Calculate optimal position size using multiple factors"""
        try:
            # Base risk amount
            risk_per_trade = account_value * self.max_position_risk
            
            # Adjust for pattern confidence
            confidence_multiplier = pattern_confidence / 100
            adjusted_risk = risk_per_trade * confidence_multiplier
            
            # Calculate position size
            price_risk = abs(entry_price - stop_loss)
            if price_risk == 0:
                return 0
            
            shares = int(adjusted_risk / price_risk)
            
            # Ensure position doesn't exceed maximum
            max_shares_by_value = int(account_value * 0.1 / entry_price)  # Max 10% of portfolio
            
            return min(shares, max_shares_by_value)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def validate_trade(self, signal: PatternSignal, current_positions: List[Position], 
                      account_value: float) -> bool:
        """Validate if trade meets risk criteria"""
        try:
            # Check maximum positions
            if len(current_positions) >= self.max_positions:
                logger.info(f"Maximum positions reached: {len(current_positions)}")
                return False
            
            # Check if already have position in this symbol
            if any(pos.symbol == signal.symbol for pos in current_positions):
                logger.info(f"Already have position in {signal.symbol}")
                return False
            
            # Check portfolio risk
            total_risk = sum(abs(pos.entry_price - pos.stop_loss) * pos.quantity 
                           for pos in current_positions)
            portfolio_risk_pct = total_risk / account_value
            
            if portfolio_risk_pct > self.max_portfolio_risk:
                logger.info(f"Portfolio risk too high: {portfolio_risk_pct:.2%}")
                return False
            
            # Check pattern confidence threshold
            if signal.confidence < 60:
                logger.info(f"Pattern confidence too low: {signal.confidence}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False

class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = 'https://paper-api.alpaca.markets'):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.pattern_detector = PatternDetector()
        self.ma_analyzer = MovingAverageAnalyzer()
        self.risk_manager = RiskManager()
        
        # Trading universe - popular stocks with good liquidity
        self.universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ORCL', 'CRM', 'ADBE', 'PYPL', 'INTC', 'AMD', 'QCOM', 'AVGO',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'BMY', 'CVS',
            'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI'
        ]
        
        self.active_signals = []
        self.positions = []
        
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'day_trade_count': int(account.day_trade_count)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_market_data(self, symbol: str, timeframe: str = '1Day', limit: int = 100) -> pd.DataFrame:
        """Get historical market data"""
        try:
            # Use yfinance for reliable data
            ticker = yf.Ticker(symbol)
            
            if timeframe == '1Day':
                period = '6mo'
                interval = '1d'
            elif timeframe == '1Hour':
                period = '1mo'
                interval = '1h'
            else:
                period = '3mo'
                interval = '1d'
            
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return pd.DataFrame()
            
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            df = df.reset_index()
            
            return df.tail(limit)
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def scan_for_patterns(self) -> List[PatternSignal]:
        """Scan universe for chart patterns"""
        signals = []
        
        logger.info(f"Scanning {len(self.universe)} symbols for patterns...")
        
        for symbol in self.universe:
            try:
                # Get market data
                df = self.get_market_data(symbol)
                if df.empty or len(df) < 50:
                    continue
                
                # Skip if current price is too low (penny stocks)
                current_price = df['close'].iloc[-1]
                if current_price < 10:
                    continue
                
                # Get moving average signals
                ma_signals = self.ma_analyzer.calculate_ma_signals(df)
                
                # Only look for bullish patterns if MA trend is bullish
                if ma_signals.get('ma_trend') != 'bullish':
                    continue
                
                # Detect patterns
                patterns = [
                    self.pattern_detector.detect_head_and_shoulders(df),
                    self.pattern_detector.detect_triangle_pattern(df),
                    self.pattern_detector.detect_flag_pattern(df),
                    self.pattern_detector.detect_cup_and_handle(df)
                ]
                
                for pattern in patterns:
                    if pattern and pattern['confidence'] >= 60:
                        signal = PatternSignal(
                            symbol=symbol,
                            pattern_type=pattern['type'],
                            confidence=pattern['confidence'],
                            entry_price=pattern['entry'],
                            stop_loss=pattern['stop_loss'],
                            target_price=pattern['target'],
                            signal_time=datetime.now(),
                            timeframe='1Day'
                        )
                        signals.append(signal)
                        logger.info(f"Found {pattern['type']} in {symbol} - Confidence: {pattern['confidence']:.1f}%")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals[:20]  # Return top 20 signals
    
    def update_positions(self):
        """Update current positions from Alpaca"""
        try:
            alpaca_positions = self.api.list_positions()
            self.positions = []
            
            for pos in alpaca_positions:
                # Get current market data for stop loss management
                df = self.get_market_data(pos.symbol, limit=5)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    
                    position = Position(
                        symbol=pos.symbol,
                        quantity=int(pos.qty),
                        entry_price=float(pos.avg_cost_basis),
                        current_price=current_price,
                        stop_loss=float(pos.avg_cost_basis) * 0.95,  # Default 5% stop
                        target_price=float(pos.avg_cost_basis) * 1.15,  # Default 15% target
                        pattern_type='unknown',
                        entry_time=datetime.now()
                    )
                    self.positions.append(position)
                    
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def execute_trade(self, signal: PatternSignal) -> bool:
        """Execute trade based on signal"""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return False
            
            # Validate trade
            if not self.risk_manager.validate_trade(signal, self.positions, account_info['equity']):
                return False
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                account_info['equity'],
                signal.entry_price,
                signal.stop_loss,
                signal.confidence
            )
            
            if position_size <= 0:
                logger.info(f"Position size too small for {signal.symbol}")
                return False
            
            # Check if market is open
            clock = self.api.get_clock()
            if not clock.is_open:
                logger.info("Market is closed")
                return False
            
            # Place order
            order = self.api.submit_order(
                symbol=signal.symbol,
                qty=position_size,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"Placed order for {position_size} shares of {signal.symbol}")
            
            # Place stop loss order
            self.api.submit_order(
                symbol=signal.symbol,
                qty=position_size,
                side='sell',
                type='stop',
                stop_price=signal.stop_loss,
                time_in_force='gtc'
            )
            
            # Place profit target order
            self.api.submit_order(
                symbol=signal.symbol,
                qty=position_size,
                side='sell',
                type='limit',
                limit_price=signal.target_price,
                time_in_force='gtc'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
            return False
    
    def manage_positions(self):
        """Manage existing positions with trailing stops and profit taking"""
        for position in self.positions:
            try:
                # Get current price
                df = self.get_market_data(position.symbol, limit=10)
                if df.empty:
                    continue
                
                current_price = df['close'].iloc[-1]
                position.current_price = current_price
                
                # Calculate profit/loss
                pnl_pct = (current_price - position.entry_price) / position.entry_price
                
                # Trailing stop logic
                if pnl_pct > 0.10:  # If profit > 10%
                    # Move stop loss to break even
                    new_stop = position.entry_price * 1.02
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        logger.info(f"Updated trailing stop for {position.symbol} to ${new_stop:.2f}")
                
                elif pnl_pct > 0.20:  # If profit > 20%
                    # Trail stop closer
                    new_stop = current_price * 0.95
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        logger.info(f"Tightened trailing stop for {position.symbol} to ${new_stop:.2f}")
                
                # Check for exit signals
                if self._should_exit_position(position, df):
                    self._close_position(position)
                    
            except Exception as e:
                logger.error(f"Error managing position {position.symbol}: {e}")
    
    def _should_exit_position(self, position: Position, df: pd.DataFrame) -> bool:
        """Determine if position should be closed"""
        try:
            # Get MA signals for exit
            ma_signals = self.ma_analyzer.calculate_ma_signals(df)
            
            # Exit if trend turns bearish
            if ma_signals.get('ma_trend') == 'bearish':
                logger.info(f"Exiting {position.symbol} - trend turned bearish")
                return True
            
            # Exit if price falls below key MA
            if ma_signals.get('price_vs_ma20') == 'below':
                logger.info(f"Exiting {position.symbol} - price below MA20")
                return True
            
            # Time-based exit (hold max 30 days)
            days_held = (datetime.now() - position.entry_time).days
            if days_held > 30:
                logger.info(f"Exiting {position.symbol} - maximum hold period reached")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating exit for {position.symbol}: {e}")
            return False
    
    def _close_position(self, position: Position):
        """Close a position"""
        try:
            # Cancel any existing orders for this symbol
            open_orders = self.api.list_orders(status='open', symbols=position.symbol)
            for order in open_orders:
                self.api.cancel_order(order.id)
            
            # Submit market sell order
            self.api.submit_order(
                symbol=position.symbol,
                qty=abs(position.quantity),
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            pnl_pct = (position.current_price - position.entry_price) / position.entry_price
            logger.info(f"Closed position in {position.symbol} - P&L: {pnl_pct:.2%}")
            
        except Exception as e:
            logger.error(f"Error closing position {position.symbol}: {e}")
    
    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        logger.info("=== Starting Trading Cycle ===")
        
        try:
            # Update current positions
            self.update_positions()
            logger.info(f"Current positions: {len(self.positions)}")
            
            # Manage existing positions
            self.manage_positions()
            
            # Scan for new opportunities
            signals = self.scan_for_patterns()
            logger.info(f"Found {len(signals)} pattern signals")
            
            # Execute top signals
            executed_count = 0
            for signal in signals[:5]:  # Limit to top 5 signals
                if self.execute_trade(signal):
                    executed_count += 1
                    time.sleep(2)  # Brief pause between orders
            
            logger.info(f"Executed {executed_count} new trades")
            
            # Print portfolio summary
            self._print_portfolio_summary()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _print_portfolio_summary(self):
        """Print current portfolio status"""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return
            
            logger.info("=== Portfolio Summary ===")
            logger.info(f"Equity: ${account_info['equity']:,.2f}")
            logger.info(f"Cash: ${account_info['cash']:,.2f}")
            logger.info(f"Positions: {len(self.positions)}")
            
            total_pnl = 0
            for pos in self.positions:
                pnl = (pos.current_price - pos.entry_price) * pos.quantity
                total_pnl += pnl
                logger.info(f"{pos.symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f} "
                          f"Current: ${pos.current_price:.2f} P&L: ${pnl:.2f}")
            
            logger.info(f"Total Unrealized P&L: ${total_pnl:.2f}")
            logger.info("========================")
            
        except Exception as e:
            logger.error(f"Error printing portfolio summary: {e}")

def main():
    """Main execution function"""
    # Configuration - Replace with your Alpaca API credentials
    API_KEY = "YOUR_ALPACA_API_KEY"
    SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
    BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading for testing
    
    # Validate credentials
    if API_KEY == "YOUR_ALPACA_API_KEY" or SECRET_KEY == "YOUR_ALPACA_SECRET_KEY":
        logger.error("Please set your Alpaca API credentials")
        return
    
    # Initialize trading system
    try:
        trading_system = TradingSystem(API_KEY, SECRET_KEY, BASE_URL)
        logger.info("Trading system initialized successfully")
        
        # Test connection
        account_info = trading_system.get_account_info()
        if not account_info:
            logger.error("Failed to connect to Alpaca API")
            return
        
        logger.info(f"Connected successfully - Account Equity: ${account_info['equity']:,.2f}")
        
        # Single trading cycle for testing
        trading_system.run_trading_cycle()
        
        # For continuous trading, uncomment the following:
        # while True:
        #     try:
        #         # Check if market is open
        #         clock = trading_system.api.get_clock()
        #         if clock.is_open:
        #             trading_system.run_trading_cycle()
        #             time.sleep(300)  # Wait 5 minutes between cycles
        #         else:
        #             logger.info("Market closed - waiting...")
        #             time.sleep(3600)  # Wait 1 hour when market closed
        #     except KeyboardInterrupt:
        #         logger.info("Trading stopped by user")
        #         break
        #     except Exception as e:
        #         logger.error(f"Error in main loop: {e}")
        #         time.sleep(60)  # Wait 1 minute before retrying
        
    except Exception as e:
        logger.error(f"Error initializing trading system: {e}")

if __name__ == "__main__":
    main()