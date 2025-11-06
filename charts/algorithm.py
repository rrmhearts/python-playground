import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.signal import find_peaks
import talib
import time
import json
from sklearn.preprocessing import StandardScaler
from collections import deque

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
    volume_confirmed: bool = False
    time_confirmed: bool = False

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
    trailing_stop_activated: bool = False

@dataclass
class MarketRegime:
    """Market regime detection"""
    regime_type: str  # 'bull', 'bear', 'sideways'
    volatility: float
    trend_strength: float
    last_updated: datetime

class PatternDetector:
    """Advanced pattern detection with multiple algorithms"""
    
    def __init__(self):
        self.min_pattern_length = 20
        self.lookback_period = 100
        self.volume_threshold = 1.5  # Volume must be 1.5x average for confirmation
        self.breakout_confirmation_bars = 3  # Wait 3 bars for confirmation
    
    def detect_inverse_head_and_shoulders(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Inverse Head and Shoulders pattern (bullish reversal)"""
        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df['volume'].values
            
            # Find peaks and troughs
            peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
            troughs, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
            
            if len(troughs) < 3 or len(peaks) < 2:
                return None
            
            # Get last 3 troughs for potential inverse H&S
            recent_troughs = troughs[-3:]
            left_shoulder, head, right_shoulder = recent_troughs
            
            # Validate inverse H&S geometry
            left_low = lows[left_shoulder]
            head_low = lows[head]
            right_low = lows[right_shoulder]
            
            # Head should be lowest, shoulders roughly equal (bullish pattern)
            if (head_low < left_low and head_low < right_low and
                abs(left_low - right_low) / left_low < 0.05):
                
                # Find neckline (resistance level)
                neckline_left = highs[left_shoulder:head].max()
                neckline_right = highs[head:right_shoulder].max()
                neckline = min(neckline_left, neckline_right)
                
                # Calculate target (bullish)
                pattern_height = neckline - head_low
                target = neckline + pattern_height
                
                # Entry above neckline (bullish breakout)
                entry = neckline * 1.02
                stop_loss = head_low * 0.98
                
                # Volume confirmation
                volume_confirmed = self._validate_volume_breakout(df, len(df) - 1)
                
                return {
                    'type': 'inverse_head_and_shoulders',
                    'confidence': self._calculate_inverse_hs_confidence(left_low, head_low, right_low, neckline),
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'target': target,
                    'neckline': neckline,
                    'volume_confirmed': volume_confirmed
                }
        except Exception as e:
            logger.error(f"Error in inverse H&S detection: {e}")
        
        return None
    
    def detect_triangle_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect triangle patterns - only bullish breakouts for long-only system"""
        try:
            highs = df['high'].values[-50:]  # Last 50 bars
            lows = df['low'].values[-50:]
            closes = df['close'].values
            volumes = df['volume'].values
            
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
                
                # Only trade bullish breakouts for ascending and symmetrical triangles
                if triangle_type in ['ascending', 'symmetrical']:
                    entry = recent_high * 1.02
                    stop_loss = recent_low * 0.98
                    target = entry + (recent_high - recent_low) * 1.5
                    
                    # Volume and time confirmation
                    volume_confirmed = self._validate_volume_breakout(df, len(df) - 1)
                    time_confirmed = self._confirm_breakout_timing(df, entry)
                    
                    return {
                        'type': f'{triangle_type}_triangle',
                        'confidence': self._calculate_triangle_confidence(peaks, troughs, highs, lows),
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'target': target,
                        'pattern_range': recent_high - recent_low,
                        'volume_confirmed': volume_confirmed,
                        'time_confirmed': time_confirmed
                    }
                # Skip descending triangles as they're bearish
                
        except Exception as e:
            logger.error(f"Error in triangle detection: {e}")
        
        return None
    
    def detect_flag_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect bullish flag/pennant patterns"""
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            if len(closes) < 30:
                return None
            
            # Look for strong upward move (flagpole)
            recent_moves = []
            for i in range(10, 25):
                if len(closes) > i + 10:
                    move_pct = (closes[-i] - closes[-i-10]) / closes[-i-10]
                    recent_moves.append(move_pct)
            
            if not recent_moves:
                return None
                
            max_move = max(recent_moves)
            
            # Require significant upward move for flagpole
            if max_move < 0.08:  # 8% minimum move
                return None
            
            # Check for consolidation after move (flag)
            flagpole_end = recent_moves.index(max_move)
            consolidation_period = closes[-flagpole_end:]
            
            if len(consolidation_period) < 5:
                return None
            
            # Flag should be sideways with declining volume
            consolidation_range = (max(consolidation_period) - min(consolidation_period)) / max(consolidation_period)
            
            if consolidation_range < 0.06:  # Tight consolidation
                entry = max(consolidation_period) * 1.01
                stop_loss = min(consolidation_period) * 0.98
                target = entry + (entry - stop_loss) * 2  # 1:2 risk/reward
                
                # Enhanced volume confirmation
                volume_confirmed = self._validate_volume_breakout(df, len(df) - 1)
                time_confirmed = self._confirm_breakout_timing(df, entry)
                
                return {
                    'type': 'bull_flag',
                    'confidence': self._calculate_flag_confidence(max_move, consolidation_range, volumes),
                    'entry': entry,
                    'stop_loss': stop_loss,
                    'target': target,
                    'flagpole_move': max_move,
                    'volume_confirmed': volume_confirmed,
                    'time_confirmed': time_confirmed
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
            volumes = df['volume'].values
            
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
                    
                    # Volume and time confirmation
                    volume_confirmed = self._validate_volume_breakout(df, len(df) - 1)
                    time_confirmed = self._confirm_breakout_timing(df, entry)
                    
                    return {
                        'type': 'cup_and_handle',
                        'confidence': self._calculate_cup_confidence(left_high, cup_bottom, right_high, handle_low),
                        'entry': entry,
                        'stop_loss': stop_loss,
                        'target': target,
                        'cup_depth': (left_high - cup_bottom) / left_high,
                        'volume_confirmed': volume_confirmed,
                        'time_confirmed': time_confirmed
                    }
        except Exception as e:
            logger.error(f"Error in cup and handle detection: {e}")
        
        return None
    
    def _validate_volume_breakout(self, df: pd.DataFrame, breakout_bar: int = -1) -> bool:
        """Validate volume confirmation for breakouts"""
        try:
            if len(df) < 21:  # Need at least 21 bars for 20-day average
                return False
                
            current_volume = df['volume'].iloc[breakout_bar]
            avg_volume = df['volume'].tail(20).mean()
            
            # Volume should be at least 1.5x average
            return current_volume > (avg_volume * self.volume_threshold)
        except Exception as e:
            logger.error(f"Error validating volume breakout: {e}")
            return False
    
    def _confirm_breakout_timing(self, df: pd.DataFrame, entry_price: float) -> bool:
        """Confirm breakout with time-based validation"""
        try:
            if len(df) < self.breakout_confirmation_bars:
                return False
                
            # Check if recent closes are above entry price
            recent_closes = df['close'].tail(self.breakout_confirmation_bars)
            confirmations = sum(close >= entry_price * 0.99 for close in recent_closes)
            
            # Require at least 2 out of 3 confirmations
            return confirmations >= 2
        except Exception as e:
            logger.error(f"Error confirming breakout timing: {e}")
            return False
    
    def _calculate_inverse_hs_confidence(self, left: float, head: float, right: float, neckline: float) -> float:
        """Calculate confidence score for inverse H&S pattern"""
        try:
            # Symmetry of shoulders
            shoulder_symmetry = 1 - abs(left - right) / max(left, right)
            
            # Head depth (how much lower the head is)
            head_depth = (min(left, right) - head) / min(left, right)
            
            # Distance to neckline
            neckline_distance = (neckline - max(left, right)) / neckline
            
            confidence = (shoulder_symmetry * 0.4 + head_depth * 0.4 + neckline_distance * 0.2) * 100
            return min(100, max(50, confidence))
        except:
            return 50
    
    def _calculate_triangle_confidence(self, peaks: np.ndarray, troughs: np.ndarray, 
                                     highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate confidence for triangle pattern"""
        try:
            if len(peaks) < 3 or len(troughs) < 3:
                return 50
                
            # Convergence quality
            peak_trend = np.polyfit(peaks[-3:], highs[peaks[-3:]], 1)[0]
            trough_trend = np.polyfit(troughs[-3:], lows[troughs[-3:]], 1)[0]
            
            convergence = abs(peak_trend - trough_trend)
            confidence = min(100, convergence * 1000)  # Scale appropriately
            
            return max(50, confidence)  # Minimum 50% confidence
        except:
            return 50
    
    def _calculate_flag_confidence(self, flagpole_move: float, consolidation_range: float, 
                                  volumes: np.ndarray) -> float:
        """Calculate confidence for flag pattern"""
        try:
            # Strong flagpole
            flagpole_score = min(100, flagpole_move * 500)  # Scale 8% move to 40 points
            
            # Tight consolidation
            consolidation_score = max(0, 100 - consolidation_range * 1000)
            
            # Volume pattern (declining during flag)
            if len(volumes) >= 10:
                volume_trend = np.polyfit(range(len(volumes[-10:])), volumes[-10:], 1)[0]
                volume_score = 100 if volume_trend < 0 else 50
            else:
                volume_score = 50
            
            confidence = (flagpole_score * 0.4 + consolidation_score * 0.4 + volume_score * 0.2)
            return min(100, max(60, confidence))
        except:
            return 60
    
    def _calculate_cup_confidence(self, left_high: float, bottom: float, 
                                 right_high: float, handle_low: float) -> float:
        """Calculate confidence for cup and handle"""
        try:
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
        except:
            return 65
    
    def _calculate_trendline_slope(self, x_points: np.ndarray, y_points: np.ndarray) -> float:
        """Calculate slope of trendline"""
        try:
            if len(x_points) < 2:
                return 0
            return np.polyfit(x_points, y_points, 1)[0]
        except:
            return 0
    
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
    """Moving average analysis and signals with regime detection"""
    
    def __init__(self):
        self.short_ma = 20
        self.long_ma = 50
        self.signal_ma = 9
    
    def calculate_ma_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate moving average signals with enhanced analysis"""
        try:
            # Calculate different MAs
            df = df.copy()  # Avoid modifying original DataFrame
            df['ma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['ma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
            df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            
            # RSI for momentum
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # Bollinger Bands for volatility
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
            
            # Current values
            current_price = df['close'].iloc[-1]
            ma_20 = df['ma_20'].iloc[-1]
            ma_50 = df['ma_50'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            # Generate enhanced signals
            signals = {
                'ma_trend': 'bullish' if ma_20 > ma_50 else 'bearish',
                'price_vs_ma20': 'above' if current_price > ma_20 else 'below',
                'price_vs_ma50': 'above' if current_price > ma_50 else 'below',
                'macd_signal': 'bullish' if macd > macd_signal else 'bearish',
                'rsi_level': self._classify_rsi(rsi),
                'ma_strength': self._calculate_ma_strength(df),
                'trend_quality': self._assess_trend_quality(df),
                'volatility_regime': self._assess_volatility_regime(df),
                'momentum_alignment': self._check_momentum_alignment(df)
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating MA signals: {e}")
            return {}
    
    def _classify_rsi(self, rsi: float) -> str:
        """Classify RSI levels"""
        if rsi > 70:
            return 'overbought'
        elif rsi < 30:
            return 'oversold'
        elif rsi > 60:
            return 'strong_bullish'
        elif rsi < 40:
            return 'weak_bearish'
        else:
            return 'neutral'
    
    def _calculate_ma_strength(self, df: pd.DataFrame) -> float:
        """Calculate moving average trend strength"""
        try:
            ma_20_slope = (df['ma_20'].iloc[-1] - df['ma_20'].iloc[-5]) / df['ma_20'].iloc[-5]
            ma_50_slope = (df['ma_50'].iloc[-1] - df['ma_50'].iloc[-10]) / df['ma_50'].iloc[-10]
            
            # Normalize slopes and combine
            strength = (abs(ma_20_slope) + abs(ma_50_slope)) * 100
            return min(100, strength)
        except:
            return 50
    
    def _assess_trend_quality(self, df: pd.DataFrame) -> str:
        """Assess overall trend quality with multiple factors"""
        try:
            recent_closes = df['close'].tail(10)
            recent_ma20 = df['ma_20'].tail(10)
            
            # Count bars above/below MA
            above_ma = sum(recent_closes > recent_ma20)
            
            # Check MA slope consistency
            ma_slopes = []
            for i in range(1, 6):
                slope = (df['ma_20'].iloc[-i] - df['ma_20'].iloc[-i-1]) / df['ma_20'].iloc[-i-1]
                ma_slopes.append(slope)
            
            consistent_slope = len([s for s in ma_slopes if s > 0]) >= 3
            
            if above_ma >= 8 and consistent_slope:
                return 'strong_bullish'
            elif above_ma >= 6:
                return 'weak_bullish'
            elif above_ma <= 2 and not consistent_slope:
                return 'strong_bearish'
            elif above_ma <= 4:
                return 'weak_bearish'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _assess_volatility_regime(self, df: pd.DataFrame) -> str:
        """Assess current volatility regime"""
        try:
            # Calculate 20-day volatility
            returns = df['close'].pct_change().tail(20)
            current_vol = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate historical volatility percentile
            historical_returns = df['close'].pct_change().tail(100)
            historical_vol = historical_returns.rolling(20).std() * np.sqrt(252)
            vol_percentile = (current_vol > historical_vol).mean()
            
            if vol_percentile > 0.8:
                return 'high_volatility'
            elif vol_percentile < 0.2:
                return 'low_volatility'
            else:
                return 'normal_volatility'
        except:
            return 'normal_volatility'
    
    def _check_momentum_alignment(self, df: pd.DataFrame) -> bool:
        """Check if multiple momentum indicators are aligned"""
        try:
            # MACD alignment
            macd_bullish = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
            
            # RSI not overbought
            rsi_ok = df['rsi'].iloc[-1] < 75
            
            # Price above key MAs
            price_above_ma = df['close'].iloc[-1] > df['ma_20'].iloc[-1]
            
            # MA trend is bullish
            ma_bullish = df['ma_20'].iloc[-1] > df['ma_50'].iloc[-1]
            
            return sum([macd_bullish, rsi_ok, price_above_ma, ma_bullish]) >= 3
        except:
            return False

class RiskManager:
    """Advanced risk management system with correlation checking"""
    
    def __init__(self, max_portfolio_risk: float = 0.05):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = 0.02  # 2% per position
        self.max_positions = 10
        self.correlation_limit = 0.7
        self.max_sector_exposure = 0.3  # Max 30% in any sector
        self.daily_loss_limit = 0.03  # Max 3% daily loss
        
        # Sector mapping for correlation checking
        self.sector_map = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'META': 'Technology',
            'NVDA': 'Technology', 'ORCL': 'Technology', 'CRM': 'Technology', 'ADBE': 'Technology',
            'PYPL': 'Technology', 'INTC': 'Technology', 'AMD': 'Technology', 'QCOM': 'Technology',
            'AVGO': 'Technology', 'NFLX': 'Technology',
            
            # Financial
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
            'MS': 'Financial', 'C': 'Financial', 'V': 'Financial', 'MA': 'Financial', 'AXP': 'Financial',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'TMO': 'Healthcare', 'DHR': 'Healthcare', 'BMY': 'Healthcare', 'CVS': 'Healthcare',
            
            # Other
            'AMZN': 'Consumer', 'TSLA': 'Automotive',
            
            # ETFs
            'SPY': 'Market', 'QQQ': 'Technology', 'IWM': 'Market', 'XLF': 'Financial',
            'XLK': 'Technology', 'XLE': 'Energy', 'XLV': 'Healthcare', 'XLI': 'Industrial'
        }
    
    def calculate_position_size(self, account_value: float, entry_price: float, 
                              stop_loss: float, pattern_confidence: float,
                              volatility_adjustment: float = 1.0) -> int:
        """Calculate optimal position size using multiple factors"""
        try:
            # Base risk amount
            risk_per_trade = account_value * self.max_position_risk
            
            # Adjust for pattern confidence
            confidence_multiplier = pattern_confidence / 100
            
            # Adjust for volatility
            volatility_multiplier = min(1.0, 1.0 / volatility_adjustment)
            
            # Combined adjustments
            adjusted_risk = risk_per_trade * confidence_multiplier * volatility_multiplier
            
            # Calculate position size
            price_risk = abs(entry_price - stop_loss)
            if price_risk == 0:
                return 0
            
            shares = int(adjusted_risk / price_risk)
            
            # Ensure position doesn't exceed maximums
            max_shares_by_value = int(account_value * 0.1 / entry_price)  # Max 10% of portfolio
            max_shares_by_risk = int(account_value * 0.05 / entry_price)   # Max 5% position size
            
            return min(shares, max_shares_by_value, max_shares_by_risk)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def validate_trade(self, signal: PatternSignal, current_positions: List[Position], 
                      account_value: float) -> bool:
        """Enhanced trade validation with correlation and sector checks"""
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
            
            # Check volume confirmation for better patterns
            if signal.confidence > 80 and not signal.volume_confirmed:
                logger.info(f"High confidence pattern {signal.symbol} lacks volume confirmation")
                return False
            
            # Check sector concentration
            if not self._validate_sector_exposure(signal.symbol, current_positions, account_value):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False
    
    def _validate_sector_exposure(self, new_symbol: str, current_positions: List[Position], 
                                 account_value: float) -> bool:
        """Validate sector exposure limits"""
        try:
            new_sector = self.sector_map.get(new_symbol, 'Other')
            
            # Calculate current sector exposure
            sector_exposure = {}
            for pos in current_positions:
                sector = self.sector_map.get(pos.symbol, 'Other')
                position_value = pos.current_price * pos.quantity
                sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value
            
            # Check if adding new position would exceed sector limit
            current_sector_exposure = sector_exposure.get(new_sector, 0)
            sector_exposure_pct = current_sector_exposure / account_value
            
            if sector_exposure_pct > self.max_sector_exposure:
                logger.info(f"Sector exposure limit exceeded for {new_sector}: {sector_exposure_pct:.2%}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating sector exposure: {e}")
            return True  # Allow trade if validation fails

class MarketRegimeDetector:
    """Detect market regimes for adaptive strategy selection"""
    
    def __init__(self):
        self.lookback_period = 50
        self.volatility_window = 20
    
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        try:
            closes = df['close'].values
            
            if len(closes) < self.lookback_period:
                return MarketRegime('neutral', 0.5, 0.5, datetime.now())
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(df)
            
            # Calculate volatility
            returns = pd.Series(closes).pct_change().tail(self.volatility_window)
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Determine regime
            if trend_strength > 0.6:
                regime_type = 'bull'
            elif trend_strength < -0.6:
                regime_type = 'bear'
            else:
                regime_type = 'sideways'
            
            return MarketRegime(
                regime_type=regime_type,
                volatility=volatility,
                trend_strength=abs(trend_strength),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime('neutral', 0.5, 0.5, datetime.now())
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            closes = df['close'].values
            
            # Price trend (20-day vs 50-day MA)
            ma_20 = np.mean(closes[-20:])
            ma_50 = np.mean(closes[-50:])
            ma_trend = (ma_20 - ma_50) / ma_50
            
            # Momentum trend (recent vs older prices)
            recent_avg = np.mean(closes[-10:])
            older_avg = np.mean(closes[-30:-20])
            momentum_trend = (recent_avg - older_avg) / older_avg
            
            # Slope trend
            x = np.arange(len(closes[-20:]))
            slope = np.polyfit(x, closes[-20:], 1)[0]
            slope_trend = slope / np.mean(closes[-20:])
            
            # Combined trend strength
            trend_strength = (ma_trend * 0.4 + momentum_trend * 0.4 + slope_trend * 0.2)
            
            return np.clip(trend_strength, -1, 1)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0

class TradingSystem:
    """Enhanced trading system with regime detection and improved risk management"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = 'https://paper-api.alpaca.markets'):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.pattern_detector = PatternDetector()
        self.ma_analyzer = MovingAverageAnalyzer()
        self.risk_manager = RiskManager()
        self.regime_detector = MarketRegimeDetector()
        
        # Enhanced trading universe with sector diversification
        self.universe = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ORCL', 'CRM', 'ADBE',
            
            # Semiconductors
            'INTC', 'AMD', 'QCOM', 'AVGO',
            
            # Payments
            'PYPL', 'V', 'MA',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'BMY', 'CVS',
            
            # Other
            'TSLA',
            
            # ETFs for diversification
            'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI'
        ]
        
        self.active_signals = []
        self.positions = []
        self.daily_pnl_tracker = deque(maxlen=30)  # Track 30 days of P&L
        self.current_regime = None
        
        # Transaction cost modeling
        self.commission_per_share = 0.005  # $0.005 per share
        self.spread_cost_pct = 0.001  # 0.1% spread cost
    
    def get_account_info(self) -> Dict:
        """Get account information with enhanced error handling"""
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'day_trade_count': int(account.day_trade_count),
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_market_data(self, symbol: str, timeframe: str = '1Day', limit: int = 100) -> pd.DataFrame:
        """Get historical market data with improved error handling"""
        try:
            # Use yfinance for reliable data
            ticker = yf.Ticker(symbol)
            
            if timeframe == '1Day':
                period = '6mo'
                interval = '1d'
            elif timeframe == '1Hour':
                period = '1mo'
                interval = '1h'
            elif timeframe == '15Min':
                period = '5d'
                interval = '15m'
            else:
                period = '3mo'
                interval = '1d'
            
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Rename columns to lowercase and clean data
            df.columns = df.columns.str.lower()
            df = df.reset_index()
            
            # Remove any invalid data
            df = df.dropna()
            df = df[df['volume'] > 0]  # Remove zero volume bars
            
            return df.tail(limit)
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def scan_for_patterns(self) -> List[PatternSignal]:
        """Enhanced pattern scanning with regime awareness"""
        signals = []
        
        logger.info(f"Scanning {len(self.universe)} symbols for patterns...")
        
        # Get market regime from SPY
        spy_data = self.get_market_data('SPY')
        if not spy_data.empty:
            self.current_regime = self.regime_detector.detect_regime(spy_data)
            logger.info(f"Current market regime: {self.current_regime.regime_type} "
                       f"(Volatility: {self.current_regime.volatility:.2f})")
        
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
                
                # Skip if volume is too low
                avg_volume = df['volume'].tail(20).mean()
                if avg_volume < 100000:  # Minimum 100K average volume
                    continue
                
                # Get moving average signals
                ma_signals = self.ma_analyzer.calculate_ma_signals(df)
                
                # Enhanced filtering based on regime and MA signals
                if not self._should_scan_symbol(ma_signals):
                    continue
                
                # Detect patterns - now includes inverse H&S
                patterns = [
                    self.pattern_detector.detect_inverse_head_and_shoulders(df),
                    self.pattern_detector.detect_triangle_pattern(df),
                    self.pattern_detector.detect_flag_pattern(df),
                    self.pattern_detector.detect_cup_and_handle(df)
                ]
                
                for pattern in patterns:
                    if pattern and self._validate_pattern_quality(pattern, ma_signals):
                        signal = PatternSignal(
                            symbol=symbol,
                            pattern_type=pattern['type'],
                            confidence=pattern['confidence'],
                            entry_price=pattern['entry'],
                            stop_loss=pattern['stop_loss'],
                            target_price=pattern['target'],
                            signal_time=datetime.now(),
                            timeframe='1Day',
                            volume_confirmed=pattern.get('volume_confirmed', False),
                            time_confirmed=pattern.get('time_confirmed', False)
                        )
                        signals.append(signal)
                        logger.info(f"Found {pattern['type']} in {symbol} - "
                                   f"Confidence: {pattern['confidence']:.1f}% "
                                   f"Vol: {signal.volume_confirmed} Time: {signal.time_confirmed}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by confidence and quality
        signals.sort(key=lambda x: (x.confidence, x.volume_confirmed, x.time_confirmed), reverse=True)
        return signals[:15]  # Return top 15 signals
    
    def _should_scan_symbol(self, ma_signals: Dict) -> bool:
        """Determine if symbol should be scanned based on MA signals and regime"""
        try:
            # Only trade bullish setups in bull/neutral markets
            if ma_signals.get('ma_trend') != 'bullish':
                return False
            
            # Require momentum alignment for better quality
            if not ma_signals.get('momentum_alignment', False):
                return False
            
            # In high volatility regimes, be more selective
            if (self.current_regime and 
                self.current_regime.volatility > 0.3 and 
                ma_signals.get('trend_quality') not in ['strong_bullish']):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating symbol scan criteria: {e}")
            return True
    
    def _validate_pattern_quality(self, pattern: Dict, ma_signals: Dict) -> bool:
        """Enhanced pattern quality validation"""
        try:
            # Minimum confidence threshold
            if pattern['confidence'] < 60:
                return False
            
            # For high-confidence patterns, require volume confirmation
            if pattern['confidence'] > 80 and not pattern.get('volume_confirmed', False):
                return False
            
            # Risk/reward ratio check
            entry = pattern['entry']
            stop = pattern['stop_loss']
            target = pattern['target']
            
            if entry <= stop:  # Invalid risk setup
                return False
            
            risk_reward_ratio = (target - entry) / (entry - stop)
            if risk_reward_ratio < 1.5:  # Minimum 1.5:1 reward/risk
                return False
            
            # Pattern-specific validations
            if pattern['type'] == 'bull_flag':
                # Flags should have strong momentum
                if ma_signals.get('rsi_level') == 'overbought':
                    return False
            
            elif pattern['type'] in ['ascending_triangle', 'symmetrical_triangle']:
                # Triangles should have time confirmation
                if not pattern.get('time_confirmed', False):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating pattern quality: {e}")
            return False
    
    def update_positions(self):
        """Enhanced position tracking with better error handling"""
        try:
            alpaca_positions = self.api.list_positions()
            updated_positions = []
            
            for pos in alpaca_positions:
                try:
                    # Get current market data for position management
                    df = self.get_market_data(pos.symbol, limit=10)
                    if not df.empty:
                        current_price = df['close'].iloc[-1]
                        
                        # Find existing position or create new one
                        existing_pos = next((p for p in self.positions if p.symbol == pos.symbol), None)
                        
                        if existing_pos:
                            # Update existing position
                            existing_pos.current_price = current_price
                            existing_pos.quantity = int(pos.qty)
                            updated_positions.append(existing_pos)
                        else:
                            # Create new position tracking
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
                            updated_positions.append(position)
                    
                except Exception as e:
                    logger.error(f"Error updating position {pos.symbol}: {e}")
                    continue
            
            self.positions = updated_positions
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def execute_trade(self, signal: PatternSignal) -> bool:
        """Enhanced trade execution with transaction cost consideration"""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return False
            
            # Enhanced trade validation
            if not self.risk_manager.validate_trade(signal, self.positions, account_info['equity']):
                return False
            
            # Calculate volatility adjustment
            volatility_adj = 1.0
            if self.current_regime:
                volatility_adj = min(2.0, max(0.5, self.current_regime.volatility / 0.2))
            
            # Calculate position size with volatility adjustment
            position_size = self.risk_manager.calculate_position_size(
                account_info['equity'],
                signal.entry_price,
                signal.stop_loss,
                signal.confidence,
                volatility_adj
            )
            
            if position_size <= 0:
                logger.info(f"Position size too small for {signal.symbol}")
                return False
            
            # Calculate estimated transaction costs
            estimated_cost = self._calculate_transaction_costs(signal.entry_price, position_size)
            logger.info(f"Estimated transaction costs for {signal.symbol}: ${estimated_cost:.2f}")
            
            # Check if market is open
            clock = self.api.get_clock()
            if not clock.is_open:
                logger.info("Market is closed")
                return False
            
            # Place primary order with limit price slightly above market for better fills
            entry_limit = signal.entry_price * 1.001  # 0.1% above signal price
            
            order = self.api.submit_order(
                symbol=signal.symbol,
                qty=position_size,
                side='buy',
                type='limit',
                limit_price=round(entry_limit, 2),
                time_in_force='day',
                extended_hours=False
            )
            
            logger.info(f"Placed limit order for {position_size} shares of {signal.symbol} "
                       f"at ${entry_limit:.2f}")
            
            # Store order info for tracking
            signal.order_id = order.id
            self.active_signals.append(signal)
            
            # Place contingent stop loss order (will be activated after fill)
            try:
                time.sleep(1)  # Brief delay
                stop_order = self.api.submit_order(
                    symbol=signal.symbol,
                    qty=position_size,
                    side='sell',
                    type='stop',
                    stop_price=round(signal.stop_loss, 2),
                    time_in_force='gtc'
                )
                logger.info(f"Placed stop loss at ${signal.stop_loss:.2f}")
                
            except Exception as e:
                logger.warning(f"Could not place stop loss for {signal.symbol}: {e}")
            
            # Place profit target order
            try:
                time.sleep(1)  # Brief delay
                target_order = self.api.submit_order(
                    symbol=signal.symbol,
                    qty=position_size,
                    side='sell',
                    type='limit',
                    limit_price=round(signal.target_price, 2),
                    time_in_force='gtc'
                )
                logger.info(f"Placed profit target at ${signal.target_price:.2f}")
                
            except Exception as e:
                logger.warning(f"Could not place profit target for {signal.symbol}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
            return False
    
    def _calculate_transaction_costs(self, price: float, quantity: int) -> float:
        """Calculate estimated transaction costs"""
        try:
            commission = quantity * self.commission_per_share
            spread_cost = price * quantity * self.spread_cost_pct
            return commission + spread_cost
        except:
            return 0
    
    def manage_positions(self):
        """Enhanced position management with improved trailing stops"""
        for position in self.positions:
            try:
                # Get current price and technical data
                df = self.get_market_data(position.symbol, limit=20)
                if df.empty:
                    continue
                
                current_price = df['close'].iloc[-1]
                position.current_price = current_price
                
                # Calculate profit/loss
                pnl_pct = (current_price - position.entry_price) / position.entry_price
                pnl_dollars = pnl_pct * position.entry_price * position.quantity
                
                # Enhanced trailing stop logic - FIXED
                if pnl_pct > 0.10:  # If profit > 10%
                    # Move stop loss to break even + small profit
                    new_stop = position.entry_price * 1.03  # 3% profit lock-in
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        self._update_stop_loss_order(position)
                        logger.info(f"Updated trailing stop for {position.symbol} to ${new_stop:.2f}")
                
                if pnl_pct > 0.20:  # If profit > 20% - separate condition
                    # Trail stop closer to current price
                    new_stop = current_price * 0.92  # 8% below current price
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        position.trailing_stop_activated = True
                        self._update_stop_loss_order(position)
                        logger.info(f"Activated tight trailing stop for {position.symbol} to ${new_stop:.2f}")
                
                if pnl_pct > 0.30:  # If profit > 30%
                    # Very tight trailing stop
                    new_stop = current_price * 0.95  # 5% below current price
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        self._update_stop_loss_order(position)
                        logger.info(f"Tightened trailing stop for {position.symbol} to ${new_stop:.2f}")
                
                # Check for technical exit signals
                if self._should_exit_position(position, df):
                    self._close_position(position)
                    
                # Log position status
                if pnl_pct > 0.05 or pnl_pct < -0.03:  # Log significant moves
                    logger.info(f"{position.symbol}: P&L {pnl_pct:.2%} (${pnl_dollars:.2f}) "
                               f"Stop: ${position.stop_loss:.2f}")
                    
            except Exception as e:
                logger.error(f"Error managing position {position.symbol}: {e}")
    
    def _update_stop_loss_order(self, position: Position):
        """Update stop loss order in the market"""
        try:
            # Cancel existing stop orders for this symbol
            open_orders = self.api.list_orders(status='open', symbols=position.symbol)
            for order in open_orders:
                if order.side == 'sell' and order.order_type == 'stop':
                    self.api.cancel_order(order.id)
                    time.sleep(0.5)
            
            # Place new stop order
            self.api.submit_order(
                symbol=position.symbol,
                qty=abs(position.quantity),
                side='sell',
                type='stop',
                stop_price=round(position.stop_loss, 2),
                time_in_force='gtc'
            )
            
        except Exception as e:
            logger.error(f"Error updating stop loss for {position.symbol}: {e}")
    
    def _should_exit_position(self, position: Position, df: pd.DataFrame) -> bool:
        """Enhanced exit signal detection"""
        try:
            # Get technical signals for exit
            ma_signals = self.ma_analyzer.calculate_ma_signals(df)
            
            # Exit if trend turns bearish
            if ma_signals.get('ma_trend') == 'bearish':
                logger.info(f"Exiting {position.symbol} - trend turned bearish")
                return True
            
            # Exit if price falls below key MA and momentum is weak
            if (ma_signals.get('price_vs_ma20') == 'below' and 
                ma_signals.get('macd_signal') == 'bearish'):
                logger.info(f"Exiting {position.symbol} - price below MA20 with weak momentum")
                return True
            
            # Exit if RSI becomes extremely overbought (profit taking)
            if ma_signals.get('rsi_level') == 'overbought':
                pnl_pct = (position.current_price - position.entry_price) / position.entry_price
                if pnl_pct > 0.15:  # Only if profitable
                    logger.info(f"Exiting {position.symbol} - taking profits on overbought RSI")
                    return True
            
            # Time-based exit (hold max 30 days)
            days_held = (datetime.now() - position.entry_time).days
            if days_held > 30:
                logger.info(f"Exiting {position.symbol} - maximum hold period reached")
                return True
            
            # Market regime change exit
            if (self.current_regime and 
                self.current_regime.regime_type == 'bear' and 
                ma_signals.get('trend_quality') in ['strong_bearish', 'weak_bearish']):
                logger.info(f"Exiting {position.symbol} - market regime turned bearish")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating exit for {position.symbol}: {e}")
            return False
    
    def _close_position(self, position: Position):
        """Enhanced position closing with better order management"""
        try:
            # Cancel all existing orders for this symbol
            open_orders = self.api.list_orders(status='open', symbols=position.symbol)
            for order in open_orders:
                try:
                    self.api.cancel_order(order.id)
                    time.sleep(0.2)
                except:
                    pass  # Order might already be filled/cancelled
            
            # Submit market sell order
            close_order = self.api.submit_order(
                symbol=position.symbol,
                qty=abs(position.quantity),
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            pnl_pct = (position.current_price - position.entry_price) / position.entry_price
            pnl_dollars = pnl_pct * position.entry_price * position.quantity
            
            logger.info(f"Closed position in {position.symbol} - "
                       f"P&L: {pnl_pct:.2%} (${pnl_dollars:.2f})")
            
            # Track daily P&L
            self.daily_pnl_tracker.append(pnl_dollars)
            
        except Exception as e:
            logger.error(f"Error closing position {position.symbol}: {e}")
    
    def run_trading_cycle(self):
        """Enhanced trading cycle with regime awareness"""
        logger.info("=== Starting Enhanced Trading Cycle ===")
        
        try:
            # Check daily loss limits
            if not self._check_daily_limits():
                logger.warning("Daily loss limit reached - halting trading")
                return
            
            # Update current positions
            self.update_positions()
            logger.info(f"Current positions: {len(self.positions)}")
            
            # Manage existing positions first
            self.manage_positions()
            
            # Clean up filled signals
            self._cleanup_filled_signals()
            
            # Scan for new opportunities
            signals = self.scan_for_patterns()
            logger.info(f"Found {len(signals)} pattern signals")
            
            if self.current_regime:
                logger.info(f"Market regime: {self.current_regime.regime_type} "
                           f"(Volatility: {self.current_regime.volatility:.2f})")
            
            # Execute top signals (fewer in high volatility)
            max_new_trades = 3 if (self.current_regime and self.current_regime.volatility > 0.3) else 5
            
            executed_count = 0
            for signal in signals[:max_new_trades]:
                if self.execute_trade(signal):
                    executed_count += 1
                    time.sleep(3)  # Longer pause between orders
            
            logger.info(f"Executed {executed_count} new trades")
            
            # Print enhanced portfolio summary
            self._print_enhanced_portfolio_summary()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _check_daily_limits(self) -> bool:
        """Check if daily loss limits are exceeded"""
        try:
            if len(self.daily_pnl_tracker) > 0:
                daily_pnl = sum(self.daily_pnl_tracker)
                account_info = self.get_account_info()
                
                if account_info:
                    daily_loss_pct = abs(daily_pnl) / account_info['equity']
                    if daily_pnl < 0 and daily_loss_pct > self.risk_manager.daily_loss_limit:
                        return False
            
            return True
        except:
            return True
    
    def _cleanup_filled_signals(self):
        """Clean up signals for filled orders"""
        try:
            remaining_signals = []
            for signal in self.active_signals:
                try:
                    order = self.api.get_order(signal.order_id)
                    if order.status not in ['filled', 'canceled', 'rejected']:
                        remaining_signals.append(signal)
                except:
                    pass  # Order not found, probably filled
            
            self.active_signals = remaining_signals
        except Exception as e:
            logger.error(f"Error cleaning up signals: {e}")
    
    def _print_enhanced_portfolio_summary(self):
        """Enhanced portfolio summary with regime and risk metrics"""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return
            
            logger.info("=== Enhanced Portfolio Summary ===")
            logger.info(f"Equity: ${account_info['equity']:,.2f}")
            logger.info(f"Cash: ${account_info['cash']:,.2f}")
            logger.info(f"Buying Power: ${account_info['buying_power']:,.2f}")
            logger.info(f"Positions: {len(self.positions)}")
            
            if self.current_regime:
                logger.info(f"Market Regime: {self.current_regime.regime_type.upper()} "
                           f"(Vol: {self.current_regime.volatility:.2f})")
            
            # Position details
            total_pnl = 0
            total_position_value = 0
            
            for pos in self.positions:
                pnl = (pos.current_price - pos.entry_price) * pos.quantity
                position_value = pos.current_price * pos.quantity
                total_pnl += pnl
                total_position_value += position_value
                
                logger.info(f"{pos.symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f} "
                           f"Current: ${pos.current_price:.2f} P&L: ${pnl:.2f} "
                           f"({(pnl/position_value)*100:.1f}%)")
            
            # Portfolio metrics
            portfolio_utilization = total_position_value / account_info['equity']
            
            logger.info(f"Total Unrealized P&L: ${total_pnl:.2f}")
            logger.info(f"Portfolio Utilization: {portfolio_utilization:.1%}")
            logger.info(f"Active Signals: {len(self.active_signals)}")
            
            # Daily P&L tracking
            if self.daily_pnl_tracker:
                daily_pnl = sum(self.daily_pnl_tracker)
                logger.info(f"Today's Realized P&L: ${daily_pnl:.2f}")
            
            logger.info("=====================================")
            
        except Exception as e:
            logger.error(f"Error printing portfolio summary: {e}")

def main():
    """Enhanced main execution function"""
    # Configuration - Replace with your Alpaca API credentials
    API_KEY = "YOUR_ALPACA_API_KEY"
    SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
    BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading for testing
    
    # Validate credentials
    if API_KEY == "YOUR_ALPACA_API_KEY" or SECRET_KEY == "YOUR_ALPACA_SECRET_KEY":
        logger.error("Please set your Alpaca API credentials")
        logger.info("To get API credentials:")
        logger.info("1. Sign up at https://alpaca.markets/")
        logger.info("2. Go to Account -> API Keys")
        logger.info("3. Generate new API key pair")
        logger.info("4. Replace the placeholder values in this script")
        return
    
    # Initialize trading system
    try:
        trading_system = TradingSystem(API_KEY, SECRET_KEY, BASE_URL)
        logger.info("Enhanced trading system initialized successfully")
        
        # Test connection
        account_info = trading_system.get_account_info()
        if not account_info:
            logger.error("Failed to connect to Alpaca API")
            return
        
        logger.info(f"Connected successfully - Account Equity: ${account_info['equity']:,.2f}")
        
        if account_info.get('pattern_day_trader'):
            logger.info("Account is marked as Pattern Day Trader")
        else:
            logger.info(f"Day trades used: {account_info['day_trade_count']}/3")
        
        # Run single trading cycle for testing
        logger.info("Running single trading cycle for testing...")
        trading_system.run_trading_cycle()
        
        # Ask user if they want to run continuous trading
        user_input = input("\nWould you like to start continuous trading? (y/n): ")
        
        if user_input.lower() in ['y', 'yes']:
            logger.info("Starting continuous trading mode...")
            logger.info("Press Ctrl+C to stop trading")
            
            # Continuous trading loop
            cycle_count = 0
            while True:
                try:
                    # Check if market is open
                    clock = trading_system.api.get_clock()
                    
                    if clock.is_open:
                        cycle_count += 1
                        logger.info(f"=== Trading Cycle #{cycle_count} ===")
                        
                        trading_system.run_trading_cycle()
                        
                        # Wait between cycles (5 minutes during market hours)
                        logger.info("Waiting 5 minutes before next cycle...")
                        time.sleep(300)
                        
                    else:
                        # Market is closed
                        next_open = clock.next_open.astimezone()
                        logger.info(f"Market closed - next open: {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                        
                        # Check every hour when market is closed
                        time.sleep(3600)
                        
                except KeyboardInterrupt:
                    logger.info("Trading stopped by user")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in main trading loop: {e}")
                    logger.info("Waiting 60 seconds before retrying...")
                    time.sleep(60)
        
        else:
            logger.info("Single cycle completed. Exiting...")
        
    except Exception as e:
        logger.error(f"Error initializing trading system: {e}")
        logger.error("Please check your API credentials and internet connection")

if __name__ == "__main__":
    main()