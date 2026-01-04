"""
Chart generation module for Stocker App
Creates interactive and static charts for stock visualization
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Set default style - will be updated based on theme
plt.style.use('seaborn-v0_8-whitegrid')


class ChartGenerator:
    """Generates stock charts with technical indicators"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.dpi = 100
    
    def create_candlestick_chart(self, history_data: dict, indicators: dict = None) -> Figure:
        """Create a candlestick chart with volume and indicators"""
        try:
            data = history_data.get('data', [])
            if not data:
                raise ValueError("No data available for charting")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, dpi=self.dpi,
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot candlesticks
            self._plot_candlesticks(ax1, df)
            
            # Add indicators if provided
            if indicators:
                self._add_indicators(ax1, df, indicators)
            
            # Plot volume
            self._plot_volume(ax2, df)
            
            # Format axes
            ax1.set_title(f"{history_data.get('symbol', 'Stock')} - Price Chart", fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.set_xlabel('Date', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            # Return empty figure on error
            fig = Figure(figsize=self.fig_size, dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error creating chart: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """Plot candlestick chart"""
        # Plot price line
        ax.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='blue')
        
        # Plot high/low range
        for i in range(len(df)):
            date = df.index[i]
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            open_price = df['open'].iloc[i]
            close = df['close'].iloc[i]
            
            # Color based on up/down
            color = 'green' if close >= open_price else 'red'
            
            # Draw high-low line
            ax.plot([date, date], [low, high], color='black', linewidth=0.5, alpha=0.5)
            
            # Draw open-close box
            box_height = abs(close - open_price)
            box_bottom = min(open_price, close)
            ax.bar(date, box_height, bottom=box_bottom, width=0.8, 
                  color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    def _add_indicators(self, ax, df: pd.DataFrame, indicators: dict):
        """Add technical indicators to chart"""
        # Moving averages
        if 'ema_20' in indicators:
            # Calculate EMA if not in data
            ema_20 = df['close'].ewm(span=20, adjust=False).mean()
            ax.plot(df.index, ema_20, label='EMA 20', color='orange', linewidth=1, alpha=0.7)
        
        if 'ema_50' in indicators:
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()
            ax.plot(df.index, ema_50, label='EMA 50', color='purple', linewidth=1, alpha=0.7)
        
        # Bollinger Bands
        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            # Calculate BB if not in data
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)
            
            ax.fill_between(df.index, bb_upper, bb_lower, alpha=0.2, color='gray', label='Bollinger Bands')
            ax.plot(df.index, sma_20, label='SMA 20', color='gray', linewidth=1, alpha=0.5)
    
    def _plot_volume(self, ax, df: pd.DataFrame):
        """Plot volume bars"""
        colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] 
                 else 'red' for i in range(len(df))]
        
        ax.bar(df.index, df['volume'], color=colors, alpha=0.6, width=0.8)
        ax.set_ylabel('Volume', fontsize=9)
    
    def create_indicator_chart(self, history_data: dict, indicators: dict) -> Figure:
        """Create a chart showing technical indicators"""
        try:
            data = history_data.get('data', [])
            if not data:
                raise ValueError("No data available")
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=self.dpi)
            
            # RSI
            if 'rsi' in indicators:
                rsi_values = df['close'].rolling(14).apply(
                    lambda x: 100 - (100 / (1 + (x.diff().fillna(0).clip(lower=0).sum() / 
                                                 abs(x.diff().fillna(0).clip(upper=0).sum()) if 
                                                 abs(x.diff().fillna(0).clip(upper=0).sum()) > 0 else 1)))
                )
                axes[0].plot(df.index, rsi_values, label='RSI', color='blue')
                axes[0].axhline(y=70, color='r', linestyle='--', label='Overbought')
                axes[0].axhline(y=30, color='g', linestyle='--', label='Oversold')
                axes[0].set_ylabel('RSI', fontsize=10)
                axes[0].set_title('RSI Indicator', fontsize=12)
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                axes[0].set_ylim(0, 100)
            
            # MACD
            if 'macd' in indicators:
                ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                signal = macd.ewm(span=9, adjust=False).mean()
                histogram = macd - signal
                
                axes[1].plot(df.index, macd, label='MACD', color='blue')
                axes[1].plot(df.index, signal, label='Signal', color='red')
                axes[1].bar(df.index, histogram, label='Histogram', alpha=0.6, color='gray')
                axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                axes[1].set_ylabel('MACD', fontsize=10)
                axes[1].set_title('MACD Indicator', fontsize=12)
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
            
            # Volume
            axes[2].bar(df.index, df['volume'], alpha=0.6, color='blue')
            axes[2].set_ylabel('Volume', fontsize=10)
            axes[2].set_xlabel('Date', fontsize=10)
            axes[2].set_title('Volume', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            logger.error(f"Error creating indicator chart: {e}")
            fig = Figure(figsize=self.fig_size, dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            return fig

