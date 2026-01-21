"""
Market Dashboard Component
Displays Market Intelligence features (Sentiment, Macro, Sector) in a visual dashboard.
"""
import logging
import tkinter as tk
from tkinter import ttk, font
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MarketDashboard(ttk.Frame):
    """
    Main container for Market Intelligence widgets
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        # self.rowconfigure(0, weight=1) # Don't expand vertically necessarily
        
        # --- Styles ---
        self._init_styles()
        
        # --- Layout ---
        # 1. Macro Card (Left)
        self.macro_card = MacroCard(self)
        self.macro_card.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # 2. Sentiment Card (Right)
        self.sentiment_card = SentimentCard(self)
        self.sentiment_card.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # 3. Sector Card (Bottom Full Width)
        self.sector_card = SectorCard(self)
        self.sector_card.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

    def _init_styles(self):
        style = ttk.Style()
        style.configure("Card.TFrame", relief="ridge", borderwidth=1)
        style.configure("CardTitle.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("CardValue.TLabel", font=("Segoe UI", 10))
        style.configure("RiskHigh.TLabel", foreground="red", font=("Segoe UI", 10, "bold"))
        style.configure("RiskLow.TLabel", foreground="green", font=("Segoe UI", 10, "bold"))
        style.configure("RiskNeutral.TLabel", foreground="#FFA500", font=("Segoe UI", 10, "bold"))

    def update_data(self, analysis_data: Dict):
        """Update all widgets with new analysis data"""
        if not analysis_data:
            return
            
        macro_info = analysis_data.get('macro_info', {})
        sentiment_info = analysis_data.get('sentiment_info', {})
        sector_analysis = analysis_data.get('sector_analysis', {})
        
        self.macro_card.update_data(macro_info)
        self.sentiment_card.update_data(sentiment_info)
        self.sector_card.update_data(sector_analysis)
        
        
class MacroCard(ttk.Frame):
    """Displays VIX, TNX, DXY and Market Regime"""
    def __init__(self, parent):
        super().__init__(parent, style="Card.TFrame", padding=15)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        
        # Title
        ttk.Label(self, text="🌍 Macro Regime", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        # Status
        self.status_var = tk.StringVar(value="Waiting for data...")
        self.status_label = ttk.Label(self, textvariable=self.status_var, font=("Segoe UI", 11))
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 15))
        
        # Grid of Indicators
        # VIX
        ttk.Label(self, text="VIX (Fear):").grid(row=2, column=0, sticky="w")
        self.vix_val = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self.vix_val, style="CardValue.TLabel").grid(row=2, column=1, sticky="e")
        
        # TNX
        ttk.Label(self, text="10Y Yield:").grid(row=3, column=0, sticky="w")
        self.tnx_val = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self.tnx_val, style="CardValue.TLabel").grid(row=3, column=1, sticky="e")
        
        # DXY
        ttk.Label(self, text="Dollar (DXY):").grid(row=4, column=0, sticky="w")
        self.dxy_val = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self.dxy_val, style="CardValue.TLabel").grid(row=4, column=1, sticky="e")

    def update_data(self, data: Dict):
        if not data or not data.get('available'):
            self.status_var.set("Data Unavailable")
            return
            
        # Update Values
        self.vix_val.set(f"{data.get('vix_value', 0):.2f}")
        self.tnx_val.set(f"{data.get('tnx_value', 0):.2f}%")
        self.dxy_val.set(f"{data.get('dxy_value', 0):.2f}")
        
        # Update Status Summary
        risk = data.get('risk_level', 'neutral')
        summary = data.get('regime_summary', 'Unknown')
        
        # Icon
        icon = "🌥️"
        if risk == 'extreme': icon = "🌪️"
        elif risk == 'high': icon = "🌩️"
        elif risk == 'low': icon = "☀️"
        
        self.status_var.set(f"{icon} {summary}")
        
        # Color coding
        if risk in ['high', 'extreme']:
            self.status_label.configure(style="RiskHigh.TLabel")
        elif risk == 'low':
            self.status_label.configure(style="RiskLow.TLabel")
        else:
            self.status_label.configure(style="RiskNeutral.TLabel")


class SentimentCard(ttk.Frame):
    """Displays News Sentiment Score and Headlines"""
    def __init__(self, parent):
        super().__init__(parent, style="Card.TFrame", padding=15)
        self.columnconfigure(0, weight=1)
        
        # Title
        ttk.Label(self, text="📰 News Sentiment", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Score / Rating
        self.rating_var = tk.StringVar(value="Waiting...")
        self.rating_label = ttk.Label(self, textvariable=self.rating_var, font=("Segoe UI", 11))
        self.rating_label.grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        # Headlines Container
        ttk.Label(self, text="Recent Headlines:", font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w", pady=(10, 5))
        self.headlines_frame = ttk.Frame(self)
        self.headlines_frame.grid(row=3, column=0, sticky="nsew")
        
    def update_data(self, data: Dict):
        if not data or not data.get('available'):
            self.rating_var.set("No News Data")
            return
            
        # Update Rating
        score = data.get('sentiment_score', 0)
        rating = data.get('sentiment_rating', 'neutral')
        
        emoji = "😐"
        style = "RiskNeutral.TLabel"
        if "bullish" in rating:
            emoji = "🐂"
            style = "RiskLow.TLabel" # Using Green for Bullish
        elif "bearish" in rating:
            emoji = "🐻"
            style = "RiskHigh.TLabel" # Using Red for Bearish
            
        self.rating_var.set(f"{emoji} {rating.replace('_', ' ').upper()} (Score: {score:.1f})")
        self.rating_label.configure(style=style)
        
        # Update Headlines
        for widget in self.headlines_frame.winfo_children():
            widget.destroy()
            
        headlines = data.get('headlines', [])
        if not headlines:
            ttk.Label(self.headlines_frame, text="No recent headlines found.", font=("Segoe UI", 9, "italic")).pack(anchor="w")
        else:
            for i, h in enumerate(headlines[:3]): # Top 3
                lbl_text = f"• {h['title'][:50]}..." if len(h['title']) > 50 else f"• {h['title']}"
                
                # Color code headline based on its specific score
                hl_color = "black"
                if h.get('score', 0) > 0: hl_color = "green"
                elif h.get('score', 0) < 0: hl_color = "red"
                
                lbl = ttk.Label(self.headlines_frame, text=lbl_text, font=("Segoe UI", 8), foreground=hl_color, anchor="w")
                lbl.pack(fill="x", pady=1)


class SectorCard(ttk.Frame):
    """Displays Sector Performance"""
    def __init__(self, parent):
        super().__init__(parent, style="Card.TFrame", padding=15)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        
        ttk.Label(self, text="🏢 Sector Analysis", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        self.info_var = tk.StringVar(value="Waiting...")
        ttk.Label(self, textvariable=self.info_var, font=("Segoe UI", 10)).grid(row=1, column=0, columnspan=2, sticky="w")
        
    def update_data(self, data: Dict):
        if not data or not data.get('available'):
            self.info_var.set("Sector data unavailable")
            return
            
        sector = data.get('sector', 'Unknown')
        signal = data.get('sector_signal', 'neutral')
        vs_sector = data.get('stock_vs_sector', 0)
        trend = data.get('sector_trend', 'unknown')
        
        emoji = "⚖️"
        if "outperform" in signal: emoji = "🚀"
        elif "underperform" in signal: emoji = "🐢"
        
        text = f"Sector: {sector}\n"
        text += f"Trend: {trend.title()}\n"
        text += f"Performance vs Sector: {vs_sector:+.2f}% {emoji}"
        
        self.info_var.set(text)
