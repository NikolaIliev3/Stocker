"""
Main GUI Application for Stocker - Stock Trading/Investing Desktop App
Enhanced with themes, animations, and predictions tracking
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import logging
from pathlib import Path
from datetime import datetime
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from config import APP_DATA_DIR, INITIAL_BALANCE
from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from investing_analyzer import InvestingAnalyzer
from chart_generator import ChartGenerator
from portfolio import Portfolio
from predictions_tracker import PredictionsTracker
from ui_themes import ThemeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnimatedButton(tk.Button):
    """Button with hover animation effects"""
    def __init__(self, parent, *args, **kwargs):
        self.original_bg = kwargs.get('bg', '#2196F3')
        self.hover_bg = kwargs.get('hover_bg', '#1976D2')
        if 'hover_bg' in kwargs:
            del kwargs['hover_bg']
        super().__init__(parent, *args, **kwargs, 
                       relief=tk.RAISED, bd=2, cursor='hand2',
                       activebackground=self.hover_bg)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
    
    def _on_enter(self, event):
        self.config(bg=self.hover_bg)
    
    def _on_leave(self, event):
        self.config(bg=self.original_bg)


class StockerApp:
    """Main application class with enhanced UI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Stocker - Stock Trading & Investing App")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        
        # Initialize components
        self.data_fetcher = StockDataFetcher()
        self.trading_analyzer = TradingAnalyzer()
        self.investing_analyzer = InvestingAnalyzer()
        self.chart_generator = ChartGenerator()
        self.portfolio = Portfolio(APP_DATA_DIR)
        self.predictions_tracker = PredictionsTracker(APP_DATA_DIR)
        self.theme_manager = ThemeManager()
        
        # State variables
        self.current_strategy = None
        self.current_symbol = None
        self.current_data = None
        self.current_analysis = None
        self.current_theme = "light"
        
        # Create UI
        self._create_ui()
        
        # Load portfolio balance
        self._update_portfolio_display()
        
        # Verify predictions on startup
        self._verify_predictions_on_startup()
    
    def _verify_predictions_on_startup(self):
        """Verify all active predictions when app starts"""
        def verify_in_background():
            try:
                logger.info("Verifying active predictions...")
                result = self.predictions_tracker.verify_all_active_predictions(self.data_fetcher)
                if result['verified'] > 0:
                    self.root.after(0, self._show_verification_results, result)
            except Exception as e:
                logger.error(f"Error verifying predictions: {e}")
        
        thread = threading.Thread(target=verify_in_background)
        thread.daemon = True
        thread.start()
    
    def _show_verification_results(self, result: dict):
        """Show prediction verification results"""
        if result['verified'] > 0:
            accuracy = result['accuracy']
            message = f"Verified {result['verified']} prediction(s). "
            message += f"Accuracy: {accuracy:.1f}% ({result['correct']}/{result['verified']} correct)"
            messagebox.showinfo("Predictions Verified", message)
            self._update_predictions_display()
    
    def _apply_theme(self, theme_name: str):
        """Apply a theme to the entire application"""
        self.current_theme = theme_name
        self.theme_manager.set_theme(theme_name)
        theme = self.theme_manager.get_theme(theme_name)
        
        # Apply theme to root
        self.root.config(bg=theme['bg'])
        
        # Apply to all widgets (simplified - in production would be more comprehensive)
        try:
            style = ttk.Style()
            style.theme_use('clam')  # Base theme
            
            # Configure ttk styles
            style.configure('TFrame', background=theme['frame_bg'])
            style.configure('TLabelFrame', background=theme['frame_bg'], foreground=theme['fg'])
            style.configure('TLabel', background=theme['frame_bg'], foreground=theme['fg'])
            style.configure('TButton', background=theme['button_bg'], foreground=theme['button_fg'])
            style.map('TButton', background=[('active', theme['accent_hover'])])
            style.configure('TEntry', fieldbackground=theme['entry_bg'], foreground=theme['entry_fg'])
            style.configure('TNotebook', background=theme['frame_bg'])
            style.configure('TNotebook.Tab', background=theme['secondary_bg'], foreground=theme['fg'])
            
            # Update matplotlib style
            if theme_name == "dark" or theme_name == "black":
                plt.style.use('dark_background')
            else:
                plt.style.use('default')
            
            logger.info(f"Theme changed to: {theme_name}")
        except Exception as e:
            logger.error(f"Error applying theme: {e}")
    
    def _create_ui(self):
        """Create the enhanced user interface"""
        # Apply initial theme
        self._apply_theme("light")
        
        # Main container with gradient effect simulation
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Top bar with theme selector
        top_bar = ttk.Frame(main_frame)
        top_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Theme selector
        theme_frame = ttk.Frame(top_bar)
        theme_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(theme_frame, text="Theme:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.theme_var = tk.StringVar(value="light")
        theme_menu = ttk.Combobox(theme_frame, textvariable=self.theme_var, 
                                 values=["light", "dark", "black"], 
                                 state="readonly", width=10)
        theme_menu.pack(side=tk.LEFT)
        theme_menu.bind('<<ComboboxSelected>>', lambda e: self._apply_theme(self.theme_var.get()))
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, padding="15")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.config(width=280)
        
        # Strategy selection with icons
        strategy_frame = ttk.LabelFrame(left_panel, text="📊 Strategy Selection", padding="12")
        strategy_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.strategy_var = tk.StringVar(value="trading")
        ttk.Radiobutton(strategy_frame, text="⚡ Trading (Short-term)", 
                       variable=self.strategy_var, value="trading",
                       command=self._on_strategy_change).pack(anchor=tk.W, pady=3)
        ttk.Radiobutton(strategy_frame, text="📈 Investing (Long-term)", 
                       variable=self.strategy_var, value="investing",
                       command=self._on_strategy_change).pack(anchor=tk.W, pady=3)
        
        # Stock input with search icon
        stock_frame = ttk.LabelFrame(left_panel, text="🔍 Stock Lookup", padding="12")
        stock_frame.pack(fill=tk.X, pady=(0, 12))
        
        ttk.Label(stock_frame, text="Stock Symbol:", font=('Arial', 9)).pack(anchor=tk.W)
        self.symbol_entry = ttk.Entry(stock_frame, width=18, font=('Arial', 10))
        self.symbol_entry.pack(fill=tk.X, pady=(5, 10))
        self.symbol_entry.bind('<Return>', lambda e: self._analyze_stock())
        
        analyze_btn = ttk.Button(stock_frame, text="Analyze Stock", 
                                 command=self._analyze_stock)
        analyze_btn.pack(fill=tk.X)
        
        # Budget input
        budget_frame = ttk.LabelFrame(left_panel, text="💰 Investment Budget", padding="12")
        budget_frame.pack(fill=tk.X, pady=(0, 12))
        
        ttk.Label(budget_frame, text="Budget ($):", font=('Arial', 9)).pack(anchor=tk.W)
        self.budget_entry = ttk.Entry(budget_frame, width=18, font=('Arial', 10))
        self.budget_entry.pack(fill=tk.X, pady=(5, 10))
        self.budget_entry.insert(0, "1000")
        
        ttk.Button(budget_frame, text="Calculate Potential", 
                  command=self._calculate_potential).pack(fill=tk.X)
        
        # Portfolio display with animated stats
        portfolio_frame = ttk.LabelFrame(left_panel, text="💼 Portfolio", padding="12")
        portfolio_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.portfolio_label = ttk.Label(portfolio_frame, text="Balance: $0.00", 
                                        font=('Arial', 10, 'bold'))
        self.portfolio_label.pack(anchor=tk.W, pady=5)
        
        self.stats_label = ttk.Label(portfolio_frame, text="Wins: 0 | Losses: 0", 
                                     font=('Arial', 9))
        self.stats_label.pack(anchor=tk.W, pady=3)
        
        ttk.Button(portfolio_frame, text="Set Initial Balance", 
                  command=self._set_balance).pack(fill=tk.X, pady=(10, 0))
        
        # Predictions quick stats
        pred_frame = ttk.LabelFrame(left_panel, text="🎯 Predictions", padding="12")
        pred_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.pred_stats_label = ttk.Label(pred_frame, text="Accuracy: N/A", 
                                         font=('Arial', 9))
        self.pred_stats_label.pack(anchor=tk.W, pady=3)
        
        ttk.Button(pred_frame, text="View All Predictions", 
                  command=self._show_predictions_tab).pack(fill=tk.X, pady=(5, 0))
        
        # Right panel - Results
        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        # Results header with animated title
        results_header = ttk.Frame(right_panel)
        results_header.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        results_header.columnconfigure(0, weight=1)
        
        self.results_title = ttk.Label(results_header, text="📊 Stock Analysis Results", 
                                      font=('Arial', 16, 'bold'))
        self.results_title.pack(side=tk.LEFT)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(self.analysis_frame, text="📋 Analysis")
        
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame, 
                                                      wrap=tk.WORD, height=20,
                                                      font=('Consolas', 10))
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.chart_frame, text="📈 Charts")
        
        self.chart_canvas = None
        
        # Indicators tab
        self.indicators_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.indicators_frame, text="📊 Indicators")
        
        self.indicators_canvas = None
        
        # Potential tab
        self.potential_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(self.potential_frame, text="💰 Potential Trade")
        
        self.potential_text = scrolledtext.ScrolledText(self.potential_frame, 
                                                       wrap=tk.WORD, height=20,
                                                       font=('Consolas', 10))
        self.potential_text.pack(fill=tk.BOTH, expand=True)
        
        # Predictions tab
        self.predictions_frame = ttk.Frame(self.notebook, padding="15")
        self.notebook.add(self.predictions_frame, text="🎯 Predictions")
        
        self._create_predictions_tab()
        
        # Update predictions display
        self._update_predictions_display()
    
    def _create_predictions_tab(self):
        """Create the predictions tracking tab"""
        # Header with stats
        header_frame = ttk.Frame(self.predictions_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        stats = self.predictions_tracker.get_statistics()
        
        stats_text = f"Total: {stats['total_predictions']} | "
        stats_text += f"Active: {stats['active']} | "
        stats_text += f"Verified: {stats['verified']} | "
        if stats['verified'] > 0:
            stats_text += f"Accuracy: {stats['accuracy']:.1f}%"
        else:
            stats_text += "Accuracy: N/A"
        
        ttk.Label(header_frame, text=stats_text, font=('Arial', 11, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(header_frame, text="Refresh", 
                  command=self._refresh_predictions).pack(side=tk.RIGHT)
        
        # Predictions list with scrollbar
        list_frame = ttk.Frame(self.predictions_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for predictions
        columns = ('ID', 'Symbol', 'Action', 'Entry', 'Target', 'Status', 'Result')
        self.predictions_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.predictions_tree.heading(col, text=col)
            self.predictions_tree.column(col, width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.predictions_tree.yview)
        self.predictions_tree.configure(yscrollcommand=scrollbar.set)
        
        self.predictions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons frame
        btn_frame = ttk.Frame(self.predictions_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Save Current Prediction", 
                  command=self._save_current_prediction).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Verify All Active", 
                  command=self._verify_all_predictions).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear Old Predictions", 
                  command=self._clear_old_predictions).pack(side=tk.LEFT, padx=5)
    
    def _show_predictions_tab(self):
        """Switch to predictions tab"""
        # Find predictions tab index
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "🎯 Predictions":
                self.notebook.select(i)
                break
    
    def _save_current_prediction(self):
        """Save current analysis as a prediction"""
        if not self.current_analysis or 'error' in self.current_analysis:
            messagebox.showwarning("No Analysis", "Please analyze a stock first")
            return
        
        if not self.current_symbol:
            messagebox.showwarning("No Symbol", "No stock symbol available")
            return
        
        recommendation = self.current_analysis.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        entry_price = recommendation.get('entry_price', 0)
        target_price = recommendation.get('target_price', 0)
        stop_loss = recommendation.get('stop_loss', 0)
        confidence = recommendation.get('confidence', 0)
        reasoning = self.current_analysis.get('reasoning', '')
        
        if entry_price <= 0:
            messagebox.showwarning("Invalid Prediction", "Cannot save prediction with invalid prices")
            return
        
        strategy = self.strategy_var.get()
        
        prediction = self.predictions_tracker.add_prediction(
            self.current_symbol, strategy, action, entry_price, 
            target_price, stop_loss, confidence, reasoning
        )
        
        messagebox.showinfo("Prediction Saved", 
                          f"Prediction #{prediction['id']} saved for {self.current_symbol}")
        self._update_predictions_display()
    
    def _verify_all_predictions(self):
        """Manually verify all active predictions"""
        def verify_in_background():
            try:
                self.root.after(0, self._update_status, "Verifying predictions...")
                result = self.predictions_tracker.verify_all_active_predictions(self.data_fetcher)
                self.root.after(0, self._show_verification_results, result)
                self.root.after(0, self._update_predictions_display)
            except Exception as e:
                logger.error(f"Error verifying predictions: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to verify: {str(e)}"))
        
        thread = threading.Thread(target=verify_in_background)
        thread.daemon = True
        thread.start()
    
    def _clear_old_predictions(self):
        """Clear old verified predictions"""
        if messagebox.askyesno("Confirm", "Clear all verified predictions?"):
            verified = self.predictions_tracker.get_verified_predictions()
            self.predictions_tracker.predictions = [
                p for p in self.predictions_tracker.predictions if p['status'] != 'verified'
            ]
            self.predictions_tracker.save()
            messagebox.showinfo("Cleared", f"Removed {len(verified)} verified predictions")
            self._update_predictions_display()
    
    def _update_predictions_display(self):
        """Update the predictions display"""
        # Update stats label
        stats = self.predictions_tracker.get_statistics()
        if stats['verified'] > 0:
            self.pred_stats_label.config(
                text=f"Accuracy: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['verified']} correct)"
            )
        else:
            self.pred_stats_label.config(text="Accuracy: N/A (No verified predictions)")
        
        # Clear tree
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
        
        # Add all predictions
        for pred in self.predictions_tracker.predictions:
            status = pred['status']
            result = "N/A"
            if pred['verified']:
                result = "✓ Correct" if pred['was_correct'] else "✗ Incorrect"
            
            values = (
                pred['id'],
                pred['symbol'],
                pred['action'],
                f"${pred['entry_price']:.2f}",
                f"${pred['target_price']:.2f}",
                status.title(),
                result
            )
            
            item = self.predictions_tree.insert('', tk.END, values=values)
            
            # Color code based on result
            if pred['verified']:
                if pred['was_correct']:
                    self.predictions_tree.set(item, 'Result', '✓ Correct')
                else:
                    self.predictions_tree.set(item, 'Result', '✗ Incorrect')
    
    def _refresh_predictions(self):
        """Refresh predictions display"""
        self._update_predictions_display()
    
    def _on_strategy_change(self):
        """Handle strategy change with animation"""
        self.current_strategy = self.strategy_var.get()
        logger.info(f"Strategy changed to: {self.current_strategy}")
        # Could add visual feedback here
    
    def _analyze_stock(self):
        """Analyze a stock based on selected strategy"""
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Input Error", "Please enter a stock symbol")
            return
        
        # Disable button during analysis with visual feedback
        self.symbol_entry.config(state='disabled')
        self._animate_loading()
        
        # Run analysis in thread to prevent UI freezing
        thread = threading.Thread(target=self._perform_analysis, args=(symbol,))
        thread.daemon = True
        thread.start()
    
    def _animate_loading(self):
        """Animate loading indicator"""
        dots = [".", "..", "..."]
        def update_dots(i=0):
            if self.symbol_entry.cget('state') == 'disabled':
                self.results_title.config(text=f"Loading{dots[i % len(dots)]}")
                self.root.after(500, lambda: update_dots(i + 1))
        
        update_dots()
    
    def _perform_analysis(self, symbol: str):
        """Perform stock analysis (runs in background thread)"""
        try:
            self.root.after(0, self._update_status, f"Fetching data for {symbol}...")
            
            # Fetch stock data
            stock_data = self.data_fetcher.fetch_stock_data(symbol)
            history_data = self.data_fetcher.fetch_stock_history(symbol)
            
            self.current_symbol = symbol
            self.current_data = stock_data
            
            strategy = self.strategy_var.get()
            
            # Perform analysis based on strategy
            if strategy == "trading":
                self.root.after(0, self._update_status, "Performing trading analysis...")
                analysis = self.trading_analyzer.analyze(stock_data, history_data)
            else:  # investing
                self.root.after(0, self._update_status, "Performing fundamental analysis...")
                financials_data = self.data_fetcher.fetch_financials(symbol)
                analysis = self.investing_analyzer.analyze(stock_data, financials_data, history_data)
            
            self.current_analysis = analysis
            
            # Update UI in main thread
            self.root.after(0, self._display_analysis, analysis, history_data)
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            self.root.after(0, self._show_error, f"Error analyzing stock: {str(e)}")
        finally:
            self.root.after(0, lambda: self.symbol_entry.config(state='normal'))
    
    def _update_status(self, message: str):
        """Update status message"""
        self.results_title.config(text=f"📊 {message}")
    
    def _display_analysis(self, analysis: dict, history_data: dict):
        """Display analysis results with animations"""
        if 'error' in analysis:
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, f"Error: {analysis['error']}")
            return
        
        # Display reasoning with formatting
        reasoning = analysis.get('reasoning', 'No reasoning available')
        self.analysis_text.delete(1.0, tk.END)
        
        # Add some formatting
        recommendation = analysis.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        confidence = recommendation.get('confidence', 0)
        
        header = f"{'='*60}\n"
        header += f"RECOMMENDATION: {action} (Confidence: {confidence}%)\n"
        header += f"{'='*60}\n\n"
        
        self.analysis_text.insert(tk.END, header)
        self.analysis_text.insert(tk.END, reasoning)
        
        # Highlight recommendation
        self.analysis_text.tag_add("header", "1.0", "3.0")
        self.analysis_text.tag_config("header", font=('Consolas', 11, 'bold'))
        
        # Update title with animation
        symbol = self.current_symbol or "Stock"
        self._fade_title(f"{symbol} - Analysis Results")
        
        # Generate and display charts
        self._display_charts(history_data, analysis.get('indicators', {}))
        
        # Store for potential calculation
        self.current_analysis = analysis
    
    def _fade_title(self, new_text: str):
        """Animate title change"""
        # Simple fade effect by updating text
        self.results_title.config(text=f"📊 {new_text}")
    
    def _display_charts(self, history_data: dict, indicators: dict):
        """Display stock charts"""
        try:
            # Main candlestick chart
            if self.chart_canvas:
                self.chart_canvas.get_tk_widget().destroy()
            
            fig = self.chart_generator.create_candlestick_chart(history_data, indicators)
            self.chart_canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            self.chart_canvas.draw()
            self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Indicators chart
            if self.indicators_canvas:
                self.indicators_canvas.get_tk_widget().destroy()
            
            fig2 = self.chart_generator.create_indicator_chart(history_data, indicators)
            self.indicators_canvas = FigureCanvasTkAgg(fig2, self.indicators_frame)
            self.indicators_canvas.draw()
            self.indicators_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            logger.error(f"Error displaying charts: {e}")
    
    def _calculate_potential(self):
        """Calculate potential win/loss for current analysis"""
        if not self.current_analysis or 'error' in self.current_analysis:
            messagebox.showwarning("No Analysis", "Please analyze a stock first")
            return
        
        try:
            budget = float(self.budget_entry.get())
            if budget <= 0:
                raise ValueError("Budget must be positive")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid budget: {str(e)}")
            return
        
        recommendation = self.current_analysis.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        entry_price = recommendation.get('entry_price', 0)
        target_price = recommendation.get('target_price', 0)
        stop_loss = recommendation.get('stop_loss', 0)
        
        if entry_price <= 0:
            messagebox.showwarning("No Recommendation", "No valid recommendation available")
            return
        
        symbol = self.current_symbol or "UNKNOWN"
        potential = self.portfolio.calculate_potential_trade(
            symbol, budget, entry_price, target_price, stop_loss, action
        )
        
        if 'error' in potential:
            self.potential_text.delete(1.0, tk.END)
            self.potential_text.insert(tk.END, f"Error: {potential['error']}")
            return
        
        # Format output with better styling
        output = f"{'='*60}\n"
        output += f"Potential Trade Analysis for {potential['symbol']}\n"
        output += f"{'='*60}\n\n"
        output += f"Action: {potential['action']}\n"
        output += f"Shares: {potential['shares']}\n"
        output += f"Budget Used: ${potential['budget_used']:.2f}\n\n"
        output += f"Entry Price: ${potential['entry_price']:.2f}\n"
        output += f"Target Price: ${potential['target_price']:.2f}\n"
        output += f"Stop Loss: ${potential['stop_loss']:.2f}\n\n"
        output += f"Potential Win: ${potential['potential_win']:.2f} "
        output += f"({potential['potential_win_percent']:.2f}%)\n"
        output += f"Potential Loss: ${potential['potential_loss']:.2f} "
        output += f"({potential['potential_loss_percent']:.2f}%)\n\n"
        output += f"Risk/Reward Ratio: {potential['risk_reward_ratio']:.2f}\n"
        
        self.potential_text.delete(1.0, tk.END)
        self.potential_text.insert(tk.END, output)
    
    def _set_balance(self):
        """Set initial portfolio balance"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Initial Balance")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Initial Balance ($):").pack(pady=10)
        balance_entry = ttk.Entry(dialog, width=20)
        balance_entry.pack(pady=5)
        balance_entry.insert(0, str(self.portfolio.balance))
        balance_entry.focus()
        
        def save_balance():
            try:
                balance = float(balance_entry.get())
                if balance < 0:
                    raise ValueError("Balance must be positive")
                self.portfolio.set_balance(balance)
                self._update_portfolio_display()
                dialog.destroy()
                messagebox.showinfo("Success", "Balance updated successfully")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid balance: {str(e)}")
        
        ttk.Button(dialog, text="Save", command=save_balance).pack(pady=10)
        balance_entry.bind('<Return>', lambda e: save_balance())
    
    def _update_portfolio_display(self):
        """Update portfolio display"""
        stats = self.portfolio.get_statistics()
        self.portfolio_label.config(text=f"Balance: ${stats['balance']:.2f}")
        self.stats_label.config(text=f"Wins: {stats['wins']} | Losses: {stats['losses']} | "
                                     f"Win Rate: {stats['win_rate']:.1f}%")
    
    def _show_error(self, message: str):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.results_title.config(text="📊 Stock Analysis Results")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = StockerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

