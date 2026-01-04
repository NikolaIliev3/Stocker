"""
Main GUI Application for Stocker - Stock Trading/Investing Desktop App
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import logging
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import APP_DATA_DIR, INITIAL_BALANCE
from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from investing_analyzer import InvestingAnalyzer
from chart_generator import ChartGenerator
from portfolio import Portfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockerApp:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Stocker - Stock Trading & Investing App")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.data_fetcher = StockDataFetcher()
        self.trading_analyzer = TradingAnalyzer()
        self.investing_analyzer = InvestingAnalyzer()
        self.chart_generator = ChartGenerator()
        self.portfolio = Portfolio(APP_DATA_DIR)
        
        # State variables
        self.current_strategy = None
        self.current_symbol = None
        self.current_data = None
        self.current_analysis = None
        
        # Create UI
        self._create_ui()
        
        # Load portfolio balance
        self._update_portfolio_display()
    
    def _create_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, padding="10")
        left_panel.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Strategy selection
        strategy_frame = ttk.LabelFrame(left_panel, text="Strategy Selection", padding="10")
        strategy_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.strategy_var = tk.StringVar(value="trading")
        ttk.Radiobutton(strategy_frame, text="Trading (Short-term)", 
                       variable=self.strategy_var, value="trading",
                       command=self._on_strategy_change).pack(anchor=tk.W)
        ttk.Radiobutton(strategy_frame, text="Investing (Long-term)", 
                       variable=self.strategy_var, value="investing",
                       command=self._on_strategy_change).pack(anchor=tk.W)
        
        # Stock input
        stock_frame = ttk.LabelFrame(left_panel, text="Stock Lookup", padding="10")
        stock_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(stock_frame, text="Stock Symbol:").pack(anchor=tk.W)
        self.symbol_entry = ttk.Entry(stock_frame, width=15)
        self.symbol_entry.pack(fill=tk.X, pady=(5, 10))
        self.symbol_entry.bind('<Return>', lambda e: self._analyze_stock())
        
        ttk.Button(stock_frame, text="Analyze Stock", 
                  command=self._analyze_stock).pack(fill=tk.X)
        
        # Budget input
        budget_frame = ttk.LabelFrame(left_panel, text="Investment Budget", padding="10")
        budget_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(budget_frame, text="Budget ($):").pack(anchor=tk.W)
        self.budget_entry = ttk.Entry(budget_frame, width=15)
        self.budget_entry.pack(fill=tk.X, pady=(5, 10))
        self.budget_entry.insert(0, "1000")
        
        ttk.Button(budget_frame, text="Calculate Potential", 
                  command=self._calculate_potential).pack(fill=tk.X)
        
        # Portfolio display
        portfolio_frame = ttk.LabelFrame(left_panel, text="Portfolio", padding="10")
        portfolio_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.portfolio_label = ttk.Label(portfolio_frame, text="Balance: $0.00")
        self.portfolio_label.pack(anchor=tk.W, pady=5)
        
        self.stats_label = ttk.Label(portfolio_frame, text="Wins: 0 | Losses: 0")
        self.stats_label.pack(anchor=tk.W, pady=5)
        
        ttk.Button(portfolio_frame, text="Set Initial Balance", 
                  command=self._set_balance).pack(fill=tk.X, pady=(10, 0))
        
        # Right panel - Results
        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        # Results header
        results_header = ttk.Frame(right_panel)
        results_header.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        results_header.columnconfigure(0, weight=1)
        
        self.results_title = ttk.Label(results_header, text="Stock Analysis Results", 
                                      font=('Arial', 14, 'bold'))
        self.results_title.pack(side=tk.LEFT)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.analysis_frame, text="Analysis")
        
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame, 
                                                      wrap=tk.WORD, height=20)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.chart_frame, text="Charts")
        
        self.chart_canvas = None
        
        # Indicators tab
        self.indicators_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.indicators_frame, text="Indicators")
        
        self.indicators_canvas = None
        
        # Potential tab
        self.potential_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.potential_frame, text="Potential Trade")
        
        self.potential_text = scrolledtext.ScrolledText(self.potential_frame, 
                                                       wrap=tk.WORD, height=20)
        self.potential_text.pack(fill=tk.BOTH, expand=True)
    
    def _on_strategy_change(self):
        """Handle strategy change"""
        self.current_strategy = self.strategy_var.get()
        logger.info(f"Strategy changed to: {self.current_strategy}")
    
    def _analyze_stock(self):
        """Analyze a stock based on selected strategy"""
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Input Error", "Please enter a stock symbol")
            return
        
        # Disable button during analysis
        self.symbol_entry.config(state='disabled')
        
        # Run analysis in thread to prevent UI freezing
        thread = threading.Thread(target=self._perform_analysis, args=(symbol,))
        thread.daemon = True
        thread.start()
    
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
        self.results_title.config(text=message)
    
    def _display_analysis(self, analysis: dict, history_data: dict):
        """Display analysis results"""
        if 'error' in analysis:
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, f"Error: {analysis['error']}")
            return
        
        # Display reasoning
        reasoning = analysis.get('reasoning', 'No reasoning available')
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, reasoning)
        
        # Update title
        symbol = self.current_symbol or "Stock"
        self.results_title.config(text=f"{symbol} - Analysis Results")
        
        # Generate and display charts
        self._display_charts(history_data, analysis.get('indicators', {}))
        
        # Store for potential calculation
        self.current_analysis = analysis
    
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
        
        # Format output
        output = f"Potential Trade Analysis for {potential['symbol']}\n"
        output += "=" * 50 + "\n\n"
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
        
        ttk.Label(dialog, text="Initial Balance ($):").pack(pady=10)
        balance_entry = ttk.Entry(dialog, width=20)
        balance_entry.pack(pady=5)
        balance_entry.insert(0, str(self.portfolio.balance))
        
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
    
    def _update_portfolio_display(self):
        """Update portfolio display"""
        stats = self.portfolio.get_statistics()
        self.portfolio_label.config(text=f"Balance: ${stats['balance']:.2f}")
        self.stats_label.config(text=f"Wins: {stats['wins']} | Losses: {stats['losses']} | "
                                     f"Win Rate: {stats['win_rate']:.1f}%")
    
    def _show_error(self, message: str):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.results_title.config(text="Stock Analysis Results")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = StockerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

