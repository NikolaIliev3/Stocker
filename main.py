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
from mixed_analyzer import MixedAnalyzer
from chart_generator import ChartGenerator
from portfolio import Portfolio
from predictions_tracker import PredictionsTracker
from ui_themes import ThemeManager
from modern_ui import ModernCard, ModernButton, GradientLabel, ModernScrollbar
from localization import Localization
from user_preferences import UserPreferences
from loading_screen import LoadingScreen
from modern_settings import ModernDropdownButton
from market_scanner import MarketScanner

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
        # Enable window controls (minimize, maximize, close)
        self.root.resizable(True, True)
        # Store process status for loading screen
        self.current_processes = []
        
        # Initialize components
        self.data_fetcher = StockDataFetcher()
        self.trading_analyzer = TradingAnalyzer()
        self.investing_analyzer = InvestingAnalyzer()
        self.mixed_analyzer = MixedAnalyzer()
        self.chart_generator = ChartGenerator()
        self.portfolio = Portfolio(APP_DATA_DIR)
        self.predictions_tracker = PredictionsTracker(APP_DATA_DIR)
        self.theme_manager = ThemeManager()
        self.preferences = UserPreferences(APP_DATA_DIR)
        self.market_scanner = MarketScanner(self.data_fetcher)
        
        # Load preferences and store as instance variables
        self.saved_language = self.preferences.get_language()
        self.saved_currency = self.preferences.get_currency()
        self.saved_theme = self.preferences.get_theme()
        self.saved_strategy = self.preferences.get_strategy()
        
        self.localization = Localization(language=self.saved_language, currency=self.saved_currency)
        
        # State variables
        self.current_strategy = self.saved_strategy
        self.current_symbol = None
        self.current_data = None
        self.current_analysis = None
        self.current_theme = self.saved_theme
        self.loading_screen = None
        
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
        """Apply a theme to the entire application with full UI refresh"""
        self.current_theme = theme_name
        self.theme_manager.set_theme(theme_name)
        self.preferences.set_theme(theme_name)
        theme = self.theme_manager.get_theme(theme_name)
        
        # Update loading screen theme
        if self.loading_screen:
            self.loading_screen.theme = theme
        
        # Update dropdowns if they exist
        if hasattr(self, 'theme_dropdown'):
            self.theme_dropdown.set_value(theme_name)
        
        # Apply theme to root
        self.root.config(bg=theme['bg'])
        
        # Recreate UI with new theme (simplified approach - full recreation would be better)
        try:
            style = ttk.Style()
            style.theme_use('clam')
            
            # Configure ttk styles with modern colors
            style.configure('TFrame', background=theme['frame_bg'])
            style.configure('TLabelFrame', background=theme['frame_bg'], foreground=theme['fg'],
                          borderwidth=0, relief=tk.FLAT)
            style.configure('TLabel', background=theme['frame_bg'], foreground=theme['fg'])
            style.configure('TButton', background=theme['button_bg'], foreground=theme['button_fg'],
                          borderwidth=0, relief=tk.FLAT, padding=10)
            style.map('TButton', 
                     background=[('active', theme['button_hover']), ('pressed', theme['accent_hover'])])
            style.configure('TEntry', fieldbackground=theme['entry_bg'], foreground=theme['entry_fg'],
                          borderwidth=0, relief=tk.FLAT)
            style.configure('TNotebook', background=theme['frame_bg'], borderwidth=0)
            style.configure('TNotebook.Tab', background=theme['secondary_bg'], foreground=theme['fg'],
                          padding=[20, 10], borderwidth=0)
            style.map('TNotebook.Tab', 
                     background=[('selected', theme['frame_bg'])],
                     expand=[('selected', [1, 1, 1, 0])])
            style.configure('TRadiobutton', background=theme['frame_bg'], foreground=theme['fg'],
                          selectcolor=theme['accent'])
            style.configure('TScrollbar', background=theme['secondary_bg'], troughcolor=theme['bg'],
                          borderwidth=0, arrowcolor=theme['fg'], darkcolor=theme['secondary_bg'],
                          lightcolor=theme['secondary_bg'])
            style.map('TScrollbar', background=[('active', theme['accent'])])
            
            # Update matplotlib style
            if theme_name == "dark" or theme_name == "black":
                plt.style.use('dark_background')
            else:
                plt.style.use('seaborn-v0_8-whitegrid')
            
            # Update all existing widgets if they exist
            self._update_widgets_theme(theme)
            
            logger.info(f"Theme changed to: {theme_name}")
        except Exception as e:
            logger.error(f"Error applying theme: {e}")
    
    def _update_widgets_theme(self, theme: dict):
        """Update existing widgets with new theme colors"""
        # This would update all widgets, but for simplicity we'll just update key ones
        try:
            if hasattr(self, 'analysis_text'):
                self.analysis_text.config(bg=theme['frame_bg'], fg=theme['fg'],
                                        insertbackground=theme['fg'])
            if hasattr(self, 'potential_text'):
                self.potential_text.config(bg=theme['frame_bg'], fg=theme['fg'],
                                         insertbackground=theme['fg'])
            if hasattr(self, 'symbol_entry'):
                self.symbol_entry.config(bg=theme['entry_bg'], fg=theme['entry_fg'],
                                       insertbackground=theme['fg'])
            if hasattr(self, 'budget_entry'):
                self.budget_entry.config(bg=theme['entry_bg'], fg=theme['entry_fg'],
                                       insertbackground=theme['fg'])
        except Exception as e:
            logger.warning(f"Could not update all widgets: {e}")
    
    def _create_ui(self):
        """Create the modern, visually stunning user interface"""
        # Apply saved theme
        self._apply_theme(self.current_theme)
        theme = self.theme_manager.get_theme()
        
        # Initialize loading screen
        self.loading_screen = LoadingScreen(self.root, theme)
        
        # Main container - full window background
        self.root.config(bg=theme['bg'])
        main_container = tk.Frame(self.root, bg=theme['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Modern top bar with gradient effect
        top_bar = tk.Frame(main_container, bg=theme['secondary_bg'], height=70)
        top_bar.pack(fill=tk.X, padx=0, pady=0)
        top_bar.pack_propagate(False)
        
        # App title with modern styling
        title_frame = tk.Frame(top_bar, bg=theme['secondary_bg'])
        title_frame.pack(side=tk.LEFT, padx=25, pady=15)
        
        self.title_label = GradientLabel(title_frame, self.localization.t('app_title'), theme=theme, 
                     font=('Segoe UI', 24, 'bold'))
        self.title_label.pack(side=tk.LEFT)
        self.subtitle_label = tk.Label(title_frame, text=self.localization.t('app_subtitle'), 
                bg=theme['secondary_bg'], fg=theme['text_secondary'],
                font=('Segoe UI', 10))
        self.subtitle_label.pack(side=tk.LEFT, padx=(15, 0))
        
        # Settings frame - Theme, Language, Currency (Modern Dropdowns)
        settings_frame = tk.Frame(top_bar, bg=theme['secondary_bg'])
        settings_frame.pack(side=tk.RIGHT, padx=25, pady=15)
        
        # Modern dropdown buttons
        self.theme_dropdown = ModernDropdownButton(
            settings_frame, 
            self.localization.t('theme'),
            ["light", "dark", "black"],
            self.current_theme,
            self._apply_theme,
            theme
        )
        self.theme_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        self.language_dropdown = ModernDropdownButton(
            settings_frame,
            self.localization.t('language'),
            ["en", "bg"],
            self.saved_language,
            self._change_language,
            theme
        )
        self.language_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        self.currency_dropdown = ModernDropdownButton(
            settings_frame,
            self.localization.t('currency'),
            ["USD", "EUR", "BGN"],
            self.saved_currency,
            self._change_currency,
            theme
        )
        self.currency_dropdown.pack(side=tk.LEFT)
        
        # Main content area
        content_frame = tk.Frame(main_container, bg=theme['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left sidebar - Modern cards with scrollbar
        sidebar_container = tk.Frame(content_frame, bg=theme['bg'], width=340)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        sidebar_container.pack_propagate(False)
        
        # Create canvas and modern scrollbar for scrollable sidebar
        sidebar_canvas = tk.Canvas(sidebar_container, bg=theme['bg'], highlightthickness=0, width=320)
        sidebar = tk.Frame(sidebar_canvas, bg=theme['bg'], width=320)
        
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Modern scrollbar - make it visible
        sidebar_scrollbar = ModernScrollbar(sidebar_container, command=sidebar_canvas.yview, theme=theme)
        sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        # Force initial update
        sidebar_canvas.update_idletasks()
        sidebar_scrollbar._update_slider()
        
        # Configure scrollbar
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        sidebar_canvas.create_window((0, 0), window=sidebar, anchor=tk.NW)
        
        def configure_scroll_region(event=None):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
        
        sidebar.bind('<Configure>', configure_scroll_region)
        sidebar_canvas.bind('<Configure>', lambda e: sidebar_canvas.itemconfig(sidebar_canvas.find_all()[0], width=e.width) if sidebar_canvas.find_all() else None)
        
        # Bind mousewheel to canvas and sidebar - fix for Windows
        def on_mousewheel(event):
            # Windows uses delta, Linux uses different values
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                sidebar_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
        
        # Bind to multiple events for compatibility
        sidebar_canvas.bind("<MouseWheel>", on_mousewheel)
        sidebar.bind("<MouseWheel>", on_mousewheel)
        sidebar_canvas.bind("<Button-4>", lambda e: sidebar_canvas.yview_scroll(-1, "units"))
        sidebar_canvas.bind("<Button-5>", lambda e: sidebar_canvas.yview_scroll(1, "units"))
        sidebar.bind("<Button-4>", lambda e: sidebar_canvas.yview_scroll(-1, "units"))
        sidebar.bind("<Button-5>", lambda e: sidebar_canvas.yview_scroll(1, "units"))
        
        # Focus handling for mousewheel
        def on_enter(event):
            sidebar_canvas.focus_set()
        sidebar_canvas.bind("<Enter>", on_enter)
        sidebar.bind("<Enter>", on_enter)
        
        # Strategy Selection Card
        self.strategy_card = ModernCard(sidebar, self.localization.t('strategy'), theme=theme, padding=20)
        self.strategy_card.pack(fill=tk.X, pady=(0, 15))
        
        self.strategy_var = tk.StringVar(value=self.saved_strategy)
        strategy_inner = tk.Frame(self.strategy_card.content_frame, bg=theme['card_bg'])
        strategy_inner.pack(fill=tk.X)
        
        # Store strategy buttons for visual feedback
        self.strategy_buttons = {}
        
        # Modern radio buttons with visual selection feedback
        self.trading_btn_frame = tk.Frame(strategy_inner, bg=theme['card_bg'], relief=tk.FLAT, bd=0)
        self.trading_btn_frame.pack(fill=tk.X, pady=4)
        self.trading_btn = tk.Radiobutton(self.trading_btn_frame, text=f"⚡ {self.localization.t('trading')}", 
                                    variable=self.strategy_var, value="trading",
                                    command=self._on_strategy_change,
                                    bg=theme['card_bg'], fg=theme['fg'],
                                    font=('Segoe UI', 11, 'bold'), selectcolor=theme['accent'],
                                    activebackground=theme['card_bg'],
                                    activeforeground=theme['accent'])
        self.trading_btn.pack(anchor=tk.W, padx=10, pady=8)
        self.strategy_buttons['trading'] = self.trading_btn_frame
        
        self.mixed_btn_frame = tk.Frame(strategy_inner, bg=theme['card_bg'], relief=tk.FLAT, bd=0)
        self.mixed_btn_frame.pack(fill=tk.X, pady=4)
        self.mixed_btn = tk.Radiobutton(self.mixed_btn_frame, text=f"🔄 {self.localization.t('mixed')}", 
                                       variable=self.strategy_var, value="mixed",
                                       command=self._on_strategy_change,
                                       bg=theme['card_bg'], fg=theme['fg'],
                                       font=('Segoe UI', 11, 'bold'), selectcolor=theme['accent'],
                                       activebackground=theme['card_bg'],
                                       activeforeground=theme['accent'])
        self.mixed_btn.pack(anchor=tk.W, padx=10, pady=8)
        self.strategy_buttons['mixed'] = self.mixed_btn_frame
        
        # Add description label for mixed
        mixed_desc = tk.Label(self.mixed_btn_frame, 
                             text=f"  ({self.localization.t('mixed_description')})",
                             bg=theme['card_bg'], fg=theme['text_secondary'],
                             font=('Segoe UI', 8))
        mixed_desc.pack(anchor=tk.W, padx=(35, 0), pady=(0, 8))
        
        self.investing_btn_frame = tk.Frame(strategy_inner, bg=theme['card_bg'], relief=tk.FLAT, bd=0)
        self.investing_btn_frame.pack(fill=tk.X, pady=4)
        self.investing_btn = tk.Radiobutton(self.investing_btn_frame, text=f"📈 {self.localization.t('investing')}", 
                                       variable=self.strategy_var, value="investing",
                                       command=self._on_strategy_change,
                                       bg=theme['card_bg'], fg=theme['fg'],
                                       font=('Segoe UI', 11, 'bold'), selectcolor=theme['accent'],
                                       activebackground=theme['card_bg'],
                                       activeforeground=theme['accent'])
        self.investing_btn.pack(anchor=tk.W, padx=10, pady=8)
        self.strategy_buttons['investing'] = self.investing_btn_frame
        
        # Update visual feedback for saved strategy
        self._update_strategy_visual_feedback()
        
        # Budget Card
        self.budget_card = ModernCard(sidebar, self.localization.t('investment_budget'), theme=theme, padding=20)
        self.budget_card.pack(fill=tk.X, pady=(0, 15))
        
        currency_symbol = self.localization.format_currency_symbol()
        self.amount_label = tk.Label(self.budget_card.content_frame, 
                text=f"{self.localization.t('amount')} ({currency_symbol})", bg=theme['card_bg'],
                fg=theme['text_secondary'], font=('Segoe UI', 9))
        self.amount_label.pack(anchor=tk.W, pady=(0, 8))
        
        budget_entry_frame = tk.Frame(self.budget_card.content_frame, bg=theme['card_bg'])
        budget_entry_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.budget_entry = tk.Entry(budget_entry_frame, font=('Segoe UI', 12),
                                     bg=theme['entry_bg'], fg=theme['entry_fg'],
                                     relief=tk.FLAT, bd=0, insertbackground=theme['fg'])
        self.budget_entry.pack(fill=tk.X, ipady=10)
        # Load saved budget
        saved_budget = self.preferences.get_budget()
        self.budget_entry.insert(0, str(saved_budget))
        # Save budget when changed
        self.budget_entry.bind('<FocusOut>', lambda e: self._save_budget())
        self.budget_entry.bind('<Return>', lambda e: self._save_budget())
        
        self.budget_underline = tk.Frame(budget_entry_frame, bg=theme['border'], height=2)
        self.budget_underline.pack(fill=tk.X, pady=(5, 0))
        
        def on_budget_focus_in(e):
            self.budget_underline.config(bg=theme['accent'], height=2)
        def on_budget_focus_out(e):
            self.budget_underline.config(bg=theme['border'], height=2)
        
        self.budget_entry.bind('<FocusIn>', on_budget_focus_in)
        self.budget_entry.bind('<FocusOut>', on_budget_focus_out)
        
        self.calc_btn = ModernButton(self.budget_card.content_frame, self.localization.t('calculate_potential'), 
                               command=self._calculate_potential, theme=theme)
        self.calc_btn.pack(fill=tk.X)
        
        # Portfolio Stats Card
        self.portfolio_card = ModernCard(sidebar, self.localization.t('portfolio'), theme=theme, padding=20)
        self.portfolio_card.pack(fill=tk.X, pady=(0, 15))
        
        self.portfolio_label = tk.Label(self.portfolio_card.content_frame, 
                                       text=self.localization.format_currency(0), bg=theme['card_bg'],
                                       fg=theme['accent'], font=('Segoe UI', 20, 'bold'))
        self.portfolio_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.stats_label = tk.Label(self.portfolio_card.content_frame, 
                                   text=f"{self.localization.t('wins')}: 0 | {self.localization.t('losses')}: 0", bg=theme['card_bg'],
                                   fg=theme['text_secondary'], font=('Segoe UI', 10))
        self.stats_label.pack(anchor=tk.W, pady=(0, 15))
        
        self.set_balance_btn = ModernButton(self.portfolio_card.content_frame, self.localization.t('set_balance'), 
                    command=self._set_balance, theme=theme)
        self.set_balance_btn.pack(fill=tk.X)
        
        # Predictions Stats Card
        self.pred_card = ModernCard(sidebar, self.localization.t('predictions'), theme=theme, padding=20)
        self.pred_card.pack(fill=tk.X)
        
        self.pred_stats_label = tk.Label(self.pred_card.content_frame, 
                                        text=f"{self.localization.t('accuracy')}: N/A", bg=theme['card_bg'],
                                        fg=theme['accent'], font=('Segoe UI', 14, 'bold'))
        self.pred_stats_label.pack(anchor=tk.W, pady=(0, 15))
        
        self.view_all_btn = ModernButton(self.pred_card.content_frame, self.localization.t('view_all'), 
                    command=self._show_predictions_tab, theme=theme)
        self.view_all_btn.pack(fill=tk.X)
        
        # Right panel - Results area
        results_panel = tk.Frame(content_frame, bg=theme['bg'])
        results_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Results header
        header_card = ModernCard(results_panel, "", theme=theme, padding=20)
        header_card.pack(fill=tk.X, pady=(0, 15))
        
        self.results_title = GradientLabel(header_card.content_frame, 
                                          self.localization.t('stock_analysis_results'), theme=theme,
                                          font=('Segoe UI', 18, 'bold'))
        self.results_title.pack(side=tk.LEFT)
        
        # Modern notebook with custom styling
        notebook_container = tk.Frame(results_panel, bg=theme['bg'])
        notebook_container.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(notebook_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        self.analysis_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.analysis_frame, text=self.localization.t('analysis'))
        
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame, 
                                                      wrap=tk.WORD, height=20,
                                                      font=('Segoe UI', 10),
                                                      bg=theme['frame_bg'], fg=theme['fg'],
                                                      relief=tk.FLAT, bd=0,
                                                      insertbackground=theme['fg'])
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Chart tab
        self.chart_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.chart_frame, text=self.localization.t('charts'))
        
        self.chart_canvas = None
        
        # Indicators tab
        self.indicators_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.indicators_frame, text=self.localization.t('indicators'))
        
        self.indicators_canvas = None
        
        # Potential tab
        self.potential_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.potential_frame, text=self.localization.t('potential_trade'))
        
        self.potential_text = scrolledtext.ScrolledText(self.potential_frame, 
                                                       wrap=tk.WORD, height=20,
                                                       font=('Segoe UI', 10),
                                                       bg=theme['frame_bg'], fg=theme['fg'],
                                                       relief=tk.FLAT, bd=0,
                                                       insertbackground=theme['fg'])
        self.potential_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Predictions tab - make it scrollable
        predictions_container = tk.Frame(self.notebook, bg=theme['bg'])
        self.notebook.add(predictions_container, text=self.localization.t('predictions'))
        
        # Create canvas and scrollbar for scrollable predictions tab
        predictions_canvas = tk.Canvas(predictions_container, bg=theme['bg'], highlightthickness=0)
        self.predictions_frame = tk.Frame(predictions_canvas, bg=theme['bg'])
        
        predictions_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Modern scrollbar for predictions tab
        predictions_scrollbar = ModernScrollbar(predictions_container, command=predictions_canvas.yview, theme=theme)
        predictions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        # Force initial update
        predictions_canvas.update_idletasks()
        predictions_scrollbar._update_slider()
        
        # Configure scrollbar
        predictions_canvas.configure(yscrollcommand=predictions_scrollbar.set)
        predictions_canvas.create_window((0, 0), window=self.predictions_frame, anchor=tk.NW)
        
        def configure_predictions_scroll(event=None):
            predictions_canvas.configure(scrollregion=predictions_canvas.bbox("all"))
        
        self.predictions_frame.bind('<Configure>', configure_predictions_scroll)
        predictions_canvas.bind('<Configure>', lambda e: predictions_canvas.itemconfig(predictions_canvas.find_all()[0], width=e.width) if predictions_canvas.find_all() else None)
        
        # Bind mousewheel - fix for Windows
        def on_predictions_mousewheel(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                predictions_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
        
        predictions_canvas.bind("<MouseWheel>", on_predictions_mousewheel)
        self.predictions_frame.bind("<MouseWheel>", on_predictions_mousewheel)
        predictions_canvas.bind("<Button-4>", lambda e: predictions_canvas.yview_scroll(-1, "units"))
        predictions_canvas.bind("<Button-5>", lambda e: predictions_canvas.yview_scroll(1, "units"))
        self.predictions_frame.bind("<Button-4>", lambda e: predictions_canvas.yview_scroll(-1, "units"))
        self.predictions_frame.bind("<Button-5>", lambda e: predictions_canvas.yview_scroll(1, "units"))
        
        # Focus handling
        def on_predictions_enter(event):
            predictions_canvas.focus_set()
        predictions_canvas.bind("<Enter>", on_predictions_enter)
        self.predictions_frame.bind("<Enter>", on_predictions_enter)
        
        self._create_predictions_tab()
        
        # Update predictions display
        self._update_predictions_display()
        
        # Update strategy visual feedback after UI is created
        self.root.after(100, self._update_strategy_visual_feedback)
    
    def _change_language(self, language: str):
        """Change application language"""
        self.localization.set_language(language)
        self.preferences.set_language(language)
        if hasattr(self, 'language_dropdown'):
            self.language_dropdown.set_value(language)
        self._refresh_ui_texts()
        logger.info(f"Language changed to: {language}")
    
    def _change_currency(self, currency: str):
        """Change currency"""
        old_currency = self.saved_currency
        self.saved_currency = currency
        self.localization.set_currency(currency)
        self.preferences.set_currency(currency)
        if hasattr(self, 'currency_dropdown'):
            self.currency_dropdown.set_value(currency)
        
        # Update budget display with new currency conversion
        if hasattr(self, 'budget_entry'):
            try:
                current_budget = float(self.budget_entry.get())
                # Convert from old currency to USD, then to new currency
                from localization import Localization
                rates = Localization.CURRENCY_RATES
                usd_budget = current_budget / rates.get(old_currency, 1.0)
                new_budget = usd_budget * rates.get(currency, 1.0)
                self.budget_entry.delete(0, tk.END)
                self.budget_entry.insert(0, str(round(new_budget, 2)))
                self._save_budget()  # Save with new currency
            except ValueError:
                pass  # Invalid budget, don't update
        
        self._update_portfolio_display()
        self._update_predictions_display()
        # Update amount label
        currency_symbol = self.localization.format_currency_symbol()
        self.amount_label.config(text=f"{self.localization.t('amount')} ({currency_symbol})")
        logger.info(f"Currency changed to: {currency}")
    
    def _save_budget(self):
        """Save budget to preferences with current currency"""
        try:
            budget = float(self.budget_entry.get())
            if budget > 0:
                self.preferences.set_budget(budget, self.saved_currency)
                logger.info(f"Budget saved: {budget} {self.saved_currency}")
        except ValueError:
            pass  # Invalid input, don't save
    
    def _update_strategy_visual_feedback(self):
        """Update visual feedback for strategy selection"""
        theme = self.theme_manager.get_theme()
        selected = self.strategy_var.get()
        
        for strategy, frame in self.strategy_buttons.items():
            if strategy == selected:
                # Highlight selected strategy
                frame.config(bg=theme['accent'], highlightbackground=theme['accent'], highlightthickness=2)
                # Update button text to show selection
                for widget in frame.winfo_children():
                    if isinstance(widget, tk.Radiobutton):
                        widget.config(fg=theme['button_fg'] if theme.get('button_fg') else '#FFFFFF')
            else:
                # Reset unselected strategies
                frame.config(bg=theme['card_bg'], highlightbackground=theme['border'], highlightthickness=0)
                for widget in frame.winfo_children():
                    if isinstance(widget, tk.Radiobutton):
                        widget.config(fg=theme['fg'])
    
    def _refresh_ui_texts(self):
        """Refresh all UI texts with current language"""
        theme = self.theme_manager.get_theme()
        
        # Update title and subtitle
        self.title_label.config(text=self.localization.t('app_title'))
        self.subtitle_label.config(text=self.localization.t('app_subtitle'))
        
        # Update card titles
        # Note: Card titles are harder to update, so we'll skip them for now
        # or recreate the UI (simpler approach)
        
        # Update buttons and labels
        self.trading_btn.config(text=f"⚡ {self.localization.t('trading')}")
        self.mixed_btn.config(text=f"🔄 {self.localization.t('mixed')}")
        self.investing_btn.config(text=f"📈 {self.localization.t('investing')}")
        self.symbol_label.config(text=self.localization.t('symbol'))
        self.analyze_btn.config(text=self.localization.t('analyze_stock'))
        currency_symbol = self.localization.format_currency_symbol()
        self.amount_label.config(text=f"{self.localization.t('amount')} ({currency_symbol})")
        self.calc_btn.config(text=self.localization.t('calculate_potential'))
        self.set_balance_btn.config(text=self.localization.t('set_balance'))
        self.view_all_btn.config(text=self.localization.t('view_all'))
        
        # Update notebook tabs
        self.notebook.tab(0, text=self.localization.t('analysis'))
        self.notebook.tab(1, text=self.localization.t('charts'))
        self.notebook.tab(2, text=self.localization.t('indicators'))
        self.notebook.tab(3, text=self.localization.t('potential_trade'))
        self.notebook.tab(4, text=self.localization.t('predictions'))
        
        # Update other displays
        self._update_portfolio_display()
        self._update_predictions_display()
        self.results_title.config(text=self.localization.t('stock_analysis_results'))
    
    def _create_predictions_tab(self):
        """Create the modern predictions tracking tab with stock lookup"""
        theme = self.theme_manager.get_theme()
        
        # Stock Lookup Card - moved from sidebar
        stock_lookup_card = ModernCard(self.predictions_frame, self.localization.t('stock_lookup'), theme=theme, padding=20)
        stock_lookup_card.pack(fill=tk.X, pady=(0, 20))
        
        self.symbol_label = tk.Label(stock_lookup_card.content_frame, text=self.localization.t('symbol'), bg=theme['card_bg'],
                fg=theme['text_secondary'], font=('Segoe UI', 9))
        self.symbol_label.pack(anchor=tk.W, pady=(0, 8))
        
        entry_frame = tk.Frame(stock_lookup_card.content_frame, bg=theme['card_bg'])
        entry_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.symbol_entry = tk.Entry(entry_frame, font=('Segoe UI', 12),
                                    bg=theme['entry_bg'], fg=theme['entry_fg'],
                                    relief=tk.FLAT, bd=0, insertbackground=theme['fg'])
        self.symbol_entry.pack(fill=tk.X, ipady=10)
        self.symbol_entry.bind('<Return>', lambda e: self._analyze_stock())
        
        # Entry underline with focus animation
        self.entry_underline = tk.Frame(entry_frame, bg=theme['border'], height=2)
        self.entry_underline.pack(fill=tk.X, pady=(5, 0))
        
        def on_entry_focus_in(e):
            self.entry_underline.config(bg=theme['accent'], height=2)
        def on_entry_focus_out(e):
            self.entry_underline.config(bg=theme['border'], height=2)
        
        self.symbol_entry.bind('<FocusIn>', on_entry_focus_in)
        self.symbol_entry.bind('<FocusOut>', on_entry_focus_out)
        
        # Button frame for analyze and market scan
        btn_frame = tk.Frame(stock_lookup_card.content_frame, bg=theme['card_bg'])
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.analyze_btn = ModernButton(btn_frame, self.localization.t('analyze_stock'), 
                                   command=self._analyze_stock, theme=theme)
        self.analyze_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.scan_market_btn = ModernButton(btn_frame, self.localization.t('scan_market'), 
                                        command=self._scan_market, theme=theme)
        self.scan_market_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Search history dropdown
        history_frame = tk.Frame(stock_lookup_card.content_frame, bg=theme['card_bg'])
        history_frame.pack(fill=tk.X)
        
        history_label = tk.Label(history_frame, text=self.localization.t('recent_searches'), 
                                bg=theme['card_bg'], fg=theme['text_secondary'], 
                                font=('Segoe UI', 9))
        history_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.history_var = tk.StringVar()
        self.history_dropdown = ttk.Combobox(history_frame, textvariable=self.history_var,
                                           state='readonly', font=('Segoe UI', 10),
                                           width=20)
        self.history_dropdown.pack(fill=tk.X)
        self.history_dropdown.bind('<<ComboboxSelected>>', self._on_history_select)
        self._update_search_history()
        
        # Header card with stats
        header_card = ModernCard(self.predictions_frame, self.localization.t('prediction_statistics'), theme=theme, padding=20)
        header_card.pack(fill=tk.X, pady=(0, 20))
        
        stats = self.predictions_tracker.get_statistics()
        
        stats_inner = tk.Frame(header_card.content_frame, bg=theme['card_bg'])
        stats_inner.pack(fill=tk.X)
        
        # Modern stats display
        stats_grid = tk.Frame(stats_inner, bg=theme['card_bg'])
        stats_grid.pack(fill=tk.X)
        
        stat_items = [
            (self.localization.t('total'), stats['total_predictions'], theme['fg']),
            (self.localization.t('active'), stats['active'], theme['accent']),
            (self.localization.t('verified'), stats['verified'], theme['success']),
            (self.localization.t('accuracy'), f"{stats['accuracy']:.1f}%" if stats['verified'] > 0 else "N/A", 
             theme['accent'] if stats['verified'] > 0 else theme['text_secondary'])
        ]
        
        for i, (label, value, color) in enumerate(stat_items):
            stat_frame = tk.Frame(stats_grid, bg=theme['card_bg'])
            stat_frame.grid(row=0, column=i, padx=20, pady=10, sticky=tk.W+tk.E)
            
            tk.Label(stat_frame, text=label, bg=theme['card_bg'], 
                    fg=theme['text_secondary'], font=('Segoe UI', 9)).pack()
            tk.Label(stat_frame, text=str(value), bg=theme['card_bg'], 
                    fg=color, font=('Segoe UI', 16, 'bold')).pack()
        
        ModernButton(header_card.content_frame, self.localization.t('refresh_stats'), 
                    command=self._refresh_predictions, theme=theme).pack(pady=(15, 0))
        
        # Predictions list card
        list_card = ModernCard(self.predictions_frame, self.localization.t('all_predictions'), theme=theme, padding=20)
        list_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Treeview with modern styling
        list_inner = tk.Frame(list_card.content_frame, bg=theme['card_bg'])
        list_inner.pack(fill=tk.BOTH, expand=True)
        
        columns = ('ID', 'Symbol', 'Action', 'Entry', 'Target', 'Status', 'Result')
        self.predictions_tree = ttk.Treeview(list_inner, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.predictions_tree.heading(col, text=col, anchor=tk.CENTER)
            self.predictions_tree.column(col, width=120, anchor=tk.CENTER)
        
        # Modern scrollbar for predictions tree
        tree_scrollbar = ModernScrollbar(list_inner, command=self.predictions_tree.yview, theme=theme)
        self.predictions_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.predictions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        # Force initial update
        list_inner.update_idletasks()
        tree_scrollbar._update_slider()
        
        # Action buttons card
        actions_card = ModernCard(self.predictions_frame, "", theme=theme, padding=15)
        actions_card.pack(fill=tk.X)
        
        btn_inner = tk.Frame(actions_card.content_frame, bg=theme['card_bg'])
        btn_inner.pack(fill=tk.X)
        
        ModernButton(btn_inner, self.localization.t('save_current'), command=self._save_current_prediction, 
                    theme=theme).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ModernButton(btn_inner, self.localization.t('verify_all'), command=self._verify_all_predictions, 
                    theme=theme).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ModernButton(btn_inner, self.localization.t('clear_old'), command=self._clear_old_predictions, 
                    theme=theme).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def _show_predictions_tab(self):
        """Switch to predictions tab"""
        # Find predictions tab index (it's the last tab)
        try:
            # Predictions tab is typically the last one
            self.notebook.select(len(self.notebook.tabs()) - 1)
        except:
            # Fallback: find by text
            for i in range(self.notebook.index("end")):
                if self.localization.t('predictions') in self.notebook.tab(i, "text"):
                    self.notebook.select(i)
                    break
    
    def _save_current_prediction(self):
        """Save current analysis as a prediction"""
        if not self.current_analysis or 'error' in self.current_analysis:
            messagebox.showwarning(self.localization.t('no_analysis'), self.localization.t('please_analyze_stock'))
            return
        
        if not self.current_symbol:
            messagebox.showwarning(self.localization.t('no_symbol'), self.localization.t('no_symbol'))
            return
        
        recommendation = self.current_analysis.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        entry_price = recommendation.get('entry_price', 0)
        target_price = recommendation.get('target_price', 0)
        stop_loss = recommendation.get('stop_loss', 0)
        confidence = recommendation.get('confidence', 0)
        reasoning = self.current_analysis.get('reasoning', '')
        
        if entry_price <= 0:
            messagebox.showwarning(self.localization.t('invalid_prediction'), self.localization.t('invalid_prediction'))
            return
        
        strategy = self.strategy_var.get()
        
        prediction = self.predictions_tracker.add_prediction(
            self.current_symbol, strategy, action, entry_price, 
            target_price, stop_loss, confidence, reasoning
        )
        
        messagebox.showinfo(self.localization.t('prediction_saved'), 
                          f"{self.localization.t('prediction_saved')} #{prediction['id']} - {self.current_symbol}")
        self._update_predictions_display()
    
    def _verify_all_predictions(self):
        """Manually verify all active predictions"""
        def verify_in_background():
            try:
                self.root.after(0, self._update_status, self.localization.t('verifying_predictions'))
                result = self.predictions_tracker.verify_all_active_predictions(self.data_fetcher)
                self.root.after(0, self._show_verification_results, result)
                self.root.after(0, self._update_predictions_display)
            except Exception as e:
                logger.error(f"Error verifying predictions: {e}")
                self.root.after(0, lambda: messagebox.showerror(self.localization.t('error'), 
                    f"{self.localization.t('error')}: {str(e)}"))
        
        thread = threading.Thread(target=verify_in_background)
        thread.daemon = True
        thread.start()
    
    def _clear_old_predictions(self):
        """Clear old verified predictions"""
        if messagebox.askyesno(self.localization.t('confirm'), self.localization.t('clear_all_verified')):
            verified = self.predictions_tracker.get_verified_predictions()
            self.predictions_tracker.predictions = [
                p for p in self.predictions_tracker.predictions if p['status'] != 'verified'
            ]
            self.predictions_tracker.save()
            messagebox.showinfo(self.localization.t('cleared'), 
                             self.localization.t('removed_predictions').format(count=len(verified)))
            self._update_predictions_display()
    
    def _update_predictions_display(self):
        """Update the predictions display with modern styling"""
        theme = self.theme_manager.get_theme()
        stats = self.predictions_tracker.get_statistics()
        
        # Update stats label
        if stats['verified'] > 0:
            self.pred_stats_label.config(
                text=f"{stats['accuracy']:.1f}% {self.localization.t('accuracy')}",
                fg=theme['success'] if stats['accuracy'] >= 70 else theme['warning'] if stats['accuracy'] >= 50 else theme['error']
            )
        else:
            self.pred_stats_label.config(text=self.localization.t('no_verified_predictions'), fg=theme['text_secondary'])
        
        # Clear tree
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
        
        # Add all predictions with modern formatting
        for pred in self.predictions_tracker.predictions:
            status = pred['status']
            result = self.localization.t('pending')
            if pred['verified']:
                result = f"✓ {self.localization.t('correct')}" if pred['was_correct'] else f"✗ {self.localization.t('incorrect')}"
            
            values = (
                pred['id'],
                pred['symbol'],
                pred['action'],
                self.localization.format_currency(pred['entry_price']),
                self.localization.format_currency(pred['target_price']),
                status.title(),
                result
            )
            
            item = self.predictions_tree.insert('', tk.END, values=values)
            
            # Tag items for color coding
            if pred['verified']:
                if pred['was_correct']:
                    self.predictions_tree.set(item, 'Result', f"✓ {self.localization.t('correct')}")
                    self.predictions_tree.item(item, tags=('success',))
                else:
                    self.predictions_tree.set(item, 'Result', f"✗ {self.localization.t('incorrect')}")
                    self.predictions_tree.item(item, tags=('error',))
            else:
                self.predictions_tree.item(item, tags=('pending',))
        
        # Configure tags for colors
        self.predictions_tree.tag_configure('success', foreground=theme['success'])
        self.predictions_tree.tag_configure('error', foreground=theme['error'])
        self.predictions_tree.tag_configure('pending', foreground=theme['text_secondary'])
    
    def _refresh_predictions(self):
        """Refresh predictions display"""
        self._update_predictions_display()
    
    def _on_strategy_change(self):
        """Handle strategy change with animation"""
        self.current_strategy = self.strategy_var.get()
        self.preferences.set_strategy(self.current_strategy)
        self._update_strategy_visual_feedback()
        logger.info(f"Strategy changed to: {self.current_strategy}")
    
    def _analyze_stock(self):
        """Analyze a stock based on selected strategy"""
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning(self.localization.t('input_error'), self.localization.t('input_error'))
            return
        
        # Save to search history
        self.preferences.add_search(symbol)
        self._update_search_history()
        
        # Show loading screen with processes
        theme = self.theme_manager.get_theme()
        if self.loading_screen:
            self.loading_screen.theme = theme
        else:
            self.loading_screen = LoadingScreen(self.root, theme)
        
        # Reset cancel flag
        self.analysis_cancelled = False
        
        # Estimate time based on strategy (rough estimates)
        strategy = self.strategy_var.get()
        if strategy == "trading":
            estimated_time = 15  # seconds
        elif strategy == "mixed":
            estimated_time = 25  # seconds
        else:  # investing
            estimated_time = 20  # seconds
        
        processes = [f"Fetching data for {symbol}"]
        self.loading_screen.show(
            f"Analyzing {symbol}...", 
            processes=processes,
            cancel_callback=self._cancel_analysis,
            estimated_time=estimated_time
        )
        
        # Disable button during analysis
        self.symbol_entry.config(state='disabled')
        self.analyze_btn.config(state='disabled')
        
        # Run analysis in thread to prevent UI freezing
        thread = threading.Thread(target=self._perform_analysis, args=(symbol,))
        thread.daemon = True
        thread.start()
    
    def _scan_market(self):
        """Scan market for recommended stocks"""
        # Reset cancel flag
        self.market_scan_cancelled = False
        
        # Show loading screen with processes
        theme = self.theme_manager.get_theme()
        if self.loading_screen:
            self.loading_screen.theme = theme
        else:
            self.loading_screen = LoadingScreen(self.root, theme)
        
        processes = ["Scanning market stocks..."]
        # Market scan takes longer - estimate 60 seconds for 10 stocks
        self.loading_screen.show(
            self.localization.t('scanning_market'), 
            processes=processes,
            cancel_callback=self._cancel_market_scan,
            estimated_time=60
        )
        
        # Disable button during scan
        self.scan_market_btn.config(state='disabled')
        
        # Run scan in thread
        thread = threading.Thread(target=self._perform_market_scan)
        thread.daemon = True
        thread.start()
    
    def _cancel_market_scan(self):
        """Cancel the market scan"""
        self.market_scan_cancelled = True
        logger.info("Market scan cancelled by user")
    
    def _perform_market_scan(self):
        """Perform market scan (runs in background thread)"""
        try:
            strategy = self.strategy_var.get()
            processes = ["Scanning market stocks..."]
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            if self.market_scan_cancelled:
                return
            
            recommendations = self.market_scanner.scan_market(strategy, max_results=10)
            
            if self.market_scan_cancelled:
                return
            
            processes.append(f"Found {len(recommendations)} recommendations")
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            # Update UI in main thread
            self.root.after(0, self._display_market_recommendations, recommendations)
        except Exception as e:
            if not self.market_scan_cancelled:
                logger.error(f"Error in market scan: {e}")
                self.root.after(0, self._show_error, f"Error scanning market: {str(e)}")
        finally:
            # Hide loading screen and re-enable button
            self.root.after(0, lambda: self.loading_screen.hide() if self.loading_screen else None)
            self.root.after(0, lambda: self.scan_market_btn.config(state='normal'))
    
    def _display_market_recommendations(self, recommendations: list):
        """Display market recommendations"""
        if not recommendations:
            messagebox.showinfo(self.localization.t('market_recommendations'), 
                              self.localization.t('no_recommendations'))
            return
        
        # Create a new window to display recommendations
        theme = self.theme_manager.get_theme()
        rec_window = tk.Toplevel(self.root)
        rec_window.title(self.localization.t('market_recommendations'))
        rec_window.config(bg=theme['bg'])
        rec_window.geometry("800x600")
        
        # Header
        header = tk.Label(rec_window, text=self.localization.t('market_recommendations'),
                         bg=theme['bg'], fg=theme['accent'], font=('Segoe UI', 18, 'bold'))
        header.pack(pady=20)
        
        # Scrollable frame
        canvas = tk.Canvas(rec_window, bg=theme['bg'], highlightthickness=0)
        scrollbar = ModernScrollbar(rec_window, command=canvas.yview, theme=theme)
        scroll_frame = tk.Frame(canvas, bg=theme['bg'])
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW)
        
        # Display each recommendation
        for i, rec in enumerate(recommendations):
            card = ModernCard(scroll_frame, f"{rec['symbol']} - {rec['name']}", theme=theme, padding=15)
            card.pack(fill=tk.X, padx=20, pady=10)
            
            info_text = f"Price: {self.localization.format_currency(rec['price'])}\n"
            info_text += f"Confidence: {rec['confidence']:.1f}%\n"
            info_text += f"Entry: {self.localization.format_currency(rec['entry_price'])}\n"
            info_text += f"Target: {self.localization.format_currency(rec['target_price'])}\n"
            info_text += f"Stop Loss: {self.localization.format_currency(rec['stop_loss'])}\n\n"
            info_text += f"Reasoning: {rec['reasoning']}"
            
            info_label = tk.Label(card.content_frame, text=info_text, bg=theme['card_bg'],
                                fg=theme['fg'], font=('Segoe UI', 10), justify=tk.LEFT)
            info_label.pack(anchor=tk.W)
            
            # Button to analyze this stock
            analyze_btn = ModernButton(card.content_frame, f"Analyze {rec['symbol']}",
                                      command=lambda s=rec['symbol']: self._analyze_symbol_from_rec(s, rec_window),
                                      theme=theme)
            analyze_btn.pack(fill=tk.X, pady=(10, 0))
        
        def configure_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scroll_frame.bind('<Configure>', configure_scroll)
    
    def _analyze_symbol_from_rec(self, symbol: str, window):
        """Analyze a symbol from recommendations window"""
        window.destroy()
        self.symbol_entry.delete(0, tk.END)
        self.symbol_entry.insert(0, symbol)
        self._analyze_stock()
    
    def _update_search_history(self):
        """Update search history dropdown"""
        history = self.preferences.get_search_history()
        if hasattr(self, 'history_dropdown'):
            self.history_dropdown['values'] = history
            if history:
                self.history_dropdown.current(0)
    
    def _on_history_select(self, event=None):
        """Handle history selection"""
        selected = self.history_var.get()
        if selected:
            self.symbol_entry.delete(0, tk.END)
            self.symbol_entry.insert(0, selected)
    
    def _animate_loading(self):
        """Animate loading indicator (deprecated - using loading screen now)"""
        # This method is kept for compatibility but loading screen is used instead
        pass
    
    def _cancel_analysis(self):
        """Cancel the current analysis"""
        self.analysis_cancelled = True
        logger.info("Analysis cancelled by user")
    
    def _perform_analysis(self, symbol: str):
        """Perform stock analysis (runs in background thread)"""
        try:
            # Check if cancelled
            if self.analysis_cancelled:
                return
            
            # Update processes
            processes = [f"Fetching stock data for {symbol}"]
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            self.root.after(0, self._update_status, self.localization.t('fetching_data', symbol=symbol))
            
            # Fetch stock data
            stock_data = self.data_fetcher.fetch_stock_data(symbol)
            
            if self.analysis_cancelled:
                return
            
            processes.append(f"Fetching price history for {symbol}")
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            history_data = self.data_fetcher.fetch_stock_history(symbol)
            
            if self.analysis_cancelled:
                return
            
            self.current_symbol = symbol
            self.current_data = stock_data
            
            strategy = self.strategy_var.get()
            
            # Perform analysis based on strategy
            if strategy == "trading":
                processes.append("Performing technical analysis...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                self.root.after(0, self._update_status, self.localization.t('performing_trading_analysis'))
                analysis = self.trading_analyzer.analyze(stock_data, history_data)
            elif strategy == "mixed":
                if self.analysis_cancelled:
                    return
                processes.append("Fetching financial data...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                financials_data = self.data_fetcher.fetch_financials(symbol)
                if self.analysis_cancelled:
                    return
                processes.append("Performing mixed analysis...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                self.root.after(0, self._update_status, self.localization.t('performing_mixed_analysis'))
                analysis = self.mixed_analyzer.analyze(stock_data, financials_data, history_data)
            else:  # investing
                if self.analysis_cancelled:
                    return
                processes.append("Fetching financial data...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                financials_data = self.data_fetcher.fetch_financials(symbol)
                if self.analysis_cancelled:
                    return
                processes.append("Performing fundamental analysis...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                self.root.after(0, self._update_status, self.localization.t('performing_fundamental_analysis'))
                analysis = self.investing_analyzer.analyze(stock_data, financials_data, history_data)
            
            if self.analysis_cancelled:
                return
            
            processes.append("Generating recommendations...")
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            self.current_analysis = analysis
            
            # Update UI in main thread
            self.root.after(0, self._display_analysis, analysis, history_data)
            
        except Exception as e:
            if not self.analysis_cancelled:
                logger.error(f"Error in analysis: {e}")
                self.root.after(0, self._show_error, f"Error analyzing stock: {str(e)}")
        finally:
            # Hide loading screen and re-enable controls
            self.root.after(0, lambda: self.loading_screen.hide() if self.loading_screen else None)
            self.root.after(0, lambda: self.symbol_entry.config(state='normal'))
            self.root.after(0, lambda: self.analyze_btn.config(state='normal'))
    
    def _update_status(self, message: str):
        """Update status message with animation"""
        theme = self.theme_manager.get_theme()
        self.results_title.config(text=message, fg=theme['accent'])
    
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
        
        theme = self.theme_manager.get_theme()
        
        header = f"{'='*60}\n"
        header += f"RECOMMENDATION: {action} (Confidence: {confidence}%)\n"
        header += f"{'='*60}\n\n"
        
        self.analysis_text.insert(tk.END, header)
        self.analysis_text.insert(tk.END, reasoning)
        
        # Highlight recommendation with modern styling
        self.analysis_text.tag_add("header", "1.0", "3.0")
        self.analysis_text.tag_config("header", font=('Segoe UI', 12, 'bold'), 
                                     foreground=theme['accent'])
        
        # Update title with animation
        symbol = self.current_symbol or "Stock"
        self._fade_title(f"{symbol} - Analysis Results")
        
        # Generate and display charts
        self._display_charts(history_data, analysis.get('indicators', {}))
        
        # Store for potential calculation
        self.current_analysis = analysis
    
    def _fade_title(self, new_text: str):
        """Animate title change"""
        theme = self.theme_manager.get_theme()
        self.results_title.config(text=new_text, fg=theme['fg'])
    
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
            messagebox.showwarning(self.localization.t('no_analysis'), self.localization.t('please_analyze_stock'))
            return
        
        try:
            budget = float(self.budget_entry.get())
            if budget <= 0:
                raise ValueError(self.localization.t('balance_must_positive'))
        except ValueError as e:
            messagebox.showerror(self.localization.t('input_error'), f"{self.localization.t('input_error')}: {str(e)}")
            return
        
        recommendation = self.current_analysis.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        entry_price = recommendation.get('entry_price', 0)
        target_price = recommendation.get('target_price', 0)
        stop_loss = recommendation.get('stop_loss', 0)
        
        if entry_price <= 0:
            messagebox.showwarning(self.localization.t('warning'), self.localization.t('warning'))
            return
        
        symbol = self.current_symbol or "UNKNOWN"
        potential = self.portfolio.calculate_potential_trade(
            symbol, budget, entry_price, target_price, stop_loss, action
        )
        
        if 'error' in potential:
            self.potential_text.delete(1.0, tk.END)
            self.potential_text.insert(tk.END, f"{self.localization.t('error')}: {potential['error']}")
            return
        
        # Format output with better styling and localization
        output = f"{'='*60}\n"
        output += f"{self.localization.t('potential_trade')} - {potential['symbol']}\n"
        output += f"{'='*60}\n\n"
        output += f"{self.localization.t('action')}: {potential['action']}\n"
        output += f"{self.localization.t('shares')}: {potential['shares']}\n"
        output += f"{self.localization.t('budget_used')}: {self.localization.format_currency(potential['budget_used'])}\n\n"
        output += f"{self.localization.t('entry_price')}: {self.localization.format_currency(potential['entry_price'])}\n"
        output += f"{self.localization.t('target_price')}: {self.localization.format_currency(potential['target_price'])}\n"
        output += f"{self.localization.t('stop_loss')}: {self.localization.format_currency(potential['stop_loss'])}\n\n"
        output += f"{self.localization.t('potential_win')}: {self.localization.format_currency(potential['potential_win'])} "
        output += f"({potential['potential_win_percent']:.2f}%)\n"
        output += f"{self.localization.t('potential_loss')}: {self.localization.format_currency(potential['potential_loss'])} "
        output += f"({potential['potential_loss_percent']:.2f}%)\n\n"
        output += f"{self.localization.t('risk_reward_ratio')}: {potential['risk_reward_ratio']:.2f}\n"
        
        self.potential_text.delete(1.0, tk.END)
        self.potential_text.insert(tk.END, output)
    
    def _set_balance(self):
        """Set initial portfolio balance"""
        dialog = tk.Toplevel(self.root)
        dialog.title(self.localization.t('set_initial_balance'))
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        currency_symbol = self.localization.format_currency_symbol()
        ttk.Label(dialog, text=f"{self.localization.t('initial_balance')} ({currency_symbol}):").pack(pady=10)
        balance_entry = ttk.Entry(dialog, width=20)
        balance_entry.pack(pady=5)
        balance_entry.insert(0, str(self.portfolio.balance))
        balance_entry.focus()
        
        def save_balance():
            try:
                balance = float(balance_entry.get())
                if balance < 0:
                    raise ValueError(self.localization.t('balance_must_positive'))
                self.portfolio.set_balance(balance)
                self._update_portfolio_display()
                dialog.destroy()
                messagebox.showinfo(self.localization.t('success'), self.localization.t('balance_updated'))
            except ValueError as e:
                messagebox.showerror(self.localization.t('error'), f"{self.localization.t('invalid_balance')}: {str(e)}")
        
        ttk.Button(dialog, text=self.localization.t('save'), command=save_balance).pack(pady=10)
        balance_entry.bind('<Return>', lambda e: save_balance())
    
    def _update_portfolio_display(self):
        """Update portfolio display with modern formatting"""
        theme = self.theme_manager.get_theme()
        stats = self.portfolio.get_statistics()
        self.portfolio_label.config(text=self.localization.format_currency(stats['balance']), fg=theme['accent'])
        self.stats_label.config(text=f"{self.localization.t('wins')}: {stats['wins']} | {self.localization.t('losses')}: {stats['losses']} | "
                                     f"{self.localization.t('win_rate')}: {stats['win_rate']:.1f}%",
                               fg=theme['text_secondary'])
    
    def _show_error(self, message: str):
        """Show error message"""
        theme = self.theme_manager.get_theme()
        messagebox.showerror(self.localization.t('error'), message)
        self.results_title.config(text=self.localization.t('stock_analysis_results'), fg=theme['fg'])


def main():
    """Main entry point"""
    root = tk.Tk()
    app = StockerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

