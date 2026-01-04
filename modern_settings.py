"""
Modern Settings Components for Stocker App
Beautiful dropdown buttons for theme, language, currency
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List


class ModernDropdownButton(tk.Frame):
    """Modern dropdown button with smooth animations"""
    
    def __init__(self, parent, label: str, options: List[str], 
                 current_value: str, callback: Callable, theme: dict):
        super().__init__(parent, bg=theme['secondary_bg'])
        
        self.theme = theme
        self.options = options
        self.current_value = current_value
        self.callback = callback
        self.is_open = False
        
        # Label
        self.label_widget = tk.Label(self, text=label, bg=theme['secondary_bg'],
                                     fg=theme['text_secondary'], font=('Segoe UI', 9))
        self.label_widget.pack(side=tk.LEFT, padx=(0, 8))
        
        # Button frame
        self.button_frame = tk.Frame(self, bg=theme['secondary_bg'])
        self.button_frame.pack(side=tk.LEFT)
        
        # Main button
        self.main_button = tk.Button(
            self.button_frame,
            text=current_value.upper(),
            command=self._toggle_dropdown,
            bg=theme['accent'],
            fg=theme['button_fg'],
            font=('Segoe UI', 9, 'bold'),
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            padx=15,
            pady=8,
            activebackground=theme['accent_hover'],
            activeforeground=theme['button_fg']
        )
        self.main_button.pack()
        
        # Dropdown menu (initially hidden)
        self.dropdown_frame = None
        self.dropdown_window = None
    
    def _toggle_dropdown(self):
        """Toggle dropdown menu"""
        if self.is_open:
            self._close_dropdown()
        else:
            self._open_dropdown()
    
    def _open_dropdown(self):
        """Open dropdown menu with animation"""
        self.is_open = True
        
        # Create dropdown window
        self.dropdown_window = tk.Toplevel(self)
        self.dropdown_window.overrideredirect(True)
        self.dropdown_window.attributes('-topmost', True)
        
        # Position it below the button
        x = self.winfo_rootx() + self.button_frame.winfo_x()
        y = self.winfo_rooty() + self.button_frame.winfo_y() + self.main_button.winfo_height()
        
        self.dropdown_window.geometry(f"120x{len(self.options) * 35}+{x}+{y}")
        self.dropdown_window.config(bg=self.theme['card_bg'])
        
        # Create dropdown frame
        self.dropdown_frame = tk.Frame(self.dropdown_window, bg=self.theme['card_bg'])
        self.dropdown_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Add option buttons
        for option in self.options:
            btn = tk.Button(
                self.dropdown_frame,
                text=option.upper(),
                command=lambda o=option: self._select_option(o),
                bg=self.theme['card_bg'],
                fg=self.theme['fg'],
                font=('Segoe UI', 9),
                relief=tk.FLAT,
                bd=0,
                cursor='hand2',
                padx=10,
                pady=8,
                anchor=tk.W,
                activebackground=self.theme['accent'],
                activeforeground=self.theme['button_fg']
            )
            btn.pack(fill=tk.X, pady=2)
            
            # Highlight current selection
            if option == self.current_value:
                btn.config(bg=self.theme['accent'], fg=self.theme['button_fg'])
        
        # Bind click outside to close
        self.dropdown_window.bind('<FocusOut>', lambda e: self._close_dropdown())
        self.dropdown_window.focus_set()
    
    def _close_dropdown(self):
        """Close dropdown menu"""
        if self.dropdown_window:
            self.dropdown_window.destroy()
            self.dropdown_window = None
            self.dropdown_frame = None
        self.is_open = False
    
    def _select_option(self, option: str):
        """Select an option"""
        self.current_value = option
        self.main_button.config(text=option.upper())
        self._close_dropdown()
        
        # Call callback
        if self.callback:
            self.callback(option)
    
    def set_value(self, value: str):
        """Set current value"""
        self.current_value = value
        self.main_button.config(text=value.upper())



