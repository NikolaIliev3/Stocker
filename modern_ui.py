"""
Modern UI Components for Stocker App
Beautiful, modern UI elements with animations
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class ModernCard(tk.Frame):
    """Modern card component with shadow effect"""
    def __init__(self, parent, title: str = "", padding: int = 20, **kwargs):
        theme = kwargs.pop('theme', None)
        super().__init__(parent, **kwargs)
        
        if theme:
            self.config(bg=theme['card_bg'], relief=tk.FLAT, bd=0)
            # Create subtle border for card effect
            self.config(highlightbackground=theme['border'], highlightthickness=1)
            title_fg = theme.get('fg', '#000000')
        else:
            self.config(bg='#FFFFFF', relief=tk.FLAT, bd=0)
            title_fg = '#000000'
        
        if title:
            title_label = tk.Label(self, text=title, font=('Segoe UI', 13, 'bold'),
                                 bg=self.cget('bg'), fg=title_fg)
            title_label.pack(anchor=tk.W, pady=(padding, 0), padx=padding)
        
        self.content_frame = tk.Frame(self, bg=self.cget('bg'))
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=padding, pady=padding)


class ModernButton(tk.Button):
    """Modern button with gradient effect and animations"""
    def __init__(self, parent, text: str, command: Optional[Callable] = None,
                 theme: dict = None, **kwargs):
        if theme:
            bg = theme.get('button_bg', '#6366F1')
            fg = theme.get('button_fg', '#FFFFFF')
            hover_bg = theme.get('button_hover', '#4F46E5')
        else:
            bg = kwargs.pop('bg', '#6366F1')
            fg = kwargs.pop('fg', '#FFFFFF')
            hover_bg = kwargs.pop('hover_bg', '#4F46E5')
        
        # Store original command
        self._command = command
        
        super().__init__(parent, text=text, command=self._animated_command,
                        bg=bg, fg=fg, font=('Segoe UI', 10, 'bold'),
                        relief=tk.FLAT, bd=0, cursor='hand2',
                        padx=20, pady=12, **kwargs)
        
        self.original_bg = bg
        self.hover_bg = hover_bg
        self.original_padx = 20
        self.original_pady = 12
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
    
    def _animated_command(self):
        """Execute command with smooth animation"""
        # Scale down animation
        self._scale_animation(0.95)
        # Execute command
        if self._command:
            self._command()
        # Scale back
        self.after(100, lambda: self._scale_animation(1.0))
    
    def _scale_animation(self, scale: float):
        """Smooth scale animation"""
        new_padx = int(self.original_padx * scale)
        new_pady = int(self.original_pady * scale)
        self.config(padx=new_padx, pady=new_pady)
    
    def _on_enter(self, event):
        """Smooth hover animation"""
        self._fade_color(self.original_bg, self.hover_bg, steps=5)
    
    def _on_leave(self, event):
        """Smooth leave animation"""
        self._fade_color(self.hover_bg, self.original_bg, steps=5)
    
    def _fade_color(self, from_color: str, to_color: str, steps: int = 5):
        """Smooth color transition"""
        # Simplified - just change color directly
        self.config(bg=to_color)


class GradientLabel(tk.Label):
    """Label with gradient text effect (simulated)"""
    def __init__(self, parent, text: str, theme: dict = None, **kwargs):
        if theme:
            fg = theme.get('accent', '#6366F1')
        else:
            fg = kwargs.get('fg', '#6366F1')
        
        super().__init__(parent, text=text, fg=fg,
                       font=kwargs.get('font', ('Segoe UI', 16, 'bold')),
                       bg=kwargs.get('bg', parent.cget('bg') if hasattr(parent, 'cget') else '#FFFFFF'),
                       **{k: v for k, v in kwargs.items() if k not in ['font', 'fg', 'bg']})


class AnimatedEntry(tk.Entry):
    """Entry with focus animation"""
    def __init__(self, parent, theme: dict = None, **kwargs):
        if theme:
            bg = theme.get('entry_bg', '#FFFFFF')
            fg = theme.get('entry_fg', '#000000')
            border = theme.get('border', '#E5E7EB')
        else:
            bg = kwargs.pop('bg', '#FFFFFF')
            fg = kwargs.pop('fg', '#000000')
            border = kwargs.pop('border', '#E5E7EB')
        
        super().__init__(parent, bg=bg, fg=fg, font=('Segoe UI', 10),
                        relief=tk.FLAT, bd=0, insertbackground=fg, **kwargs)
        
        self.normal_border = border
        self.focus_border = theme.get('accent', '#6366F1') if theme else '#6366F1'
        
        # Create border effect
        self.border_frame = tk.Frame(parent, bg=self.normal_border, height=2)
        self.border_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.bind('<FocusIn>', self._on_focus_in)
        self.bind('<FocusOut>', self._on_focus_out)
    
    def _on_focus_in(self, event):
        self.border_frame.config(bg=self.focus_border)
    
    def _on_focus_out(self, event):
        self.border_frame.config(bg=self.normal_border)


class ModernScrollbar(tk.Canvas):
    """Modern, sleek scrollbar with smooth animations"""
    def __init__(self, parent, command, theme: dict = None, **kwargs):
        self.command = command
        self.theme = theme or {}
        
        # Get colors from theme - make scrollbar more visible
        # Use slightly different colors for better visibility
        bg_color = self.theme.get('bg', '#1F2937')
        # Make trough slightly lighter/darker than bg for visibility
        # Check if it's a dark theme by looking at bg color brightness
        bg_str = str(bg_color).lower()
        if '#000' in bg_str or '#1' in bg_str or '#2' in bg_str or '#3' in bg_str:
            # Dark theme
            self.trough_color = '#2A2A2A'  # Slightly lighter than bg
        else:
            # Light theme
            self.trough_color = '#E5E5E5'  # Slightly darker than bg for light theme
        
        self.slider_color = self.theme.get('secondary_bg', '#374151')
        self.slider_hover_color = self.theme.get('accent', '#6366F1')
        self.slider_active_color = self.theme.get('accent_hover', '#4F46E5')
        
        # Scrollbar dimensions - make it more visible
        self.width = 16
        self.min_slider_height = 40
        
        super().__init__(parent, width=self.width, bg=self.trough_color,
                        highlightthickness=0, relief=tk.FLAT, bd=0, **kwargs)
        
        self.slider = None
        self.slider_y = 0
        self.slider_height = 0
        self.is_dragging = False
        self.drag_start_y = 0
        self.hover_state = False
        
        self.bind('<Configure>', self._on_configure)
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        
        # Bind mousewheel
        self.bind('<MouseWheel>', self._on_mousewheel)
    
    def _on_configure(self, event=None):
        """Update scrollbar when size changes"""
        self.update_idletasks()
        self._update_slider()
    
    def _update_slider(self):
        """Update slider position and size"""
        self.delete("all")
        
        canvas_height = self.winfo_height()
        if canvas_height <= 0:
            # Draw empty scrollbar background
            self.create_rectangle(0, 0, self.width, canvas_height, fill=self.trough_color, outline="")
            return
        
        # Get scroll info from command
        try:
            first, last = self.command("get")
        except:
            first, last = 0.0, 1.0
        
        # Always draw the trough (background)
        self.create_rectangle(0, 0, self.width, canvas_height, fill=self.trough_color, outline="")
        
        if first == 0.0 and last == 1.0:
            # No scrolling needed, but still show the trough
            return
        
        # Calculate slider dimensions
        total_height = canvas_height
        visible_ratio = last - first
        self.slider_height = max(self.min_slider_height, int(total_height * visible_ratio))
        self.slider_y = int(total_height * first)
        
        # Draw slider with rounded corners effect
        slider_color = self.slider_active_color if self.is_dragging else (
            self.slider_hover_color if self.hover_state else self.slider_color
        )
        
        # Draw rounded rectangle (simulated with oval corners)
        padding = 3
        x1, y1 = padding, self.slider_y + padding
        x2, y2 = self.width - padding, self.slider_y + self.slider_height - padding
        
        # Main rectangle
        self.create_rectangle(x1, y1, x2, y2, fill=slider_color, outline="", width=0)
        
        # Rounded corners effect (small circles)
        corner_radius = 3
        self.create_oval(x1, y1, x1 + corner_radius * 2, y1 + corner_radius * 2,
                        fill=slider_color, outline="")
        self.create_oval(x2 - corner_radius * 2, y1, x2, y1 + corner_radius * 2,
                        fill=slider_color, outline="")
        self.create_oval(x1, y2 - corner_radius * 2, x1 + corner_radius * 2, y2,
                        fill=slider_color, outline="")
        self.create_oval(x2 - corner_radius * 2, y2 - corner_radius * 2, x2, y2,
                        fill=slider_color, outline="")
    
    def set(self, first, last):
        """Set scrollbar position (called by scrollable widget)"""
        self._update_slider()
    
    def _on_click(self, event):
        """Handle click on scrollbar"""
        y = event.y
        canvas_height = self.winfo_height()
        
        # Check if clicked on slider
        if self.slider_y <= y <= self.slider_y + self.slider_height:
            self.is_dragging = True
            self.drag_start_y = y - self.slider_y
        else:
            # Jump to clicked position
            new_first = (y - self.slider_height / 2) / canvas_height
            new_first = max(0.0, min(1.0, new_first))
            self.command("moveto", new_first)
        
        self._update_slider()
    
    def _on_drag(self, event):
        """Handle drag on scrollbar"""
        if self.is_dragging:
            y = event.y - self.drag_start_y
            canvas_height = self.winfo_height()
            new_first = y / canvas_height
            new_first = max(0.0, min(1.0 - (self.slider_height / canvas_height), new_first))
            self.command("moveto", new_first)
            self._update_slider()
    
    def _on_release(self, event):
        """Handle release on scrollbar"""
        self.is_dragging = False
        self._update_slider()
    
    def _on_enter(self, event):
        """Handle mouse enter"""
        self.hover_state = True
        self._update_slider()
    
    def _on_leave(self, event):
        """Handle mouse leave"""
        self.hover_state = False
        self._update_slider()
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        delta = -1 * (event.delta / 120)
        self.command("scroll", delta, "units")
        self._update_slider()

