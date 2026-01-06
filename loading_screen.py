"""
Loading Screen component for Stocker App
Beautiful animated loading overlay with smooth animations
"""
import tkinter as tk
from typing import Optional, Callable
import math
import time


class LoadingScreen:
    """Animated loading screen overlay with smooth animations and process status"""
    
    def __init__(self, parent, theme: dict):
        self.parent = parent
        self.theme = theme
        self.overlay = None
        self.canvas = None
        self.animation_id = None
        self.angle = 0
        self.pulse_scale = 1.0
        self.pulse_direction = 1
        self.fade_alpha = 0.0
        self.fade_direction = 1
        self._current_message = "Loading..."
        self._processes = []  # List of current processes
        self.process_frame = None
        self.cancel_callback = None  # Callback function for cancel
        self.start_time = None
        self.estimated_time = None
        self.time_label = None
        self.cancel_button = None
        self.is_cancelled = False
        
    def show(self, message: str = "Loading...", processes: list = None, 
             cancel_callback: Callable = None, estimated_time: float = None):
        """Show loading screen with smooth fade-in and process status
        
        Args:
            message: Main loading message
            processes: List of current processes
            cancel_callback: Function to call when cancel is clicked
            estimated_time: Estimated time in seconds until completion
        """
        if self.overlay:
            self.hide()
        
        # Store message and processes for animation
        self._current_message = message
        self._processes = processes or []
        self.cancel_callback = cancel_callback
        self.estimated_time = estimated_time
        self.start_time = time.time()
        self.is_cancelled = False
        
        # Reset animation state
        self.angle = 0
        self.pulse_scale = 1.0
        self.pulse_direction = 1
        self.fade_alpha = 0.0
        self.fade_direction = 1
        
        # Create overlay (non-blocking, minimizable window)
        self.overlay = tk.Toplevel(self.parent)
        self.overlay.title("Loading...")
        self.overlay.attributes('-topmost', False)  # Don't force on top
        # Don't use grab_set() - allows other windows to be used
        # Don't use overrideredirect - allows normal window controls (minimize, close)
        # Make it minimizable and allow normal window operations
        self.overlay.protocol("WM_DELETE_WINDOW", self._on_minimize)  # Handle minimize/close
        
        # Get parent dimensions
        self.parent.update_idletasks()
        width = self.parent.winfo_width() or 1400
        height = self.parent.winfo_height() or 900
        x = self.parent.winfo_x()
        y = self.parent.winfo_y()
        
        # Make it a smaller, minimizable window instead of full screen
        # Position it in the center, but allow it to be moved/minimized
        window_width = 500
        window_height = 400
        center_x = x + (width - window_width) // 2
        center_y = y + (height - window_height) // 2
        self.overlay.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.overlay.minsize(400, 300)  # Allow resizing but set minimum size
        self.overlay.resizable(True, True)  # Allow resizing
        
        # Create semi-transparent backdrop
        bg_color = self.theme['bg']
        self.overlay.config(bg=bg_color)
        
        # Main container
        main_container = tk.Frame(self.overlay, bg=bg_color)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for animation (centered)
        canvas_container = tk.Frame(main_container, bg=bg_color)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg=bg_color, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Process status frame (below spinner)
        self.process_frame = tk.Frame(main_container, bg=bg_color)
        self.process_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Time estimation frame
        time_frame = tk.Frame(main_container, bg=bg_color)
        time_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.time_label = tk.Label(time_frame, 
                                   text="",
                                   bg=bg_color, fg=self.theme['text_secondary'],
                                   font=('Segoe UI', 10))
        self.time_label.pack()
        
        # Cancel button frame
        if self.cancel_callback:
            button_frame = tk.Frame(main_container, bg=bg_color)
            button_frame.pack(fill=tk.X, pady=(0, 50))
            
            cancel_btn = tk.Button(button_frame, 
                                   text="Cancel",
                                   command=self._on_cancel,
                                   bg=self.theme['error'],
                                   fg='#FFFFFF',
                                   font=('Segoe UI', 11, 'bold'),
                                   relief=tk.FLAT,
                                   padx=30, pady=10,
                                   cursor='hand2')
            cancel_btn.pack()
            self.cancel_button = cancel_btn
        
        # Draw loading animation
        self._draw_loading(message)
        self._update_processes()
        self._update_time_estimate()
        
        # Start animation with smooth fade-in
        self._animate()
    
    def _on_cancel(self):
        """Handle cancel button click"""
        self.is_cancelled = True
        if self.cancel_callback:
            self.cancel_callback()
        self.hide()
    
    def _on_minimize(self):
        """Handle window minimize/close - just minimize instead of closing"""
        if self.overlay:
            self.overlay.iconify()  # Minimize instead of closing
    
    def _update_time_estimate(self):
        """Update time estimation display"""
        if not self.time_label or not self.estimated_time:
            return
        
        if self.is_cancelled:
            self.time_label.config(text="Cancelling...", fg=self.theme['error'])
            return
        
        elapsed = time.time() - self.start_time
        remaining = max(0, self.estimated_time - elapsed)
        
        if remaining > 0:
            if remaining < 60:
                time_text = f"Estimated time remaining: {int(remaining)} seconds"
            else:
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                time_text = f"Estimated time remaining: {minutes}m {seconds}s"
            self.time_label.config(text=time_text, fg=self.theme['text_secondary'])
        else:
            self.time_label.config(text="Finishing up...", fg=self.theme['accent'])
    
    def update_processes(self, processes: list):
        """Update the process list"""
        self._processes = processes or []
        if self.process_frame:
            self._update_processes()
    
    def _update_processes(self):
        """Update the process status display"""
        if not self.process_frame:
            return
        
        # Clear existing process labels
        for widget in self.process_frame.winfo_children():
            widget.destroy()
        
        if not self._processes:
            return
        
        # Create process status display
        bg_color = self.theme['bg']
        fg_color = self.theme['fg']
        accent_color = self.theme['accent']
        secondary_color = self.theme['text_secondary']
        
        # Title
        title_label = tk.Label(self.process_frame, 
                              text="Current Processes:",
                              bg=bg_color, fg=secondary_color,
                              font=('Segoe UI', 10, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Process list
        for i, process in enumerate(self._processes):
            process_frame = tk.Frame(self.process_frame, bg=bg_color)
            process_frame.pack(fill=tk.X, padx=50, pady=2)
            
            # Process indicator (animated dot)
            dot_canvas = tk.Canvas(process_frame, width=20, height=20, 
                                  bg=bg_color, highlightthickness=0)
            dot_canvas.pack(side=tk.LEFT, padx=(0, 10))
            
            # Animated dot
            dot_alpha = (math.sin(self.angle * math.pi / 180 + i * 2) + 1) / 2
            dot_size = 4 + (dot_alpha * 2)
            dot_x, dot_y = 10, 10
            dot_canvas.create_oval(
                dot_x - dot_size, dot_y - dot_size,
                dot_x + dot_size, dot_y + dot_size,
                fill=accent_color, outline=""
            )
            
            # Process text
            process_label = tk.Label(process_frame, 
                                    text=process,
                                    bg=bg_color, fg=fg_color,
                                    font=('Segoe UI', 10),
                                    anchor=tk.W)
            process_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def _draw_loading(self, message: str):
        """Draw loading spinner and message with smooth effects"""
        self.canvas.update_idletasks()
        width = self.canvas.winfo_width() or 400
        height = self.canvas.winfo_height() or 300
        
        center_x = width // 2
        center_y = height // 2
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw semi-transparent backdrop with fade
        alpha = min(1.0, self.fade_alpha)
        if alpha < 1.0:
            # Create a semi-transparent rectangle overlay
            bg_color = self.theme['bg']
            self.canvas.create_rectangle(0, 0, width, height, fill=bg_color, outline="", stipple="gray50")
        
        # Spinner parameters with pulse effect
        base_radius = 35
        radius = base_radius * self.pulse_scale
        spinner_radius = 25 * self.pulse_scale
        
        # Draw multiple rotating arcs for smoother effect
        num_arcs = 3
        for i in range(num_arcs):
            arc_angle = (self.angle + i * 120) % 360
            arc_extent = 60 + (i * 10)  # Varying arc sizes
            
            # Calculate opacity based on position
            opacity_factor = 1.0 - (i * 0.2)
            
            # Draw rotating arc with varying opacity
            self.canvas.create_arc(
                center_x - spinner_radius, center_y - spinner_radius,
                center_x + spinner_radius, center_y + spinner_radius,
                start=arc_angle, extent=arc_extent,
                outline=self.theme['accent'], width=3, style=tk.ARC
            )
        
        # Draw outer circle with pulse effect
        self.canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            outline=self.theme['accent'], width=2, fill=""
        )
        
        # Draw inner pulsing dot
        dot_radius = 4 * self.pulse_scale
        self.canvas.create_oval(
            center_x - dot_radius, center_y - dot_radius,
            center_x + dot_radius, center_y + dot_radius,
            fill=self.theme['accent'], outline=""
        )
        
        # Draw message with fade effect
        message_alpha = min(1.0, self.fade_alpha * 1.5)
        if message_alpha > 0:
            self.canvas.create_text(
                center_x, center_y + 80,
                text=message,
                fill=self.theme['fg'],
                font=('Segoe UI', 14, 'normal')
            )
        
        # Draw subtle animated dots after message
        dots_y = center_y + 105
        for i in range(3):
            dot_x = center_x - 20 + (i * 20)
            dot_alpha = (math.sin(self.angle * math.pi / 180 + i * 2) + 1) / 2
            dot_size = 3 + (dot_alpha * 2)
            self.canvas.create_oval(
                dot_x - dot_size, dots_y - dot_size,
                dot_x + dot_size, dots_y + dot_size,
                fill=self.theme['accent'], outline=""
            )
    
    def _animate(self):
        """Animate the loading spinner with smooth motion"""
        if self.overlay and self.canvas:
            # Smooth rotation (smaller increments for smoother motion)
            self.angle = (self.angle + 4) % 360
            
            # Pulse animation (breathing effect)
            pulse_speed = 0.03
            self.pulse_scale += pulse_speed * self.pulse_direction
            if self.pulse_scale >= 1.15:
                self.pulse_direction = -1
            elif self.pulse_scale <= 0.95:
                self.pulse_direction = 1
            
            # Fade-in animation
            if self.fade_alpha < 1.0:
                self.fade_alpha += 0.1
                if self.fade_alpha > 1.0:
                    self.fade_alpha = 1.0
            
            # Redraw with updated animation
            self._draw_loading(self._current_message)
            # Update process indicators
            if self.process_frame and self._processes:
                self._update_process_indicators()
            # Update time estimate every second (approximately)
            if self.time_label and hasattr(self, '_last_time_update'):
                if time.time() - self._last_time_update >= 1.0:
                    self._update_time_estimate()
                    self._last_time_update = time.time()
            elif self.time_label:
                self._last_time_update = time.time()
                self._update_time_estimate()
            
            # Higher frame rate for smoother animation (16ms = ~60 FPS)
            self.animation_id = self.overlay.after(16, self._animate)
    
    def _update_process_indicators(self):
        """Update animated process indicators"""
        if not self.process_frame:
            return
        
        for widget in self.process_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Canvas):
                        child.delete("all")
                        dot_alpha = (math.sin(self.angle * math.pi / 180) + 1) / 2
                        dot_size = 4 + (dot_alpha * 2)
                        dot_x, dot_y = 10, 10
                        child.create_oval(
                            dot_x - dot_size, dot_y - dot_size,
                            dot_x + dot_size, dot_y + dot_size,
                            fill=self.theme['accent'], outline=""
                        )
                        break
    
    def hide(self):
        """Hide loading screen with smooth fade-out"""
        if self.animation_id:
            self.overlay.after_cancel(self.animation_id)
            self.animation_id = None
        
        if self.overlay:
            # Quick fade-out
            self.fade_alpha = 1.0
            self.fade_direction = -1
            
            def fade_out():
                if self.overlay and self.canvas:
                    self.fade_alpha -= 0.15
                    if self.fade_alpha > 0:
                        self._draw_loading(self._current_message)
                        self.overlay.after(16, fade_out)
                    else:
                        if self.overlay:
                            self.overlay.destroy()
                            self.overlay = None
                            self.canvas = None
            
            fade_out()

