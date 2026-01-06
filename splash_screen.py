"""
Startup Splash Screen for Stocker App
Shows app icon and disclaimer before main UI loads
"""
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import time


class SplashScreen:
    """Startup splash screen with disclaimer"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Stocker")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Try to set icon
        icon_path = Path(__file__).parent / "stocker_icon.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except:
                pass
        
        # Make window stay on top initially
        self.root.attributes('-topmost', True)
        
        # Create UI
        self._create_ui()
        
        # Auto-close after 5 seconds or when user clicks Continue
        self.auto_close_id = None
        self.continue_clicked = False
    
    def _create_ui(self):
        """Create splash screen UI"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1E293B')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # App icon/logo area
        icon_frame = tk.Frame(main_frame, bg='#1E293B')
        icon_frame.pack(fill=tk.X, pady=(40, 20))
        
        # Try to load icon image
        icon_path = Path(__file__).parent / "stocker_icon.png"
        icon_label = None
        
        if icon_path.exists():
            try:
                from PIL import Image, ImageTk
                img = Image.open(icon_path)
                img = img.resize((120, 120), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                icon_label = tk.Label(icon_frame, image=photo, bg='#1E293B')
                icon_label.image = photo  # Keep a reference
                icon_label.pack()
            except ImportError:
                # PIL not available, use text logo
                icon_label = tk.Label(icon_frame, text="📈", font=('Segoe UI', 72), 
                                    bg='#1E293B', fg='#6366F1')
                icon_label.pack()
            except Exception:
                # Image load failed, use text logo
                icon_label = tk.Label(icon_frame, text="📈", font=('Segoe UI', 72), 
                                    bg='#1E293B', fg='#6366F1')
                icon_label.pack()
        else:
            # No icon file, use text logo
            icon_label = tk.Label(icon_frame, text="📈", font=('Segoe UI', 72), 
                                bg='#1E293B', fg='#6366F1')
            icon_label.pack()
        
        # App name
        title_label = tk.Label(main_frame, text="Stocker", 
                             font=('Segoe UI', 32, 'bold'),
                             bg='#1E293B', fg='#FFFFFF')
        title_label.pack(pady=(10, 30))
        
        # Disclaimer section
        disclaimer_frame = tk.Frame(main_frame, bg='#1E293B')
        disclaimer_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=(0, 20))
        
        disclaimer_title = tk.Label(disclaimer_frame, 
                                   text="⚠️ IMPORTANT DISCLAIMER",
                                   font=('Segoe UI', 14, 'bold'),
                                   bg='#1E293B', fg='#F59E0B')
        disclaimer_title.pack(pady=(0, 15))
        
        disclaimer_text = """The projections, predictions, and recommendations provided by this application are for informational and educational purposes only.

This application is NOT financial advice, and the creator of this application:
• Takes NO responsibility for any financial decisions or losses
• Is NOT a trained financial professional
• Does NOT guarantee the accuracy of any predictions or recommendations

All trading and investment decisions are made at your own risk. Always consult with a qualified financial advisor before making any investment decisions.

By continuing, you acknowledge that you understand and accept these terms."""
        
        disclaimer_label = tk.Label(disclaimer_frame, 
                                   text=disclaimer_text,
                                   font=('Segoe UI', 10),
                                   bg='#1E293B', fg='#E5E7EB',
                                   justify=tk.LEFT,
                                   wraplength=520)
        disclaimer_label.pack(pady=(0, 20))
        
        # Continue button
        button_frame = tk.Frame(main_frame, bg='#1E293B')
        button_frame.pack(fill=tk.X, pady=(0, 30))
        
        continue_btn = tk.Button(button_frame,
                                text="I Understand - Continue",
                                command=self._on_continue,
                                bg='#6366F1',
                                fg='#FFFFFF',
                                font=('Segoe UI', 12, 'bold'),
                                relief=tk.FLAT,
                                padx=30,
                                pady=12,
                                cursor='hand2',
                                activebackground='#4F46E5',
                                activeforeground='#FFFFFF')
        continue_btn.pack()
        
        # Auto-close timer (5 seconds)
        self.auto_close_id = self.root.after(5000, self._auto_close)
    
    def _on_continue(self):
        """Handle continue button click"""
        self.continue_clicked = True
        if self.auto_close_id:
            self.root.after_cancel(self.auto_close_id)
        self.root.destroy()
    
    def _auto_close(self):
        """Auto-close after timer"""
        if not self.continue_clicked:
            self.root.destroy()
    
    def show(self):
        """Show splash screen and wait for user or timeout"""
        self.root.deiconify()
        self.root.focus_force()
        self.root.update()
        # Start mainloop and wait for window to be destroyed
        self.root.mainloop()

