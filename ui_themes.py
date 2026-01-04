"""
Modern Theme system for Stocker App
Beautiful, modern themes with gradients and visual effects
"""
from typing import Dict


class ThemeManager:
    """Manages application themes with modern styling"""
    
    THEMES = {
        "light": {
            "name": "Light",
            "bg": "#F8F9FA",
            "fg": "#212529",
            "secondary_bg": "#FFFFFF",
            "accent": "#6366F1",
            "accent_hover": "#4F46E5",
            "accent_light": "#818CF8",
            "success": "#10B981",
            "warning": "#F59E0B",
            "error": "#EF4444",
            "border": "#E5E7EB",
            "text_secondary": "#6B7280",
            "button_bg": "#6366F1",
            "button_fg": "#FFFFFF",
            "button_hover": "#4F46E5",
            "entry_bg": "#FFFFFF",
            "entry_fg": "#111827",
            "frame_bg": "#FFFFFF",
            "card_bg": "#FFFFFF",
            "card_shadow": "#E5E7EB",
            "gradient_start": "#667EEA",
            "gradient_end": "#764BA2",
        },
        "dark": {
            "name": "Dark",
            "bg": "#0F172A",
            "fg": "#F1F5F9",
            "secondary_bg": "#1E293B",
            "accent": "#818CF8",
            "accent_hover": "#6366F1",
            "accent_light": "#A5B4FC",
            "success": "#34D399",
            "warning": "#FBBF24",
            "error": "#F87171",
            "border": "#334155",
            "text_secondary": "#94A3B8",
            "button_bg": "#818CF8",
            "button_fg": "#0F172A",
            "button_hover": "#6366F1",
            "entry_bg": "#1E293B",
            "entry_fg": "#F1F5F9",
            "frame_bg": "#1E293B",
            "card_bg": "#1E293B",
            "card_shadow": "#0F172A",
            "gradient_start": "#667EEA",
            "gradient_end": "#764BA2",
        },
        "black": {
            "name": "Black",
            "bg": "#000000",
            "fg": "#FFFFFF",
            "secondary_bg": "#0A0A0A",
            "accent": "#00FF88",
            "accent_hover": "#00CC6A",
            "accent_light": "#33FFAA",
            "success": "#00FF88",
            "warning": "#FFD700",
            "error": "#FF1744",
            "border": "#1A1A1A",
            "text_secondary": "#888888",
            "button_bg": "#00FF88",
            "button_fg": "#000000",
            "button_hover": "#00CC6A",
            "entry_bg": "#0A0A0A",
            "entry_fg": "#FFFFFF",
            "frame_bg": "#0A0A0A",
            "card_bg": "#0A0A0A",
            "card_shadow": "#000000",
            "gradient_start": "#00FF88",
            "gradient_end": "#00D4AA",
        }
    }
    
    def __init__(self):
        self.current_theme = "light"
    
    def get_theme(self, theme_name: str = None) -> Dict:
        """Get theme colors"""
        if theme_name is None:
            theme_name = self.current_theme
        return self.THEMES.get(theme_name, self.THEMES["light"])
    
    def set_theme(self, theme_name: str):
        """Set current theme"""
        if theme_name in self.THEMES:
            self.current_theme = theme_name
    
    def get_theme_names(self) -> list:
        """Get list of available theme names"""
        return list(self.THEMES.keys())
