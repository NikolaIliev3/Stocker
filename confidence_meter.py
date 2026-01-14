"""
Confidence Meter Widget - Visual confidence display with color gradient
"""
import tkinter as tk
from tkinter import ttk


class ConfidenceMeter(tk.Frame):
    """Visual confidence meter with color-coded gradient"""
    
    def __init__(self, parent, width=300, height=30, **kwargs):
        super().__init__(parent, **kwargs)
        self.width = width
        self.height = height
        
        # Create canvas for custom drawing
        self.canvas = tk.Canvas(self, width=width, height=height, 
                               highlightthickness=0, bg=self['bg'])
        self.canvas.pack()
        
        # Store current confidence
        self.confidence = 0
        
    def set_confidence(self, confidence: float):
        """Update the confidence meter
        
        Args:
            confidence: Confidence value (0-100)
        """
        self.confidence = max(0, min(100, confidence))
        self._draw_meter()
    
    def _get_color(self, confidence: float) -> str:
        """Get color based on confidence level
        
        Args:
            confidence: Confidence value (0-100)
            
        Returns:
            Hex color string
        """
        if confidence >= 70:
            # Green gradient (70-100)
            # #4CAF50 (green) to #2E7D32 (dark green)
            intensity = int(((confidence - 70) / 30) * 30)
            return f'#{76-intensity:02x}{175-intensity:02x}{80-intensity:02x}'
        elif confidence >= 50:
            # Yellow-green gradient (50-70)
            # #FFC107 (yellow) to #4CAF50 (green)
            ratio = (confidence - 50) / 20
            r = int(255 - (255 - 76) * ratio)
            g = int(193 + (175 - 193) * ratio)
            b = int(7 + (80 - 7) * ratio)
            return f'#{r:02x}{g:02x}{b:02x}'
        else:
            # Red-yellow gradient (0-50)
            # #F44336 (red) to #FFC107 (yellow)
            ratio = confidence / 50
            r = int(244 + (255 - 244) * ratio)
            g = int(67 + (193 - 67) * ratio)
            b = int(54 + (7 - 54) * ratio)
            return f'#{r:02x}{g:02x}{b:02x}'
    
    def _draw_meter(self):
        """Draw the confidence meter"""
        self.canvas.delete('all')
        
        # Draw background (empty bar)
        bg_color = '#E0E0E0'
        self.canvas.create_rectangle(0, 0, self.width, self.height,
                                     fill=bg_color, outline='#BDBDBD', width=1)
        
        # Draw filled portion
        fill_width = int((self.confidence / 100) * self.width)
        if fill_width > 0:
            color = self._get_color(self.confidence)
            self.canvas.create_rectangle(0, 0, fill_width, self.height,
                                         fill=color, outline='', width=0)
        
        # Draw percentage text
        text_color = '#FFFFFF' if self.confidence > 40 else '#424242'
        self.canvas.create_text(self.width / 2, self.height / 2,
                               text=f'{self.confidence:.1f}%',
                               font=('Segoe UI', 10, 'bold'),
                               fill=text_color)
        
        # Draw border
        self.canvas.create_rectangle(0, 0, self.width, self.height,
                                     fill='', outline='#9E9E9E', width=1)


class ConfidenceMeterWithLabel(tk.Frame):
    """Confidence meter with label and description"""
    
    def __init__(self, parent, label="Confidence", **kwargs):
        super().__init__(parent, **kwargs)
        
        # Label
        self.label = tk.Label(self, text=label, font=('Segoe UI', 9, 'bold'))
        self.label.pack(anchor='w', pady=(0, 2))
        
        # Meter
        self.meter = ConfidenceMeter(self, width=250, height=25, bg=self['bg'])
        self.meter.pack(anchor='w')
        
        # Description label
        self.desc_label = tk.Label(self, text='', font=('Segoe UI', 8),
                                   fg='#666666')
        self.desc_label.pack(anchor='w', pady=(2, 0))
    
    def set_confidence(self, confidence: float, description: str = ''):
        """Update confidence and description
        
        Args:
            confidence: Confidence value (0-100)
            description: Optional description text
        """
        self.meter.set_confidence(confidence)
        
        if description:
            self.desc_label.config(text=description)
        else:
            # Auto-generate description
            if confidence >= 85:
                desc = 'Very High Confidence'
            elif confidence >= 70:
                desc = 'High Confidence'
            elif confidence >= 50:
                desc = 'Moderate Confidence'
            elif confidence >= 30:
                desc = 'Low Confidence'
            else:
                desc = 'Very Low Confidence'
            self.desc_label.config(text=desc)


# Example usage
if __name__ == '__main__':
    root = tk.Tk()
    root.title('Confidence Meter Demo')
    root.geometry('400x300')
    
    # Test different confidence levels
    levels = [25, 45, 60, 75, 90]
    
    for i, level in enumerate(levels):
        meter = ConfidenceMeterWithLabel(root, label=f'Test {i+1}')
        meter.pack(padx=20, pady=10, fill='x')
        meter.set_confidence(level)
    
    root.mainloop()
