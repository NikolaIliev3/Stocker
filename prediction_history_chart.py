"""
Prediction History Chart - Interactive matplotlib charts for prediction performance
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PredictionHistoryChart(tk.Frame):
    """Interactive charts showing prediction performance over time"""
    
    def __init__(self, parent, predictions_tracker, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.predictions_tracker = predictions_tracker
        
        # Create UI
        self._create_ui()
    
    def _create_ui(self):
        """Create chart UI"""
        # Create notebook for multiple charts
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Accuracy over time
        self.accuracy_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.accuracy_frame, text='Accuracy Trend')
        
        # Tab 2: Correct vs Incorrect
        self.breakdown_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.breakdown_frame, text='Win/Loss Breakdown')
        
        # Tab 3: Confidence Calibration
        self.calibration_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.calibration_frame, text='Confidence Calibration')
        
        # Refresh button
        refresh_btn = tk.Button(self, text="🔄 Refresh Charts",
                               command=self.refresh,
                               font=('Segoe UI', 9),
                               bg='#2196F3', fg='white',
                               relief='flat', cursor='hand2', padx=15, pady=5)
        refresh_btn.pack(pady=5)
        
        # Initial draw
        self.refresh()
    
    def refresh(self):
        """Refresh all charts with latest data"""
        try:
            self._draw_accuracy_trend()
            self._draw_breakdown()
            self._draw_calibration()
            logger.debug("Prediction history charts refreshed")
        except Exception as e:
            logger.error(f"Error refreshing charts: {e}")
    
    def _draw_accuracy_trend(self):
        """Draw accuracy over time line chart"""
        # Clear previous
        for widget in self.accuracy_frame.winfo_children():
            widget.destroy()
        
        # Get verified predictions
        verified = self.predictions_tracker.get_verified_predictions()
        
        if not verified:
            label = tk.Label(self.accuracy_frame, text="No verified predictions yet",
                           font=('Segoe UI', 12), bg='white', fg='#666666')
            label.pack(expand=True)
            return
        
        # Group by date
        from collections import defaultdict
        by_date = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred in verified:
            try:
                date_str = pred.get('verification_date', '')
                if date_str:
                    date = datetime.fromisoformat(date_str).date()
                    by_date[date]['total'] += 1
                    if pred.get('was_correct'):
                        by_date[date]['correct'] += 1
            except:
                continue
        
        # Sort by date
        sorted_dates = sorted(by_date.keys())
        accuracies = []
        
        for date in sorted_dates:
            data = by_date[date]
            accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            accuracies.append(accuracy)
        
        # Create figure
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(sorted_dates, accuracies, marker='o', linewidth=2, 
               color='#2196F3', markersize=6)
        ax.axhline(y=50, color='#FF9800', linestyle='--', linewidth=1, alpha=0.7, label='50% Baseline')
        ax.fill_between(sorted_dates, accuracies, 50, where=[a >= 50 for a in accuracies],
                       color='#4CAF50', alpha=0.2, label='Above Baseline')
        ax.fill_between(sorted_dates, accuracies, 50, where=[a < 50 for a in accuracies],
                       color='#F44336', alpha=0.2, label='Below Baseline')
        
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Prediction Accuracy Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(0, 100)
        
        # Rotate x-axis labels
        fig.autofmt_xdate()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.accuracy_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def _draw_breakdown(self):
        """Draw correct vs incorrect bar chart by strategy"""
        # Clear previous
        for widget in self.breakdown_frame.winfo_children():
            widget.destroy()
        
        # Get statistics
        stats = self.predictions_tracker.get_statistics()
        strategy_stats = stats.get('strategy_stats', {})
        
        if not strategy_stats:
            label = tk.Label(self.breakdown_frame, text="No strategy data yet",
                           font=('Segoe UI', 12), bg='white', fg='#666666')
            label.pack(expand=True)
            return
        
        # Prepare data
        strategies = list(strategy_stats.keys())
        correct = [strategy_stats[s]['correct'] for s in strategies]
        incorrect = [strategy_stats[s]['total'] - strategy_stats[s]['correct'] for s in strategies]
        
        # Create figure
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        x = range(len(strategies))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], correct, width, label='Correct',
                      color='#4CAF50', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], incorrect, width, label='Incorrect',
                      color='#F44336', alpha=0.8)
        
        ax.set_xlabel('Strategy', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Correct vs Incorrect Predictions by Strategy', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in strategies])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=8)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.breakdown_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def _draw_calibration(self):
        """Draw confidence vs actual accuracy scatter plot"""
        # Clear previous
        for widget in self.calibration_frame.winfo_children():
            widget.destroy()
        
        # Get verified predictions
        verified = self.predictions_tracker.get_verified_predictions()
        
        if not verified:
            label = tk.Label(self.calibration_frame, text="No verified predictions yet",
                           font=('Segoe UI', 12), bg='white', fg='#666666')
            label.pack(expand=True)
            return
        
        # Group by confidence buckets
        from collections import defaultdict
        buckets = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred in verified:
            confidence = pred.get('confidence', 50)
            bucket = int(confidence // 10) * 10  # 0-10, 10-20, etc.
            buckets[bucket]['total'] += 1
            if pred.get('was_correct'):
                buckets[bucket]['correct'] += 1
        
        # Prepare data
        confidences = []
        accuracies = []
        sizes = []
        
        for bucket in sorted(buckets.keys()):
            data = buckets[bucket]
            accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            confidences.append(bucket + 5)  # Center of bucket
            accuracies.append(accuracy)
            sizes.append(data['total'] * 20)  # Size proportional to count
        
        # Create figure
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(confidences, accuracies, s=sizes, alpha=0.6,
                           c=confidences, cmap='RdYlGn', edgecolors='black', linewidth=1)
        
        # Perfect calibration line
        ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')
        
        ax.set_xlabel('Predicted Confidence (%)', fontsize=10)
        ax.set_ylabel('Actual Accuracy (%)', fontsize=10)
        ax.set_title('Confidence Calibration (size = sample count)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Confidence Level', fontsize=8)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.calibration_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)


# Example usage
if __name__ == '__main__':
    # Mock predictions tracker for demo
    class MockTracker:
        def get_verified_predictions(self):
            import random
            predictions = []
            for i in range(50):
                date = datetime.now() - timedelta(days=random.randint(0, 30))
                predictions.append({
                    'verification_date': date.isoformat(),
                    'was_correct': random.random() > 0.4,
                    'confidence': random.randint(40, 90),
                    'strategy': random.choice(['trading', 'mixed', 'investing'])
                })
            return predictions
        
        def get_statistics(self):
            return {
                'strategy_stats': {
                    'trading': {'total': 20, 'correct': 13},
                    'mixed': {'total': 15, 'correct': 10},
                    'investing': {'total': 15, 'correct': 11}
                }
            }
    
    root = tk.Tk()
    root.title('Prediction History Chart Demo')
    root.geometry('900x600')
    
    chart = PredictionHistoryChart(root, MockTracker(), bg='white')
    chart.pack(fill='both', expand=True, padx=10, pady=10)
    
    root.mainloop()
