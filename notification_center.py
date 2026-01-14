"""
Notification Center - Persistent notification panel for important events
"""
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class NotificationCenter(tk.Frame):
    """Notification center widget with badge counter"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Notification storage
        self.notifications: List[Dict] = []
        self.unread_count = 0
        self.max_notifications = 50
        
        # Create UI
        self._create_ui()
    
    def _create_ui(self):
        """Create notification center UI"""
        # Header with badge
        header_frame = tk.Frame(self, bg=self['bg'])
        header_frame.pack(fill='x', padx=5, pady=5)
        
        title_label = tk.Label(header_frame, text="📬 Notifications",
                               font=('Segoe UI', 10, 'bold'), bg=self['bg'])
        title_label.pack(side='left')
        
        self.badge_label = tk.Label(header_frame, text="", 
                                    font=('Segoe UI', 8, 'bold'),
                                    bg='#F44336', fg='white',
                                    padx=6, pady=2, borderwidth=0)
        self.badge_label.pack(side='left', padx=5)
        
        # Clear all button
        clear_btn = tk.Button(header_frame, text="Clear All",
                             command=self.clear_all,
                             font=('Segoe UI', 8),
                             bg='#E0E0E0', relief='flat', cursor='hand2')
        clear_btn.pack(side='right')
        
        # Scrollable notification list
        list_frame = tk.Frame(self, bg=self['bg'])
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        # Listbox
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                  font=('Segoe UI', 9), height=10,
                                  selectmode='single', relief='flat',
                                  bg='#FAFAFA', highlightthickness=1,
                                  highlightbackground='#E0E0E0')
        self.listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        # Bind click to mark as read
        self.listbox.bind('<<ListboxSelect>>', self._on_select)
        
        self._update_badge()
    
    def add_notification(self, message: str, notification_type: str = 'info'):
        """Add a new notification
        
        Args:
            message: Notification message
            notification_type: Type of notification (info, success, warning, error)
        """
        notification = {
            'message': message,
            'type': notification_type,
            'timestamp': datetime.now().isoformat(),
            'read': False
        }
        
        self.notifications.insert(0, notification)  # Add to beginning
        self.unread_count += 1
        
        # Limit total notifications
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[:self.max_notifications]
        
        self._update_display()
        self._update_badge()
        
        logger.debug(f"Notification added: {message[:50]}...")
    
    def _update_display(self):
        """Update the listbox display"""
        self.listbox.delete(0, tk.END)
        
        for i, notif in enumerate(self.notifications):
            # Format notification
            timestamp = datetime.fromisoformat(notif['timestamp'])
            time_str = timestamp.strftime('%H:%M')
            
            # Add emoji based on type
            emoji = {
                'success': '✅',
                'warning': '⚠️',
                'error': '❌',
                'info': 'ℹ️'
            }.get(notif['type'], 'ℹ️')
            
            # Mark unread with bullet
            unread_marker = '● ' if not notif['read'] else '  '
            
            display_text = f"{unread_marker}{emoji} [{time_str}] {notif['message']}"
            
            self.listbox.insert(tk.END, display_text)
            
            # Color code by type
            if notif['type'] == 'success':
                self.listbox.itemconfig(i, fg='#4CAF50')
            elif notif['type'] == 'warning':
                self.listbox.itemconfig(i, fg='#FF9800')
            elif notif['type'] == 'error':
                self.listbox.itemconfig(i, fg='#F44336')
            
            # Bold if unread
            if not notif['read']:
                self.listbox.itemconfig(i, font=('Segoe UI', 9, 'bold'))
    
    def _update_badge(self):
        """Update the unread badge"""
        if self.unread_count > 0:
            self.badge_label.config(text=str(self.unread_count))
            self.badge_label.pack(side='left', padx=5)
        else:
            self.badge_label.pack_forget()
    
    def _on_select(self, event):
        """Mark notification as read when selected"""
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.notifications):
                if not self.notifications[index]['read']:
                    self.notifications[index]['read'] = True
                    self.unread_count = max(0, self.unread_count - 1)
                    self._update_display()
                    self._update_badge()
    
    def clear_all(self):
        """Clear all notifications"""
        self.notifications.clear()
        self.unread_count = 0
        self._update_display()
        self._update_badge()
        logger.debug("All notifications cleared")
    
    def get_unread_count(self) -> int:
        """Get number of unread notifications"""
        return self.unread_count


# Example usage
if __name__ == '__main__':
    root = tk.Tk()
    root.title('Notification Center Demo')
    root.geometry('400x400')
    
    notif_center = NotificationCenter(root, bg='white')
    notif_center.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Add test notifications
    notif_center.add_notification("AAPL reached target price (+5.2%)", 'success')
    notif_center.add_notification("2 predictions verified (1 correct)", 'info')
    notif_center.add_notification("Market scan found 3 BUY signals", 'success')
    notif_center.add_notification("High volatility detected in TSLA", 'warning')
    notif_center.add_notification("Failed to fetch data for GOOGL", 'error')
    
    root.mainloop()
