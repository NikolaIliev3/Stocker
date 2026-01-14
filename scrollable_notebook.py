"""
Scrollable Notebook - Tkinter Notebook with scrollable tabs
Handles many tabs by adding scroll buttons or arranging in multiple rows
"""
import tkinter as tk
from tkinter import ttk


class ScrollableNotebook(ttk.Notebook):
    """Notebook with scrollable tabs when there are too many"""
    
    def __init__(self, parent, **kwargs):
        # Extract custom options
        self.tab_rows = kwargs.pop('tab_rows', 1)  # Number of rows for tabs (1 or 2)
        
        super().__init__(parent, **kwargs)
        
        if self.tab_rows == 2:
            # Configure for 2-row layout
            self.configure(style='TwoRow.TNotebook')
            self._setup_two_row_style()
    
    def _setup_two_row_style(self):
        """Setup style for two-row tabs"""
        style = ttk.Style()
        
        # This is a simplified approach - tkinter doesn't natively support multi-row tabs
        # But we can make tabs smaller to fit more
        try:
            style.configure('TwoRow.TNotebook.Tab', padding=[5, 2])
        except:
            pass


class TwoRowNotebook(tk.Frame):
    """A notebook-like widget that displays tabs in two rows and manages content in a shared area."""
    
    def __init__(self, parent, **kwargs):
        theme_bg = kwargs.get('bg', '#f0f0f0')
        super().__init__(parent, bg=theme_bg)
        
        # Tab Rows (Notebooks with height=0 content or similar)
        # We use a trick: notebooks with only one tab visible or just very small
        self.nb1 = ttk.Notebook(self)
        self.nb1.pack(fill='x', side='top')
        
        self.nb2 = ttk.Notebook(self)
        self.nb2.pack(fill='x', side='top')
        
        # Shared Content Area
        self.content_container = tk.Frame(self, bg=theme_bg)
        self.content_container.pack(fill='both', expand=True)
        
        self._tabs = [] # List of {'widget': w, 'text': t, 'nb': nb}
        self.max_tabs_per_row = 6
        
        # Setup styles to handle selection visual
        self._setup_selection_styles()
        
        # Initial styles
        self.nb1.configure(style='ActiveRow.TNotebook')
        self.nb2.configure(style='InactiveRow.TNotebook')
        
        # Add index-0 placeholders (not hidden, but textless/tiny)
        self.p1 = tk.Frame(self.nb1, height=0, width=0)
        self.nb1.add(self.p1, text=" ")
        self.p2 = tk.Frame(self.nb2, height=0, width=0)
        self.nb2.add(self.p2, text=" ")
        
        # Bind events
        self.nb1.bind("<<NotebookTabChanged>>", lambda e: self._on_tab_changed(self.nb1))
        self.nb2.bind("<<NotebookTabChanged>>", lambda e: self._on_tab_changed(self.nb2))
        
        self._switching = False
        

    def add(self, child, **kwargs):
        """Add a tab to the appropriate notebook row"""
        text = kwargs.get('text', f"Tab {len(self._tabs)+1}")
        
        # We don't actually add the child to the notebook if we want to manage it ourselves?
        # No, easier: add a dummy frame to the notebook and keep the real child in content_container
        dummy = tk.Frame(self, height=1, bg=self['bg']) # Don't pack it anywhere but the notebook
        
        if len(self._tabs) < self.max_tabs_per_row:
            self.nb1.add(dummy, text=text)
            self._tabs.append({'widget': child, 'dummy': dummy, 'nb': self.nb1, 'text': text})
        else:
            self.nb2.add(dummy, text=text)
            self._tabs.append({'widget': child, 'dummy': dummy, 'nb': self.nb2, 'text': text})
            
        # Place child in content container but hide it
        child.master = self.content_container
        child.pack_forget()

    def _on_tab_changed(self, active_nb):
        if self._switching: return
        
        # Get selected dummy
        try:
            active_dummy_name = active_nb.select()
            if not active_dummy_name: return
            
            # If we selected a placeholder, do nothing
            if active_dummy_name in [str(self.p1), str(self.p2)]:
                return

            self._switching = True
            
            # Find other notebook and its placeholder
            other_nb = self.nb2 if active_nb == self.nb1 else self.nb1
            other_placeholder = self.p2 if active_nb == self.nb1 else self.p1
            
            # 1. Update styles (immediate visual fix)
            active_nb.configure(style='ActiveRow.TNotebook')
            other_nb.configure(style='InactiveRow.TNotebook')
            
            # 2. Clear logical selection in the other row
            try:
                other_nb.select(other_placeholder)
            except:
                pass

            # 3. Handle visibility of content
            for tab_info in self._tabs:
                tab_info['widget'].pack_forget()
                
            for tab_info in self._tabs:
                if str(tab_info['dummy']) == active_dummy_name:
                    tab_info['widget'].pack(fill='both', expand=True)
                    break
        except Exception as e:
            pass
        finally:
            self._switching = False
            
    def _setup_selection_styles(self):
        """Setup styles to make inactive row tabs look unselected"""
        style = ttk.Style()
        
        # Standard Active Style
        style.configure('ActiveRow.TNotebook', padding=[0, 0, 0, 0])
        style.configure('ActiveRow.TNotebook.Tab', padding=[10, 4])
        
        # Inactive Style (selected tab should look like a normal tab)
        style.configure('InactiveRow.TNotebook', padding=[0, 0, 0, 0])
        style.configure('InactiveRow.TNotebook.Tab', padding=[10, 4])
        
        # Get normal colors
        bg = style.lookup('TNotebook.Tab', 'background')
        fg = style.lookup('TNotebook.Tab', 'foreground')
        
        # Map selected state in inactive row to look non-selected
        style.map('InactiveRow.TNotebook.Tab',
                 background=[('selected', bg)],
                 foreground=[('selected', fg)],
                 expand=[('selected', [0, 0, 0, 0])]) # KEY: prevent growing/pressing

    def select(self, tab_id=None):
        if tab_id is None:
            # Return selected real widget
            for tab_info in self._tabs:
                if tab_info['widget'].winfo_viewable():
                    return str(tab_info['widget'])
            return ""
            
        if isinstance(tab_id, int):
            if tab_id < len(self._tabs):
                tab_info = self._tabs[tab_id]
                tab_info['nb'].select(tab_info['dummy'])
        else:
            # tab_id is a widget
            for tab_info in self._tabs:
                if tab_info['widget'] == tab_id or str(tab_info['widget']) == tab_id:
                    tab_info['nb'].select(tab_info['dummy'])
                    break

    def tab(self, index, option=None, **kwargs):
        # Forward to the correct notebook and dummy
        if isinstance(index, int):
            if index < len(self._tabs):
                info = self._tabs[index]
                return info['nb'].tab(info['dummy'], option, **kwargs)
        else:
            # tab_id is a widget
            for info in self._tabs:
                if info['widget'] == index or str(info['widget']) == index:
                    return info['nb'].tab(info['dummy'], option, **kwargs)
        return None

    def tabs(self):
        return [str(t['widget']) for t in self._tabs]

    def index(self, tab_id):
        if tab_id == "end": return len(self._tabs)
        for i, t in enumerate(self._tabs):
            if t['widget'] == tab_id or str(t['widget']) == tab_id:
                return i
        return -1

    def bind(self, sequence=None, func=None, add=None):
        """Bind events to this frame and both notebooks"""
        self.nb1.bind(sequence, func, add)
        self.nb2.bind(sequence, func, add)
        return super().bind(sequence, func, add)


class CompactNotebook(ttk.Notebook):
    """Notebook with compact tab styling to fit more tabs"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_compact_style()
    
    def _setup_compact_style(self):
        """Setup compact tab style"""
        style = ttk.Style()
        
        # Create compact tab style
        try:
            # Reduce padding to make tabs smaller
            style.configure('Compact.TNotebook.Tab', 
                          padding=[8, 3],  # Reduced from default [12, 5]
                          font=('Segoe UI', 8))  # Smaller font
            
            self.configure(style='Compact.TNotebook')
        except Exception as e:
            # If styling fails, continue with default
            pass


# Recommended solution: Use CompactNotebook
# Example usage in main.py:
# Replace: self.tools_notebook = ttk.Notebook(tools_frame)
# With: self.tools_notebook = CompactNotebook(tools_frame)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Scrollable Notebook Demo')
    root.geometry('800x600')
    
    # Test CompactNotebook
    notebook = CompactNotebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Add many tabs
    for i in range(12):
        frame = tk.Frame(notebook, bg='white')
        label = tk.Label(frame, text=f"Tab {i+1} Content", font=('Segoe UI', 12))
        label.pack(expand=True)
        notebook.add(frame, text=f"Tab {i+1}")
    
    root.mainloop()
