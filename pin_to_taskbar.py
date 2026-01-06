"""
Helper script to pin Stocker to Windows taskbar with custom icon
This creates a proper shortcut that can be pinned to the taskbar
"""
import os
import sys
from pathlib import Path

try:
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("Note: pywin32 not installed. Will use VBScript method instead.")
    print("   Install with: pip install pywin32")

def create_taskbar_shortcut():
    """Create a shortcut optimized for taskbar pinning"""
    script_dir = Path(__file__).parent.absolute()
    main_py = script_dir / "main.py"
    icon_file = script_dir / "stocker_icon.ico"
    
    # Create shortcut in user's AppData\Roaming\Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar
    # Or simpler: create on Desktop and user can pin it manually
    desktop = Path.home() / "Desktop"
    shortcut_path = desktop / "Stocker.lnk"
    
    if not main_py.exists():
        print(f"Error: {main_py} not found!")
        return False
    
    if not icon_file.exists():
        print(f"Warning: {icon_file} not found. Creating icon first...")
        try:
            from create_icon import create_icon
            create_icon()
        except Exception as e:
            print(f"Could not create icon: {e}")
            icon_file = None
    
    # Create Python launcher script that sets icon
    launcher_script = script_dir / "stocker_launcher.pyw"
    launcher_content = '''"""
Stocker Launcher - Runs with proper icon
"""
import sys
import os
from pathlib import Path

# Change to script directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Add current directory to path
sys.path.insert(0, str(script_dir))

# Import and run main
if __name__ == "__main__":
    import tkinter as tk
    from main import StockerApp
    
    root = tk.Tk()
    
    # Set icon before creating app (already set in StockerApp.__init__, but ensure it's set)
    icon_path = script_dir / "stocker_icon.ico"
    if icon_path.exists():
        try:
            root.iconbitmap(str(icon_path))
        except:
            pass
    
    app = StockerApp(root)
    root.mainloop()
'''
    
    with open(launcher_script, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    print(f"Created launcher script: {launcher_script}")
    
    if HAS_WIN32:
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))
            
            # Use pythonw.exe (no console window) with launcher script
            python_exe = sys.executable
            pythonw = python_exe.replace('python.exe', 'pythonw.exe')
            if not Path(pythonw).exists():
                # Try alternative locations
                python_dir = Path(python_exe).parent
                pythonw = python_dir / "pythonw.exe"
                if not pythonw.exists():
                    pythonw = python_exe  # Fallback to python.exe (will show console)
            
            shortcut.Targetpath = pythonw
            shortcut.Arguments = f'"{launcher_script}"'
            shortcut.WorkingDirectory = str(script_dir)
            shortcut.Description = "Stocker - Stock Trading & Investing App"
            
            if icon_file and icon_file.exists():
                shortcut.IconLocation = str(icon_file)
            
            shortcut.save()
            print(f"\n✅ Shortcut created: {shortcut_path}")
            print(f"   Icon: {icon_file if icon_file else 'Default'}")
            print(f"\n📌 TO PIN TO TASKBAR:")
            print(f"   1. Right-click the shortcut on your Desktop")
            print(f"   2. Select 'Pin to taskbar'")
            print(f"   3. The icon should appear on your taskbar!")
            return True
        except Exception as e:
            print(f"Error creating shortcut with win32com: {e}")
            return False
    else:
        # Use VBScript method
        return create_shortcut_vbs(script_dir, pythonw, launcher_script, icon_file, shortcut_path)

def create_shortcut_vbs(script_dir, pythonw, launcher_script, icon_file, shortcut_path):
    """Create shortcut using VBScript"""
    vbs_script = script_dir / "create_taskbar_shortcut.vbs"
    
    icon_location = f'"{icon_file}"' if icon_file and icon_file.exists() else '""'
    pythonw_path = pythonw.replace('\\', '\\\\')
    launcher_path = str(launcher_script).replace('\\', '\\\\')
    shortcut_path_str = str(shortcut_path).replace('\\', '\\\\')
    
    vbs_content = f'''Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path_str}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{pythonw_path}"
oLink.Arguments = "{launcher_path}"
oLink.WorkingDirectory = "{script_dir}"
oLink.Description = "Stocker - Stock Trading & Investing App"
If {icon_location} <> "" Then
    oLink.IconLocation = {icon_location}
End If
oLink.Save
WScript.Echo "Shortcut created successfully!"
'''
    
    with open(vbs_script, 'w', encoding='utf-8') as f:
        f.write(vbs_content)
    
    print(f"VBS script created: {vbs_script}")
    print("Running VBS script to create shortcut...")
    
    import subprocess
    try:
        result = subprocess.run(['cscript', '//nologo', str(vbs_script)], 
                              capture_output=True, text=True, cwd=str(script_dir))
        if result.returncode == 0:
            print(f"\n✅ Shortcut created: {shortcut_path}")
            print(f"   Icon: {icon_file if icon_file and icon_file.exists() else 'Default'}")
            print(f"\n📌 TO PIN TO TASKBAR:")
            print(f"   1. Right-click the shortcut on your Desktop")
            print(f"   2. Select 'Pin to taskbar'")
            print(f"   3. The icon should appear on your taskbar!")
            return True
        else:
            print(f"Error: {result.stderr}")
            print(f"\nPlease run manually: cscript \"{vbs_script}\"")
            return False
    except Exception as e:
        print(f"Error running VBS script: {e}")
        print(f"\nPlease run manually: cscript \"{vbs_script}\"")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  Stocker - Taskbar Icon Setup")
    print("=" * 60)
    print()
    create_taskbar_shortcut()

