"""
Create a Windows desktop shortcut with icon for Stocker App
"""
import os
import sys
from pathlib import Path

try:
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

def create_shortcut():
    """Create a desktop shortcut with icon"""
    # Get paths
    script_dir = Path(__file__).parent.absolute()
    start_bat = script_dir / "START_HERE.bat"
    icon_file = script_dir / "stocker_icon.ico"
    desktop = Path.home() / "Desktop"
    shortcut_path = desktop / "Stocker.lnk"
    
    if not start_bat.exists():
        print(f"Error: {start_bat} not found!")
        return False
    
    if not icon_file.exists():
        print(f"Warning: {icon_file} not found. Creating icon first...")
        # Try to create icon
        try:
            from create_icon import create_icon
            create_icon()
        except Exception as e:
            print(f"Could not create icon: {e}")
            icon_file = None
    
    if HAS_WIN32:
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = str(start_bat)
            shortcut.WorkingDirectory = str(script_dir)
            shortcut.Description = "Stocker - Stock Trading & Investing App"
            if icon_file and icon_file.exists():
                shortcut.IconLocation = str(icon_file)
            shortcut.save()
            print(f"Shortcut created: {shortcut_path}")
            print("Icon:", icon_file if icon_file else "Default")
            return True
        except Exception as e:
            print(f"Error creating shortcut with win32com: {e}")
            return False
    else:
        # Fallback: Create VBS script to create shortcut
        print("win32com not available. Creating VBS script...")
        return create_shortcut_vbs(script_dir, start_bat, icon_file, shortcut_path)

def create_shortcut_vbs(script_dir, start_bat, icon_file, shortcut_path):
    """Create shortcut using VBScript (works without pywin32)"""
    vbs_script = script_dir / "create_shortcut.vbs"
    
    icon_location = f'"{icon_file}"' if icon_file and icon_file.exists() else '""'
    
    vbs_content = f'''Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{start_bat}"
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
            print(f"Shortcut created: {shortcut_path}")
            print("Icon:", icon_file if icon_file and icon_file.exists() else "Default")
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
    create_shortcut()

