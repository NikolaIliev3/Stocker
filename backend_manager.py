"""
Backend Manager
Manages the Flask backend server process (Start, Stop, Health Check).
Decouples backend logic from the main GUI (God Object Refactoring).
"""
import socket
import logging
import threading
import subprocess
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

class BackendManager:
    """
    Manages the lifecycle of the Flask backend server.
    """
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.host = '127.0.0.1'
        self.port = 5000
        
    def is_running(self) -> bool:
        """Check if backend server is already running on port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Error checking backend status: {e}")
            return False

    def start(self):
        """Start the backend server in a background process"""
        if self.is_running():
            logger.info(f"Backend server is already running on port {self.port}")
            return

        def run_in_background():
            try:
                # Use the same python interpreter as the main app
                python_exe = sys.executable
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_api.py")
                
                # Start process detached
                # Use subprocess.CREATE_NO_WINDOW on Windows if available to hide console
                creationflags = 0
                if os.name == 'nt':
                     creationflags = 0x08000000 # CREATE_NO_WINDOW
                
                self.process = subprocess.Popen(
                    [python_exe, script_path],
                    cwd=os.path.dirname(script_path),
                    creationflags=creationflags
                )
                logger.info(f"Started backend server (PID: {self.process.pid})")
                
            except Exception as e:
                logger.error(f"Failed to start backend server: {e}")

        # Although Popen is non-blocking, wrapping in thread ensures no UI freeze during process creation overhead
        thread = threading.Thread(target=run_in_background)
        thread.daemon = True
        thread.start()

    def stop(self):
        """Stop the backend server if we started it"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("Backend server stopped")
            except Exception as e:
                logger.error(f"Error stopping backend server: {e}")
                # Force kill if needed
                try:
                    self.process.kill()
                except:
                    pass
            finally:
                self.process = None
