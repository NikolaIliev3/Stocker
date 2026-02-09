
import sys
import time
import subprocess
import os

print("--- DEBUG WRAPPER STARTING ---", flush=True)
print(f"Python executable: {sys.executable}", flush=True)
print(f"Current working directory: {os.getcwd()}", flush=True)

# Command to run
cmd = [sys.executable, "-u", "heavy_bootstrap.py", "--stage", "1"]

print(f"Executing: {' '.join(cmd)}", flush=True)
print("------------------------------", flush=True)

try:
    # Run process and stream output line by line
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)
        
    process.wait()
    print(f"--- PROCESS FINISHED with return code {process.returncode} ---", flush=True)
    
except Exception as e:
    print(f"--- ERROR executing script: {e} ---", flush=True)
    import traceback
    traceback.print_exc()
