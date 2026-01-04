# SSL Certificate Issue - Workaround Guide

## The Problem

yfinance uses curl (a C library) to fetch data, and curl on Windows is having trouble finding SSL certificates. This is a known issue with yfinance on Windows.

## Quick Fix Options

### Option 1: Use a Different Python Installation (Easiest)

If you have Anaconda or Miniconda installed:
```bash
conda install -c conda-forge yfinance
```

Conda's version of yfinance sometimes works better with SSL.

### Option 2: Reinstall yfinance with pip

Try reinstalling:
```bash
pip uninstall yfinance
pip install yfinance --no-cache-dir
```

### Option 3: Use Alternative Data Source (If yfinance keeps failing)

We can modify the backend to use a different stock data API that doesn't have SSL issues.

## Current Status

The backend is trying to work around this, but curl's SSL handling is outside Python's control. The app will show errors when fetching stock data until this is resolved.

## What's Happening

- yfinance uses curl (C library) for HTTP requests
- curl can't find/use SSL certificates on your Windows system
- Python-level SSL patches don't affect curl
- This is a system-level issue, not a code issue

## Temporary Solution

For now, the app will show errors. The backend is trying multiple workarounds, but they may not work if curl itself is broken.

## Long-term Solution

1. Fix Windows SSL certificate store
2. Use a different stock data provider
3. Use conda's yfinance (often works better)
4. Switch to a Linux/Mac environment (where this issue is less common)


