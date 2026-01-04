# Quick Setup Guide

## Step 1: Install Dependencies

Open a terminal/command prompt and run:

```bash
pip install -r requirements.txt
```

## Step 2: Start the Backend Server

**Windows:**
Double-click `start_backend.bat` or run:
```bash
python backend_proxy.py
```

**Linux/Mac:**
```bash
chmod +x start_backend.sh
./start_backend.sh
```

Or run directly:
```bash
python3 backend_proxy.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
```

**Keep this terminal open!**

## Step 3: Start the Desktop App

Open a **new** terminal/command prompt:

**Windows:**
Double-click `start_app.bat` or run:
```bash
python main.py
```

**Linux/Mac:**
```bash
chmod +x start_app.sh
./start_app.sh
```

Or run directly:
```bash
python3 main.py
```

## Step 4: Use the App

1. Select a strategy (Trading or Investing)
2. Enter a stock symbol (e.g., AAPL, MSFT, TSLA)
3. Click "Analyze Stock"
4. View results in the tabs
5. Enter a budget and click "Calculate Potential" to see win/loss estimates

## Troubleshooting

### "Connection refused" or "Cannot connect to backend"
- Make sure the backend server is running (Step 2)
- Check that port 5000 is not blocked by firewall
- Verify the backend URL in `config.py`

### "Module not found" errors
- Run `pip install -r requirements.txt` again
- Make sure you're using Python 3.8 or higher

### No data returned
- Check your internet connection
- Verify the stock symbol is correct
- Check the backend terminal for error messages

## Configuration

Edit `config.py` to change:
- Backend API URL (if using a remote server)
- Security settings
- Default balance
- Chart settings

## Security Notes

- The backend proxy runs on `localhost:5000` by default
- For production, configure HTTPS and certificate pinning
- Never expose API keys in the client application
- Always use the backend proxy for API calls

