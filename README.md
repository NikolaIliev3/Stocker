# Stocker - Stock Trading & Investing Desktop App

A secure, feature-rich desktop application for stock market analysis, trading, and investing with comprehensive security measures.

## Features

### Strategy Modes
- **Trading Strategy**: Short-term technical analysis focusing on:
  - Price action and trends
  - Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
  - Support and resistance levels
  - Volume analysis
  - Momentum indicators
  
- **Investing Strategy**: Long-term fundamental analysis focusing on:
  - Company fundamentals and business model
  - Financial health and statements
  - Valuation metrics (P/E, P/S, etc.)
  - Growth prospects
  - Risk assessment

### Core Functionality
- Stock symbol lookup and analysis
- Real-time stock data fetching
- Interactive charts with technical indicators
- Buy/Sell/Hold recommendations with confidence scores
- Potential win/loss calculations based on budget
- Portfolio tracking (balance, wins, losses, win rate)
- Detailed reasoning for all recommendations

### Security Features
- **HTTPS Enforcement**: All API calls use HTTPS only
- **Certificate Pinning**: Prevents man-in-the-middle attacks
- **Backend Proxy**: API keys never exposed to client
- **Data Validation**: Sanity checks on all received data
- **Rate Limiting**: Prevents API abuse
- **Secure Storage**: Encrypted local data storage

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend proxy server (in one terminal):
```bash
python backend_proxy.py
```

4. Start the desktop application (in another terminal):
```bash
python main.py
```

## Usage

### Starting the Application

1. **Start Backend Server First**:
   ```bash
   python backend_proxy.py
   ```
   The server will run on `http://localhost:5000`

2. **Start Desktop App**:
   ```bash
   python main.py
   ```

### Using the App

1. **Select Strategy**:
   - Choose "Trading" for short-term technical analysis
   - Choose "Investing" for long-term fundamental analysis

2. **Analyze a Stock**:
   - Enter a stock symbol (e.g., "AAPL", "MSFT", "TSLA")
   - Click "Analyze Stock" or press Enter
   - Wait for analysis to complete

3. **View Results**:
   - **Analysis Tab**: Detailed recommendation with reasoning
   - **Charts Tab**: Price chart with candlesticks and indicators
   - **Indicators Tab**: Technical indicator charts (RSI, MACD, Volume)
   - **Potential Trade Tab**: Calculate potential wins/losses

4. **Calculate Potential Trade**:
   - Enter your investment budget
   - Click "Calculate Potential"
   - View potential win/loss based on recommendation

5. **Manage Portfolio**:
   - Set initial balance
   - Track wins and losses
   - Monitor total return

## Architecture

### Security Architecture

```
Desktop App (Client)
   |
   | HTTPS (TLS 1.2+)
   | Certificate Pinning
   v
Backend Proxy Server
   |
   | Secure API Keys
   | Rate Limiting
   | Caching
   v
Stock Data Providers (Yahoo Finance, etc.)
```

### Key Components

- **main.py**: Main GUI application
- **backend_proxy.py**: Secure backend proxy server
- **security.py**: Security measures (HTTPS, certificate pinning, validation)
- **data_fetcher.py**: Secure data fetching from backend
- **trading_analyzer.py**: Technical analysis engine
- **investing_analyzer.py**: Fundamental analysis engine
- **chart_generator.py**: Chart generation and visualization
- **portfolio.py**: Portfolio tracking and management
- **config.py**: Configuration and settings

## Security Measures

### Implemented Security Features

1. **HTTPS Enforcement**
   - All API calls use HTTPS only
   - HTTP requests are blocked
   - TLS 1.2+ required

2. **Certificate Pinning**
   - Prevents fake data injection
   - Validates server certificates
   - Configurable per endpoint

3. **Backend Proxy**
   - API keys never exposed to client
   - Rate limiting prevents abuse
   - Caching reduces API calls
   - Centralized security control

4. **Data Validation**
   - Price range validation
   - Volume validation
   - Outlier detection
   - Sanity checks on all data

5. **Secure Storage**
   - Portfolio data stored securely
   - No sensitive data in plaintext

### Security Best Practices

- Never embed API keys in the desktop app
- Always use the backend proxy for API calls
- Validate all data before processing
- Use HTTPS for all network communication
- Implement rate limiting on backend
- Monitor for suspicious activity

## Configuration

Edit `config.py` to customize:

- Backend API URL
- Security settings (HTTPS enforcement, certificate pinning)
- Data validation thresholds
- Cache duration
- Chart settings
- Analysis parameters

## Building for Distribution

### Code Signing (Windows)

For production distribution, you should code sign the executable:

```bash
# Create executable
pyinstaller --onefile --windowed --name Stocker main.py

# Sign with Authenticode (requires certificate)
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist/Stocker.exe
```

### Code Signing (macOS)

```bash
# Create app bundle
pyinstaller --onefile --windowed --name Stocker main.py

# Notarize (requires Apple Developer account)
xcrun notarytool submit Stocker.dmg --apple-id your@email.com --team-id TEAM_ID --password APP_PASSWORD
```

## Troubleshooting

### Backend Server Not Starting
- Check if port 5000 is available
- Ensure all dependencies are installed
- Check firewall settings

### No Data Retrieved
- Verify backend server is running
- Check internet connection
- Verify stock symbol is correct
- Check backend logs for errors

### Security Warnings
- Ensure backend is using HTTPS in production
- Configure certificate pinning for your backend
- Verify TLS version is 1.2 or higher

## Development

### Running Tests

```bash
# Test backend server
python -m pytest tests/

# Test security module
python -c "from security import get_secure_session; print('Security OK')"
```

### Adding New Indicators

1. Add calculation in `trading_analyzer.py` or `investing_analyzer.py`
2. Update chart generation in `chart_generator.py`
3. Add to reasoning output

## License

This project is provided as-is for educational and personal use.

## Disclaimer

**IMPORTANT**: This application provides informational predictions and analysis only. It is NOT financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results. Trading and investing involve risk of loss.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review backend server logs
3. Verify all dependencies are installed correctly

