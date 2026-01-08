# 🔔 How to Add Alerts to Your Alert System

## Overview

The Stocker app includes a comprehensive alert system that allows you to create alerts for:
- **Price alerts** - Get notified when a stock reaches a target price
- **Prediction alerts** - Get notified about new buy/sell recommendations
- **Portfolio alerts** - Get notified about portfolio events

---

## Method 1: Using the UI (Easiest)

### Step 1: Open the Alerts Tab
1. Launch the Stocker app
2. Click on the **"🔔 Alerts"** tab in the main window

### Step 2: Fill in Alert Details
In the "Create Alert" section, you'll see:

- **Alert Type**: Choose from dropdown
  - `price_alert` - Alert when stock price reaches target
  - `prediction_alert` - Alert for new predictions
  - `portfolio_alert` - Alert for portfolio events

- **Symbol**: Enter stock symbol (e.g., `AAPL`, `MSFT`)
  - Required for `price_alert`
  - Optional for other types

- **Message**: Enter your alert message
  - Example: "AAPL reached $150"
  - Example: "New BUY signal for MSFT"

- **Priority**: Choose priority level
  - `low` - Low priority alerts
  - `medium` - Medium priority (default)
  - `high` - High priority alerts
  - `critical` - Critical alerts

### Step 3: Create the Alert
1. Click the **"Create Alert"** button
2. Your alert will be created and appear in the alerts list
3. Unread alerts will show a badge count in the tab title

### Example: Creating a Price Alert
1. **Alert Type**: Select `price_alert`
2. **Symbol**: Enter `AAPL`
3. **Message**: Enter "AAPL price alert"
4. **Priority**: Select `high`
5. Click **"Create Alert"**

The system will automatically fetch the current price and create an alert.

---

## Method 2: Programmatically (Python Code)

### Basic Alert Creation

```python
from alert_system import AlertSystem
from pathlib import Path

# Initialize alert system
data_dir = Path("data")  # Your data directory
alert_system = AlertSystem(data_dir)

# Create a simple alert
alert_system.add_alert(
    alert_type='price_alert',
    message='AAPL reached $150',
    priority='high',
    data={'symbol': 'AAPL', 'price': 150.00}
)
```

### Price Alert

```python
# Create price alert (automatically fetches current price)
alert_system.create_price_alert(
    symbol='AAPL',
    current_price=145.50,
    target_price=150.00,
    direction='above'  # or 'below'
)
```

### Prediction Alert

```python
# Create prediction alert
alert_system.create_prediction_alert(
    symbol='MSFT',
    action='BUY',
    confidence=85.5
)
```

### Portfolio Alert

```python
# Create portfolio alert
alert_system.create_portfolio_alert(
    message='Portfolio value exceeded $10,000',
    portfolio_data={
        'total_value': 10500.00,
        'profit': 500.00
    }
)
```

---

## Method 3: Automatic Alerts (Built-in)

The app automatically creates alerts in certain situations:

### 1. **Price Alerts** (When analyzing stocks)
- Created when you analyze a stock and set price targets
- Automatically monitors price movements

### 2. **Prediction Alerts** (When predictions are made)
- Created when new BUY/SELL recommendations are generated
- Includes confidence level and reasoning

### 3. **Portfolio Alerts** (Portfolio events)
- Created when portfolio value changes significantly
- Created when trades are executed
- Created when profit targets are reached

---

## Alert Types Explained

### 1. Price Alert (`price_alert`)
**Purpose**: Get notified when a stock reaches a target price

**When to use**:
- You want to buy when price drops to a certain level
- You want to sell when price reaches a target
- You're monitoring a stock for entry/exit points

**Example**:
```python
alert_system.create_price_alert(
    symbol='TSLA',
    current_price=200.00,
    target_price=220.00,
    direction='above'
)
# Alert: "TSLA price is above $220.00 (current: $200.00)"
```

### 2. Prediction Alert (`prediction_alert`)
**Purpose**: Get notified about new trading recommendations

**When to use**:
- You want alerts for new BUY signals
- You want alerts for new SELL signals
- You want to track high-confidence predictions

**Example**:
```python
alert_system.create_prediction_alert(
    symbol='NVDA',
    action='BUY',
    confidence=92.5
)
# Alert: "New BUY recommendation for NVDA (confidence: 92.5%)"
```

### 3. Portfolio Alert (`portfolio_alert`)
**Purpose**: Get notified about portfolio events

**When to use**:
- Portfolio value milestones
- Profit/loss thresholds
- Trade executions
- Risk warnings

**Example**:
```python
alert_system.create_portfolio_alert(
    message='Portfolio profit exceeded 10%',
    portfolio_data={
        'total_value': 11000.00,
        'initial_value': 10000.00,
        'profit_pct': 10.0
    }
)
```

---

## Managing Alerts

### View Alerts
- All alerts are displayed in the **"🔔 Alerts"** tab
- Unread alerts are highlighted
- Alerts are sorted by timestamp (newest first)

### Mark as Read
- Click on an alert to mark it as read
- Click **"Mark All Read"** to mark all alerts as read

### Delete Alerts
- Right-click on an alert to delete it
- Old alerts are automatically cleared after 30 days

### Filter Alerts
- Filter by type (price, prediction, portfolio)
- Filter by priority (low, medium, high, critical)
- Filter by read/unread status

---

## Alert Priorities

### Low Priority
- Informational alerts
- Non-urgent notifications
- Background updates

### Medium Priority (Default)
- Standard alerts
- Regular notifications
- Normal importance

### High Priority
- Important alerts
- Action required
- Significant events

### Critical Priority
- Urgent alerts
- Immediate action needed
- Critical events

---

## Advanced Usage

### Register Callbacks

You can register callbacks to be notified when alerts are created:

```python
def on_alert_created(alert):
    print(f"New alert: {alert.message}")
    # Send email, show notification, etc.

alert_system.register_callback(on_alert_created)
```

### Get Alerts by Type

```python
# Get all price alerts
price_alerts = alert_system.get_alerts_by_type('price_alert')

# Get all unread alerts
unread_alerts = alert_system.get_unread_alerts()
```

### Custom Alert Data

```python
# Create alert with custom data
alert_system.add_alert(
    alert_type='custom_alert',
    message='Custom alert message',
    priority='medium',
    data={
        'symbol': 'AAPL',
        'price': 150.00,
        'indicator': 'RSI',
        'value': 70
    }
)
```

---

## Best Practices

### 1. **Use Descriptive Messages**
- Include stock symbol
- Include relevant price/values
- Include context

### 2. **Set Appropriate Priorities**
- Use `critical` sparingly
- Use `high` for important alerts
- Use `medium` for standard alerts
- Use `low` for informational alerts

### 3. **Clean Up Old Alerts**
- Review alerts regularly
- Delete alerts you no longer need
- Use "Mark All Read" to clear unread badges

### 4. **Use Price Alerts for Entry/Exit**
- Set price alerts at support/resistance levels
- Set alerts for profit targets
- Set alerts for stop-loss levels

### 5. **Monitor Prediction Alerts**
- Pay attention to high-confidence predictions
- Review prediction alerts before trading
- Track prediction accuracy over time

---

## Troubleshooting

### "Alert system not available"
- Make sure `alert_system.py` is in your project
- Check that `HAS_NEW_MODULES` is True
- Restart the app

### Alerts not showing
- Check the Alerts tab
- Click "Refresh" button
- Check if alerts are marked as read

### Price alerts not triggering
- Price alerts are created but not automatically monitored
- You need to check prices manually or integrate with monitoring system
- Consider using prediction alerts instead

---

## Example Workflow

### Scenario: Monitor AAPL for Buy Entry

1. **Create Price Alert**:
   ```
   Alert Type: price_alert
   Symbol: AAPL
   Message: "AAPL buy entry alert"
   Priority: high
   ```

2. **Create Prediction Alert** (automatic):
   - When analyzing AAPL, if BUY signal appears
   - Alert is automatically created

3. **Monitor Alerts**:
   - Check Alerts tab regularly
   - Review unread alerts
   - Take action when alerts trigger

---

## Summary

✅ **UI Method**: Use the Alerts tab → Fill form → Click "Create Alert"  
✅ **Code Method**: Use `alert_system.add_alert()` or helper methods  
✅ **Automatic**: Alerts are created automatically for predictions and portfolio events  
✅ **Manage**: View, mark as read, delete alerts in the Alerts tab  

**Quick Start**: Go to **"🔔 Alerts"** tab → Fill in the form → Click **"Create Alert"**! 🎯
