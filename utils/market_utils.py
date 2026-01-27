import datetime
import pytz

def is_market_open():
    """
    Checks if the US market (NYSE/NASDAQ) is currently open.
    Standard hours: 9:30 AM - 4:00 PM ET, Monday - Friday.
    Does not account for holidays.
    """
    # Define Wall Street timezone
    eastern = pytz.timezone('US/Eastern')
    
    # Get current time in Eastern Time
    now_et = datetime.datetime.now(eastern)
    
    # Check if it's a weekday (0=Monday, 6=Sunday)
    if now_et.weekday() >= 5:
        return False
        
    # Define market hours
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_et <= market_close

def get_market_status_message():
    """Returns a user-friendly message about the market status."""
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.datetime.now(eastern)
    
    if now_et.weekday() >= 5:
        return "Market is closed (Weekend)"
    
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now_et < market_open:
        return f"Market is closed (Opens at 9:30 AM ET, currently {now_et.strftime('%I:%M %p')} ET)"
    elif now_et > market_close:
        return f"Market is closed (Closed at 4:00 PM ET, currently {now_et.strftime('%I:%M %p')} ET)"
    else:
        return "Market is open"
