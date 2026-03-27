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


def add_trading_days(start_date, trading_days: int):
    """Add N trading days to a date, skipping weekends and major US holidays.
    
    This ensures predictions get the full number of TRADING days to play out,
    instead of being shortchanged by weekends.
    
    Args:
        start_date: datetime object or date to start from
        trading_days: Number of trading days to add
        
    Returns:
        datetime with N trading days added
    """
    if trading_days <= 0:
        return start_date
    
    # Major US market holidays (month, day) — fixed-date holidays
    # For floating holidays (Thanksgiving, etc.), we approximate
    US_HOLIDAYS_FIXED = {
        (1, 1),   # New Year's Day
        (1, 15),  # MLK Day (approximate - 3rd Monday)
        (2, 19),  # Presidents Day (approximate - 3rd Monday)
        (7, 4),   # Independence Day
        (9, 1),   # Labor Day (approximate - 1st Monday)
        (11, 28), # Thanksgiving (approximate - 4th Thursday)
        (12, 25), # Christmas Day
    }
    
    current = start_date
    days_added = 0
    
    while days_added < trading_days:
        current += datetime.timedelta(days=1)
        # Skip weekends (5=Saturday, 6=Sunday)
        if current.weekday() >= 5:
            continue
        # Skip major holidays
        if (current.month, current.day) in US_HOLIDAYS_FIXED:
            continue
        days_added += 1
    
    return current
