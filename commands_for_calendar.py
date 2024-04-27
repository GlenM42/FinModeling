from datetime import datetime, timedelta


def find_next_first_friday():
    """Calculate the next first Friday of the month."""
    today = datetime.today()
    current_month = today.month
    next_first_friday = today + timedelta(days=(4 - today.weekday() + 7) % 7)  # Next Friday
    while next_first_friday.month == current_month:
        next_first_friday += timedelta(days=7)
    return next_first_friday
