import datetime

def format_time_now(format_str: str = "%Y%m%d%H%M%S") -> str:
    """Format current time as a string."""
    return datetime.datetime.now().strftime(format_str)

