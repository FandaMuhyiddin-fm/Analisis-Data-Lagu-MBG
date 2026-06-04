import datetime
import pandas as pd

def format_number(num):
    """
    Format large numbers into reader-friendly string (e.g., 1.2M, 45.3K, 120).
    Handles integers, floats, and robustly handles NaN/None values.
    """
    if num is None or pd.isna(num):
        return "-"
    
    try:
        val = float(num)
        if val >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        elif val >= 1_000:
            return f"{val / 1_000:.1f}K"
        elif val.is_integer():
            return f"{int(val):,}"
        else:
            return f"{val:.1f}"
    except (ValueError, TypeError):
        return str(num)

def format_duration(seconds):
    """
    Format video duration in seconds into 'MM:SS' or 'SS.S s'.
    """
    if seconds is None or pd.isna(seconds):
        return "-"
    
    try:
        sec = float(seconds)
        if sec >= 60:
            minutes = int(sec // 60)
            remaining_seconds = int(sec % 60)
            return f"{minutes:02d}:{remaining_seconds:02d}"
        else:
            return f"{sec:.1f}s"
    except (ValueError, TypeError):
        return str(seconds)

def format_date(iso_string):
    """
    Format ISO timestamp into 'YYYY-MM-DD HH:MM' or simply 'YYYY-MM-DD'.
    """
    if iso_string is None or pd.isna(iso_string):
        return "-"
    
    try:
        # Convert string to datetime
        dt = pd.to_datetime(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(iso_string)
