import redis
import json
import math

r = redis.Redis(host="localhost", port=6380, decode_responses=True)

def clean_dict_nans(obj):
    """Recursively replaces NaN with None for valid JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: clean_dict_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_nans(i) for i in obj]
    return obj

def push_forecast(city, data):
    clean_data = clean_dict_nans(data)
    key = f"aqi:forecast:{city.lower()}"
    r.set(key, json.dumps(clean_data))
    print(f" Pushed clean data to {key}")