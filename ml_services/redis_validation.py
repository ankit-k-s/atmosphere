import redis
import json
import math

r = redis.Redis(host="localhost", port=6380, decode_responses=True)


def validate_json(obj):
    if isinstance(obj, float) and math.isnan(obj):
        raise ValueError("❌ Found NaN in JSON")

    if isinstance(obj, dict):
        for v in obj.values():
            validate_json(v)

    if isinstance(obj, list):
        for v in obj:
            validate_json(v)


def validate_redis(city):
    key = f"aqi:forecast:{city.lower()}"

    print(f"\n🔍 Checking Redis key: {key}")

    data = r.get(key)

    if data is None:
        print("❌ Key not found in Redis")
        return

    print("✅ Key exists")

    try:
        parsed = json.loads(data)
        print("✅ JSON parsing successful")
    except Exception as e:
        print("❌ JSON parsing failed:", e)
        return

    # Check NaN issues
    try:
        validate_json(parsed)
        print("✅ No NaN values found")
    except Exception as e:
        print(str(e))
        return

    # Basic structure check
    if "data" not in parsed:
        print("❌ Missing 'data' field")
        return

    if len(parsed["data"]) == 0:
        print("❌ No records inside data")
        return

    print(f"✅ Total records: {len(parsed['data'])}")

    # Print sample
    print("\n📊 Sample record:")
    print(json.dumps(parsed["data"][0], indent=2))


if __name__ == "__main__":
    validate_redis("Delhi")