import numpy as np
import pandas as pd


# -------------------------------
# SAFE AQI FUNCTIONS
# -------------------------------
def safe(x):
    return 0 if pd.isna(x) or x < 0 else x


def sub_pm25(x):
    x = safe(x)
    if x <= 30: return x * (50/30)
    elif x <= 60: return 50 + (x-30)*(50/30)
    elif x <= 90: return 100 + (x-60)*(100/30)
    elif x <= 120: return 200 + (x-90)*(100/30)
    elif x <= 250: return 300 + (x-120)*(100/130)
    else: return 400 + (x-250)


def sub_pm10(x):
    x = safe(x)
    if x <= 50: return x
    elif x <= 100: return 50 + (x-50)
    elif x <= 250: return 100 + (x-100)*(100/150)
    elif x <= 350: return 200 + (x-250)
    elif x <= 430: return 300 + (x-350)*(100/80)
    else: return 400 + (x-430)


def sub_no2(x):
    x = safe(x)
    if x <= 40: return x*(50/40)
    elif x <= 80: return 50+(x-40)*(50/40)
    elif x <= 180: return 100+(x-80)
    elif x <= 280: return 200+(x-180)
    elif x <= 400: return 300+(x-280)
    else: return 400+(x-400)


def sub_so2(x):
    x = safe(x)
    if x <= 40: return x*(50/40)
    elif x <= 80: return 50+(x-40)*(50/40)
    elif x <= 380: return 100+(x-80)*(100/300)
    elif x <= 800: return 200+(x-380)*(100/420)
    elif x <= 1600: return 300+(x-800)*(100/800)
    else: return 400+(x-1600)


def sub_co(x):
    x = safe(x)
    if x <= 1: return x*50
    elif x <= 2: return 50+(x-1)*50
    elif x <= 10: return 100+(x-2)*(100/8)
    elif x <= 17: return 200+(x-10)*(100/7)
    elif x <= 34: return 300+(x-17)*(100/17)
    else: return 400+(x-34)


def compute_aqi(row):
    return int(max(
        sub_pm25(row.get("PM2_5_ug_m3")),
        sub_pm10(row.get("PM10_ug_m3")),
        sub_no2(row.get("NO2_ug_m3")),
        sub_so2(row.get("SO2_ug_m3")),
        sub_co(row.get("CO_mg_m3"))
    ))


# -------------------------------
# FINAL FORMATTER
# -------------------------------
def format_for_redis(df, city):

    df_city = df[df["city"].str.lower() == city.lower()].copy()

    output = {
        "city": city,
        "generated_at": str(pd.Timestamp.utcnow()),
        "data": []
    }

    for _, row in df_city.iterrows():

        # wind calculation
        wx = 0 if pd.isna(row.get("wind_x")) else row["wind_x"]
        wy = 0 if pd.isna(row.get("wind_y")) else row["wind_y"]

        wind_speed = np.sqrt(wx**2 + wy**2)
        angle = np.degrees(np.arctan2(wy, wx))
        wind_dir = (270 - angle) % 360

        record = {
            "timestamp": str(row["ds"]),
            "city": row["city"],
            "hotspot": row["station"],

            "latitude": round(row.get("latitude", 0), 5),
            "longitude": round(row.get("longitude", 0), 5),

            "aqi": compute_aqi(row),

            # FLAT STRUCTURE (safe)
            "pm25": round(row.get("PM2_5_ug_m3", 0), 2),
            "pm10": round(row.get("PM10_ug_m3", 0), 2),
            "no2": round(row.get("NO2_ug_m3", 0), 2),
            "so2": round(row.get("SO2_ug_m3", 0), 2),
            "co": round(row.get("CO_mg_m3", 0), 2),

            "windSpeed": round(wind_speed, 2),
            "windDirection": round(wind_dir, 2),

            # keep schema stable
            "no": None,
            "nox": None,
            "nh3": None,
            "ozone": None
        }

        output["data"].append(record)

    return output