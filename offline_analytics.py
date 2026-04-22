import pandas as pd
import numpy as np
import json
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.signal import correlate

# --- MOCK DB FETCHER ---
def fetch_timeseries_data():
    """
    In production, this queries TimescaleDB. 
    Here, we generate mock data to prove the mathematics work.
    """
    np.random.seed(42)
    t = np.arange(100)
    
    # Station A creates a spike. Station B catches that spike 2 hours later.
    station_a_pm25 = 50 + 20 * np.sin(t / 5) + np.random.normal(0, 5, 100)
    station_b_pm25 = 50 + 20 * np.sin((t - 2) / 5) + np.random.normal(0, 5, 100) # Lagged by 2
    
    # Wind speed strongly correlates with Station B's PM2.5 (Advection)
    wind_speed = station_b_pm25 * 0.5 + np.random.normal(0, 2, 100)
    
    return {
        "Anand Vihar": station_a_pm25,
        "ITO": station_b_pm25
    }, wind_speed

def calculate_causality(pm25_series, wind_speed_series, station_name):
    """Feature 2: Does Wind (Advection) or Local Emissions cause the pollution?"""
    print(f"\n[CAUSALITY TEST] Analyzing {station_name}...")
    df = pd.DataFrame({'pm25': pm25_series, 'wind': wind_speed_series})
    max_lag = 3 
    
    try:
        # Does Wind Granger-cause PM2.5?
        wind_gc = grangercausalitytests(df[['pm25', 'wind']], maxlag=max_lag, verbose=False)
        p_values = [wind_gc[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
        best_p_val = min(p_values)
        
        if best_p_val < 0.05:
            print(f"Result: ADVECTION-DRIVEN (Imported). p-value = {best_p_val:.4f}")
            return "Advection-Driven"
        else:
            print(f"Result: EMISSION-DRIVEN (Local Violation). p-value = {best_p_val:.4f}")
            return "Emission-Driven"
    except Exception as e:
        print("Error computing causality.")

def build_influence_graph(timeseries_dict):
    """Feature 3: Station Influence Graph using Lagged Cross-Correlation"""
    print("\n[INFLUENCE GRAPH] Calculating propagation edges...")
    stations = list(timeseries_dict.keys())
    edges = []
    
    for i in range(len(stations)):
        for j in range(len(stations)):
            if i == j: continue
                
            station_a, station_b = stations[i], stations[j]
            series_a, series_b = timeseries_dict[station_a], timeseries_dict[station_b]
            
            # Normalize
            a_norm = (series_a - np.mean(series_a)) / (np.std(series_a) + 1e-6)
            b_norm = (series_b - np.mean(series_b)) / (np.std(series_b) + 1e-6)
            
            correlation = correlate(a_norm, b_norm, mode='full')
            lags = np.arange(-len(a_norm) + 1, len(a_norm))
            
            max_corr_idx = np.argmax(correlation)
            best_lag = lags[max_corr_idx]
            max_corr_value = correlation[max_corr_idx] / len(series_a)
            
            # If A correlates with B at a lag > 0, pollution drifted from A to B.
            if max_corr_value > 0.5 and best_lag > 0:
                print(f"Edge Created: {station_a} -> {station_b} | Weight: {max_corr_value:.2f} | Delay: {best_lag} hours")
                edges.append({"source": station_a, "target": station_b, "weight": max_corr_value, "lag_hours": int(best_lag)})
                
    return edges

if __name__ == "__main__":
    # 1. Fetch Mock DB Data
    station_data, wind_data = fetch_timeseries_data()
    
    # 2. Run Causal Analysis on a specific station
    calculate_causality(station_data["ITO"], wind_data, "ITO")
    
    # 3. Build the directed influence graph between all stations
    graph_edges = build_influence_graph(station_data)
    
    # In production, save graph_edges to Redis for the UI to query