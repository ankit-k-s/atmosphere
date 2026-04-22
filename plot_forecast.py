import pandas as pd
import plotly.graph_objects as go
import os

# --- CONFIGURATION ---
# Pointing to the original Foundation Model results!
RESULTS_FILE = "data/processed/evaluation/tide_96hr_results.parquet"
OUTPUT_DIR = "data/processed/evaluation/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_station_forecast(target_station_id, pollutant='PM2_5_ug_m3'):
    print(f"Loading evaluation data for {pollutant}...")
    df = pd.read_parquet(RESULTS_FILE)
    
    # Filter for the specific pollutant
    df_pol = df[df['pollutant'] == pollutant]
    
    # Find all unique IDs that contain our target station
    # (Because the IDs look like: Delhi_ITO_c0_PM2_5_ug_m3)
    station_chunks = [uid for uid in df_pol['unique_id'].unique() if target_station_id in uid]
    
    if not station_chunks:
        print(f"❌ Error: Could not find station '{target_station_id}' in the results.")
        print("Available cities/stations include:")
        # Print a small sample of available names to help the user
        sample_ids = df_pol['unique_id'].head(10).tolist()
        for s in sample_ids:
            print(f"  - {s.split('_c')[0]}")
        return

    # Grab the first valid continuous chunk for this station
    target_uid = station_chunks[0]
    plot_df = df_pol[df_pol['unique_id'] == target_uid].sort_values('ds')
    
    print(f"Plotting 96-hour horizon for: {target_uid}")
    
    # --- BUILD THE INTERACTIVE PLOT ---
    fig = go.Figure()

    # 1. Plot the Ground Truth (What actually happened)
    fig.add_trace(go.Scatter(
        x=plot_df['ds'], y=plot_df['y'],
        mode='lines+markers',
        name='Actual Reality (Ground Truth)',
        line=dict(color='black', width=2),
        marker=dict(size=4)
    ))

    # 2. Plot the TiDE Prediction
    fig.add_trace(go.Scatter(
        x=plot_df['ds'], y=plot_df['prediction'],
        mode='lines',
        name='TiDE 96-hr Forecast',
        line=dict(color='red', width=3, dash='dot')
    ))

    # Formatting the dashboard
    city_name = target_station_id.split('_')[0]
    fig.update_layout(
        title=f"Atmospheric Intelligence: {city_name} ({target_station_id}) - {pollutant}",
        xaxis_title="Time (Future 96 Hours)",
        yaxis_title=f"Concentration ({pollutant})",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Save and display
    output_path = os.path.join(OUTPUT_DIR, f"{target_station_id}_{pollutant}_forecast.html")
    fig.write_html(output_path)
    print(f"✅ Success! Interactive graph saved to: {output_path}")

if __name__ == "__main__":
    # Change 'Delhi_ITO' to any station name that exists in your dataset!
    # Example: 'Mumbai_Bandra', 'Kolkata_FortWilliam', 'Bengaluru_BTM', etc.
    TARGET_STATION = "Delhi_ITO" 
    
    # Generate plots for both PM2.5 and NO2 for comparison
    plot_station_forecast(TARGET_STATION, pollutant='PM2_5_ug_m3')
    plot_station_forecast(TARGET_STATION, pollutant='NO2_ug_m3')