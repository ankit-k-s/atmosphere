import numpy as np
import json
import math

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/visualizations/Delhi_96hr_digital_twin.json"
OUTPUT_FILE = "data/processed/visualizations/Delhi_96hr_digital_twin_physics.json"

# Physics Parameters
DT_HOUR = 3600
SUB_STEPS = 60
DT_SUB = DT_HOUR / SUB_STEPS

# Tuning as per CFD critique
DIFFUSION_COEFF = 12.0  # Reduced from 25.0 to keep plumes sharp and realistic
BACKGROUND_PM25 = 80.0  # Open Boundary Inflow: Ambient pollution outside city limits

def calculate_grid_dimensions(spatial_meta):
    lat_min = spatial_meta['lat_min']
    lat_max = spatial_meta['lat_max']
    lon_min = spatial_meta['lon_min']
    lon_max = spatial_meta['lon_max']
    res_x, res_y = spatial_meta['grid_resolution']
    
    dy = ((lat_max - lat_min) * 111320.0) / res_y
    mean_lat = math.radians((lat_min + lat_max) / 2.0)
    dx = ((lon_max - lon_min) * 111320.0 * math.cos(mean_lat)) / res_x
    return dx, dy

def advect_and_diffuse(C, U, V, dx, dy, dt):
    res_y, res_x = C.shape
    next_C = np.copy(C)
    
    # 1. Map physical wind to matrix index directions
    # East (+U) = +x direction. North (+V) = -y direction (since matrix row 0 is North).
    U_idx = U
    V_idx = -V 
    
    # 2. X-Fluxes (East-West Faces)
    flux_x = np.zeros((res_y, res_x + 1))
    for y in range(res_y):
        for x in range(res_x + 1):
            if x == 0: # Western Boundary Inflow/Outflow
                u_face = U_idx[y, 0]
                flux_x[y, x] = u_face * BACKGROUND_PM25 if u_face > 0 else u_face * C[y, 0]
            elif x == res_x: # Eastern Boundary Inflow/Outflow
                u_face = U_idx[y, -1]
                flux_x[y, x] = u_face * C[y, -1] if u_face > 0 else u_face * BACKGROUND_PM25
            else: # Internal Upwind Scheme
                u_face = (U_idx[y, x-1] + U_idx[y, x]) / 2.0
                flux_x[y, x] = u_face * C[y, x-1] if u_face > 0 else u_face * C[y, x]
                
    # 3. Y-Fluxes (North-South Faces)
    flux_y = np.zeros((res_y + 1, res_x))
    for y in range(res_y + 1):
        for x in range(res_x):
            if y == 0: # Northern Boundary Inflow/Outflow
                v_face = V_idx[0, x]
                flux_y[y, x] = v_face * BACKGROUND_PM25 if v_face > 0 else v_face * C[0, x]
            elif y == res_y: # Southern Boundary Inflow/Outflow
                v_face = V_idx[-1, x]
                flux_y[y, x] = v_face * C[-1, x] if v_face > 0 else v_face * BACKGROUND_PM25
            else: # Internal Upwind Scheme
                v_face = (V_idx[y-1, x] + V_idx[y, x]) / 2.0
                flux_y[y, x] = v_face * C[y-1, x] if v_face > 0 else v_face * C[y, x]
                
    # 4. Apply Net Flux & Diffusion
    for y in range(res_y):
        for x in range(res_x):
            # Mass Conservation: Amount entering minus amount leaving
            net_adv = (flux_x[y, x] - flux_x[y, x+1]) / dx + (flux_y[y, x] - flux_y[y+1, x]) / dy
            
            c_center = C[y, x]
            c_left   = C[y, x-1] if x > 0 else BACKGROUND_PM25
            c_right  = C[y, x+1] if x < res_x - 1 else BACKGROUND_PM25
            c_up     = C[y-1, x] if y > 0 else BACKGROUND_PM25
            c_down   = C[y+1, x] if y < res_y - 1 else BACKGROUND_PM25
            
            diff_x = DIFFUSION_COEFF * (c_right - 2*c_center + c_left) / (dx**2)
            diff_y = DIFFUSION_COEFF * (c_down - 2*c_center + c_up) / (dy**2)
            
            next_C[y, x] += dt * (net_adv + diff_x + diff_y)
            next_C[y, x] = max(0.0, next_C[y, x]) # Clip impossible negative mass
            
    return next_C

def process_physics():
    print("Loading Base Digital Twin...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        
    res_x, res_y = data['spatial_meta']['grid_resolution']
    dx, dy = calculate_grid_dimensions(data['spatial_meta'])
    
    print(f"Calculated Geodesic Grid: DX = {dx:.1f}m, DY = {dy:.1f}m")
    print("Executing Mass-Conserving CFD Engine...")
    
    physical_state = np.array(data['timeline'][0]['pm25']).reshape(res_y, res_x)
    
    for i in range(1, len(data['timeline'])):
        curr_hour = data['timeline'][i]
        
        u_matrix = np.array(curr_hour['wind_u']).reshape(res_y, res_x)
        v_matrix = np.array(curr_hour['wind_v']).reshape(res_y, res_x)
        
        tide_prev = np.array(data['timeline'][i-1]['pm25']).reshape(res_y, res_x)
        tide_curr = np.array(curr_hour['pm25']).reshape(res_y, res_x)
        source_rate = (tide_curr - tide_prev) / SUB_STEPS
        
        # --- MASS CONSERVATION TRACKER ---
        mass_before = np.sum(physical_state)
        
        for step in range(SUB_STEPS):
            physical_state = advect_and_diffuse(physical_state, u_matrix, v_matrix, dx, dy, DT_SUB)
            physical_state += source_rate
            physical_state = np.clip(physical_state, 0.0, 1000.0) 
            
        mass_after = np.sum(physical_state)
        source_added = np.sum(source_rate) * SUB_STEPS
        boundary_exchange = mass_after - (mass_before + source_added)
        
        data['timeline'][i]['pm25'] = np.round(physical_state.flatten(), 2).tolist()
        
        if i % 12 == 0:
            print(f"Frame [{i}/{len(data['timeline'])}] | Total Mass: {mass_after:.1f} | " 
                  f"TiDE Generated: {source_added:+.1f} | Boundary Exchange: {boundary_exchange:+.1f}")

    print("Packaging Research-Grade Payload...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f)
        
    print(f" CFD Simulation Complete. Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_physics()