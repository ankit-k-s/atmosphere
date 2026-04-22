import numpy as np
import json
from physics_engine import calculate_grid_dimensions, advect_and_diffuse_vectorized

def run_adjoint_tracer(input_file, target_x, target_y, target_time_idx, hours_back=6):
    """
    Adjoint Tracer utilizing the high-speed vectorized physics engine.
    Limits backward tracing to 6 hours to minimize diffusion-loss inaccuracy.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    res_x, res_y = data['spatial_meta']['grid_resolution']
    dx, dy = calculate_grid_dimensions(data['spatial_meta'])
    
    # 1. Inject Virtual Tracer Mass
    tracer_grid = np.zeros((res_y, res_x))
    tracer_grid[target_y, target_x] = 1000.0 
    
    start_t = target_time_idx
    end_t = max(0, start_time_idx - hours_back)
    
    # 2. Step backward through history
    for t in range(start_t, end_t, -1):
        # Reverse wind vectors to blow tracer backwards
        rev_u = -np.array(data['timeline'][t]['wind_u']).reshape(res_y, res_x)
        rev_v = -np.array(data['timeline'][t]['wind_v']).reshape(res_y, res_x)
        
        for _ in range(60):
            # Run the vectorized physics engine backwards
            tracer_grid = advect_and_diffuse_vectorized(tracer_grid, rev_u, rev_v, dx, dy, 60)
            
    # Locate highest remaining tracer concentration
    origin_y, origin_x = np.unravel_index(np.argmax(tracer_grid), tracer_grid.shape)
    
    # Confidence drops exponentially as 'hours_back' increases due to diffusion loss
    confidence = max(0, 100 - (hours_back * 12.5)) 
    
    print(f"--- TRACE RESULTS ---")
    print(f"Spike at ({target_x}, {target_y}) likely originated near ({origin_x}, {origin_y})")
    print(f"Confidence: ~{confidence}% (Subject to atmospheric diffusion loss)")
    
    return origin_x, origin_y

if __name__ == "__main__":
    run_adjoint_tracer("Delhi_96hr_digital_twin.json", 25, 30, 48, hours_back=4)