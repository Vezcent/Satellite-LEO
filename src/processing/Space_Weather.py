import os
import pandas as pd
import numpy as np

def process_space_weather():
    input_path = r"E:\Satellite LEO\dataset\SpaceWeather_Raw.csv"
    output_path = r"E:\Satellite LEO\preprocessed-data\space_weather.csv"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Cut Header: Delete introductory lines
    # Find the line that starts with 'YEAR'
    header_idx = 0
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('YEAR'):
                header_idx = i
                break
                
    # Read CSV. delim_whitespace=True handles the irregular spacing.
    df = pd.read_csv(input_path, delim_whitespace=True, skiprows=header_idx)
    
    # 2. Filter columns: Keep YEAR, DOY, HR, 5, 6, 7, 8
    cols_to_keep = ['YEAR', 'DOY', 'HR', '5', '6', '7', '8']
    df = df[cols_to_keep]
    
    # Rename columns to the final desired format
    df.columns = ['Year', 'DOY', 'Hour', 'Kp', 'Dst', 'Ap', 'F10.7']
    
    # 3. Eliminate NaN traps
    # The numbers 999.9, 99999.99 and variations are used for missing data.
    missing_flags = [99.0, 99.9, 999.0, 999.9, 9999.0, 9999.9, 99999.0, 99999.9, 99999.99, 999999.0, 999999.9, 999999.99]
    for col in ['Kp', 'Dst', 'Ap', 'F10.7']:
        df[col] = df[col].replace(missing_flags, np.nan)
        
    # 4. Linear interpolation
    df[['Kp', 'Dst', 'Ap', 'F10.7']] = df[['Kp', 'Dst', 'Ap', 'F10.7']].interpolate(method='linear')
    # Use bfill and ffill to handle edge cases if the very first or last row is NaN
    df[['Kp', 'Dst', 'Ap', 'F10.7']] = df[['Kp', 'Dst', 'Ap', 'F10.7']].bfill().ffill()
    
    # 5. Normalize Kp: Divide the Kp column by 10 (Example 53 -> 5.3)
    df['Kp'] = df['Kp'] / 10.0
    
    # Convert fields to appropriate numeric types for cleanly formatted output
    df['Year'] = df['Year'].astype(int)
    df['DOY'] = df['DOY'].astype(int)
    df['Hour'] = df['Hour'].astype(int)
    df['Dst'] = df['Dst'].round().astype(int)
    df['Ap'] = df['Ap'].round().astype(int)
    df['F10.7'] = df['F10.7'].round(1)
    df['Kp'] = df['Kp'].round(1)
    
    # 6. Save Output
    df.to_csv(output_path, index=False)
    print(f"Space Weather data processed and saved to {output_path}")

if __name__ == "__main__":
    process_space_weather()
