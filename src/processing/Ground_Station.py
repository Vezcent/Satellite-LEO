import csv
import json
import os

def process_ground_stations():
    input_path = r"E:\Satellite LEO\dataset\ground_stations_raw.txt"
    output_path = r"E:\Satellite LEO\preprocessed-data\ground_stations.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ground_stations = []
    
    with open(input_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            station = {
                "id": row["ID"],
                "name": row["Name"],
                "country": row["Country"],
                "latitude_deg": float(row["Latitude"]),
                "longitude_deg": float(row["Longitude"]),
                "altitude_m": float(row["Altitude"]),
                "min_elevation_mask_deg": float(row["Min_Elevation"]),
                "role": row["Role"]
            }
            ground_stations.append(station)
            
    output_dict = {
        "ground_stations": ground_stations
    }
    
    with open(output_path, mode='w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2)
        
    print(f"Successfully processed ground stations and saved to {output_path}")

if __name__ == "__main__":
    process_ground_stations()
