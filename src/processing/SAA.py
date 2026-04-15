import os

def process_files():
    base_dir = r"E:\Satellite LEO\dataset"
    out_dir = r"E:\Satellite LEO\preprocessed-data"
    os.makedirs(out_dir, exist_ok=True)
    
    spenvis_path = os.path.join(base_dir, "spenvis_world.csv")
    flux_10_path = os.path.join(base_dir, "flux_10.csv")
    flux_30_path = os.path.join(base_dir, "flux_30.csv")
    out_path = os.path.join(out_dir, "saa_heatmap_600km.csv")

    def get_data_lines(filepath):
        if not os.path.exists(filepath):
            return []
        lines = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.split(',')]
                if not parts:
                    continue
                # check if first element is a scientific number like altitude and it has enough columns
                try:
                    float(parts[0])
                    if len(parts) >= 5:
                        lines.append(parts)
                except ValueError:
                    continue
        return lines

    spenvis_lines = get_data_lines(spenvis_path)
    flux_10_lines = get_data_lines(flux_10_path)
    flux_30_lines = get_data_lines(flux_30_path)

    with open(out_path, 'w') as out_f:
        out_f.write("Latitude,Longitude,Flux_10MeV,Flux_30MeV\n")
        
        for i, spenvis in enumerate(spenvis_lines):
            try:
                lat = float(spenvis[1]) if len(spenvis) > 1 else 0.0
                lon = float(spenvis[2]) if len(spenvis) > 2 else 0.0
            except ValueError:
                continue
            
            f10 = 0.0
            if i < len(flux_10_lines) and len(flux_10_lines[i]) > 2:
                try:
                    f10 = float(flux_10_lines[i][2])
                except ValueError:
                    pass
                    
            f30 = 0.0
            if i < len(flux_30_lines) and len(flux_30_lines[i]) > 2:
                try:
                    f30 = float(flux_30_lines[i][2])
                except ValueError:
                    pass
                    
            out_f.write(f"{lat},{lon},{f10},{f30}\n")
            
if __name__ == '__main__':
    process_files()
    print("Processing complete!")
