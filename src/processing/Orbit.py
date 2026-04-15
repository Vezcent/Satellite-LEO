import os

def process_orbit():
    input_path = r"E:\Satellite LEO\dataset\PROBA-1_Orbit_Raw.txt"
    output_path = r"E:\Satellite LEO\preprocessed-data\initial_state.txt"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # Read the first two lines
        line1 = f.readline()
        line2 = f.readline()
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(line1)
        f.write(line2)

    print(f"Successfully extracted initial state to {output_path}")

if __name__ == "__main__":
    process_orbit()
