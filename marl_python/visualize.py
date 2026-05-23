import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_latest_log():
    # Search in controller_csharp/logs or build bin/Release/Debug folders
    paths = [
        "../controller_csharp/logs/session_*.csv",
        "../controller_csharp/bin/*/net10.0/logs/session_*.csv",
        "controller_csharp/logs/session_*.csv",
        "logs/session_*.csv"
    ]
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        return None
    # Sort by modification time
    files.sort(key=os.path.getmtime)
    return files[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize S-MAS Telemetry CSV Logs")
    parser.add_argument("--file", type=str, default=None, help="Path to telemetry CSV log")
    args = parser.parse_args()

    csv_path = args.file
    if csv_path is None:
        csv_path = get_latest_log()
        if csv_path is None:
            print("Error: No telemetry log found. Please specify --file <path>.")
            exit(1)
        print(f"Loading latest log: {csv_path}")
    else:
        print(f"Loading log: {csv_path}")

    df = pd.read_csv(csv_path)

    # If sim_time_h is not present, calculate it from sim_time_s
    if "sim_time_h" not in df.columns and "sim_time_s" in df.columns:
        df["sim_time_h"] = df["sim_time_s"] / 3600.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # 1. Altitude
    axes[0].plot(df["sim_time_h"], df["altitude_km"], label="Altitude")
    axes[0].set_ylabel("Altitude (km)")
    if "target_alt_km" in df.columns:
        axes[0].plot(df["sim_time_h"], df["target_alt_km"], color='r', linestyle='--', label='Target')
    else:
        # Fallback
        axes[0].axhline(y=600, color='r', linestyle='--', label='Default Target (600km)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Battery SoC
    axes[1].plot(df["sim_time_h"], df["battery_soc"] * 100, color='g', label="SoC")
    axes[1].set_ylabel("SoC (%)")
    axes[1].grid(True, alpha=0.3)

    # 3. Fuel
    if "fuel_fraction" in df.columns:
        axes[2].plot(df["sim_time_h"], df["fuel_fraction"] * 100, color='orange', label="Fuel")
        axes[2].axhline(y=10, color='r', linestyle='--', label='Critical Threshold (10%)')
        axes[2].set_ylabel("Fuel (%)")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "Fuel data not logged", transform=axes[2].transAxes, ha='center')
    axes[2].grid(True, alpha=0.3)

    # 4. Temperature
    if "temp_bus" in df.columns:
        axes[3].plot(df["sim_time_h"], df["temp_bus"], label="Bus", alpha=0.8)
        axes[3].plot(df["sim_time_h"], df["temp_battery"], label="Battery", alpha=0.8)
        axes[3].plot(df["sim_time_h"], df["temp_payload"], label="Payload", alpha=0.8)
        axes[3].axhspan(-10, 45, alpha=0.1, color='green', label='Battery Safe zone [-10, 45]')
        axes[3].set_ylabel("Temp (°C)")
        axes[3].legend()
    else:
        axes[3].text(0.5, 0.5, "Thermal data not logged", transform=axes[3].transAxes, ha='center')
    axes[3].grid(True, alpha=0.3)

    axes[3].set_xlabel("Time (hours)")
    plt.tight_layout()
    plot_name = "telemetry_plot.png"
    plt.savefig(plot_name, dpi=150)
    print(f"Saved visualization plot to: {plot_name}")
    plt.show()
