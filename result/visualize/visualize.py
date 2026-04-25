import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob
from datetime import datetime

def visualize_results(log_path=None):
    # 1. Setup paths
    base_dir = r"E:\Satellite LEO"
    save_dir = os.path.join(base_dir, "result", "save")
    os.makedirs(save_dir, exist_ok=True)

    # 2. Find log file
    if not log_path:
        log_pattern = os.path.join(base_dir, "controller_csharp", "bin", "Release", "net10.0", "logs", "session_*.csv")
        logs = glob.glob(log_pattern)
        if not logs:
            print("No log files found in default location.")
            return
        log_path = max(logs, key=os.path.getmtime) # Get latest

    print(f"Loading log: {log_path}")
    df = pd.read_csv(log_path)
    
    if df.empty:
        print("Log file is empty.")
        return

    # 3. Process Metrics
    last_step = df.iloc[-1]
    lifetime_s = last_step['sim_time_s']
    days = int(lifetime_s // 86400)
    hours = int((lifetime_s % 86400) // 3600)
    mins = int((lifetime_s % 3600) // 60)
    
    avg_soc = df['battery_soc'].mean() * 100
    payload_on_pct = (df['payload_on'].sum() / len(df)) * 100
    avg_alt = df['altitude_km'].mean()
    
    # Estimate Fuel Remaining (assuming initial 100kg and some burn rate)
    # Total throttle sum as a proxy for fuel spent
    fuel_spent_proxy = df['throttle'].sum() * 0.01 # Arbitrary scaling
    
    # 4. Generate Plots
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"S-MAS Mission Analysis\n{os.path.basename(log_path)}", fontsize=16)

    # Subplot 1: Altitude
    plt.subplot(3, 1, 1)
    plt.plot(df['sim_time_s']/3600, df['altitude_km'], color='#2ecc71', label='Altitude')
    plt.axhline(y=600, color='r', linestyle='--', alpha=0.3, label='Target (600km)')
    plt.axhline(y=200, color='black', linestyle='-', alpha=0.5, label='Re-entry')
    plt.ylabel("Alt (km)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: Power (SoC)
    plt.subplot(3, 1, 2)
    plt.fill_between(df['sim_time_s']/3600, df['battery_soc']*100, color='#3498db', alpha=0.3)
    plt.plot(df['sim_time_s']/3600, df['battery_soc']*100, color='#2980b9', label='Battery SoC %')
    plt.ylabel("SoC (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 3: Mission Activity
    plt.subplot(3, 1, 3)
    plt.fill_between(df['sim_time_s']/3600, df['payload_on'], color='#e67e22', alpha=0.5, label='Payload ON')
    plt.plot(df['sim_time_s']/3600, df['throttle'], color='#e74c3c', label='Throttle', alpha=0.7)
    plt.ylabel("Activity (0-1)")
    plt.xlabel("Time (Hours)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_name = f"analysis_{timestamp}.png"
    plt.savefig(os.path.join(save_dir, plot_name))
    print(f"Plot saved to: {os.path.join(save_dir, plot_name)}")

    # 5. Save Summary CSV
    summary_data = {
        "timestamp": [timestamp],
        "log_file": [os.path.basename(log_path)],
        "lifetime_days": [days],
        "lifetime_hours": [hours],
        "lifetime_mins": [mins],
        "avg_soc_pct": [round(avg_soc, 2)],
        "payload_duty_cycle_pct": [round(payload_on_pct, 2)],
        "avg_altitude_km": [round(avg_alt, 2)],
        "final_status": ["DECEASED" if last_step['is_done'] else "ACTIVE"],
        "death_reason": [last_step['done_reason']]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(save_dir, f"summary_{timestamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")

    # 6. Display Stats
    print("\n" + "="*40)
    print("      MISSION SUMMARY")
    print("="*40)
    print(f"Lifetime:    {days}d {hours}h {mins}m")
    print(f"Avg SoC:     {avg_soc:.1f}%")
    print(f"Payload Use: {payload_on_pct:.1f}%")
    print(f"Final Alt:   {last_step['altitude_km']:.2f} km")
    print(f"Status:      {'RE-ENTERED' if last_step['is_done'] else 'OPERATIONAL'}")
    print("="*40)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_results(path)
