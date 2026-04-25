import pandas as pd
import glob
import os

base_dir = r"E:\Satellite LEO"
log_pattern = os.path.join(base_dir, "controller_csharp", "bin", "Release", "net10.0", "logs", "session_*.csv")
logs = glob.glob(log_pattern)
if not logs:
    print("No logs")
    exit()

latest_log = max(logs, key=os.path.getmtime)
df = pd.read_csv(latest_log)

print(f"Log: {os.path.basename(latest_log)}")
print(f"Total Steps: {len(df)}")
print(f"Payload ON:  {df['payload_on'].sum()}")
print(f"Deep Sleep:  {df['deep_sleep'].sum()}")
print(f"Avg Altitude: {df['altitude_km'].mean():.2f}")
print(f"Avg SoC:      {df['battery_soc'].mean()*100:.2f}%")
