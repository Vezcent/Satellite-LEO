"""
S-MAS Mission Analysis & Visualization
Upgraded: 5 subplots basic, plus 3 advanced panels (Thermal, ADCS, Propulsion) 
tracking Phase A, B, and C environment & vehicle states.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import argparse
from datetime import datetime

BASE_DIR = r"E:\Satellite LEO"
SAVE_DIR = os.path.join(BASE_DIR, "result", "save")

FDIR_COLORS = {0: '#22c55e', 1: '#facc15', 2: '#ef4444', 3: '#3b82f6'}
FDIR_LABELS = {0: 'NOMINAL', 1: 'DEGRADED', 2: 'SAFE', 3: 'RECOVERY'}


def find_latest_log():
    """Find the most recent session CSV from the controller."""
    log_pattern = os.path.join(BASE_DIR, "controller_csharp", "bin", "Release", "net10.0", "logs", "session_*.csv")
    logs = glob.glob(log_pattern)
    if not logs:
        # Fallback to local logs directory just in case
        log_pattern_fallback = os.path.join(BASE_DIR, "controller_csharp", "logs", "session_*.csv")
        logs = glob.glob(log_pattern_fallback)
        
    if not logs:
        print("No log files found in default location.")
        return None
    return max(logs, key=os.path.getmtime)


def plot_thermal(df, time_hours, timestamp, seu_mult, title_suffix):
    """Generate and save thermal subsystem analysis plot."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"S-MAS Thermal Subsystem Analysis (SEU {seu_mult}x)\n{title_suffix}", fontsize=14, fontweight='bold')
    
    # Subplot 1: Temperatures
    ax1 = axes[0]
    if 'temp_bus' in df.columns:
        ax1.plot(time_hours, df['temp_bus'], color='#ec4899', linewidth=1.0, label='Bus Temp (°C)')
    if 'temp_battery' in df.columns:
        ax1.plot(time_hours, df['temp_battery'], color='#f97316', linewidth=1.0, label='Battery Temp (°C)')
    if 'temp_payload' in df.columns:
        ax1.plot(time_hours, df['temp_payload'], color='#ef4444', linewidth=1.0, label='Payload Temp (°C)')
    
    ax1.axhline(y=0, color='blue', linestyle=':', alpha=0.5, label='0°C Limit')
    ax1.axhline(y=45, color='red', linestyle=':', alpha=0.5, label='45°C Limit')
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper right', fontsize=8)
    
    # Subplot 2: Heater
    ax2 = axes[1]
    if 'heater_on' in df.columns:
        ax2.fill_between(time_hours, df['heater_on'], color='#ec4899', alpha=0.3, label='Heater ON', step='mid')
        ax2.plot(time_hours, df['heater_on'], color='#db2777', linewidth=0.8, drawstyle='steps-mid')
    ax2.set_ylabel("Heater Status (0/1)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Subplot 3: Eclipse & Solar Power
    ax3 = axes[2]
    ax3.plot(time_hours, df['solar_power_w'], color='#eab308', linewidth=1.0, label='Solar Power (W)')
    ax3.set_ylabel("Solar Power (W)", color='#eab308')
    ax3.tick_params(axis='y', labelcolor='#eab308')
    
    if 'in_eclipse' in df.columns:
        ax3b = ax3.twinx()
        ax3b.fill_between(time_hours, df['in_eclipse'], color='#6b7280', alpha=0.2, label='In Eclipse', step='mid')
        ax3b.plot(time_hours, df['in_eclipse'], color='#374151', linewidth=0.5, linestyle='--', drawstyle='steps-mid')
        ax3b.set_ylabel("Eclipse State (0/1)", color='#374151')
        ax3b.set_ylim(-0.1, 1.1)
        ax3b.tick_params(axis='y', labelcolor='#374151')
        
    ax3.set_xlabel("Time (Hours)")
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"thermal_{timestamp}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Thermal plot saved to: {out_path}")
    plt.close()


def plot_adcs(df, time_hours, timestamp, seu_mult, title_suffix):
    """Generate and save ADCS subsystem analysis plot."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"S-MAS ADCS Subsystem Analysis (SEU {seu_mult}x)\n{title_suffix}", fontsize=14, fontweight='bold')
    
    # Subplot 1: Nadir Error
    ax1 = axes[0]
    if 'nadir_error' in df.columns:
        ax1.plot(time_hours, df['nadir_error'], color='#a855f7', linewidth=1.0, label='Nadir Error')
        ax1.axhline(y=0.1, color='#ef4444', linestyle='--', alpha=0.5, label='Pointing Tolerance')
    ax1.set_ylabel("Nadir Error (rad)")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper right', fontsize=8)
    
    # Subplot 2: Sun Sensor Angle
    ax2 = axes[1]
    if 'sun_angle' in df.columns:
        ax2.plot(time_hours, df['sun_angle'], color='#eab308', linewidth=1.0, label='Sun Angle (rad)')
    ax2.set_ylabel("Sun Angle (rad)")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Subplot 3: Reaction Wheel Momentum
    ax3 = axes[2]
    if 'wheel_momentum_x' in df.columns:
        ax3.plot(time_hours, df['wheel_momentum_x'], color='#3b82f6', linewidth=0.8, label='Wheel X')
    if 'wheel_momentum_y' in df.columns:
        ax3.plot(time_hours, df['wheel_momentum_y'], color='#22c55e', linewidth=0.8, label='Wheel Y')
    if 'wheel_momentum_z' in df.columns:
        ax3.plot(time_hours, df['wheel_momentum_z'], color='#ec4899', linewidth=0.8, label='Wheel Z')
    ax3.set_xlabel("Time (Hours)")
    ax3.set_ylabel("Momentum (Nms)")
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"adcs_{timestamp}.png")
    plt.savefig(out_path, dpi=150)
    print(f"ADCS plot saved to: {out_path}")
    plt.close()


def plot_propulsion(df, time_hours, timestamp, seu_mult, title_suffix):
    """Generate and save propulsion and orbit decay plot."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"S-MAS Propulsion & Fuel Analysis (SEU {seu_mult}x)\n{title_suffix}", fontsize=14, fontweight='bold')
    
    # Subplot 1: Fuel Fraction
    ax1 = axes[0]
    if 'fuel_fraction' in df.columns:
        ax1.fill_between(time_hours, df['fuel_fraction'] * 100, color='#06b6d4', alpha=0.2)
        ax1.plot(time_hours, df['fuel_fraction'] * 100, color='#0891b2', linewidth=1.2, label='Fuel Fraction %')
        ax1.set_ylim(-5, 105)
    ax1.set_ylabel("Fuel (%)")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper right', fontsize=8)
    
    # Subplot 2: Thrust vector & Throttle
    ax2 = axes[1]
    if 'thrust_x' in df.columns:
        ax2.plot(time_hours, df['thrust_x'], color='#ef4444', linewidth=0.8, alpha=0.8, label='Thrust X')
    if 'thrust_y' in df.columns:
        ax2.plot(time_hours, df['thrust_y'], color='#22c55e', linewidth=0.8, alpha=0.8, label='Thrust Y')
    if 'thrust_z' in df.columns:
        ax2.plot(time_hours, df['thrust_z'], color='#3b82f6', linewidth=0.8, alpha=0.8, label='Thrust Z')
    ax2.set_ylabel("Thrust Vector (N)")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Subplot 3: Drag vs Density
    ax3 = axes[2]
    if 'atm_density' in df.columns:
        ax3.plot(time_hours, df['atm_density'], color='#6b7280', linewidth=1.0, label='Atm Density (kg/m³)')
        ax3.set_ylabel("Density (kg/m³)", color='#6b7280')
        ax3.tick_params(axis='y', labelcolor='#6b7280')
        
    if 'drag_coeff' in df.columns:
        ax3b = ax3.twinx()
        ax3b.plot(time_hours, df['drag_coeff'], color='#ef4444', linestyle='--', linewidth=0.8, label='Cd')
        ax3b.set_ylabel("Cd", color='#ef4444')
        ax3b.tick_params(axis='y', labelcolor='#ef4444')
        
    ax3.set_xlabel("Time (Hours)")
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"propulsion_{timestamp}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Propulsion plot saved to: {out_path}")
    plt.close()


def visualize_results(log_path=None, seu_mult=1.0, title=None, advanced=False):
    """Generate basic and optional advanced subsystem analysis plots."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not log_path:
        log_path = find_latest_log()
        if not log_path:
            return

    print(f"Loading log: {log_path}")
    df = pd.read_csv(log_path)

    if df.empty:
        print("Log file is empty.")
        return

    # ── Process Metrics ──
    last_step = df.iloc[-1]
    lifetime_s = last_step['sim_time_s']
    days = int(lifetime_s // 86400)
    hours = int((lifetime_s % 86400) // 3600)
    mins = int((lifetime_s % 3600) // 60)

    avg_soc = df['battery_soc'].mean() * 100
    payload_on_pct = (df['payload_on'].sum() / len(df)) * 100
    avg_alt = df['altitude_km'].mean()

    # Environment config (from new columns if present)
    if 'seu_mult' in df.columns:
        seu_mult = df['seu_mult'].iloc[-1]

    total_seu = int(df['seu_active'].sum()) if 'seu_active' in df.columns else 0

    # FDIR mode percentages
    fdir_degraded_pct = 0.0
    fdir_safe_pct = 0.0
    if 'fdir_mode' in df.columns:
        fdir_degraded_pct = (df['fdir_mode'] == 1).sum() / len(df) * 100
        fdir_safe_pct = (df['fdir_mode'] == 2).sum() / len(df) * 100

    time_hours = df['sim_time_s'] / 3600
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title_suffix = os.path.basename(log_path)

    # ── Generate Plots (5 subplots basic) ──
    fig_title = title or f"S-MAS Mission Analysis (SEU {seu_mult}x)\n{title_suffix}"
    fig, axes = plt.subplots(5, 1, figsize=(15, 16), sharex=True)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')

    # Subplot 1: Altitude
    ax1 = axes[0]
    ax1.plot(time_hours, df['altitude_km'], color='#22c55e', linewidth=0.8, label='Altitude')
    ax1.axhline(y=600, color='cyan', linestyle='--', alpha=0.4, label='Target (600km)')
    ax1.axhline(y=200, color='#ef4444', linestyle='-', alpha=0.5, label='Re-entry')
    ax1.set_ylabel("Alt (km)")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper right', fontsize=8)

    # Subplot 2: Battery SoC + Capacity (dual y-axis)
    ax2 = axes[1]
    ax2.fill_between(time_hours, df['battery_soc'] * 100, color='#3b82f6', alpha=0.2)
    ax2.plot(time_hours, df['battery_soc'] * 100, color='#2563eb', linewidth=0.8, label='SoC %')
    ax2.set_ylabel("SoC (%)", color='#2563eb')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.2)

    if 'battery_capacity_j' in df.columns:
        ax2b = ax2.twinx()
        ax2b.plot(time_hours, df['battery_capacity_j'], color='#f97316', linestyle='--',
                  linewidth=0.8, alpha=0.8, label='Capacity (J)')
        ax2b.set_ylabel("Capacity (J)", color='#f97316')
        ax2b.legend(loc='lower right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # Subplot 3: Mission Activity
    ax3 = axes[2]
    ax3.fill_between(time_hours, df['payload_on'], color='#f59e0b', alpha=0.4, label='Payload ON')
    ax3.plot(time_hours, df['throttle'], color='#ef4444', label='Throttle', alpha=0.7, linewidth=0.8)
    ax3.set_ylabel("Activity (0-1)")
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc='upper right', fontsize=8)

    # Subplot 4: FDIR Mode Timeline
    ax4 = axes[3]
    if 'fdir_mode' in df.columns:
        fdir = df['fdir_mode'].values
        for mode, color in FDIR_COLORS.items():
            mask = (fdir == mode)
            if mask.any():
                ax4.fill_between(time_hours, 0, 1, where=mask, color=color,
                                 alpha=0.6, label=FDIR_LABELS.get(mode, f'Mode {mode}'),
                                 step='mid')
    ax4.set_ylabel("FDIR Mode")
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.legend(loc='upper right', fontsize=8, ncol=4)
    ax4.grid(True, alpha=0.2)

    # Subplot 5: SEU Events
    ax5 = axes[4]
    if 'seu_active' in df.columns:
        seu_mask = df['seu_active'] == 1
        if seu_mask.any():
            ax5.scatter(time_hours[seu_mask], [1] * seu_mask.sum(),
                        c='#ef4444', s=8, alpha=0.7, label=f'SEU Events ({total_seu} total)')
        else:
            ax5.text(0.5, 0.5, 'No SEU events', transform=ax5.transAxes,
                     ha='center', va='center', color='gray', fontsize=12)
    ax5.set_ylabel("SEU")
    ax5.set_xlabel("Time (Hours)")
    ax5.set_ylim(0, 2)
    ax5.set_yticks([])
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save Basic Plot
    plot_name = f"analysis_{timestamp}.png"
    plt.savefig(os.path.join(SAVE_DIR, plot_name), dpi=150)
    print(f"Basic plot saved to: {os.path.join(SAVE_DIR, plot_name)}")
    plt.close()

    # ── Advanced Diagnostic Plots ──
    if advanced:
        print("Generating advanced phase-C subsystem diagnostic plots...")
        plot_thermal(df, time_hours, timestamp, seu_mult, title_suffix)
        plot_adcs(df, time_hours, timestamp, seu_mult, title_suffix)
        plot_propulsion(df, time_hours, timestamp, seu_mult, title_suffix)

    # ── Save Summary CSV ──
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
        "death_reason": [last_step['done_reason']],
        # Environmental config
        "seu_multiplier": [seu_mult],
        "total_seu_events": [total_seu],
        "fdir_degraded_pct": [round(fdir_degraded_pct, 2)],
        "fdir_safe_pct": [round(fdir_safe_pct, 2)],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(SAVE_DIR, f"summary_{timestamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to: {summary_csv}")

    # ── Console Stats ──
    print("\n" + "=" * 50)
    print("      MISSION SUMMARY")
    print("=" * 50)
    print(f"Lifetime:      {days}d {hours}h {mins}m")
    print(f"Avg SoC:       {avg_soc:.1f}%")
    print(f"Payload Use:   {payload_on_pct:.1f}%")
    print(f"Final Alt:     {last_step['altitude_km']:.2f} km")
    print(f"SEU Events:    {total_seu}")
    print(f"FDIR DEGRADED: {fdir_degraded_pct:.1f}%")
    print(f"FDIR SAFE:     {fdir_safe_pct:.1f}%")
    print(f"SEU Mult:      {seu_mult}x")
    print(f"Status:        {'RE-ENTERED' if last_step['is_done'] else 'OPERATIONAL'}")
    print("=" * 50)


def survival_curve():
    """
    Read all summary_*.csv files, group by seu_multiplier,
    and plot Lifetime vs SEU Multiplier with error bars.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    pattern = os.path.join(SAVE_DIR, "summary_*.csv")
    files = glob.glob(pattern)

    if not files:
        print("No summary CSV files found. Run simulations first.")
        return

    print(f"Found {len(files)} summary files.")

    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'seu_multiplier' in df.columns and 'lifetime_days' in df.columns:
                all_data.append(df)
        except Exception as e:
            print(f"  Warning: skipping {f}: {e}")

    if not all_data:
        print("No valid summary files with seu_multiplier column.")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Group by SEU multiplier
    grouped = combined.groupby('seu_multiplier')['lifetime_days'].agg(['mean', 'std', 'count'])
    grouped = grouped.sort_index()
    grouped['lifetime_years'] = grouped['mean'] / 365.0
    grouped['std_years'] = grouped['std'].fillna(0) / 365.0

    print("\nSurvival Data:")
    print(grouped.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(grouped.index, grouped['lifetime_years'], yerr=grouped['std_years'],
                fmt='o-', color='#3b82f6', capsize=5, capthick=2, linewidth=2,
                markersize=8, label='Mean Lifetime ± 1σ')

    # Reference lines
    ax.axhline(y=10, color='#22c55e', linestyle='--', alpha=0.5, label='10-year target')
    ax.axhline(y=5, color='#f59e0b', linestyle='--', alpha=0.5, label='5-year minimum')

    ax.set_xscale('log')
    ax.set_xlabel('SEU Multiplier', fontsize=13)
    ax.set_ylabel('Lifetime (years)', fontsize=13)
    ax.set_title('S-MAS Survival Curve: Lifetime vs SEU Rate', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, "survival_curve.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSurvival curve saved to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="S-MAS Mission Analysis & Visualization")
    parser.add_argument('log_path', nargs='?', default=None, help='Path to session CSV log')
    parser.add_argument('--seu-mult', type=float, default=1.0, help='SEU multiplier for labeling')
    parser.add_argument('--title', type=str, default=None, help='Custom plot title')
    parser.add_argument('--advanced', action='store_true', help='Generate advanced Phase-C thermal, ADCS, and propulsion plots')
    parser.add_argument('--survival-curve', action='store_true', help='Generate survival curve from summaries')

    args = parser.parse_args()

    if args.survival_curve:
        survival_curve()
    else:
        visualize_results(args.log_path, args.seu_mult, args.title, args.advanced)


if __name__ == "__main__":
    main()
