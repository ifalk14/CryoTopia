"""
read_plotly_results.py

Reads `results/simulation_results.csv` (produced by `CryTopia2.py`) and recreates
the four Plotly figures from the original script: main (temperatures + pressure + states),
mass, quality+enthalpy, and heat transfer.

Usage:
    python read_plotly_results.py [path/to/simulation_results.csv]

If no path is provided it will use `./results/simulation_results.csv`.

This script prefers `pandas` for convenience; if pandas is not available it will fall
back to the stdlib `csv` + `numpy`.
"""

import sys
import os
import argparse
import json
import tempfile
import zipfile


def find_csv_from_path(path):
    """Accepts a path that may be: directory, CSV, JSON metadata, or ZIP.
    Returns path to a CSV file (possibly in a temp dir) or None.
    """
    path = os.path.abspath(path)
    # directory: look for metadata.json then simulation_results.csv
    if os.path.isdir(path):
        meta = os.path.join(path, 'simulation_metadata.json')
        csvp = os.path.join(path, 'simulation_results.csv')
        if os.path.exists(csvp):
            return csvp
        if os.path.exists(meta):
            # metadata won't contain csv path; assume csv in same dir
            return csvp if os.path.exists(csvp) else None
        return None

    # file
    if os.path.isfile(path):
        lower = path.lower()
        if lower.endswith('.csv'):
            return path
        if lower.endswith('.json'):
            # look in same dir for csv
            d = os.path.dirname(path)
            csvp = os.path.join(d, 'simulation_results.csv')
            return csvp if os.path.exists(csvp) else None
        if lower.endswith('.zip'):
            # extract CSV or metadata from zip to temp dir
            try:
                tmp = tempfile.mkdtemp()
                with zipfile.ZipFile(path, 'r') as zf:
                    names = zf.namelist()
                    # prefer CSV
                    csv_names = [n for n in names if n.endswith('simulation_results.csv')]
                    if csv_names:
                        zf.extract(csv_names[0], tmp)
                        return os.path.join(tmp, csv_names[0])
                    # else try metadata then expect csv in same dir in zip
                    meta_names = [n for n in names if n.endswith('simulation_metadata.json')]
                    if meta_names:
                        zf.extract(meta_names[0], tmp)
                        # look for csv in zip (same directory)
                        base_dir = os.path.dirname(meta_names[0])
                        candidate = os.path.join(tmp, base_dir, 'simulation_results.csv')
                        if os.path.exists(candidate):
                            return candidate
                return None
            except Exception:
                return None
    return None


parser = argparse.ArgumentParser(description='Plot CryTopia2 results (CSV or results directory or ZIP or metadata JSON)')
parser.add_argument('path', nargs='?', default=os.path.join(os.getcwd(), 'results'), help='Path to CSV, results directory, metadata JSON, or ZIP')
args = parser.parse_args()
parser.add_argument('--save-html', action='store_true', help='Save Plotly figures to HTML files (in addition to displaying)')
parser.add_argument('--outdir', help='Directory to save HTML files (defaults to <csv_dir>/plots)')

csv_path = find_csv_from_path(args.path)
if not csv_path or not os.path.exists(csv_path):
    print(f"CSV not found at: {args.path} (tried {csv_path})")
    sys.exit(1)

# robust loading that skips duplicate header rows if present
expected_cols = [
    'time_s', 'time_hr', 'gas_temp_K', 'liquid_temp_K', 'gas_pressure_psi',
    'gas_mass_kg', 'liquid_mass_kg', 'liquid_quality', 'liquid_enthalpy_J_per_kg',
    'liquid_heat_conv_J', 'liquid_heat_rad_J', 'gas_heat_rad_J', 'gas_heat_conv_J',
    'gas_heat_strut_J', 'liq_gas_heat_J', 'gas_heat_J', 'liquid_heat_J',
    'pressurization_events', 'relief_events', 'venting_events', 'draining_events'
]

try:
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str)
    # drop any rows that look like header repeats: if time_s equals 'time_s' or non-numeric
    df['time_s_num'] = pd.to_numeric(df['time_s'], errors='coerce')
    df = df[df['time_s_num'].notna()].copy()
    # coerce numeric columns
    for col in expected_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.NA

    time_hr = df['time_hr'].astype(float).tolist()
    gas_temp_data = df['gas_temp_K'].astype(float).tolist()
    liquid_temp_data = df['liquid_temp_K'].astype(float).tolist()
    gas_pressure_data = df['gas_pressure_psi'].astype(float).tolist()
    gas_mass_data = df['gas_mass_kg'].astype(float).tolist()
    liquid_mass_data = df['liquid_mass_kg'].astype(float).tolist()
    liquid_quality = df['liquid_quality'].astype(float).tolist()
    liquid_enthalpy_data = df['liquid_enthalpy_J_per_kg'].astype(float).tolist()
    liquid_heat_conv = df['liquid_heat_conv_J'].fillna(0).astype(float).tolist()
    liquid_heat_rad = df['liquid_heat_rad_J'].fillna(0).astype(float).tolist()
    gas_heat_rad = df['gas_heat_rad_J'].fillna(0).astype(float).tolist()
    gas_heat_conv = df['gas_heat_conv_J'].fillna(0).astype(float).tolist()
    gas_heat_strut = df['gas_heat_strut_J'].fillna(0).astype(float).tolist()
    liq_gas_heat = df['liq_gas_heat_J'].fillna(0).astype(float).tolist()
    gas_heat = df['gas_heat_J'].fillna(0).astype(float).tolist()
    liquid_heat = df['liquid_heat_J'].fillna(0).astype(float).tolist()
    pressurization_events = df['pressurization_events'].fillna(0).astype(float).tolist()
    relief_events = df['relief_events'].fillna(0).astype(float).tolist()
    venting_events = df['venting_events'].fillna(0).astype(float).tolist()
    draining_events = df['draining_events'].fillna(0).astype(float).tolist()
except Exception:
    import csv
    import numpy as np
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # skip header repeats
            if len(row) == 0:
                continue
            if row[0].strip() == header[0]:
                continue
            rows.append(row)
    if len(rows) == 0:
        print('No data rows found in CSV')
        sys.exit(1)
    data = np.array(rows)
    def col(name):
        try:
            idx = header.index(name)
        except ValueError:
            return np.zeros(data.shape[0]).astype(float).tolist()
        # coerce non-numeric to float (will fail if garbage)
        out = []
        for v in data[:, idx]:
            try:
                out.append(float(v))
            except Exception:
                out.append(float('nan'))
        return out

    time_hr = col('time_hr')
    gas_temp_data = col('gas_temp_K')
    liquid_temp_data = col('liquid_temp_K')
    gas_pressure_data = col('gas_pressure_psi')
    gas_mass_data = col('gas_mass_kg')
    liquid_mass_data = col('liquid_mass_kg')
    liquid_quality = col('liquid_quality')
    liquid_enthalpy_data = col('liquid_enthalpy_J_per_kg')
    liquid_heat_conv = col('liquid_heat_conv_J')
    liquid_heat_rad = col('liquid_heat_rad_J')
    gas_heat_rad = col('gas_heat_rad_J')
    gas_heat_conv = col('gas_heat_conv_J')
    gas_heat_strut = col('gas_heat_strut_J')
    liq_gas_heat = col('liq_gas_heat_J')
    gas_heat = col('gas_heat_J')
    liquid_heat = col('liquid_heat_J')
    pressurization_events = col('pressurization_events')
    relief_events = col('relief_events')
    venting_events = col('venting_events')
    draining_events = col('draining_events')

for i in range(len(draining_events)):
    if draining_events[i] >= 1:
        draining_events[i] = 140
    elif venting_events[i] >= 1:
        venting_events[i] = 140
    elif relief_events[i] >= 1:
        relief_events[i] = 140
    elif pressurization_events[i] >= 1:
        pressurization_events[i] = 140


# Create Plotly figures matching the originals
import plotly.graph_objects as go

# Main figure - Temperature and Pressure with state markers
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_hr, y=gas_temp_data, mode='lines',
                         name='Gas Temperature (K)', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=time_hr, y=liquid_temp_data, mode='lines',
                         name='Liquid Temperature (K)', line=dict(color='darkblue')))
fig.add_trace(go.Scatter(x=time_hr, y=gas_pressure_data, mode='lines',
                         name='Gas Pressure (psi)', line=dict(color='green'), yaxis='y2'))

fig.update_layout(
    title="Temperature and Pressure Changes Over Time",
    xaxis=dict(title="Time (hr)"),
    yaxis=dict(title="Temperature (K)", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
    yaxis2=dict(title="Pressure (psi)", titlefont=dict(color="green"),
                tickfont=dict(color="green"), overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99),
)

# Add state traces
fig.add_trace(go.Scatter(x=time_hr, y=pressurization_events, mode='lines',
                         name='Pressurizing State', line=dict(dash='dash', color="yellow")))
fig.add_trace(go.Scatter(x=time_hr, y=relief_events, mode='lines',
                         name='Relief State', line=dict(dash='dash', color='Purple')))
fig.add_trace(go.Scatter(x=time_hr, y=venting_events, mode='lines',
                         name='Venting State', line=dict(dash='dash', color='Orange')))
fig.add_trace(go.Scatter(x=time_hr, y=draining_events, mode='lines',
                         name='Draining State', line=dict(dash='dash', color='Red')))

# Mass figure
fig_mass = go.Figure()
fig_mass.add_trace(go.Scatter(x=time_hr, y=gas_mass_data, mode='lines',
                              name='Gas Mass (kg)', line=dict(color='cadetblue')))
fig_mass.add_trace(go.Scatter(x=time_hr, y=liquid_mass_data, mode='lines',
                              name='Liquid Mass (kg)', line=dict(color='darkblue'), yaxis='y2'))

fig_mass.update_layout(
    title="Mass Changes Over Time",
    xaxis=dict(title="Time (hr)"),
    yaxis=dict(title="Gas Mass (kg)", titlefont=dict(color="cadetblue"), tickfont=dict(color="cadetblue")),
    yaxis2=dict(title="Liquid Mass (kg)", titlefont=dict(color="darkblue"),
                tickfont=dict(color="darkblue"), overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99),
    template="plotly_white"
)

# Quality + enthalpy
fig_quality = go.Figure()
fig_quality.add_trace(go.Scatter(x=time_hr, y=liquid_quality, mode='lines',
                                 name='Liquid Quality', line=dict(color='purple')))
fig_quality.add_trace(go.Scatter(x=time_hr, y=liquid_enthalpy_data, mode='lines',
                                 name='Liquid Enthalpy (J/kg)', line=dict(color='red'), yaxis='y2'))

fig_quality.update_layout(
    title="Liquid Quality Over Time",
    xaxis=dict(title="Time (hr)"),
    yaxis=dict(title="Quality (dimensionless)", titlefont=dict(color="purple"), tickfont=dict(color="purple")),
    yaxis2=dict(title="Enthalpy (J/kg)", titlefont=dict(color="red"),
                tickfont=dict(color="red"), overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99),
    template="plotly_white"
)

# Heat flows
fig_heat = go.Figure()
fig_heat.add_trace(go.Scatter(x=time_hr, y=liquid_heat, mode='lines',
                              name='Heat into Liquid (J)', line=dict(color='cadetblue')))
fig_heat.add_trace(go.Scatter(x=time_hr, y=gas_heat, mode='lines',
                              name='Heat into Gas (J)', line=dict(color='orange')))
fig_heat.add_trace(go.Scatter(x=time_hr, y=liquid_heat_conv, mode='lines',
                              name='Liquid Convective Heat (J)', line=dict(color='purple')))
fig_heat.add_trace(go.Scatter(x=time_hr, y=liquid_heat_rad, mode='lines',
                              name='Liquid Radiative Heat (J)', line=dict(color='pink')))
fig_heat.add_trace(go.Scatter(x=time_hr, y=gas_heat_rad, mode='lines',
                              name='Gas Radiative Heat (J)', line=dict(color='green')))
fig_heat.add_trace(go.Scatter(x=time_hr, y=gas_heat_conv, mode='lines',
                              name='Gas Convective Heat (J)', line=dict(color='cyan')))
fig_heat.add_trace(go.Scatter(x=time_hr, y=gas_heat_strut, mode='lines',
                              name='Gas Structural Conductive Heat (J)', line=dict(color='brown')))
fig_heat.add_trace(go.Scatter(x=time_hr, y=liq_gas_heat, mode='lines',
                              name='Liquid-Gas Heat Transfer (J)', line=dict(color='red')))

fig_heat.update_layout(
    title="Heat Transfer into Liquid and Gas Over Time",
    xaxis=dict(title="Time (hr)"),
    yaxis=dict(title="Heat (J)", titlefont=dict(color="black"), tickfont=dict(color="black")),
    legend=dict(x=0.01, y=0.99),
    template="plotly_white"
)

# Show all figures
fig_heat.show()
fig_quality.show()
fig_mass.show()
fig.show()

print(f"Loaded CSV: {csv_path}")
print('Plotly figures displayed in your browser or default viewer.')
# Optionally save figures to HTML
if args.save_html:
    try:
        csv_dir = os.path.dirname(csv_path)
        out_dir = args.outdir if args.outdir else os.path.join(csv_dir, 'plots')
        os.makedirs(out_dir, exist_ok=True)
        heat_path = os.path.join(out_dir, 'figure_heat.html')
        quality_path = os.path.join(out_dir, 'figure_quality.html')
        mass_path = os.path.join(out_dir, 'figure_mass.html')
        main_path = os.path.join(out_dir, 'figure_main.html')
        fig_heat.write_html(heat_path)
        fig_quality.write_html(quality_path)
        fig_mass.write_html(mass_path)
        fig.write_html(main_path)
        print(f"Saved HTML plots to: {out_dir}")
        print(f"  - {heat_path}")
        print(f"  - {quality_path}")
        print(f"  - {mass_path}")
        print(f"  - {main_path}")
    except Exception as e:
        print('Warning: failed to save HTML plots:', e)

