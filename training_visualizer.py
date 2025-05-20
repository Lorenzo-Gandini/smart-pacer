import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from tqdm import tqdm
import os

# === CONFIG ===
csv_path = "training_logs/runner_fartlek_20250519_111624.csv"
athlete = csv_path.split("/")[-1].split("_")[0]
training = csv_path.split("/")[-1].split("_")[1]
output_video = f"data/video/{athlete}_{training}_video.mp4"

# === LOAD & PROCESS DATA ===
df = pd.read_csv(csv_path)
df = df.dropna(subset=["lat", "lon", "fatigue_level"])
color_map = {"low": "green", "medium": "orange", "high": "red"}
df["color"] = df["fatigue_level"].map(color_map)

gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)
df["x"] = gdf.geometry.x
df["y"] = gdf.geometry.y

# === FIGURE SETUP ===
padding_factor = 0.1
x_pad = (df["x"].max() - df["x"].min()) * padding_factor
y_pad = (df["y"].max() - df["y"].min()) * padding_factor
x_min, x_max = df["x"].min() - x_pad, df["x"].max() + x_pad
y_min, y_max = df["y"].min() - y_pad, df["y"].max() + y_pad

fig, (ax_map, ax_text) = plt.subplots(1, 2, figsize=(10, 12), gridspec_kw={'width_ratios': [3, 1]})
fig.suptitle("Smart Pacing Simulation", fontsize=16)
ax_map.set_xlim(x_min, x_max)
ax_map.set_ylim(y_min, y_max)
ctx.add_basemap(ax_map, source=ctx.providers.OpenStreetMap.Mapnik)

# === GRAFICA ===
scat, = ax_map.plot([], [], 'ro', label='Athlete')
line = LineCollection([], linewidths=2)
ax_map.add_collection(line)
text_box = ax_text.text(0.05, 0.95, "", transform=ax_text.transAxes, fontsize=9, va='top', family='monospace')
ax_text.axis('off')

# === ANIMATION LOGIC ===
def update(frame):
    x = df["x"].values
    y = df["y"].values

    # Aggiorna linea colorata
    segments = [[[x[i], y[i]], [x[i+1], y[i+1]]] for i in range(frame)]
    colors = [color_map[df["fatigue_level"].iloc[i+1]] for i in range(frame)]
    line.set_segments(segments)
    line.set_color(colors)

    # Punto atleta
    scat.set_data([x[frame]], [y[frame]])

    # Lavagna info
    row = df.iloc[frame]
    total_time = df["second"].max()
    info = f"""
Second       : {int(row['second'])} / {int(total_time)} sec
Phase        : {row['phase']}
Fatigue      : {row['fatigue_level']}
Action       : {row['action']}
HR Zone      : {row['HR_zone']}
Power Zone   : {row['power_zone']}
Target HR    : {row['target_HR']}
Target Power : {row['target_power']}
Slope        : {row['slope']}
Reward       : {row['reward']:.2f}
"""
    text_box.set_text(info)
    return [scat, line, text_box]

# === VIDEO EXPORT ===
writer = FFMpegWriter(fps=20, metadata=dict(artist='SmartPacer'), codec='libx264')

with writer.saving(fig, output_video, dpi=100):
    for frame in tqdm(range(len(df)), desc="🎞 Saving video", ncols=70):
        update(frame)
        writer.grab_frame()

plt.close(fig)
print("✅ Video saved:", output_video)
