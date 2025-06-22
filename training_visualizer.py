import os
import glob
import json

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from runner_env import load_json

# === CONFIG ===
CSV_DIR       = "data/training_log_DEF_20062025"
ATHLETES_JSON = "data/athletes.json"
TRAININGS_JSON= "data/trainings.json"
MAPS_DIR      = "data/maps"
OUT_DIR       = "data/video/final_delivery"

os.makedirs(OUT_DIR, exist_ok=True)

athletes = load_json(ATHLETES_JSON)
trainings = load_json(TRAININGS_JSON)
map_files = glob.glob(os.path.join(MAPS_DIR, "*.json"))

for profile_label, athlete in athletes.items():
    for training_name, training in trainings.items():
        print(f"  * Allenamento: {training_name}")
        for map_path in map_files:
            circuit_name = os.path.splitext(os.path.basename(map_path))[0]
            print(f"    - Circuito: {circuit_name}", end=" ... ")
    
            csv_path     = os.path.join(CSV_DIR, f"{profile_label}_{training_name}_{circuit_name}.csv")
            output_video = os.path.join(OUT_DIR,  f"{profile_label}_{training_name}_{circuit_name}.mp4")

            print(csv_path)

            # DataFrame and downsample
            df = pd.read_csv(csv_path)
            STEP = 2 #print every 2 steps, to reduce video size and speed up animation
            df = df[df.second % STEP == 0].reset_index(drop=True)
            df = df.dropna(subset=["lat", "lon", "fatigue"])

            # Color map for fatigue levels
            color_map = {"low":"green", "medium":"orange", "high":"red"}
            df["color"] = df["fatigue"].map(color_map)

            #GeodataFrame and projection
            gdf = gpd.GeoDataFrame(
                df,
                geometry=[Point(xy) for xy in zip(df.lon, df.lat)],
                crs="EPSG:4326" #this is the default WGS84 coordinate system
            ).to_crs(epsg=3857) #apply projection for compatibility with basemaps
            df["x"] = gdf.geometry.x
            df["y"] = gdf.geometry.y

            # Extract array vlaues
            xs     = df["x"].values
            ys     = df["y"].values
            colors = df["color"].tolist()

            #white padding for the video
            pad = 0.1
            x_pad = (xs.max() - xs.min()) * pad
            y_pad = (ys.max() - ys.min()) * pad
            x_min, x_max = xs.min() - x_pad, xs.max() + x_pad
            y_min, y_max = ys.min() - y_pad, ys.max() + y_pad

            #create figure 
            fig, (ax_map, ax_text) = plt.subplots(
                1, 2, figsize=(10, 12),
                gridspec_kw={'width_ratios':[3,1]}
            )
            fig.suptitle(f"{profile_label} – {training_name} – {circuit_name}", fontsize=16)

            ax_map.set_xlim(x_min, x_max)
            ax_map.set_ylim(y_min, y_max)
            ctx.add_basemap(ax_map, source=ctx.providers.OpenStreetMap.Mapnik)
            ax_map.set_autoscale_on(False)

            #draw the red dot and the lines 
            scat, = ax_map.plot([], [], 'ro', label='Athlete')
            line = LineCollection([], linewidths=2)
            ax_map.add_collection(line)

            # textbox on the right side
            text_box = ax_text.text(
                0.05, 0.95, "",
                transform=ax_text.transAxes,
                fontsize=9, va='top', family='monospace'
            )
            ax_text.axis('off')

            def init():
                scat.set_data([], [])
                line.set_segments([])
                text_box.set_text("")
                return scat, line, text_box

            def animate(i):
                segs = [[[xs[j], ys[j]], [xs[j+1], ys[j+1]]] for j in range(i)]
                line.set_segments(segs)
                line.set_color(colors[:i])

                # Correctly set as sequences
                scat.set_data([xs[i]], [ys[i]])

                row = df.iloc[i]
                info = (
                    f"Second: {int(row.second)}/{int(df.second.max())}\n"
                    f"Phase:   {row.phase}\n"
                    f"Action:  {row.action}\n"
                    f"Fatigue: {row.fatigue}\n"
                    f"HR Z:    {row.HR_zone}   Power Z: {row.power_zone}"
                )
                text_box.set_text(info)
                return scat, line, text_box

            frame_indices = list(range(0, len(df), 2))

            #Animation with blit, which is a performance optimization
            anim = FuncAnimation(
                fig, animate,
                frames=frame_indices,
                init_func=init,
                blit=True,
                interval=100  # milli -> 10 fps
            )

            anim.save(
                output_video,
                writer='ffmpeg',
                fps=10,
                dpi=100,
                metadata={'artist':'SmartPacer'},
                bitrate=2000
            )

            plt.close(fig)
            print("✅ Video saved:", output_video)
