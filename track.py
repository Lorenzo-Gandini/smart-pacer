'''
This file contains functions to extract and process GPX data, calculate distances, slopes, and save the data to a JSON file.
'''

import gpxpy
import matplotlib.pyplot as plt
import folium
import json
from datetime import datetime
from math import radians, cos, sin, asin, sqrt


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # metri
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c


def slope_level(delta_elev):
    if delta_elev > 0.5:
        return "uphill"
    elif delta_elev < -0.5:
        return "downhill"
    else:
        return "flat"


def parse_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    data = []
    last_point = None

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                entry = {
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "elevation": point.elevation,
                    "time": point.time.isoformat() if point.time else None
                }

                if last_point:
                    dist = haversine(
                        last_point.latitude, last_point.longitude,
                        point.latitude, point.longitude
                    )
                    if point.time and last_point.time:
                        delta_t = (point.time - last_point.time).total_seconds()
                        entry["delta_time"] = delta_t
                        entry["speed_mps"] = dist / delta_t if delta_t > 0 else 0.0
                    else:
                        entry["delta_time"] = None
                        entry["speed_mps"] = None

                    delta_elev = point.elevation - last_point.elevation
                    entry["slope"] = slope_level(delta_elev)
                else:
                    entry["delta_time"] = None
                    entry["speed_mps"] = None
                    entry["slope"] = "flat"

                last_point = point
                data.append(entry)

    return data

def save_to_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Dati salvati in '{output_path}'")


if __name__ == "__main__":
    circuit = "tre_laghi"  
    gpx_file = f"data/maps/{circuit}.GPX"
    output_path = f"data/maps/{circuit}.json"
    data = parse_gpx(gpx_file)

    elevations = [p["elevation"] for p in data if p["elevation"] is not None]
    save_to_json(data, output_path)
