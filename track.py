'''
This file reads a GPX (GPS Exchange Format) and processes its track data to extract detailed information about each GPS point. 
For each point, it extracts latitude, longitude, elevation, and timestamp and calculates distance between two points using the haversine formula, computes the time difference in seconds, and derives the speed in meters per second. Extracts also elevation and slope.
'''

import gpxpy
from utils import haversine, slope_level, save_to_json

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


if __name__ == "__main__":

    circuit = "tre_laghi"                       # Define the name of the circuit
    gpx_file = f"data/maps/{circuit}.GPX"       # Name of the gpx file
    output_path = f"data/maps/{circuit}.json"   # Output file
    data = parse_gpx(gpx_file)                  # Parse the GPX file

    elevations = [p["elevation"] for p in data if p["elevation"] is not None]
    save_to_json(data, output_path)
