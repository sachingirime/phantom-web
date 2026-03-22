"""
Convert emission CSV data (UTM Zone 13N) to WGS84 lat/lon JSON
for use in the Vue.js frontend static site.
Includes plume polygon data from pre-computed pickle cache.
"""
import json
import csv
import os
import sys
import pickle

try:
    from pyproj import Transformer
except ImportError:
    print("pyproj not installed. Run: pip install pyproj")
    sys.exit(1)

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "emission_quantification_results.csv")
PLUME_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "plumes")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "data", "emissions.json")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# UTM Zone 13N — matches Permian Basin AVIRIS-NG data (same as Streamlit app, EPSG:32613)
transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)

def load_plume_data(folder_name):
    """Load pre-computed plume data from pickle cache. Polygons already in [lat, lon] format."""
    cache_file = os.path.join(PLUME_DIR, f"{folder_name}.pkl")
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        polygons = data.get("polygons", [])
        rounded = []
        for poly in polygons:
            rounded.append([[round(pt[0], 6), round(pt[1], 6)] for pt in poly])
        return {
            "polygons": rounded,
            "det_pixels": int(data.get("det_pixels", 0)),
            "max_conf": round(float(data.get("max_conf", 0)), 6),
            "mean_conf": round(float(data.get("mean_conf", 0)), 6),
        }
    except Exception as e:
        print(f"  Warning: could not load plume for {folder_name}: {e}")
        return None

records = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            utm_x = float(row["center_lon"])  # easting
            utm_y = float(row["center_lat"])  # northing
            lon, lat = transformer.transform(utm_x, utm_y)

            # Parse date/time from folder name: ang20191018t141549...
            folder = row["folder_name"]
            date_str = folder[3:11]   # 20191018
            time_str = folder[12:18]  # 141549
            date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            time_fmt = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]} UTC"

            q_kg_hr = round(float(row["Q_kg_hr"]), 2)
            plume_data = load_plume_data(folder)

            records.append({
                "folder_name": folder,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "Q_kg_hr": q_kg_hr,
                "Q_uncertainty_kg_hr": round(float(row["Q_uncertainty_kg_hr"]), 2),
                "emission_category": "Critical" if q_kg_hr >= 100 else "Standard",
                "max_probability": round(float(row["max_probability"]), 4),
                "U_10_ms": round(float(row["U_10_ms"]), 2),
                "wind_direction_deg": round(float(row["wind_direction_deg"]), 1),
                "date_formatted": date_fmt,
                "time_formatted": time_fmt,
                "plume": plume_data,  # { polygons, det_pixels, max_conf, mean_conf } or null
            })
        except (ValueError, KeyError, TypeError) as e:
            print(f"Skipping row {row.get('folder_name', '?')}: {e}", flush=True)

# Sort by emission rate descending
records.sort(key=lambda r: r["Q_kg_hr"], reverse=True)

total = sum(r["Q_kg_hr"] for r in records)
critical = sum(1 for r in records if r["emission_category"] == "Critical")
median = sorted(r["Q_kg_hr"] for r in records)[len(records) // 2] if records else 0
with_plumes = sum(1 for r in records if r["plume"])

output = {
    "emissions": records,
    "summary": {
        "total_sites": len(records),
        "critical_count": critical,
        "total_kg_hr": round(total, 1),
        "median_kg_hr": round(median, 1),
    }
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, separators=(',', ':'))  # compact to reduce file size

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"Wrote {len(records)} records ({with_plumes} with plumes) → {OUT_PATH} ({size_kb:.1f} KB)")
print(f"Summary: {output['summary']}")
