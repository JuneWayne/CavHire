import pandas as pd
import requests
import time
import json

# Input CSV file with column: "location"
csv_file = "../datacollection/uvajobsdata.csv"
output_json = "locations.json"

# Use Nominatim API for free geocoding (OpenStreetMap)
def geocode(location_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{location_name}, University of Virginia",
        "format": "json",
        "limit": 1
    }
    response = requests.get(url, params=params, headers={"User-Agent": "UVAMapBot/1.0"})
    data = response.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None

# Load CSV
df = pd.read_csv(csv_file)

# Create JSON structure
output = {}

for _, row in df.iterrows():
    loc_name = row["addressLocality"]
    print(f"Geocoding {loc_name}...")
    lat, lon = geocode(loc_name)
    if lat and lon:
        output[loc_name] = {
            "lat": lat,
            "lng": lon
        }
    else:
        print(f"⚠️ Could not geocode: {loc_name}")
    time.sleep(1)  # To avoid being blocked by the API

# Save to JSON
with open(output_json, "w") as f:
    json.dump(output, f, indent=2)

print("✅ Saved to locations.json")
