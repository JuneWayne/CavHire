import pandas as pd
import requests
import time
import json

csv_file = "../datacollection/Parsed_Data_Science_Internships.csv"
output_json = "locations_US.json"

def geocode(location_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{location_name}, United States",
        "format": "json",
        "limit": 1
    }
    response = requests.get(url, params=params, headers={"User-Agent": "USMapBot/1.0"})
    data = response.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None

df = pd.read_csv(csv_file)

output = {}

for _, row in df.iterrows():
    loc_name = row["location"]
    print(f"Geocoding {loc_name}...")
    lat, lon = geocode(loc_name)
    if lat and lon:
        output[loc_name] = {
            "lat": lat,
            "lng": lon
        }
    else:
        print(f" Could not geocode: {loc_name}")
    time.sleep(1)  

with open(output_json, "w") as f:
    json.dump(output, f, indent=2)

print("Saved to locations.json")
