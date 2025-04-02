import pandas as pd
import json

# --- Step 1: Set input and output file paths ---
csv_file_path = "../datacollection/Parsed_DF.csv"  # Replace with your CSV file path
json_file_path = "Parsed_Data_Science_Internships.json" 

df = pd.read_csv(csv_file_path)

data_as_json = df.to_dict(orient="records")

with open(json_file_path, "w") as json_file:
    json.dump(data_as_json, json_file, indent=2)

print(f"✅ Converted CSV to JSON: {json_file_path}")
