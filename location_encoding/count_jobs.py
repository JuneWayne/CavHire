import pandas as pd
import json
import re

# Load CSV
df = pd.read_csv("../datacollection/Parsed_DF.csv")

# Clean and standardize location
df['location'] = df['location'].fillna("Unknown").apply(lambda x: x.strip())

# Convert number of applicants to numeric
def extract_number(text):
    if pd.isna(text):
        return 0
    match = re.search(r'\d+', text.replace(",", ""))
    return int(match.group()) if match else 0

df['num_applicants_clean'] = df['num_applicants'].apply(extract_number)

# Group and aggregate
location_summary = df.groupby('location').agg(
    number_of_jobs=('job_title', 'count'),
    average_applicants=('num_applicants_clean', 'mean')
).reset_index()

# Convert to JSON
location_data = []
for _, row in location_summary.iterrows():
    location_data.append({
        "location": row['location'],
        "number_of_jobs": int(row['number_of_jobs']),
        "average_applicants": round(row['average_applicants'], 2)
    })

# Save JSON
with open("location_job_applicants_summary_US.json", "w") as f:
    json.dump(location_data, f, indent=2)
