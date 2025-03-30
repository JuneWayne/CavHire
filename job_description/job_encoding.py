import pandas as pd
import openai
import json
import time
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv('../.env')
api_key = os.getenv("OPENAI_API_KEY")

# Use new client interface
client = openai.OpenAI(api_key=api_key)

# Load job data
df = pd.read_csv("../datacollection/uvajobsdata.csv")
structured_data = {}

# Function to extract structured job info
def extract_job_info(location, title, desc):
    prompt = f"""
    Extract the following from this job posting:
    - title
    - description
    - time of posting
    - wage
    - requirements
    - key responsibilities
    - start date
    - eligibility

    Job title: {title}
    Location: {location}
    Description: {desc}

    Respond in JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"❌ Error processing {title} at {location}: {e}")
        return None

# Process each row
for _, row in df.iterrows():
    location = row["addressLocality"]
    title = row["title"]
    description = row["description"]
    
    print(f"📦 Parsing: {location} - {title}")
    info = extract_job_info(location, title, description)
    
    if info:
        structured_data.setdefault(location, []).append(info)
    
    time.sleep(1)  # Respect rate limits

# Save to JSON
with open("jobData.json", "w") as f:
    json.dump(structured_data, f, indent=2)

print("✅ Job data successfully saved to jobData.json")
