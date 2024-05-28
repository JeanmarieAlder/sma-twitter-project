import pandas as pd
import re

df = pd.read_csv("C:\\Users\\vadre\\Desktop\\SMA\\Project\\GIT\\sma-twitter-project-master\\experiments\\202405231328_mod_gain_output.csv")

def clean_location(location):
    if pd.isnull(location):
        return ""
    non_location_phrases = [
        "Worldwide", "Global", "Everywhere", "Remote", "Online", "Earth",
        "Moon", "Mars", "Home", "Here", "There", "Somewhere", "Nowhere",
        "US", "UK", "CA", "NY", "TX"
    ]
    for phrase in non_location_phrases:
        location = re.sub(rf'\b{phrase}\b', '', location, flags=re.IGNORECASE)

    location = re.sub(r'[^a-zA-Z\s,]', '', location)
    location = re.sub(r'\s+', ' ', location).strip()
    location = re.sub(r',+', ',', location).strip(',')

    location = location.title()

    return location

df['cleaned_location'] = df['user_location'].apply(clean_location)

output_path = 'cleaned_location_data1.csv'
df.to_csv(output_path, index=False)

print(f"Cleaned data saved to {output_path}")


