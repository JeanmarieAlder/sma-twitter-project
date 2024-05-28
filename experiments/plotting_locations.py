import folium
import pandas as pd

df = pd.read_csv('cleaned_location_with_coordinates.csv')

world_map = folium.Map(location=[0, 0], zoom_start=2)

for idx, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['cleaned_location']
    ).add_to(world_map)

world_map.save('world_map.html')
print("World map saved to 'world_map.html'")
