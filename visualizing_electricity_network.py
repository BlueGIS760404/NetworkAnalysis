import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import contextily as ctx
import os
import numpy as np
import pyproj

# Disable PROJ network access to avoid version conflicts
os.environ['PROJ_NETWORK'] = 'OFF'

# Set workspace
workspace = r"C:\GIS\Projects\ElectricityNetwork_SanFrancisco"
os.makedirs(workspace, exist_ok=True)

# Create transformer for coordinate conversion
transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# =============================================
# SAN FRANCISCO ELECTRICITY NETWORK (12 POLES)
# =============================================

# Central point (Downtown SF)
center_x, center_y = -122.4194, 37.7749

# Generate 12 electric poles in a radial pattern around downtown
np.random.seed(42)  # For reproducibility
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
distances = np.random.uniform(0.002, 0.005, 12)  # ~200-500m in degrees

pole_data = []
for i in range(12):
    pole_id = f"P{i+1:03d}"
    x = center_x + distances[i] * np.cos(angles[i])
    y = center_y + distances[i] * np.sin(angles[i])
    voltage = 11.0 if i < 8 else 33.0  # First 8 poles at 11kV, last 4 at 33kV
    install_date = f"2023-{(i % 12) + 1:02d}-{15 + (i * 3):02d}"
    
    pole_data.append({
        "PoleID": pole_id,
        "x": x,
        "y": y,
        "Voltage": voltage,
        "InstallDate": install_date
    })

# Create GeoDataFrame for poles
poles_df = pd.DataFrame(pole_data)
poles_gdf = gpd.GeoDataFrame(
    poles_df,
    geometry=[Point(xy) for xy in zip(poles_df["x"], poles_df["y"])],
    crs="EPSG:4326"
)

# Create power lines (connecting poles in a ring with some cross-connections)
line_data = [
    {"LineID": "L001", "Capacity_kV": 11.0, "coords": [(pole_data[i]["x"], pole_data[i]["y"]) for i in [0, 1, 2, 3, 4, 5, 6, 7, 0]]},
    {"LineID": "L002", "Capacity_kV": 33.0, "coords": [(pole_data[8]["x"], pole_data[8]["y"]), (pole_data[10]["x"], pole_data[10]["y"])]},
    {"LineID": "L003", "Capacity_kV": 33.0, "coords": [(pole_data[9]["x"], pole_data[9]["y"]), (pole_data[11]["x"], pole_data[11]["y"])]},
    {"LineID": "L004", "Capacity_kV": 33.0, "coords": [(pole_data[8]["x"], pole_data[8]["y"]), (pole_data[2]["x"], pole_data[2]["y"])]},
    {"LineID": "L005", "Capacity_kV": 33.0, "coords": [(pole_data[10]["x"], pole_data[10]["y"]), (pole_data[5]["x"], pole_data[5]["y"])]}
]

# Create GeoDataFrame for lines
lines_df = pd.DataFrame(line_data)
lines_gdf = gpd.GeoDataFrame(
    lines_df,
    geometry=[LineString(coords) for coords in lines_df["coords"]],
    crs="EPSG:4326"
)

# =============================================
# VISUALIZATION
# =============================================

# Reproject to Web Mercator
poles_gdf_mercator = poles_gdf.to_crs("EPSG:3857")
lines_gdf_mercator = lines_gdf.to_crs("EPSG:3857")

# Create figure
fig, ax = plt.subplots(figsize=(14, 12))

# Plot basemap first (so other layers appear on top)
try:
    bounds = poles_gdf_mercator.total_bounds
    if bounds is None or not all(np.isfinite(bounds)):
        raise ValueError("Invalid bounds for basemap")
    padding = 800  # ~800 meters padding
    img, ext = ctx.bounds2img(
        bounds[0] - padding,
        bounds[1] - padding,
        bounds[2] + padding,
        bounds[3] + padding,
        source=ctx.providers.Esri.WorldStreetMap,  # Changed to Esri for reliability
        zoom=15
    )
    ax.imshow(img, extent=ext, alpha=0.9)
    print("Basemap loaded successfully")
except Exception as e:
    print(f"Basemap error: {e}. Plotting without basemap.")
    ax.set_facecolor('lightgray')  # Fallback background color

# Plot power lines with voltage-based styling
for voltage in lines_gdf_mercator['Capacity_kV'].unique():
    subset = lines_gdf_mercator[lines_gdf_mercator['Capacity_kV'] == voltage]
    linewidth = 3 if voltage == 33.0 else 2
    color = 'gold' if voltage == 33.0 else 'deepskyblue'
    subset.plot(ax=ax, color=color, linewidth=linewidth, linestyle='dashed' if voltage == 33.0 else 'solid')

# Plot poles with voltage-based styling
poles_gdf_mercator.plot(
    ax=ax,
    column="Voltage",
    cmap='viridis',
    markersize=200,
    legend=True,
    legend_kwds={
        'label': "Voltage (kV)",
        'orientation': "horizontal",
        'ticks': [11.0, 33.0]  # Explicit ticks for clarity
    }
)

# Add pole labels
for idx, row in poles_gdf_mercator.iterrows():
    ax.annotate(
        text=row["PoleID"],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(15, 15),  # Increased offset for better readability
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)
    )

# Add line labels at midpoints
for idx, row in lines_gdf_mercator.iterrows():
    midpoint = row.geometry.interpolate(0.5, normalized=True)
    ax.annotate(
        text=f"{row['LineID']}\n{row['Capacity_kV']}kV",
        xy=(midpoint.x, midpoint.y),
        xytext=(0, 5),  # Slight offset for better placement
        textcoords="offset points",
        fontsize=9,
        ha='center',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
    )

# Customize plot with proper coordinate labels
plt.title("San Francisco Electricity Network\nDowntown Distribution System", fontsize=14, pad=20)
plt.xlabel("Easting (meters, EPSG:3857)")
plt.ylabel("Northing (meters, EPSG:3857)")

# Create secondary axes with proper transformation
def transform_x(x):
    lon, _ = transformer.transform(x, np.zeros_like(x))
    return lon

def transform_y(y):
    _, lat = transformer.transform(np.zeros_like(y), y)
    return lat

secax = ax.secondary_xaxis('top', functions=(transform_x, transform_x))
secax.set_xlabel("Longitude (degrees)")
secay = ax.secondary_yaxis('right', functions=(transform_y, transform_y))
secay.set_ylabel("Latitude (degrees)")

# Adjust bounds with padding
ax.set_xlim(bounds[0] - padding, bounds[2] + padding)
ax.set_ylim(bounds[1] - padding, bounds[3] + padding)

plt.tight_layout()
output_path = os.path.join(workspace, "SF_Electricity_Network.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)  # Close figure to free memory

print(f"Visualization complete. Output saved to: {output_path}")
