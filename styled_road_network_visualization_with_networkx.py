import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import matplotlib.patches as mpatches

# === Load Shapefile ===
shapefile_path = 'roads.shp'  # <- Change this to your actual shapefile
gdf = gpd.read_file(shapefile_path)
gdf = gdf[gdf.geometry.type == 'LineString']

# === Choose Attribute for Road Type Classification ===
color_attr = 'road_class'
if color_attr not in gdf.columns:
    raise ValueError(f"Column '{color_attr}' not found in shapefile attributes.")

# Get unique road types
unique_road_types = gdf[color_attr].dropna().unique()
unique_road_types.sort()

# Define a color map
predefined_colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'gray', 'black']
color_map = {
    road_type: predefined_colors[i % len(predefined_colors)]
    for i, road_type in enumerate(unique_road_types)
}

# === Build Graph ===
G = nx.Graph()
for idx, row in gdf.iterrows():
    line: LineString = row.geometry
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        u = coords[i]
        v = coords[i + 1]
        G.add_edge(u, v, attr_dict=row.to_dict())

# === Styling and Positioning ===
pos = {node: node for node in G.nodes}
edge_colors = []
edge_widths = []
legend_labels = {}

for u, v, data in G.edges(data=True):
    attr = data.get('attr_dict', {})
    road_type = attr.get(color_attr, 'unknown')
    color = color_map.get(road_type, 'gray')

    # Assign width based on road type
    if road_type == 'major':
        width = 3.5
    elif road_type == 'minor':
        width = 2
    elif road_type == 'local':
        width = 1
    else:
        width = 1.5

    edge_colors.append(color)
    edge_widths.append(width)

    pretty_label = road_type.replace('_', ' ').title()
    if pretty_label not in legend_labels:
        legend_labels[pretty_label] = color

# === Plot Network ===
plt.figure(figsize=(12, 10))
nx.draw(
    G,
    pos,
    node_size=0,
    edge_color=edge_colors,
    width=edge_widths,
    with_labels=False
)

# === Create Legend ===
legend_handles = [
    mpatches.Patch(color=color, label=label)
    for label, color in legend_labels.items()
]
plt.legend(handles=legend_handles, title='Road Type', loc='lower left')
plt.title("Styled Road Network from Shapefile")
plt.axis('off')
plt.tight_layout()
plt.savefig("roads_networkx.png")
plt.show()
