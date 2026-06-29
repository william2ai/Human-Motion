import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

# Define the sensors and their respective files
sensor_files = {
    'Head': 'head_sensor.csv',
    'Neck': 'neck_sensor.csv',
    'Hand': 'hand_sensor.csv',
    'Leg': 'leg_sensor.csv'
}

# Function to read and extract features from a sensor file
def read_sensor_data(file_path):
    df = pd.read_csv(file_path)
    # Return the acceleration features without averaging
    return df[['Acc X', 'Acc Y', 'Acc Z']].values

# Prepare the node features
node_features = []
for sensor in sensor_files:
    features = read_sensor_data(sensor_files[sensor])
    # Store the entire feature set for each sensor
    node_features.append(features)

# Convert the list of arrays to a tensor with shape (num_sensors, time_steps, features)
node_features_tensor = [torch.tensor(features, dtype=torch.float) for features in node_features]

# Check the shape of the data
for i, features in enumerate(node_features_tensor):
    print(f"{list(sensor_files.keys())[i]} features shape: {features.shape}")
    

# Initialize a similarity matrix
num_sensors = len(sensor_files)
similarity_matrix = torch.zeros((num_sensors, num_sensors))

# Compute cosine similarity for each pair of sensors
for i in range(num_sensors):
    for j in range(i + 1, num_sensors):
        # Compute the cosine similarity for the full time series
        sim = cosine_similarity(node_features_tensor[i].numpy(), node_features_tensor[j].numpy())
        # Average the similarity across all time steps
        avg_similarity = sim.mean()
        similarity_matrix[i, j] = avg_similarity
        similarity_matrix[j, i] = avg_similarity  # Symmetric matrix

# Print the similarity matrix
print("Similarity Matrix:")
print(similarity_matrix)

# Define a threshold for similarity
similarity_threshold = 0.8  # Example threshold

# Create edges based on similarity
edges = []
sensor_names = list(sensor_files.keys())

for i in range(num_sensors):
    for j in range(i + 1, num_sensors):
        if similarity_matrix[i, j] > similarity_threshold:
            edges.append((i, j))

# Convert edges to tensor format
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Create the graph data object
graph_data = Data(x=node_features_tensor, edge_index=edge_index)

# Check the graph data
print("Graph Data:")
print(graph_data)