import os
import sys
import trimesh
import torch  # Replaces TensorFlow
from sklearn.neighbors import KDTree
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'train_logmap'))

import delaunay_pytorch  # Replace with PyTorch equivalent
import torch_dataset  # Replace with PyTorch dataset implementation
import pointnet_seg_torch as model  # Replace with PyTorch-based PointNet implementation

BATCH_SIZE = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_NEIGHBORS = 120
N_NEAREST_NEIGHBORS = 30
N_TRIG_NEIGHBORS = 10

RESIZE = True

def resize(neighbor_points):
    diag = safe_norm(torch.max(neighbor_points, dim=1).values - torch.min(neighbor_points, dim=1).values, axis=-1)
    diag = diag.unsqueeze(1).unsqueeze(2).repeat(1, neighbor_points.shape[1], neighbor_points.shape[2])
    return neighbor_points / diag

def geodesic_patch(neighbor_coordinates, neighbors_index, classifier_model, device):
    center_index = neighbors_index[:, :1]
    center_coordinates = neighbor_coordinates[:, :1]

    # Forward pass through classifier
    geo_distances = classifier_model(neighbor_coordinates.to(device)).squeeze(-1)
    geo_distances = torch.sigmoid(geo_distances)

    # Find closest neighbors based on geodesic distances
    closest = torch.topk(geo_distances[:, 1:], k=N_NEAREST_NEIGHBORS, dim=-1).indices
    neighbor_coordinates = torch.gather(neighbor_coordinates[:, 1:], 1, closest.unsqueeze(-1).expand(-1, -1, neighbor_coordinates.shape[-1]))
    neighbors_index = torch.gather(neighbors_index[:, 1:], 1, closest)

    # Add the center point back
    neighbor_coordinates = torch.cat([center_coordinates, neighbor_coordinates], dim=1)
    neighbors_index = torch.cat([center_index, neighbors_index], dim=1)

    return neighbor_coordinates, neighbors_index

def init_graph_pytorch(X3D, n_neighbors, classifier_model, logmap_model, device):
    n_points = X3D.shape[0]
    coord_3D = torch.tensor(X3D, dtype=torch.float32, device=device)
    
    # Placeholder equivalent
    points_neighbors0 = torch.empty((BATCH_SIZE, n_neighbors + 1), dtype=torch.int32, device=device)
    first_index = torch.empty((BATCH_SIZE,), dtype=torch.int32, device=device)
    
    # Compute neighbor points
    neighbor_points0 = coord_3D[points_neighbors0]
    neighbor_points0 = neighbor_points0 - neighbor_points0[:, :1, :]

    # Resize
    if RESIZE:
        diag = torch.norm(torch.max(neighbor_points0, dim=1)[0] - torch.min(neighbor_points0, dim=1)[0], dim=-1)
        diag = diag.unsqueeze(-1).unsqueeze(-1).expand_as(neighbor_points0)
        neighbor_points0 = neighbor_points0 / diag

    # Geodesic patch generation
    neighbor_points, points_neighbors = geodesic_patch(neighbor_points0, points_neighbors0, classifier_model,device)

    # Logmap prediction
    predicted_map = logmap_model(neighbor_points)

    # Placeholder for Delaunay triangulation (use a PyTorch equivalent)
    target_triangles, target_indices = delaunay_pytorch(predicted_map, points_neighbors[:, 1:], first_index)
    
    return {
        "points_neighbors0": points_neighbors0,
        "predicted_map": predicted_map,
        "target_triangles": target_triangles,
        "target_indices": target_indices,
    }


def reconstruct(name, classifier_model, logmap_model, in_path, res_path, device):
    # Load the input point cloud
    logmap_points = np.loadtxt(os.path.join(in_path, name))
    name = name.replace('.xyz', "")
    X3D = logmap_points[:, :3]

    # Build KDTree for neighborhood queries
    tree = KDTree(X3D)

    # Initialize normals (if required for downstream processing)
    X3D_normals = np.zeros([X3D.shape[0], 3])
    X3D_normals[:, 2] = 1

    # Number of points in the point cloud
    n_points = len(X3D)

    # Initialize the PyTorch graph for processing
    ops = init_graph_pytorch(torch.tensor(X3D, dtype=torch.float32, device=device), 
                             n_neighbors=120, 
                             classifier_model=classifier_model, 
                             logmap_model=logmap_model, 
                             device=device)

    # Prepare batch processing
    points_indices = list(range(n_points))
    predicted_map, triangles, indices, predicted_neighborhood_indices = [], [], [], []

    # Batch processing loop
    for step in range(1 + len(X3D) // BATCH_SIZE):
        if step % 10 == 0:
            print(f"step: {step}/{n_points // BATCH_SIZE}")

        # Define current points batch
        if (step + 1) * BATCH_SIZE > n_points:
            current_points = points_indices[-BATCH_SIZE:]
        else:
            current_points = points_indices[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]

        # Query neighbors for the current batch
        center_points = np.array(X3D[current_points])
        points_neighbors = tree.query(center_points, 120 + 1)[1]

        # Move data to PyTorch tensors on the specified device
        points_neighbors_tensor = torch.tensor(points_neighbors, dtype=torch.int64, device=device)
        current_points_tensor = torch.tensor(current_points, dtype=torch.int64, device=device)

        # Perform inference using the PyTorch graph
        with torch.no_grad():
            predicted_map_batch = ops['predicted_map'](points_neighbors_tensor)
            target_triangles_batch = ops['target_triangles'](points_neighbors_tensor, current_points_tensor)
            target_indices_batch = ops['target_indices'](points_neighbors_tensor, current_points_tensor)
            predicted_neighborhood_indices_batch = ops['predicted_neighborhood_indices'](points_neighbors_tensor)

        # Handle the last batch (truncate if needed)
        if (step + 1) * BATCH_SIZE > n_points:
            batch_size = n_points % BATCH_SIZE
            predicted_map_batch = predicted_map_batch[-batch_size:]
            target_triangles_batch = target_triangles_batch[-batch_size:]
            target_indices_batch = target_indices_batch[-batch_size:]
            predicted_neighborhood_indices_batch = predicted_neighborhood_indices_batch[-batch_size:]

        # Append results to lists
        predicted_map.append(predicted_map_batch.cpu().numpy())
        triangles.append(target_triangles_batch.cpu().numpy())
        indices.append(target_indices_batch.cpu().numpy())
        predicted_neighborhood_indices.append(predicted_neighborhood_indices_batch.cpu().numpy())

    # Concatenate results from all batches
    predicted_map = np.concatenate(predicted_map)
    triangles = np.concatenate(triangles)
    indices = np.concatenate(indices)
    predicted_neighborhood_indices = np.concatenate(predicted_neighborhood_indices)

    # Save results
    np.save(os.path.join(res_path, f"predicted_map_{name}.npy"), predicted_map)
    np.save(os.path.join(res_path, f"predicted_neighborhood_indices_{name}.npy"), predicted_neighborhood_indices)

    # Export the raw mesh as a PLY file
    trimesh.Trimesh(X3D, indices[triangles > 0.5]).export(os.path.join(res_path, f"predicted_raw_mesh_{name}.ply"))


if __name__ == '__main__':
    in_path = "data/test_data"
    res_path = "data/test_data/raw_prediction"
    os.makedirs(res_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logmap_model = LogMapModel().to(device)
    classifier_model = ClassifierModel().to(device)

    # Load pre-trained models
    logmap_model.load_state_dict(torch.load("data/pretrained_models/pretrained_logmap/model.pth"))
    classifier_model.load_state_dict(torch.load("data/pretrained_models/pretrained_classifier/model.pth"))

    files = [x for x in os.listdir(in_path) if x.endswith('.xyz')]

    for name in files:
        reconstruct(name, classifier_model, logmap_model, in_path, res_path, device)