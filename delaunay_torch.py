import torch
import torch.nn.functional as F
import numpy as np
import config

def get_couples_matrix_sparse(shape):
    couples = []
    for i in range(1, shape):
        for j in range(i):
            couples.append([i, j])
    couples = np.array(couples)
    return couples

def safe_norm(x, epsilon=config.EPS, dim=None):
    return torch.sqrt(torch.clamp(torch.sum(x**2, dim=dim), min=epsilon))

def get_middle_points(center_point, B):
    center_point = center_point.unsqueeze(1).repeat(1, B.shape[1], 1)
    return (center_point + B) / 2.0

def comp_half_planes(nn_coord, center_point):
    n_points = nn_coord.shape[0]
    n_neighbors = nn_coord.shape[1]
    middle_points = get_middle_points(center_point, nn_coord)
    dir_vec = nn_coord - center_point.unsqueeze(1)
    half_planes_normal = dir_vec / torch.clamp(safe_norm(dir_vec, dim=-1, epsilon=config.EPS).unsqueeze(-1), min=config.EPS)
    col3 = -(middle_points[:, :, 0] * half_planes_normal[:, :, 0] + middle_points[:, :, 1] * half_planes_normal[:, :, 1])
    half_planes = torch.cat([half_planes_normal, col3.unsqueeze(-1)], dim=-1)
    return half_planes

def get_is_trig_exact(inter_dist, n_neighbors):
    n_points = inter_dist.shape[0]
    inter_dist = -torch.sign(inter_dist)
    is_triangle = torch.sum(inter_dist, dim=2)
    is_triangle = torch.where(is_triangle < n_neighbors, torch.zeros_like(is_triangle), torch.ones_like(is_triangle))
    return is_triangle

def compute_intersections(half_planes, couples):
    inter = torch.cross(half_planes[:, couples[:, 0]], half_planes[:, couples[:, 1]], dim=-1)
    mask = torch.abs(inter[:, :, 2]) < config.EPS
    inter = inter / torch.clamp(inter[:, :, 2].unsqueeze(-1), min=1e-10)
    inter[mask] = 10.0  # set far away points
    return inter

def compute_triangles_local_geodesic_distances(nn_coord, center_point, couples):
    n_neighbors = nn_coord.shape[1]
    n_trigs = couples.shape[0]
    nn_coord = nn_coord[:, :, :2]
    center_point = center_point[:, :2]
    half_planes = comp_half_planes(nn_coord, center_point)
    intersections = compute_intersections(half_planes, couples)
    intersection_couples = couples.unsqueeze(0).repeat(center_point.shape[0], 1, 1)

    inter_dist0 = torch.sum(
        half_planes.unsqueeze(1) * intersections.unsqueeze(2),
        dim=-1
    )
    to_ignore = torch.zeros_like(inter_dist0)
    to_ignore.scatter_(
        -1,
        intersection_couples.view(-1, 3).t(),
        1.0
    )
    inter_dist0[to_ignore > 0.5] = -1e10
    inter_dist = torch.where(torch.abs(inter_dist0) < config.EPS, -1e10, inter_dist0)
    is_triangle_exact = get_is_trig_exact(inter_dist, n_neighbors)
    return is_triangle_exact, intersection_couples

def get_triangles_geo_batches(n_neighbors=60, gdist=None, gdist_neighbors=None, first_index=None):
    couples = torch.tensor(get_couples_matrix_sparse(n_neighbors), dtype=torch.int32)
    nn_coord = gdist[:, 1:]
    center_point = gdist[:, 0]
    exact_triangles, local_indices = compute_triangles_local_geodesic_distances(nn_coord, center_point, couples)
    global_indices = gdist_neighbors.gather(1, local_indices)
    first_index = first_index.unsqueeze(-1).unsqueeze(-1).repeat(1, global_indices.shape[1], 1)
    global_indices = torch.cat([first_index, global_indices], dim=2)
    return exact_triangles, global_indices

