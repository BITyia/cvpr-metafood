import torch
import open3d as o3d
import torch.optim as optim
import trimesh
import json
import argparse
import tqdm
import os
import numpy as np

torch.set_printoptions(precision=10)

def check_for_nans(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}!")

def chamfer_distance(p1, p2, batch_size=1000):
    """
    Calculate the Chamfer Distance between two point clouds p1 and p2.
    Both p1 and p2 should be tensors of shape (N, 3) and (M, 3).
    """
    num_points_p1 = p1.shape[0]
    num_points_p2 = p2.shape[0]

    def pairwise_distances(a, b):
        a_sq = a.pow(2).sum(dim=1, keepdim=True)
        b_sq = b.pow(2).sum(dim=1, keepdim=True)
        dist = a_sq - 2 * a @ b.T + b_sq.T
        return dist * 100

    min_dist_p1 = torch.full((num_points_p1,), float('inf'), device=p1.device)
    min_dist_p2 = torch.full((num_points_p2,), float('inf'), device=p2.device)

    for i in range(0, num_points_p1, batch_size):
        end_i = min(i + batch_size, num_points_p1)
        batch_p1 = p1[i:end_i]
        dist = pairwise_distances(batch_p1, p2)
        min_dist_p1[i:end_i] = dist.min(dim=1)[0]

    for j in range(0, num_points_p2, batch_size):
        end_j = min(j + batch_size, num_points_p2)
        batch_p2 = p2[j:end_j]
        dist = pairwise_distances(p1, batch_p2)
        min_dist_p2[j:end_j] = dist.min(dim=0)[0]

    min_dist_p1 = torch.sqrt(min_dist_p1)
    min_dist_p2 = torch.sqrt(min_dist_p2)

    chamfer_dist = torch.mean(min_dist_p1) + torch.mean(min_dist_p2)
    check_for_nans(chamfer_dist, "chamfer_distance")
    return chamfer_dist

def load_mesh(file_path):
    mesh = trimesh.load(file_path)
    print("Volume:", mesh.volume)
    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices), triangles=o3d.utility.Vector3iVector(mesh.faces))
    return mesh

def get_point_cloud(mesh, number_of_points=100000):
    print("Number of vertices:", len(mesh.vertices))
    print("Number of points:", number_of_points)
    if number_of_points > len(mesh.vertices):
        return mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))
    return mesh.sample_points_uniformly(number_of_points=number_of_points)

def apply_transformation(pc, transform_matrix):
    homogenous_pc = torch.cat((pc, torch.ones((pc.shape[0], 1), device=pc.device)), dim=1)
    transformed_pc = homogenous_pc @ transform_matrix.T
    return transformed_pc[:, :3]

def load_initial_transformation(json_root, Food_id):
    with open(json_root, 'r') as f:
        data = json.load(f)
    
    records = data.get(Food_id, [])
    if not records:
        return np.eye(4)

    min_record = min(records, key=lambda x: x['chamfer_distance'])
    print("Min chamfer distance:", min_record['chamfer_distance'])
    return np.array(min_record['transformation'])

def save_transformed_obj(mesh, transformation, output_path):
    mesh.transform(transformation)
    o3d.io.write_triangle_mesh(output_path, mesh)

def save_transformation_matrix(transform_json_path, Food_id, transformation):
    if os.path.exists(transform_json_path):
        with open(transform_json_path, 'r') as f:
            transform_data = json.load(f)
    else:
        transform_data = {}
    
    transform_data[f"{Food_id}.obj"] = transformation.tolist()

    with open(transform_json_path, 'w') as f:
        json.dump(transform_data, f, indent=4)

def se3_to_SE3(wu):
    device = wu.device
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=device)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
    return Rt

def skew_symmetric(w):
    w0, w1, w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O, -w2, w1], dim=-1),
                      torch.stack([w2, O, -w0], dim=-1),
                      torch.stack([-w1, w0, O], dim=-1)], dim=-2)
    return wx

def taylor_A(x, nth=10):
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        if i > 0: denom *= (2 * i) * (2 * i + 1)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans

def taylor_B(x, nth=10):
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        denom *= (2 * i + 1) * (2 * i + 2)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans

def taylor_C(x, nth=10):
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        denom *= (2 * i + 2) * (2 * i + 3)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans

def apply_gradient_clipping(optimizer, clip_value=1.0):
    for param in optimizer.param_groups[0]['params']:
        if param.grad is not None:
            param.grad.data.clamp_(-clip_value, clip_value)

def main():
    parser = argparse.ArgumentParser(description="Align two .obj files interactively and save the transformation matrix.")
    parser.add_argument('--Food_id', type=str, default='14', help="Food id")
    parser.add_argument('--number_of_points', type=int, default=100000, help="Number of points to sample")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for ICP")
    parser.add_argument('--max_iter', type=int, default=2000, help="Max iteration for ICP")
    parser.add_argument('--align_times', type=int, default=1, help="Align times")
    parser.add_argument('--json_root', type=str, default="D://Code//record.json", help="Path to the JSON record file")
    parser.add_argument('--transform_json_path', type=str, default="C://Users//TY//Desktop//cvpr-metafood//phase2//example_transform//transform.json", help="Path to the transformation JSON file")
    parser.add_argument('--output_obj_dir', type=str, default="C://Users//TY//Desktop//cvpr-metafood//phase2//predict_transform", help="Directory to save transformed objects")
    parser.add_argument('--ground_truth_path', type=str, default="C://Users//TY//Desktop//cvpr-metafood//phase2//ground_truth//", help="Directory of ground truth objects")
    parser.add_argument('--predict_path', type=str, default="C://Users//TY//Desktop//cvpr-metafood//phase2//predict//", help="Directory of predicted objects")
    args = parser.parse_args()
    
    Food_id = args.Food_id
    json_root = args.json_root
    transform_json_path = args.transform_json_path
    output_obj_dir = args.output_obj_dir
    ground_truth_path = args.ground_truth_path
    predict_path = args.predict_path

    with open(json_root, 'r') as json_file:
        json_data = json.load(json_file)

    if Food_id not in json_data:
        json_data[Food_id] = []

    current_record = {
        "sample_points": args.number_of_points,
        "threshold": args.threshold
    }

    original_mesh1 = load_mesh(os.path.join(ground_truth_path, f"{Food_id}.obj"))
    original_mesh2 = load_mesh(os.path.join(predict_path, f"{Food_id}.obj"))

    pcd1 = get_point_cloud(original_mesh1, args.number_of_points)
    pcd2 = get_point_cloud(original_mesh2, args.number_of_points)

    pcd1_tensor = torch.tensor(np.asarray(pcd1.points), dtype=torch.float64).cuda().requires_grad_(True)
    pcd2_tensor = torch.tensor(np.asarray(pcd2.points), dtype=torch.float64).cuda().requires_grad_(True)

    initial_transformation = load_initial_transformation(json_root, Food_id)
    print("Initial transformation:", initial_transformation)
    initial_transformation_tensor = torch.tensor(initial_transformation, dtype=torch.float64).cuda()

    se3_params = torch.zeros((1, 6), dtype=torch.float64, device='cuda', requires_grad=True)
    optimizer = optim.Adam([se3_params], lr=1e-3)

    for epoch in tqdm.tqdm(range(15)):
        optimizer.zero_grad()

        se3_matrix = se3_to_SE3(se3_params)
        se3_matrix_4x4 = torch.eye(4, dtype=torch.float64, device=se3_matrix.device)
        se3_matrix_4x4[:3, :] = se3_matrix.squeeze()

        transform_matrix = initial_transformation_tensor @ se3_matrix_4x4
        transformed_pcd2 = apply_transformation(pcd2_tensor, transform_matrix)
        loss = chamfer_distance(pcd1_tensor, transformed_pcd2, batch_size=1000)
        
        loss.backward()
        apply_gradient_clipping(optimizer) 
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Chamfer Distance: {loss.item()}")

    final_transform = transform_matrix.detach().cpu().numpy()
    current_record["transformation"] = final_transform.tolist()
    current_record["chamfer_distance"] = loss.item()

    print("Final transformation:")
    print(final_transform)
    print("Final chamfer distance:", loss.item())

    json_data[Food_id].append(current_record)
    with open(json_root, 'w') as f:
        json.dump(json_data, f)

    output_obj_path = os.path.join(output_obj_dir, f"{Food_id}.ply")
    save_transformed_obj(original_mesh2, final_transform, output_obj_path)

    save_transformation_matrix(transform_json_path, Food_id, final_transform)

if __name__ == "__main__":
    main()
