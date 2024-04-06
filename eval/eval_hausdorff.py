"""
Earth Movers Distance (EMD)
"""

import os
import torch
from tqdm import tqdm
import json
from pytorch3d.io import IO, load_obj
from scipy.spatial.distance import directed_hausdorff
# scipy.stats.wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None)[source]

truth_obj_dir = '/home/charlie/Documents/polygen/val_meshes_even_dist'
#test_obj_dir = '/home/charlie/Documents/AutoSDF/demo_data/objs' 
#test_obj_dir = '/home/charlie/Documents/Pixel2Mesh/Data/objs'
test_obj_dir = '/home/charlie/Documents/polygen/objs_with_label'

classes = os.listdir(truth_obj_dir)

batch_size = 1
device_type = 'cuda'
device = torch.device(device_type)

IO = IO()

dists = dict()
for key in classes:
    dists[key] = []

for obj_class in classes:
    truth_objs_path = os.path.join(truth_obj_dir, obj_class)
    test_objs_path = os.path.join(test_obj_dir, obj_class)

    n_objs = len(os.listdir(truth_objs_path))
    for i in tqdm(range(0, n_objs, batch_size)):
        #meshes = pytorch3d.io.load_objs_as_meshes()
        truth_pc = torch.load(os.path.join(truth_objs_path, f"model_{i}/processed_model.pt"))['vertices'] #IO.load_pointcloud(os.path.join(truth_objs_path, f"model_{i}/processed_model.pt"), device=device)
        #test_pc = IO.load_pointcloud(os.path.join(test_objs_path, f"model_{i}.obj"), device=device)
        test_pc = load_obj(os.path.join(test_objs_path, f"model_{i}.obj"), load_textures=False)[0]
        #loss, loss_normals = chamfer_distance(truth_pc.unsqueeze(0), test_pc.unsqueeze(0))
        hausdorff_dist, _, _ = directed_hausdorff(truth_pc, test_pc)
        dists[obj_class].append(hausdorff_dist)

results_save_path = './results/hausdorff/with_labels.json'
os.makedirs('./results/hausdorff', exist_ok=True)
# Save to a file
with open(results_save_path, 'w') as json_file:
    json.dump(dists, json_file)