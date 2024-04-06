import os
import torch
import numpy as np
from tqdm import tqdm
import json
from torchvision.io import read_image
from torcheval.metrics import FrechetInceptionDistance


truth_img_dir = './ground_truth_objs_images'
test_img_dir = './pixel2mesh_objs_images'

classes = os.listdir(truth_img_dir)

batch_size = 1 #8
device_type = 'cuda'
device = torch.device(device_type)

losses = dict()
for key in classes:
    losses[key] = []


n_imgs = 6

for img_class in classes:
    metric = FrechetInceptionDistance(device=device)

    truth_img_path = os.path.join(truth_img_dir, img_class)
    test_img_path = os.path.join(test_img_dir, img_class)

    n_models = len(os.listdir(truth_img_path))
    for i in tqdm(range(0, n_models, batch_size)):
        inp = []
        target = []
        for j in range(n_imgs):
            test_img = read_image(os.path.join(truth_img_path, f"model_{i}", f"var_{j}.png"))[:3] / 255
            real_img = read_image(os.path.join(test_img_path, f"model_{i}", f"var_{j}.png"))[:3] / 255
            inp.append(test_img.unsqueeze(0).to(device))
            target.append(real_img.unsqueeze(0).to(device))
        # for j in range(n_imgs):
        # inp = read_image(os.path.join(truth_img_path, f"model_{i}", f"var_{j}.png")).to(device)
        # target = read_image(os.path.join(test_img_path, f"model_{i}", f"var_{j}.png")).to(device)

        metric.update(torch.cat(inp), False)
        metric.update(torch.cat(target), True)
        #metric.update(inp, target)
    frechet_val = metric.compute()
    print(frechet_val)
    losses[img_class].append(frechet_val.item())

results_save_path = './results/frechet/pixel2mesh.json'
os.makedirs('./results/frechet', exist_ok=True)
# Save to a file
with open(results_save_path, 'w') as json_file:
    json.dump(losses, json_file)