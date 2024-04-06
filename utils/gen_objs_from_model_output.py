from contextlib import nullcontext
import torch
import os
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale
from torch.utils.data import Dataset

random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

class ImageToMeshDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        files = next(os.walk(dataset_dir))[1]

        self.len = len(files)
        self.n_variations = len(os.listdir(os.path.join(dataset_dir, files[0]))) - 1 # ignore processed_model.pt

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mesh_dict = torch.load(os.path.join(self.dataset_dir, f"model_{idx}", "processed_model.pt"))
        
        # Single image for resnet
        img_idx = 0
        img = read_image(os.path.join(self.dataset_dir, f"model_{idx}", f"var_{img_idx}.png"))
        mask = data_utils.background_mask(img)
        mesh_dict['image'] = rgb_to_grayscale(img[0:3]) * mask # ignore alpha channel

        return mesh_dict

    def map(self, func):
        for idx in range(self.len):
            path = os.path.join(self.dataset_dir, f"model_{idx}", "processed_model.pt")
            mesh_dict = func(torch.load(path))
            torch.save(mesh_dict, path)
        return self
        
def pad_batch(batch):
    # group matching keys in batch
    items = list(zip(*[item.values() for item in batch]))
    packed_dict = {}
    for i, key in enumerate(batch[0].keys()):
        if items[i][0].dim() == 0:
            padded_values = torch.tensor(items[i]).contiguous()
        else:
            padded_values = torch.nn.utils.rnn.pad_sequence(items[i], batch_first=True, padding_value=0.).contiguous()
            
        packed_dict[key] = padded_values
    return packed_dict

if __name__ == "__main__":
    from tqdm import tqdm
    import shared.data_utils as data_utils
    import modules

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    compile = False
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    device_type = 'cuda'
    device = torch.device(device_type)
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
    
    class_labels = {'bench': 0, 'chair': 1, 'table': 2}

    # with open('./config.json', 'r') as f:
    #     config = load(f)
    config = torch.load('/home/charlie/Documents/polygen/final_no_labels/config.pt')

    quantization_bits = 8

    # Create vertex model
    vertex_model = modules.ImageToVertexModel(
        decoder_config=config["vertex_decoder_config"],
        res_net_config=config["res_net_config"],
        max_num_input_verts=config["max_num_input_verts"],
        quantization_bits=quantization_bits,
        class_conditional=False,
        num_classes=len(class_labels),
        device=device
    ).to(device=device)

    # Create face model
    face_model = modules.FaceModel(
        encoder_config=config["face_encoder_config"],
        decoder_config=config["face_decoder_config"],
        class_conditional=False,
        max_seq_length=config["max_num_face_sequence"],
        quantization_bits=quantization_bits,
        max_num_input_verts=config["max_num_input_verts"],
        decoder_cross_attention=True,
        use_discrete_vertex_embeddings=True,
        device=device
    ).to(device=device)

    # compile the model

    if compile:
        # requires PyTorch 2.0 and Linux
        print("compiling the model... (takes a ~minute)")
        vertex_model = torch.compile(vertex_model, dynamic=True)
        face_model = torch.compile(face_model, dynamic=True)

    chkpt = torch.load("./final_no_labels/ckpt.pt")
    vertex_model.load_state_dict(chkpt['vertex_model'])
    face_model.load_state_dict(chkpt['face_model'])

    os.makedirs('./objs_no_labels/bench', exist_ok=True)
    os.makedirs('./objs_no_labels/chair', exist_ok=True)
    os.makedirs('./objs_no_labels/table', exist_ok=True)

    dataset_path = "./val_meshes_even_dist/"
    save_path = "./objs_no_labels/"

    n_examples = 2000
    batch_size = 3
    classes = list(class_labels.keys())
    for i in tqdm(range(0, n_examples)):
        images = []
        masks = []
        for b in range(0, batch_size):
            img = read_image(os.path.join(dataset_path, classes[b], f"model_{i}", f"var_0.png"))
            mask = data_utils.background_mask(img)
            img = rgb_to_grayscale(img[0:3]) * mask
            images.append(img.unsqueeze(0))
        images = torch.vstack(images).to(device)
        batch = {'image': images, 'class_label': torch.tensor([class_labels[classes[b]] for b in range(batch_size)], device=device)} # ignore alpha channel

        with ctx:
            vertex_samples = vertex_model.sample(
                batch_size, context=batch, max_sample_length=config['max_num_input_verts'], top_p=0.1,
                recenter_verts=False, only_return_complete=False)

            face_samples = face_model.sample(
            context=vertex_samples, max_sample_length=config['max_num_face_sequence'], top_p=0.1,
            only_return_complete=False) 

        mesh_list = []
        for batch in range(batch_size):
            mesh_list.append(
                {
                'vertices': vertex_samples['vertices'][batch][:vertex_samples['num_vertices'][batch]].cpu(),
                'faces': data_utils.unflatten_faces(
                    face_samples['faces'][batch][:face_samples['num_face_indices'][batch]].cpu())
                }
            )
                    
        for batch in range(batch_size):
            path = os.path.join(save_path, classes[batch], f'model_{i}.obj')
            data_utils.write_obj(
                mesh_list[batch]['vertices'], mesh_list[batch]['faces'], 
                path)
