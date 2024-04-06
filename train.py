import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from random import randint
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
import shared.data_utils as data_utils

class ImageToMeshDataset(Dataset):
    def __init__(self, dataset_dir, multiview=False):
        """
        dataset_dir should be a directory of folders with the names "model_X" where X is an integer.
        Each model subdirectory should contain a processed_model.pt file with the preprocessed mesh and
        the input image of that mesh with the name "var_X.png". These files can be obtained with the generate_dataset.py script.
        """
        self.dataset_dir = dataset_dir
        files = next(os.walk(dataset_dir))[1]

        self.len = len(files)
        self.n_variations = len(os.listdir(os.path.join(dataset_dir, files[0]))) - 1 # ignore processed_model.pt

        self.multiview = multiview
        if multiview:
            self.img_idxs = np.empty((self.len, self.n_variations), dtype=np.int16)
            self.img_counts = np.zeros(self.len, dtype=np.int8) # count how many images have been seen for each mesh, so that they can be reshuffled once all have been seen
            for i in range(self.len):
                self.img_idxs[i] = np.random.permutation(self.n_variations) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mesh_dict = torch.load(os.path.join(self.dataset_dir, f"model_{idx}", "processed_model.pt"))

        if self.multiview:
            img_idx = self.img_idxs[idx, self.img_counts[idx]]
            self.img_counts[idx] += 1
            if self.img_counts[idx] == self.n_variations:
                self.img_idxs[idx] = np.random.permutation(self.n_variations)
                self.img_counts[idx] = 0
            
            # Stack multiple views of the mesh for multi-view resnet
            images = []
            #for n in np.random.permutation(self.n_variations): # shuffle images
            for n in range(self.n_variations):
                img = read_image(os.path.join(self.dataset_dir, f"model_{idx}", f"var_{n}.png"))
                images.append(rgb_to_grayscale(img[0:3]))
            mesh_dict['images'] = torch.stack(images) # ignore alpha channel
        else:
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
    import matplotlib
    matplotlib.use('agg') # use non-gui backend to avoid threading errors
    import matplotlib.pyplot as plt

    import shared.data_utils as data_utils
    import modules
    import time
    from shared.math_utils import LinearWarmupCosineAnnealingLR
    import wandb
    from contextlib import nullcontext
    from torch.utils.data import DataLoader
    from json import load
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    torch.compiler.reset()
    torch._dynamo.reset()
    #torch._dynamo.config.cache_size_limit = 8 # may need to increase to avoid recompiling during training

    random_seed = 21
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    batch_size = 8
    shuffle = True
    num_workers = 4
    compile = True
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    device_type = 'cuda'
    device = torch.device(device_type)
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
    
    #torch.autograd.set_detect_anomaly(True, check_nan=True)

    print(f"Using {device_type} device")
    torch.multiprocessing.set_start_method('spawn')

    def cycle(iterable):
        # cycle through a dataloader indefinitely
        while True:
            for x in iterable:
                yield x

    class_labels = {'bench': 0, 'chair': 1, 'table': 2}

    ### Create Dataset ###
    dataset_path = "./processed_meshes/"
    chkpt_path = "./checkpoints_meshes/"
    synthetic_dataset = ImageToMeshDataset(dataset_path)

    # Create vertex and face datasets
    is_preprocessed = 'faces_mask' in synthetic_dataset[0].keys() and 'num_vertices' in synthetic_dataset[0].keys()
    if is_preprocessed:
        dataset = synthetic_dataset
    else:
        # Dataset has already been processed for vertex and face model training
        dataset = data_utils.make_vertex_model_dataset(
            synthetic_dataset, apply_random_shift=False)
        dataset = data_utils.make_face_model_dataset(
            synthetic_dataset, apply_random_shift=False)

    # Create train and test splits
    num_examples = len(synthetic_dataset)
    train_idxs, test_idxs = torch.utils.data.random_split(range(num_examples), [0.95, 0.05])
    train_v_ds = torch.utils.data.Subset(dataset, train_idxs)
    test_v_ds = torch.utils.data.Subset(dataset, test_idxs)

    vertex_train_loader = iter(cycle(DataLoader(train_v_ds, 
                                            shuffle=shuffle, 
                                            batch_size=batch_size, 
                                            collate_fn=pad_batch,
                                            num_workers=num_workers,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            drop_last=True)))

    vertex_test_loader = iter(cycle(DataLoader(test_v_ds, 
                                            shuffle=shuffle, 
                                            batch_size=batch_size, 
                                            collate_fn=pad_batch,
                                            num_workers=0,
                                            pin_memory=True)))

    # Create vertex model
    with open('./config.json', 'r') as f:
        config = load(f)

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


    # Autotuner runs a short benchmark and selects the kernel with 
    # the best performance on a given hardware for a given input size for computing convolutions
    # disable for highly variable input sizes
    #torch.backends.cudnn.benchmark = False

    ### Train models ###

    # Optimization settings
    vertex_learning_rate = 3e-4
    face_learning_rate = 1e-4
    training_steps = 600001
    log_step = 1
    save_step = 50000
    sample_step = 20000
    n_samples = 4

    # wandb logging
    wandb_config = {
        "vertex_learning_rate": vertex_learning_rate, 
        "face_learning_rate": face_learning_rate,
        "batch_size": batch_size,
        "training_steps": training_steps,
        "vertex_decoder_config": config['vertex_decoder_config'],
        "res_net_config": config['res_net_config'],
        "face_encoder_config": config['face_encoder_config'],
        "face_decoder_config": config['face_decoder_config'],
    }

    wandb.init(project="polygen", config=wandb_config)
    wandb_config = None

    linear_warmup_steps = 5000
    # Create an optimizer an minimize the summed log probability of the mesh sequences
    face_model_optim = torch.optim.AdamW(face_model.parameters(), lr=face_learning_rate)

    # linear warmpup to reduce early overfitting if the data is skewed 
    # or if the early batches of data are shuffled so that they contain similar data and overfit to these features.
    # it gives less weight to the first {linear_warmup_steps} steps.
    # sort of like a weight initialization based on the dataset.
    # Also helps stabilize training for small datasets since the training will be more unstable initially with random initial parameters
    face_schedular = LinearWarmupCosineAnnealingLR(
            face_model_optim,
            linear_warmup_steps,
            training_steps,
            0.0,
            )
    
    vertex_model_optim = torch.optim.AdamW(vertex_model.parameters(), lr=vertex_learning_rate)

    vertex_schedular = LinearWarmupCosineAnnealingLR(
        vertex_model_optim,
        linear_warmup_steps,
        training_steps,
        0.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device_type == 'cuda')

    start_step = 0

    # Uncomment when resuming from checkpoint
    #chkpt= torch.load("path/to/chkpt.pt")
    #vertex_model.load_state_dict(chkpt['vertex_model'])
    #face_model.load_state_dict(chkpt['face_model'])
    #vertex_model_optim.load_state_dict(chkpt['vertex_optimizer'])
    #face_model_optim.load_state_dict(chkpt['face_optimizer'])
    #vertex_schedular.load_state_dict(chkpt['vertex_schedular'])
    #face_schedular.load_state_dict(chkpt['face_schedular'])
    #scaler.load_state_dict(chkpt['scaler'])
    #start_step = chkpt['step']+1
    #chkpt_vertex = None
    #chkpt_face = None

    start_step = 0

    os.makedirs(chkpt_path, exist_ok=True)

    model_config = {
        "init_vertex_learning_rate":vertex_learning_rate,
        "init_face_learning_rate": face_learning_rate,
        "vertex_decoder_config": config['vertex_decoder_config'],
        "res_net_config": config['res_net_config'],
        "face_encoder_config": config['face_encoder_config'],
        "face_decoder_config": config['face_decoder_config'],
        "max_num_input_verts": config['max_num_input_verts'],
        "max_num_face_sequence": config['max_num_face_sequence']
    }

    torch.save(model_config, os.path.join(chkpt_path, "model_config.pt"))

    max_num_input_verts = config['max_num_input_verts']
    max_num_face_sequence = config['max_num_face_sequence']
    model_config = None
    config = None
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1) # use -1 as the padding index to ignore

    # Training loop
    for n in range(start_step, training_steps+1):
        vertex_model_batch = next(vertex_train_loader)
        for key, value in vertex_model_batch.items():            
            vertex_model_batch[key] = value.to(device)
            
        t = time.time()

        with ctx:
            vertex_model_pred_dist = vertex_model(vertex_model_batch)
            # vertex_model_loss = -torch.sum(
            #         vertex_model_pred_dist.log_prob(vertex_model_batch['vertices_flat']) * 
            #         vertex_model_batch['vertices_flat_mask'])  
        
        # Keep the first zero in the sequence as it is the stopping token, 
        # convert the following zeros to -1 as they are padding that should be ignored 
        seq_lengths = vertex_model_batch['vertices_flat_mask'].sum(-1).long()
        inp = vertex_model_pred_dist.logits * vertex_model_batch['vertices_flat_mask'].unsqueeze(-1)
        pad_mask = torch.zeros_like(vertex_model_batch['vertices_flat'], device=device)
        for i in range(inp.shape[0]):
            pad_mask[i, seq_lengths[i]:] -= 1
        targets = vertex_model_batch['vertices_flat'] + pad_mask
        inp += pad_mask.unsqueeze(-1)

        vertex_model_loss = criterion(inp.permute(0,2,1), targets.long())
            
        scaler.scale(vertex_model_loss).backward()
        scaler.unscale_(vertex_model_optim)
        torch.nn.utils.clip_grad_norm_(vertex_model.parameters(), 1.0)
        scaler.step(vertex_model_optim)
        vertex_model_optim.zero_grad(set_to_none=True) # zero grads after step to free memory
       
        with ctx:
            face_model_pred_dist  = face_model(vertex_model_batch)
            # face_model_loss = -torch.sum(
            #     face_model_pred_dist.log_prob(vertex_model_batch['faces'].to(device)) * 
            #         vertex_model_batch['faces_mask'].to(device))

        seq_lengths = vertex_model_batch['faces_mask'].sum(-1).long()
        inp = face_model_pred_dist.logits * vertex_model_batch['faces_mask'].unsqueeze(-1)
        pad_mask = torch.zeros_like(vertex_model_batch['faces'], device=device)
        for i in range(inp.shape[0]):
            pad_mask[i, seq_lengths[i]:] -= 1
        targets = vertex_model_batch['faces'] + pad_mask
        inp += pad_mask.unsqueeze(-1)

        face_model_loss = criterion(inp.permute(0,2,1), targets.long())
            
        vertex_model_batch = None # Free memory
        
        scaler.scale(face_model_loss).backward()
        scaler.unscale_(face_model_optim) # grad norm is scaled so we unscale it
        torch.nn.utils.clip_grad_norm_(face_model.parameters(), 1.0)
        scaler.step(face_model_optim)
        face_model_optim.zero_grad(set_to_none=True) # setting to none can use less memory 
        
        face_schedular.step()
        vertex_schedular.step()
        scaler.update()

        dt = time.time() - t
        
        if (n % save_step == 0 and n > 0) or n == training_steps-1:
            checkpoint = {
                "vertex_model": vertex_model.state_dict(),
                "face_model": face_model.state_dict(),
                "vertex_optimizer": vertex_model_optim.state_dict(),
                "face_optimizer": face_model_optim.state_dict(),
                "vertex_schedular": vertex_schedular.state_dict(),
                "face_schedular": face_schedular.state_dict(),
                "scaler": scaler.state_dict(),
                "step": n
            }
            torch.save(checkpoint, os.path.join(chkpt_path, f"chkpt_{n}.pt"))
            checkpoint = None # free memory

        # time forward pass
        if n % log_step == 0:
            log = {
                    "Loss (vertices)": vertex_model_loss,
                    "Loss (faces)": face_model_loss,
                    "Vertex_lr": vertex_model_optim.param_groups[0]['lr'],
                    "Face_lr": face_model_optim.param_groups[0]['lr'],
                    "Time (ms)": dt * 1000,
                }

        if n % sample_step == 0: # and n > 0:
            vertex_test_batch = next(vertex_test_loader)

            for key, value in vertex_test_batch.items():            
                vertex_test_batch[key] = value.to(device)

            with ctx:
                with torch.no_grad():
                    vertex_val_pred_dist = vertex_model(vertex_test_batch)
                    # Using the same loss function as in the original paper for simplicity, can be changed to use cross-entropy
                    vertex_val_loss = -torch.sum(
                            vertex_val_pred_dist.log_prob(vertex_test_batch['vertices_flat']) * 
                            vertex_test_batch['vertices_flat_mask'])  
                    
                    face_val_pred_dist  = face_model(vertex_test_batch)
                    face_val_loss = -torch.sum(
                    face_val_pred_dist.log_prob(vertex_test_batch['faces'].to(device)) * 
                            vertex_test_batch['faces_mask'].to(device))

                
                vertex_samples = vertex_model.sample(
                n_samples, context=vertex_test_batch, max_sample_length=max_num_input_verts, top_p=0.95,
                recenter_verts=False, only_return_complete=False)

                face_samples = face_model.sample(
                context=vertex_samples, max_sample_length=max_num_face_sequence, top_p=0.95,
                only_return_complete=False) 

            mesh_list = []
            for batch in range(min(n_samples, batch_size)):
                mesh_list.append(
                    {
                    'vertices': vertex_samples['vertices'][batch][:vertex_samples['num_vertices'][batch]].cpu(),
                    'faces': data_utils.unflatten_faces(
                        face_samples['faces'][batch][:face_samples['num_face_indices'][batch]].cpu())
                    }
                )
                
            fig_size = 4
            n_cols = 4
            n_rows = 1
            fig = data_utils.plot_meshes(mesh_list, ax_lims=0.5, return_fig=True)
            scale = 1.0
            fig_adjusted_height = fig_size * (n_rows + scale)  # Increase the height slightly more for larger images
            fig.set_figheight(fig_adjusted_height)

            # Add an extra row in the figure above the meshes to display the image
            # Doesn't work with n_samples > 4
            # Would be much easier if i changed the plot_meshes function but oh well...

            # Adjust positions of existing mesh axes to make space for larger images
            for ax in fig.axes:
                pos = ax.get_position()
                # Adjust the bottom position more significantly to make space for larger images
                new_bottom = pos.y0 - (scale / n_rows)  # Increase the offset for the bottom position
                new_bottom = max(new_bottom, 0)
                # Optionally adjust the height of the mesh plots if needed
                ax.set_position([pos.x0, new_bottom, pos.width, pos.height])

            # Now add your image axes above the mesh plots
            for batch in range(min(n_samples, batch_size)):
                # Calculate position for the new image axes
                mesh_ax = fig.axes[batch]
                mesh_pos = mesh_ax.get_position()

                image_height = mesh_pos.height # Adjust this multiplier based on desired image size
                image_bottom = mesh_pos.y0 + mesh_pos.height - (image_height - mesh_pos.height)
                image_pos = [mesh_pos.x0, mesh_pos.y0 + mesh_pos.height, mesh_pos.width, mesh_pos.height / n_rows]
                
                # Create new axes for the image
                axins = fig.add_axes(image_pos)
                if vertex_val_batch['image'][batch].shape[0] == 3:
                #if vertex_val_batch['images'][batch][0].shape[0] == 3: # for multi-view
                    cmap=None
                else:
                    cmap='gray'
                
                axins.imshow(vertex_val_batch['image'][batch].permute(1,2,0).cpu(), cmap=cmap)
                #axins.imshow(vertex_val_batch['images'][batch][4].permute(1,2,0).cpu(), cmap=cmap) # for multi-view
                axins.axis('off')

            log['Sample'] = wandb.Image(fig)
            plt.close(fig)
            
            log['Val Loss (vertices)'] = vertex_val_loss
            log['Val Loss (faces)'] = face_val_loss

            vertex_val_pred_dist = None
            face_val_pred_dist = None
            vertex_val_loss = None
            face_val_loss = None
            fig = None
            mesh_list = None
            vertex_samples = None
            face_samples = None
            vertex_val_batch = None # free memory

        wandb.log(log)
        log = None


    wandb.finish()
