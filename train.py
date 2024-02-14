import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from random import randint
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np


class ImageToMeshDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        files = next(os.walk(dataset_dir))[1]

        self.len = len(files)
        self.n_variations = 2 #len(os.listdir(os.path.join(dataset_dir, files[0]))) - 2 # ignore processed_model.pt and make it zero indexed
        #self.img_idxs = np.zeros(self.len, dtype=np.int8) # start all image idx at zero and increment each time 
        self.img_idxs = np.empty((self.len, self.n_variations), dtype=np.int16)
        self.img_counts = np.ones(self.len, dtype=np.int8) # count how many images have been seen for each mesh, so that they can be reshuffled once all have been seen
        for i in range(self.len):
            self.img_idxs[i] = np.random.permutation(self.n_variations) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        mesh_dict = torch.load(os.path.join(self.dataset_dir, f"model_{idx}", "processed_model.pt"))
        #img_idx = randint(0, 5) #)self.n_variations)
        img_idx = self.img_idxs[idx, self.img_counts[idx]]
        self.img_counts[idx] += 1
        #self.img_idxs[idx] += 1
        if self.img_counts[idx] == self.n_variations: #self.img_idxs[idx] == 2: #
            #self.img_idxs[idx] = 0
            self.img_idxs[idx] = np.random.permutation(self.n_variations)
            self.img_counts[idx] = 1
        
        img_idx = 0
        img = read_image(os.path.join(self.dataset_dir, f"model_{idx}", f"var_{img_idx}.png"))
        #img = rgb_to_grayscale(img[0:3])
        mesh_dict['image'] = img[0:3] #rgb_to_grayscale(img[0:3]) #img[0:3] # ignore alpha channel
        #mesh_dict['face_seq_lens'] = torch.tensor(mesh_dict['faces'].shape[0])
        #mesh_dict['vert_flat_seq_lens'] = torch.tensor(mesh_dict['vertices_flat'].shape[0])
        #mesh_dict['vert_seq_lens'] = torch.tensor(mesh_dict['vertices'].shape[0])
        #mesh_dict['class_label'] = torch.tensor(idx)
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
    #print(items)
    for i, key in enumerate(batch[0].keys()):
        if items[i][0].dim() == 0:
            padded_values = torch.tensor(items[i])
            #padded_values = torch.tensor(items[i], device=device)
        else:
            padded_values = torch.nn.utils.rnn.pad_sequence(items[i], batch_first=True, padding_value=0.)
            #padded_values = torch.nn.utils.rnn.pad_sequence(items[i], batch_first=True, padding_value=0.).to(device=device)
    
        packed_dict[key] = padded_values
    return packed_dict

if __name__ == "__main__":
    import shared.data_utils as data_utils
    import modules
    import time
    from shared.math_utils import LinearWarmupCosineAnnealingLR
    import matplotlib.pyplot as plt
    import wandb
    from contextlib import nullcontext
    import torch._inductor.codecache
    from torch.utils.data import DataLoader
    from json import load

    #torch.backends.cuda.enable_flash_sdp(True)
    #torch.backends.cuda.enable_mem_efficient_sdp(True)
    #torch.backends.cuda.enable_math_sdp(False)

    #print(torch._inductor.codecache.CacheBase.get_local_cache_path())
    #print(torch._inductor.codecache.CacheBase.get_global_cache_path())

    torch.compiler.reset()
    torch._dynamo.reset()

    random_seed = 100
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    batch_size = 2
    shuffle = True
    num_workers = 2
    compile = True # Compiling throws an assertion error for some reason, bug?
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

    dataset_path = "./processed_chairs_and_benches/"
    chkpt_path = "./checkpoints_chairs_and_benches"
    synthetic_dataset = ImageToMeshDataset(dataset_path)

    # Create vertex and face datasets
    is_preprocessed = 'faces_mask' in synthetic_dataset[0].keys() and 'num_vertices' in synthetic_dataset[0].keys()
    if is_preprocessed:
        dataset = synthetic_dataset
        """ vertex_model_dataset = synthetic_dataset
        face_model_dataset = synthetic_dataset """
    else:
        # Dataset has already been processed for vertex and face model training
        dataset = data_utils.make_vertex_model_dataset(
            synthetic_dataset, apply_random_shift=False)
        dataset = data_utils.make_face_model_dataset(
            synthetic_dataset, apply_random_shift=False)
        """ vertex_model_dataset = data_utils.make_vertex_model_dataset(
            synthetic_dataset, apply_random_shift=False)
        face_model_dataset = data_utils.make_face_model_dataset(
            synthetic_dataset, apply_random_shift=False) """


    num_examples = len(synthetic_dataset)
    train_idxs, val_idxs, test_idxs = torch.utils.data.random_split(range(num_examples), [0.925, 0.025, 0.05])

    train_idxs = list(range(4)) # [1,2,3,4] # trains well on tiny dataset
    val_idxs = list(range(4)) #[1,2,3,4]
    train_v_ds = torch.utils.data.Subset(dataset, train_idxs)
    val_v_ds = torch.utils.data.Subset(dataset, val_idxs)
    test_v_ds = torch.utils.data.Subset(dataset, test_idxs)

    vertex_train_loader = iter(cycle(DataLoader(train_v_ds, 
                                            shuffle=shuffle, 
                                            batch_size=batch_size, 
                                            collate_fn=pad_batch,
                                            num_workers=num_workers,
                                            persistent_workers=True,
                                            pin_memory=True)))

    vertex_val_loader = iter(cycle(DataLoader(val_v_ds, 
                                            shuffle=shuffle, 
                                            batch_size=batch_size, 
                                            collate_fn=pad_batch,
                                            num_workers=0,
                                            pin_memory=True)))

    # Create vertex model

    #config = torch.load("./checkpoints_reduced_similar_mesh_dataset_1/model_config.pt")
    with open('./config.json', 'r') as f:
        config = load(f)

    # Create vertex model
    vertex_model = modules.ImageToVertexModel(
        decoder_config=config["vertex_decoder_config"],
        res_net_config=config["res_net_config"],
        max_num_input_verts=config["max_num_input_verts"],
        quantization_bits=8,
        device=device
    ).to(device=device)

    # vertex_model = modules.VertexModel(
    #     decoder_config=config["vertex_decoder_config"],
    #     max_num_input_verts=config["max_num_input_verts"],
    #     context_type='label',
    #     class_conditional=True,
    #     quantization_bits=8,
    #     num_classes=len(train_idxs),
    #     device=device
    # ).to(device=device)

    # Create face model
    face_model = modules.FaceModel(
        encoder_config=config["face_encoder_config"],
        decoder_config=config["face_decoder_config"],
        class_conditional=False,
        max_seq_length=config["max_num_face_sequence"],
        quantization_bits=8,
        max_num_input_verts=config["max_num_input_verts"],
        decoder_cross_attention=True,
        use_discrete_vertex_embeddings=True,
        device=device
    ).to(device=device)

    # compile the model

    if compile:
        # requires PyTorch 2.0 and Linux
        print("compiling the model... (takes a ~minute)")
        vertex_model = torch.compile(vertex_model)#mode="reduce-overhead") #, backend="cudagraphs"
        face_model = torch.compile(face_model)# mode="reduce-overhead")

    # Train models

    # Autotuner runs a short benchmark and selects the kernel with 
    # the best performance on a given hardware for a given input size for computing convolutions
    # disable for highly variable input sizes
    #torch.backends.cudnn.benchmark = False

    # Optimization settings
    learning_rate = 3e-4
    training_steps = 5000 #300001
    log_step = 1
    save_step = 50000
    sample_step = 10000
    n_samples = 4

    # wandb logging
    wandb_config = {
        "learning_rate": learning_rate, 
        "batch_size": batch_size,
        "training_steps": training_steps,
        "vertex_decoder_config": config['vertex_decoder_config'],
        "res_net_config": config['res_net_config'],
        "face_encoder_config": config['face_encoder_config'],
        "face_decoder_config": config['face_decoder_config'],
    }
    """ config = {
        "learning_rate": learning_rate, 
        "batch_size": batch_size,
        "training_steps": training_steps,
        "vertex_decoder_config": vertex_decoder_config,
        "res_net_config": res_net_config,
        "face_encoder_config": face_encoder_config,
        "face_decoder_config": face_decoder_config,
    } """
    wandb.init(project="polygen", config=wandb_config)
    wandb_config = None
    #wandb.init(project="polygen", config=config, id="097wwgo9", resume=True)

    linear_warmup_steps = 0 #5000 # 5000 in paper
    # Create an optimizer an minimize the summed log probability of the mesh sequences
    #face_model_optim = torch.optim.AdamW(face_model.parameters(), lr=learning_rate)
    face_model_optim = torch.optim.AdamW(face_model.parameters(), lr=learning_rate)

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

    #vertex_model_optim = torch.optim.AdamW(vertex_model.parameters(), lr=learning_rate)
    vertex_model_optim = torch.optim.AdamW(vertex_model.parameters(), lr=learning_rate)
    vertex_schedular = LinearWarmupCosineAnnealingLR(
            vertex_model_optim,
            linear_warmup_steps,
            training_steps,
            0.0,
            )

    scaler = torch.cuda.amp.GradScaler(enabled=device_type == 'cuda')

    start_step = 0

    """ chkpt = torch.load("./checkpoints_reduced_similar_mesh_dataset/chkpt_45000.pt")
    vertex_model.load_state_dict(chkpt['vertex_model'])
    face_model.load_state_dict(chkpt['face_model'])
    vertex_model_optim.load_state_dict(chkpt['vertex_optimizer'])
    face_model_optim.load_state_dict(chkpt['face_optimizer'])
    vertex_schedular.load_state_dict(chkpt['vertex_schedular'])
    face_schedular.load_state_dict(chkpt['face_schedular'])
    scaler.load_state_dict(chkpt['scaler'])
    start_step = chkpt['step']+1 """

    os.makedirs(chkpt_path, exist_ok=True)
    #os.makedirs("./checkpoints_broken_textures", exist_ok=True)

    model_config = {
        "init_lr":learning_rate,
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
    
    #next(vertex_train_loader) # skip first batch, it is all zeros when num_workers>0, wtf...

    # Training loop
    for n in range(start_step, training_steps+1):
        
        # Sample a batch from the dataloader, reset dataloader if at end
        # vertex and face loaders should have same size
        """ vertex_model_batch = next(vertex_train_loader)
        face_model_batch = next(face_train_loader) """
        """ for k in vertex_model_batch.keys():
            vertex_model_batch[k] = vertex_model_batch[k].to(device=device, dtype=data_type)
        for k in face_model_batch.keys():
            face_model_batch[k] = face_model_batch[k].to(device=device, dtype=data_type) """
        
        #with torch.autocast(device_type=device_type, dtype=data_type, enabled=use_amp):
        vertex_model_batch = next(vertex_train_loader)
        #vertex_model_batch['class_label'] = torch.tensor([1,2,3,4]).repeat(4,1)
        for key, value in vertex_model_batch.items():
            vertex_model_batch[key] = value.to(device)# .contiguous()

        t = time.time()
        with ctx:
            vertex_model_pred_dist = vertex_model(vertex_model_batch)
            # vertex_model_loss = -torch.sum(
            #     vertex_model_pred_dist.log_prob(vertex_model_batch['vertices_flat'].to(device)) * 
            #     vertex_model_batch['vertices_flat_mask'].to(device))  
            #vertex_model_batch = None # free memory
            
            #face_model_batch = next(face_train_loader)
            face_model_pred_dist  = face_model(vertex_model_batch)
            # face_model_loss = -torch.sum(face_model_pred_dist.log_prob(vertex_model_batch['faces'].to(device)) * 
            #     vertex_model_batch['faces_mask'].to(device))
            #face_model_batch = None # free memory
            #vertex_model_batch = None # free memory

            vertex_model_loss = -torch.sum(
                    vertex_model_pred_dist.log_prob(vertex_model_batch['vertices_flat'].to(device)) * 
                    vertex_model_batch['vertices_flat_mask'].to(device))  
            

            face_model_loss = -torch.sum(
                face_model_pred_dist.log_prob(vertex_model_batch['faces'].to(device)) * 
                    vertex_model_batch['faces_mask'].to(device))
            
        vertex_model_batch = None

        print("XX")#, vertex_model_loss.shape)
        if torch.isnan(vertex_model_loss).any():
            print("NaN in vertex model loss!!!")

        # vertex_model_loss.backward()
        # torch.nn.utils.clip_grad_norm_(vertex_model.parameters(), 1.0)
        # vertex_model_optim.step()
        # vertex_model_optim.zero_grad(set_to_none=True)
        
        # face_model_loss.backward()
        # torch.nn.utils.clip_grad_norm_(face_model.parameters(), 1.0)
        # face_model_optim.step()
        # face_model_optim.zero_grad(set_to_none=True)

        scaler.scale(vertex_model_loss).backward()
        scaler.unscale_(vertex_model_optim)
        torch.nn.utils.clip_grad_norm_(vertex_model.parameters(), 1.0)
        scaler.step(vertex_model_optim)
        vertex_model_optim.zero_grad(set_to_none=True)
        
        #if n % 2 == 0:
        scaler.scale(face_model_loss).backward()
        scaler.unscale_(face_model_optim) # grad norm is scaled so we unscale it
        torch.nn.utils.clip_grad_norm_(face_model.parameters(), 1.0)
        scaler.step(face_model_optim)
        face_model_optim.zero_grad(set_to_none=True) # setting to none can use less memory 
        
        face_schedular.step()
        vertex_schedular.step()
        scaler.update()


        dt = time.time() - t
        
        if n % save_step == 0:
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

        if n % sample_step == 0:# and n > 0:
            vertex_val_batch = next(vertex_val_loader)

            with ctx:
                vertex_samples = vertex_model.sample(
                n_samples, context=vertex_val_batch, max_sample_length=max_num_input_verts, top_p=0.95,
                recenter_verts=False, only_return_complete=False)

                face_samples = face_model.sample(
                context=vertex_samples, max_sample_length=max_num_face_sequence, top_p=0.95,
                only_return_complete=False) 

            try:
                mesh_list = []
                for batch in range(min(n_samples, batch_size)):
                    mesh_list.append(
                        {
                        'vertices': vertex_samples['vertices'][batch][:vertex_samples['num_vertices'][batch]].cpu(),
                        'faces': data_utils.unflatten_faces(
                            face_samples['faces'][batch][:face_samples['num_face_indices'][batch]].cpu())
                        }
                    )
                fig = data_utils.plot_meshes(mesh_list, ax_lims=0.5, return_fig=True)
                log['Sample'] = wandb.Image(fig)
                plt.close(fig)
            except:
                print(f"Error plotting mesh at step {n}")

            fig = None
            mesh_list = None
            vertex_samples = None
            face_samples = None
            vertex_val_batch = None # free memory

        wandb.log(log)


    wandb.finish()
