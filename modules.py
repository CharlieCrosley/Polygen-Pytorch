import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from shared.data_utils import dequantize_verts, quantize_verts, create_causal_mask, convert_to_additive_mask
from shared.math_utils import top_k_logits, top_p_logits

def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out 

def init_weights_kaiming_uniform(m, mode='fan_in', nonlinearity='relu'):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity)

def init_weights_xavier_uniform(m, nonlinearity='relu'):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlinearity))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlinearity))

def init_weights_xavier_normal(m, nonlinearity='relu'):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(nonlinearity))

def init_weights_zeros(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.zeros_(m.weight) 


class VertexModel(nn.Module):
    """Autoregressive generative model of quantized mesh vertices.

    Operates on flattened vertex sequences with a stopping token:

    [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, z_n, STOP]

    Input vertex coordinates are embedded and tagged with learned coordinate and
    position indicators. A transformer decoder outputs logits for a quantized
    vertex distribution.
    """

    def __init__(self,
               decoder_config,
               quantization_bits,
               context_type,
               class_conditional=False,
               num_classes=55,
               max_num_input_verts=2500,
               use_discrete_embeddings=True,
               device='cpu'):
        """Initializes VertexModel.

        Args:
        decoder_config: Dictionary with TransformerDecoder config
        quantization_bits: Number of quantization used in mesh preprocessing.
        context_type: String indicating type of context for input to be conditioned on. One of 'label' or 'image'.
        class_conditional: If True, then condition on learned class embeddings.
        num_classes: Number of classes to condition on.
        max_num_input_verts: Maximum number of vertices. Used for learned position
            embeddings.
        use_discrete_embeddings: If True, use discrete rather than continuous
            vertex embeddings.
        """
        if context_type not in ['label', 'image', None]:
            raise ValueError('context_type must be one of "label", "image" or None.')
        
        super(VertexModel, self).__init__()
        self.embedding_dim = decoder_config['embd_size']
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.quantization_bits = quantization_bits
        self.use_discrete_embeddings = use_discrete_embeddings
        self.device = device

        # Embedding initialization
        if context_type == 'label':
            self.label_embd = nn.Embedding(self.num_classes, self.embedding_dim)

        self.coord_embd = nn.Embedding(3, self.embedding_dim) # 3 in to represent (x, y, z) 
        self.pos_embd = nn.Embedding(self.max_num_input_verts, self.embedding_dim)

        self.project_to_logits = nn.Linear(self.embedding_dim, 
                                           2**self.quantization_bits + 1, 
                                           bias=True)

        if self.use_discrete_embeddings:
            self.vertex_embd = nn.Embedding(2**self.quantization_bits + 1, 
                                            self.embedding_dim)
        else:
            self.vertex_embd = nn.Linear(self.max_num_input_verts, 
                                         self.embedding_dim, 
                                         bias=True)
        
        self.embed_zeros = nn.Parameter(torch.rand((1, 1, self.embedding_dim)))
        self.decoder = TransformerDecoder(**decoder_config, bias=False)
        
        self.apply(lambda m: init_weights_kaiming_uniform(m, nonlinearity='relu'))
        self.to(device)

    def prepare_context(self, context):
        """Prepare class label context."""
        if self.class_conditional:
            global_context_embedding = self.label_embd(context['class_label'].to(self.device))
        else:
            global_context_embedding = None
        return global_context_embedding, None
    
    def _embed_inputs(self, vertices, global_context_embedding=None):
        """Embeds flat vertices and adds position and coordinate information."""
        # Dequantize inputs and get shapes
        input_shape = vertices.shape
        batch_size, seq_length = input_shape[0], input_shape[1]
        seq_range = torch.arange(seq_length, device=self.device)
      
        # indicates whether the input token is an x, y, or z coordinate
        coord_embeddings = self.coord_embd(seq_range % 3) #[0,1,2,0,1,2,...] 
 
        # indicates which vertex in the sequence the token belongs to
        pos_embeddings = self.pos_embd(torch.floor_divide(seq_range, 3)) # [0,0,0,1,1,1,...]

        # Discrete vertex value embeddings
        # which expresses a token’s quantized coordinate value
        if self.use_discrete_embeddings:
            vert_embeddings = self.vertex_embd(vertices)
        # Continuous vertex value embeddings
        else:
            # Pad vertices to max_num_input_verts for the linear layer
            n_pad = self.max_num_input_verts - input_shape[1]
            pad_vertices = F.pad(vertices, (0, n_pad))
            vert_embeddings = self.vertex_embd(dequantize_verts(pad_vertices.unsqueeze(1), self.quantization_bits))
        
        if global_context_embedding is None: 
            zero_embed_tiled = torch.tile(self.embed_zeros, [batch_size, 1, 1])
        else: # global_context_embedding can be the embedded class label for example
            zero_embed_tiled = global_context_embedding.unsqueeze(1)

        # Aggregate embeddings
        embeddings = vert_embeddings + (coord_embeddings + pos_embeddings).unsqueeze(0)
        embeddings = torch.cat([zero_embed_tiled, embeddings], dim=1)
        return embeddings
    
    def create_vertex_coordinate_dist(self,
                   vertices,
                   vertices_mask=None,
                   global_context_embedding=None,
                   sequential_context_embeddings=None,
                   temperature=1.,
                   top_k=0,
                   top_p=1.,
                   cache=None):
        """Outputs categorical dist for quantized vertex coordinates."""

        # Embed inputs
        decoder_inputs = self._embed_inputs(vertices, global_context_embedding)
        if vertices_mask is not None:
            # append 0 to start of mask to account for concatenation in _embed_inputs
            vertices_mask = torch.logical_not(vertices_mask).float()
            vertices_mask = F.pad(vertices_mask, (1, 0), value=0.)

        if cache is not None:
            decoder_inputs = decoder_inputs[:, -1:]

        # pass through decoder
        is_causal = True if cache is None else False

        outputs = self.decoder(
            decoder_inputs,
            tgt_key_padding_mask=vertices_mask, 
            memory=sequential_context_embeddings, 
            cache=cache,
            is_causal=is_causal
        )
        
        # Get logits and optionally process for sampling
        logits = self.project_to_logits(outputs)
        logits /= temperature
        logits = top_k_logits(logits.float(), top_k)
        logits = top_p_logits(logits, top_p)
        cat_dist = torch.distributions.Categorical(logits=logits)
        return cat_dist

    def __call__(self, batch):
        """Pass batch through vertex model and get log probabilities under model.

        Args:
        batch: Dictionary containing:
            'vertices_flat': int32 vertex tensors of shape [batch_size, seq_length].
        is_training: If True, use dropout.

        Returns:
        pred_dist: tfd.Categorical predictive distribution with batch shape
            [batch_size, seq_length].
        """
        global_context, seq_context = self.prepare_context(batch)
        pred_dist = self.create_vertex_coordinate_dist(
            batch['vertices_flat'][:, :-1],  # Last element not used for preds
            vertices_mask=batch['vertices_flat_mask'][:, :-1],
            global_context_embedding=global_context,
            sequential_context_embeddings=seq_context)
        return pred_dist
    
    @torch.no_grad()
    def sample(self,
             num_samples,
             context=None,
             max_sample_length=None,
             temperature=1.,
             top_k=0,
             top_p=1.,
             recenter_verts=True,
             only_return_complete=True):
        """Autoregressive sampling with caching.

        Args:
        num_samples: Number of samples to produce.
        context: Dictionary of context, such as class labels. See _prepare_context
            for details.
        max_sample_length: Maximum length of sampled vertex sequences. Sequences
            that do not complete are truncated.
        temperature: Scalar softmax temperature > 0.
        top_k: Number of tokens to keep for top-k sampling.
        top_p: Proportion of probability mass to keep for top-p sampling.
        recenter_verts: If True, center vertex samples around origin. This should
            be used if model is trained using shift augmentations.
        only_return_complete: If True, only return completed samples. Otherwise
            return all samples along with completed indicator.

        Returns:
        outputs: Output dictionary with fields:
            'completed': Boolean tensor of shape [num_samples]. If True then
            corresponding sample completed within max_sample_length.
            'vertices': Tensor of samples with shape [num_samples, num_verts, 3].
            'num_vertices': Tensor indicating number of vertices for each example
            in padded vertex samples.
            'vertices_mask': Tensor of shape [num_samples, num_verts] that masks
            corresponding invalid elements in 'vertices'.
        """
        self.eval()
        # Obtain context for decoder
        global_context, seq_context = self.prepare_context(context)

        # num_samples is the minimum value of num_samples and the batch size of
        # context inputs (if present).
        if global_context is not None:
            num_samples = min(num_samples, global_context.shape[0])
            global_context = global_context[:num_samples]
            if seq_context is not None:
                seq_context = seq_context[:num_samples]
        elif seq_context is not None:
            num_samples = min(num_samples, seq_context.shape[0])
            seq_context = seq_context[:num_samples]

        # Initial values for loop variables
        samples = torch.zeros([num_samples, 0], dtype=torch.int32, device=self.device)

        if max_sample_length is not None and max_sample_length <= self.max_num_input_verts:
            max_sample_length = max_sample_length
        else:
            max_sample_length = self.max_num_input_verts
        
        cache = self.decoder.create_init_cache(num_samples)
        stop_cond = False
        i = 0
        max_iters = max_sample_length * 3 + 1
        while not stop_cond and i < max_iters:
            cat_dist = self.create_vertex_coordinate_dist(
                samples,
                global_context_embedding=global_context,
                sequential_context_embeddings=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p)
            next_sample = cat_dist.sample()[:, -1:]
            samples = torch.cat([samples, next_sample], dim=1)
            stop_cond = torch.eq(samples, 0).any(-1).all() # stop once all samples have a 0 (stop token)
            i += 1
        del cache
        # Check if samples completed. Samples are complete if the stopping token
        # is produced.
        completed = torch.eq(samples, 0).any(-1)

        # Get the number of vertices in the sample. This requires finding the
        # index of the stopping token. For complete samples use to argmax to get
        # first nonzero index.
        stop_index_completed = torch.eq(samples, 0).int().argmax(dim=-1)
        # For incomplete samples the stopping index is just the maximum index.
        stop_index_incomplete = (
            max_sample_length * 3 * torch.ones_like(stop_index_completed, device=self.device))
    
        # if completed use the completed index else use incomplete index
        stop_index = torch.where(
            completed, stop_index_completed, stop_index_incomplete)
        num_vertices = torch.floor_divide(stop_index, 3) # each vertex has 3 coordinates

        # Convert to 3D vertices by reshaping and re-ordering x -> y -> z
        samples = samples[:, :(torch.max(num_vertices) * 3)] - 1
        verts_dequantized = dequantize_verts(samples, self.quantization_bits)
        vertices = verts_dequantized.view(num_samples, -1, 3)
        # veriteces are currently in order z -> y -> x so we need to reorder them
        vertices = torch.stack(
            [vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1)

        # Pad samples to max sample length. This is required in order to concatenate
        # Samples across different replicator instances. Pad with stopping tokens
        # for incomplete samples.
        pad_size = max_sample_length - vertices.shape[1]
        vertices = F.pad(vertices, (0,0,0,pad_size))

        # 3D Vertex mask
        vertices_mask = torch.where(torch.arange(max_sample_length, device=self.device) < 
                                    num_vertices.unsqueeze(1), 1.0, 0.0)

        if recenter_verts:
            vert_max = torch.max(
                vertices - 1e10 * (1. - vertices_mask).unsqueeze(0), dim=1,
                keepdims=True)
            vert_min = torch.min(
                vertices + 1e10 * (1. - vertices_mask).unsqueeze(0), dim=1,
                keepdims=True)
            vert_centers = 0.5 * (vert_max + vert_min)
            vertices -= vert_centers
        vertices *= vertices_mask.unsqueeze(-1)

        if only_return_complete: # mask out incomplete samples
            vertices = vertices[completed]
            num_vertices = num_vertices[completed]
            vertices_mask = vertices_mask[completed]
            completed = completed[completed]

        # Outputs
        outputs = {
            'completed': completed,
            'vertices': vertices,
            'num_vertices': num_vertices,
            'vertices_mask': vertices_mask,
        }
        self.train()
        return outputs
    
    def to(self, device=None, **kwargs):
        module = super(VertexModel, self).to(device, **kwargs)
        module.device = device
        return module
    

class FaceModel(nn.Module):
    """Autoregressive generative model of n-gon meshes.

    Operates on sets of input vertices as well as flattened face sequences with
    new face and stopping tokens:

    [f_0^0, f_0^1, f_0^2, NEW, f_1^0, f_1^1, ..., STOP]

    Input vertices are encoded using a Transformer encoder.

    Input face sequences are embedded and tagged with learned position indicators,
    as well as their corresponding vertex embeddings. A transformer decoder
    outputs a pointer which is compared to each vertex embedding to obtain a
    distribution over vertex indices.
    """

    def __init__(self,
                encoder_config,
                decoder_config,
                class_conditional=True,
                num_classes=55,
                decoder_cross_attention=True,
                use_discrete_vertex_embeddings=True,
                quantization_bits=8,
                max_num_input_verts=2500,
                max_seq_length=5000,
                device='cpu'):
        """Initializes FaceModel.

        Args:
        encoder_config: Dictionary with TransformerEncoder config.
        decoder_config: Dictionary with TransformerDecoder config.
        class_conditional: If True, then condition on learned class embeddings.
        num_classes: Number of classes to condition on.
        decoder_cross_attention: If True, the use cross attention from decoder
            querys into encoder outputs.
        use_discrete_vertex_embeddings: If True, use discrete vertex embeddings.
        quantization_bits: Number of quantization bits for discrete vertex
            embeddings.
        max_num_input_verts: Maximum number of vertices.
        max_seq_length: Maximum face sequence length. Used for learned position
            embeddings.
        """
        super(FaceModel, self).__init__()
        self.embedding_dim = decoder_config['embd_size']
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.decoder_cross_attention = decoder_cross_attention
        self.use_discrete_vertex_embeddings = use_discrete_vertex_embeddings
        self.quantization_bits = quantization_bits
        self.max_num_input_verts = max_num_input_verts
        self.device = device

        self.label_embd = nn.Embedding(self.num_classes, self.embedding_dim)
        self.pos_embd = nn.Embedding(self.max_seq_length, self.embedding_dim)
        self.project_to_pointers = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        if use_discrete_vertex_embeddings:
            self.vertex_embd_1 = nn.Embedding(256, self.embedding_dim)
            self.vertex_embd_2 = nn.Embedding(256, self.embedding_dim)
            self.vertex_embd_3 = nn.Embedding(256, self.embedding_dim)
        else:
            self.vertex_embd = nn.Linear(self.max_num_input_verts, self.embedding_dim, bias=True)

        # learnable stopping token embeddings
        self.stopping_embeddings = torch.nn.Parameter(torch.rand([1, 2, self.embedding_dim]))
        self.embed_zeros = torch.nn.Parameter(torch.rand([1, 1, self.embedding_dim]))

        self.decoder = TransformerDecoder(**decoder_config, bias=False)
        self.encoder = TransformerEncoder(**encoder_config, bias=False)

        self.apply(lambda m: init_weights_kaiming_uniform(m, nonlinearity='relu'))
        self.to(device)
    
    def prepare_context(self, context):
        """Prepare class label context."""
        if self.class_conditional:
            global_context_embedding = self.label_embd(context['class_label'].to(self.device))
        else:
            global_context_embedding = None
        
        vertex_embeddings = self.embed_vertices(
            context['vertices'].to(self.device), context['vertices_mask'])
       
        if self.decoder_cross_attention:
            sequential_context_embeddings = (
                vertex_embeddings *
                F.pad(context['vertices_mask'], (2,0)).unsqueeze(-1))
        else:
            sequential_context_embeddings = None
        return (vertex_embeddings, global_context_embedding,
                sequential_context_embeddings)
    
    def embed_vertices(self, vertices, vertices_mask):
        """Embeds vertices with transformer encoder."""
        if self.use_discrete_vertex_embeddings:
            vertex_embeddings = 0.
            if torch.is_floating_point(vertices):
                verts = quantize_verts(vertices, self.quantization_bits)
            else:
                verts = vertices
            vertex_embeddings += self.vertex_embd_1(verts[..., 0])
            vertex_embeddings += self.vertex_embd_2(verts[..., 1])
            vertex_embeddings += self.vertex_embd_3(verts[..., 2])
        else:
            n_pad = self.max_num_input_verts - vertices.shape[1]
            pad_vertices = F.pad(vertices, (0, n_pad))
            vertex_embeddings = self.vertex_embd(pad_vertices.unsqueeze(1))

        vertex_embeddings = vertex_embeddings * vertices_mask.unsqueeze(-1)
        
        # Pad vertex embeddings with learned embeddings for stopping and new face tokens
        stopping_embeddings = torch.tile(self.stopping_embeddings, [vertices.shape[0], 1, 1])
        vertex_embeddings = torch.cat(
            [stopping_embeddings, vertex_embeddings], dim=1)
        
        # Pass through Transformer encoder
        vertices_mask = F.pad(vertices_mask, (2,0), value=1.) # pad for stopping and new face tokens
        vertices_mask = torch.logical_not(vertices_mask).float()
        
        vertex_embeddings = self.encoder(
            vertex_embeddings,
            src_key_padding_mask=vertices_mask,
            is_causal=False
        )
        return vertex_embeddings
    
    def embed_inputs(self, faces_long, vertex_embeddings,
                    global_context_embedding=None):
        """Embeds face sequences and adds within and between face positions."""

        # faces_long is the indices of the vertices in the face
        # gather those vertex embeddings according to the indices
        face_embeddings = batched_index_select(vertex_embeddings, 1, faces_long)
  
        # Position of vertex in face
        pos_embeddings = self.pos_embd(torch.arange(faces_long.shape[1], device=self.device)) 

        # Step zero embeddings
        batch_size = face_embeddings.shape[0]
        if global_context_embedding is None:
            # zero embedding is used to keep the shape the same when not using global context.
            zero_embed_tiled = torch.tile(self.embed_zeros, [batch_size, 1, 1])
        else:
            zero_embed_tiled = global_context_embedding.unsqueeze(1)
        
        # Aggregate embeddings
        embeddings = face_embeddings + pos_embeddings.unsqueeze(0)
        embeddings = torch.cat([zero_embed_tiled, embeddings], dim=1)

        return embeddings
    
    def create_vertex_indices_dist(self,
                   vertex_embeddings,
                   faces_long,
                   vertices_mask=None,
                   faces_mask=None,
                   global_context_embedding=None,
                   sequential_context_embeddings=None,
                   temperature=1.,
                   top_k=0,
                   top_p=1.,
                   cache=None):
        """Outputs categorical dist for vertex indices."""

        # Embed inputs
        decoder_inputs = self.embed_inputs(
            faces_long, vertex_embeddings, global_context_embedding)
        
        # Pass through Transformer decoder
        if cache is not None:
            decoder_inputs = decoder_inputs[:, -1:]

        if vertices_mask is not None:
            # append 1 to start of mask to account for the input embedding
            vertices_mask = F.pad(vertices_mask, (2, 0), value=1.)
            encoder_inp_mask = torch.logical_not(vertices_mask).float().contiguous()
        if faces_mask is not None:
            # append 0 to start of mask to account for the input embedding
            faces_mask = torch.logical_not(faces_mask).float()
            faces_mask = F.pad(faces_mask, (1, 0), value=0.).contiguous()
            
        is_causal = True if cache is None else False
        decoder_outputs = self.decoder(
            decoder_inputs,
            tgt_key_padding_mask=faces_mask, 
            memory=sequential_context_embeddings, 
            memory_key_padding_mask=encoder_inp_mask,
            cache=cache,
            is_causal=is_causal
        )

        # Get pointers by projecting transformer outputs to pointer vectors.
        pred_pointers = self.project_to_pointers(decoder_outputs)
        
        # pointer vector is compared to the input embeddings to get vertex scores
        logits = torch.matmul(pred_pointers, vertex_embeddings.transpose(1,2)).float()
        logits /= math.sqrt(self.embedding_dim)

        f_verts_mask = vertices_mask.unsqueeze(1)
        logits *= f_verts_mask
        logits -= (1. - f_verts_mask) * 1e9
        logits /= temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
        return torch.distributions.Categorical(logits=logits)
    
    def __call__(self, batch):
        """Pass batch through face model and get log probabilities.

        Args:
        batch: Dictionary containing:
            'vertices_dequantized': Tensor of shape [batch_size, num_vertices, 3].
            'faces': int32 tensor of shape [batch_size, seq_length] with flattened
            faces.
            'vertices_mask': float32 tensor with shape
            [batch_size, num_vertices] that masks padded elements in 'vertices'.
        is_training: If True, use dropout.

        Returns:
        pred_dist: tfd.Categorical predictive distribution with batch shape
            [batch_size, seq_length].
        """
    
        batch['vertices_mask'] = batch['vertices_mask'].to(self.device)
        vertex_embeddings, global_context, seq_context = self.prepare_context(batch)
        
        faces_mask = batch['faces'][:, :-1].float().to(self.device)
        pred_dist = self.create_vertex_indices_dist( #new_cache
            vertex_embeddings,
            faces_long=batch['faces'][:, :-1].to(self.device),
            vertices_mask=batch['vertices_mask'],
            faces_mask=faces_mask,
            global_context_embedding=global_context,
            sequential_context_embeddings=seq_context)
        return pred_dist

    @torch.no_grad()
    def sample(self,
             context,
             max_sample_length=None,
             temperature=1.,
             top_k=0,
             top_p=1.,
             only_return_complete=True):
        """Sample from face model using caching.

        Args:
        context: Dictionary of context, including 'vertices' and 'vertices_mask'.
            See _prepare_context for details.
        max_sample_length: Maximum length of sampled vertex sequences. Sequences
            that do not complete are truncated.
        temperature: Scalar softmax temperature > 0.
        top_k: Number of tokens to keep for top-k sampling.
        top_p: Proportion of probability mass to keep for top-p sampling.
        only_return_complete: If True, only return completed samples. Otherwise
            return all samples along with completed indicator.

        Returns:
        outputs: Output dictionary with fields:
            'completed': Boolean tensor of shape [num_samples]. If True then
            corresponding sample completed within max_sample_length.
            'faces': Tensor of samples with shape [num_samples, num_verts, 3].
            'num_face_indices': Tensor indicating number of vertices for each
            example in padded vertex samples.
        """
        self.eval()
        vertex_embeddings, global_context, seq_context = self.prepare_context(context)
        num_samples = vertex_embeddings.shape[0]

        # Initial values for loop variables
        samples = torch.zeros([num_samples, 0], dtype=torch.int32, device=self.device)
        max_sample_length = max_sample_length or self.max_seq_length
        
        cache = self.decoder.create_init_cache(num_samples)
        stop_cond = False
        i = 0
        while not stop_cond and i < max_sample_length:

            cat_dist = self.create_vertex_indices_dist(
                vertex_embeddings,
                samples,
                vertices_mask=context['vertices_mask'],
                global_context_embedding=global_context,
                sequential_context_embeddings=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p)
            next_sample = cat_dist.sample()[:, -1:]
            samples = torch.cat([samples, next_sample], dim=1)
            stop_cond = torch.eq(samples, 0).any(-1).all() # stop once all samples have a 0 (stop token)
            i += 1
        del cache
        # Record completed samples
        complete_samples = torch.eq(samples, 0).any(-1)
        # Find number of faces
        sample_length = samples.shape[-1]
        samples_range = torch.arange(sample_length, device=self.device).unsqueeze(0)
  
        # Get largest new face (1 is new face token) index as stopping point for incomplete samples.
        max_one_ind = torch.max(
            samples_range * (torch.eq(samples, 1)).int(),
            dim=-1)[1]
        zero_inds = torch.argmax(torch.eq(samples, 0).int(), axis=-1) # completed sample indices
        num_face_indices = torch.where(complete_samples, zero_inds, max_one_ind) + 1
        # Mask faces beyond stopping token with zeros
        # This mask has a -1 in order to replace the last new face token with zero
        faces_mask = (samples_range < num_face_indices.unsqueeze(-1) - 1).int()
     
        samples = samples * faces_mask
        # This is the real mask which keeps the last new face token
        faces_mask = (samples_range < num_face_indices.unsqueeze(-1)).int()

        # Pad to maximum size with zeros
        pad_size = max_sample_length - sample_length
        samples = F.pad(samples, (0, pad_size))

        if only_return_complete: # mask out incomplete samples
            samples = samples[complete_samples]
            num_face_indices = num_face_indices[complete_samples]
            for key in context:
                context[key] = context[key][complete_samples]
            complete_samples = complete_samples[complete_samples]

        # outputs
        outputs = {
            'context': context,
            'completed': complete_samples,
            'faces': samples,
            'num_face_indices': num_face_indices,
        }
        self.train()
        return outputs
    
    def to(self, device=None, **kwargs):
        module = super(FaceModel, self).to(device, **kwargs)
        module.device = device
        return module


class ConvResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_size=None, dropout_rate=0.1, kernel_size=(3,3), 
                 downsample=False, re_zero=True, bias=True, device='cpu'):
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.re_zero = re_zero
        self.downsample = downsample

        if re_zero:
            self.re_zero_alpha = nn.Parameter(torch.zeros(1, device=device))

        if downsample:
            # stride 2 to downsample
            out_channels = in_channels if hidden_size is None else hidden_size
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, bias=bias, device=device)
            self.bn_skip = nn.BatchNorm2d(out_channels, device=device)
        else:
            self.skip = None

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, bias=bias, device=device, padding='same')
        
        stride = 2 if downsample else 1
        padding = 0 if downsample else 'same'
        self.conv2 = nn.Conv2d(in_channels, hidden_size, kernel_size=kernel_size,
                                stride=stride, bias=bias, padding=padding, device=device)
        
        self.bn1 = nn.BatchNorm2d(in_channels, device=device)
        self.bn2 = nn.BatchNorm2d(hidden_size, device=device)

        self.relu = torch.nn.ReLU(inplace = False)

    def forward(self, inputs):
        if self.downsample:
            skip = self.skip(inputs)
            skip = self.bn_skip(skip)
        else:
            skip = inputs

        res = inputs

        res = self.conv1(res)
        res = self.bn1(res)
        res = F.relu(res)

        res = self.dropout(res)

        res = self.conv2(res)
        res = self.bn2(res)

        if self.re_zero:
            res *= self.re_zero_alpha
      
        out = skip + res
        # may be able to remove clone but compiling causes inplace operation errors
        return self.relu(out).clone() 
    
    
class MultiViewResNet(nn.Module):
    """ ResNet architecture for 2D image """

    def __init__(self, hidden_sizes=(64, 128, 256), dropout_rate=0.1, num_blocks=(1,2,2), re_zero=True, kernel_size=(3,3), bias=True, device='cpu'):
        super().__init__()

        self.re_zero = re_zero
        self.num_blocks = num_blocks
        self.device = device

        input_channels = 1
        self.conv1 = nn.Conv2d(input_channels, hidden_sizes[0], kernel_size=(7,7), stride=2, bias=bias, device=device)

        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1)
        self.fc = nn.Linear(14, 14, bias=bias)

        #self.conv_layers = []
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            # Image initially has 3 channels for RGB
            self.conv_layers.append(
                # Downsample at the start of each collection of blocks.
                # Skip fist block and only downsample at the start of every other block
                ConvResidualBlock(hidden_sizes[i] if i == 0 else hidden_sizes[i-1],
                                  hidden_sizes[i], 
                                  dropout_rate=dropout_rate, 
                                  kernel_size=kernel_size, 
                                  downsample=False if i == 0 else True,
                                  re_zero=re_zero, 
                                  device=device)
            )
            for _ in range(num_blocks[i]-1):
                self.conv_layers.append(
                    ConvResidualBlock(hidden_sizes[i], 
                                      hidden_sizes[i], 
                                      dropout_rate=dropout_rate, 
                                      kernel_size=kernel_size,
                                      re_zero=re_zero, 
                                      device=device)
                )


    def forward(self, inputs):
        # inputs = (Batch, View, C, Width, Height)
        inputs = inputs.to(self.device)

        # Swap batch and views dims -> (View, Batch, C, Width, Height)
        inputs = inputs.transpose(0, 1)

        # View pool
        view_pool = []
        for view in inputs:
            # view = (Batch, C, Width, Height)
            x = self.conv1(view)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            for layer in self.conv_layers:
                x = layer(x)
            view_pool.append(x) 
        
        view_pool = torch.stack(view_pool)
        pooled_view = torch.mean(view_pool, dim=0)
        pooled_view = self.fc(pooled_view)
        return pooled_view

    
class ResNet(nn.Module):
    """ ResNet architecture for 2D image """

    def __init__(self, hidden_sizes=(64, 128, 256), dropout_rate=0.1, num_blocks=(1,2,2), re_zero=True, kernel_size=(3,3), bias=True, device='cpu'):
        super().__init__()

        self.re_zero = re_zero
        self.num_blocks = num_blocks
        self.device = device

        input_channels = 1
        self.conv1 = nn.Conv2d(input_channels, hidden_sizes[0], kernel_size=(7,7), stride=2, bias=bias, device=device)
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            # Image initially has 3 channels for RGB
            self.conv_layers.append(
                # Downsample at the start of each collection of blocks.
                # Skip fist block and only downsample at the start of every other block
                ConvResidualBlock(hidden_sizes[i] if i == 0 else hidden_sizes[i-1],
                                  hidden_sizes[i], 
                                  dropout_rate=dropout_rate, 
                                  kernel_size=kernel_size, 
                                  downsample=False if i == 0 else True,
                                  re_zero=re_zero, 
                                  device=device)
            )
            for _ in range(num_blocks[i]-1):
                self.conv_layers.append(
                    ConvResidualBlock(hidden_sizes[i], 
                                      hidden_sizes[i], 
                                      dropout_rate=dropout_rate, 
                                      kernel_size=kernel_size,
                                      re_zero=re_zero, 
                                      device=device)
                )

    def forward(self, inputs):
        x = inputs.to(self.device)
        # Initial projection with large kernel as in original resnet architecture
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(x)

        for layer in self.conv_layers:
            x = layer(x)
        return x

from torchvision.models import resnet18, ResNet18_Weights

class CustomPretrainedResNet(nn.Module):
    def __init__(self, output_channels=256, device='cpu'):
        super(CustomPretrainedResNet, self).__init__()

        self.device = device

        weights = ResNet18_Weights.DEFAULT
        # Load a pre-trained resnet model
        self.resnet = resnet18(weights=weights)
        for param in self.resnet.parameters(): # freeze pretrained network
            param.requires_grad = False

        self.transforms = weights.transforms()

        # Remove the average pooling and fully connected layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
        # Example of adding a custom layer to change the output size
        # Here we add a Conv2d layer to change the number of output channels to 256
        # Assuming the input size is 256x256, the output size of the modified ResNet will be (batch_size, 256, 8, 8)
        # You can adjust the kernel size, stride, and padding to change the output spatial dimensions (x, y)
        self.custom_layers = nn.Sequential(
            nn.Conv2d(512, output_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.transforms(x.to(self.device))
        with torch.no_grad():
            # Pass input through the modified ResNet
            x = self.resnet(x)
        # Pass through the custom layers
        x = self.custom_layers(x)
        return x
    
from torchvision.transforms import v2

class ImageToVertexModel(VertexModel):
    """Generative model of quantized mesh vertices with image conditioning.

    Operates on flattened vertex sequences with a stopping token:

    [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, z_n, STOP]

    Input vertex coordinates are embedded and tagged with learned coordinate and
    position indicators. A transformer decoder outputs logits for a quantized
    vertex distribution. Image inputs are encoded and used to condition the
    vertex decoder.
    """

    def __init__(self, 
                 decoder_config,
                 res_net_config,
                 max_num_input_verts,
                 num_classes=4,
                 quantization_bits=8,
                 use_discrete_embeddings=True,
                 class_conditional=False,
                 device='cpu'):
        super().__init__(
            decoder_config=decoder_config,
            max_num_input_verts=max_num_input_verts,
            num_classes=num_classes,
            context_type='image',
            quantization_bits=quantization_bits,
            use_discrete_embeddings=use_discrete_embeddings,
            class_conditional=class_conditional,
            device=device
        )

        self.res_net = ResNet(**res_net_config, device=device).to(device)
        #self.res_net = CustomPretrainedResNet(256, device)
        #self.res_net = MultiViewResNet(**res_net_config, device=device).to(device)

        self.image_embd = nn.Linear(2, self.embedding_dim, bias=True, device=device)
        
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.class_conditional = class_conditional
        if class_conditional:
            self.label_embd = nn.Embedding(num_classes, self.embedding_dim)

        self.apply(lambda m: init_weights_kaiming_uniform(m, nonlinearity='relu'))


    def prepare_context(self, context):
        # -0.5 to center around zero
        # it may be better to subtract mean instead
        image = self.transforms(context['image']) - 0.5
        #image = context['images'] # 'images' for multiview


        # Pass images through encoder
        image_embeddings = self.res_net(image)

        # Add 2D coordinate grid embedding
        processed_image_resolution = image_embeddings.shape[-1]
        x = torch.linspace(-1., 1., processed_image_resolution, device=self.device)
        image_coords = torch.stack(torch.meshgrid(x, x, indexing='xy'), axis=-1)
        
        image_coord_embeddings = self.image_embd(image_coords)
     
        image_embeddings += image_coord_embeddings.unsqueeze(0).permute(0,3,2,1)

        # Reshape spatial grid to sequence
        batch_size = image_embeddings.shape[0]
        sequential_context_embedding = image_embeddings.view(batch_size, -1, self.embedding_dim)

        if self.class_conditional: # if using class labels alongside the images
            global_context_embedding = self.label_embd(context['class_label'])
        else:
            global_context_embedding = None

        return global_context_embedding, sequential_context_embedding


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class DotProductAttention(nn.Module):

    def __init__(self, embd_size=256, num_heads=4, bias=True, dropout_rate=0.2):
        super().__init__()
        # Ensure that the embedding can be split up into n heads
        assert embd_size % num_heads == 0

        self.n_head = num_heads
        self.n_embd = embd_size
        self.dropout = dropout_rate

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.q_proj = nn.Linear(embd_size, embd_size, bias=bias)
        self.v_proj = nn.Linear(embd_size, embd_size, bias=bias)
        self.k_proj = nn.Linear(embd_size, embd_size, bias=bias)
        # output projection
        self.out_proj = nn.Linear(embd_size, embd_size, bias=bias)

    def forward(self, 
                q,
                kv=None, 
                kv_padding_mask=None,
                attn_mask=None,
                cache=None,
                is_causal=True):
        
        if is_causal and attn_mask is not None:
            raise Exception("attn_mask is expected to be None if is_causal hint is true.")

        # if context_embedding is None then self attention is being used
        # otherwise the encoder output is being used for cross attention
        b, q_seq_len, n_embd = q.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.q_proj(q)
        
        if kv is None or cache is None:
            # use query for both key and value if context_embedding is not given (self attention)
            kv = kv if kv is not None else q
            k = self.k_proj(kv)
            v = self.v_proj(kv)

        # reuse k and v if cache is given to avoid recomputing everytime
        if cache is not None:
            k_old, v_old = cache['k'], cache['v']
            if k_old.shape[1] == 0: # dim 1 is the sequence length, 0 means first iter so no cache saved
                cache['k'] = k
                cache['v'] = v
            else:
                k = cache['k'] = torch.cat((k_old, k), dim=1)
                v = cache['v'] = torch.cat((v_old, v), dim=1)
        
        kv_seq_len = k.shape[1]

        # split into heads and reshape for attention
        q = q.view(b, q_seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
        k = k.view(b, kv_seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)
        v = v.view(b, kv_seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2).contiguous() # (B, nh, T, hs)

        if attn_mask is not None:
            attn_mask = attn_mask.view(b, 1, -1, q_seq_len).contiguous()
        elif kv_padding_mask is not None:
            attn_mask = kv_padding_mask.view(b, 1, -1, kv_seq_len).contiguous()

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # faster overall compared to other implementation but takes more steps to learn
            attn_weight = torch.nn.functional.scaled_dot_product_attention(
                q, 
                k, 
                v, 
                attn_mask=attn_mask, 
                dropout_p=self.dropout if self.training else 0,
                is_causal=is_causal)
        else:
            if attn_mask is None:
                attn_mask = torch.zeros(1,1,1,1, device=q.device)
            # scaled dot product attention between query and key to see how much they relate to each other
            attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
            attn_weight = torch.dropout(attn_weight, self.dropout, self.training) 
            # multiplying by value gives proportions of value according to the attention weights
            # this produces the predicting vector given q,v,k
            attn_weight  = attn_weight @ v # (b, nh, seq_len, seq_len) x (b, nh, seq_len, hs) -> (b, nh, seq_len, hs)
        attn_weight = attn_weight.transpose(1, 2).reshape(b, q_seq_len, n_embd)# re-assemble all head outputs side by side #

        # output projection
        out = self.out_proj(attn_weight)
        return out


class MLP(nn.Module):

    def __init__(self, embd_size=256, fc_size=1024, bias=True):
        super().__init__()
        self.fc = nn.Linear(embd_size, fc_size, bias=bias)
        self.out_proj  = nn.Linear(fc_size, embd_size, bias=bias)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.out_proj(x)
        return x

    
class TransformerEncoderBlock(nn.Module):

    def __init__(self, 
                 embd_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 dropout_rate=0.2,
                 bias=True,
                 re_zero=True):
        super().__init__()

        self.self_attention = DotProductAttention(embd_size, num_heads, bias, 0.)
        self.feed_forward = MLP(embd_size, fc_size, bias)
        self.register_parameter('feed_forward_alpha', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('self_attention_alpha', nn.Parameter(torch.tensor(0.)))

        self.re_zero = re_zero
        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norm_1 = LayerNorm(embd_size, bias=bias)
            self.layer_norm_2 = LayerNorm(embd_size, bias=bias)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src, src_mask=None, is_causal=False):
        residual = self.layer_norm_1(src) if self.layer_norm else src
        residual = self.self_attention(
            residual,
            attn_mask=src_mask, 
            is_causal=is_causal
        )
        if self.re_zero:
            residual = residual * self.self_attention_alpha
        src = src + residual
        residual = self.layer_norm_2(src) if self.layer_norm else src
        residual = self.feed_forward(residual)
        if self.re_zero:
            residual = residual * self.feed_forward_alpha
        residual = self.dropout(residual)
        return src + residual
        

class TransformerEncoder(nn.Module):

    def __init__(self, 
                 embd_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 bias=True,
                 re_zero=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                 embd_size,
                 fc_size,
                 num_heads,
                 layer_norm,
                 dropout_rate,
                 bias,
                 re_zero) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(embd_size, bias=bias) if layer_norm else None
     

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        if is_causal:
            if mask is not None:
                # use given mask rather than a causal mask
                is_causal = False

        if src_key_padding_mask is not None:
            if mask is None:
                mask = src_key_padding_mask.unsqueeze(1)
            else:
                mask += src_key_padding_mask.unsqueeze(1) # combine masks

        if mask is not None:
            mask = convert_to_additive_mask(mask)
        
        for layer in self.layers:
            src = layer(src,
                        src_mask=mask,
                        is_causal=is_causal)
        
        if self.layer_norm:
            src = self.layer_norm(src)
        return src
    
class TransformerDecoderBlock(nn.Module):

    def __init__(self, 
                 embd_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 dropout_rate=0.2,
                 bias=True, 
                 re_zero=True,
                 take_context_embedding=False):
        super().__init__()

        self.layer_norm = layer_norm
        self.take_context_embedding = take_context_embedding
        self.re_zero = re_zero

        self.masked_self_attn = DotProductAttention(embd_size, num_heads, bias, 0.)

        self.feed_forward = MLP(embd_size, fc_size, bias)
        if layer_norm:
            self.layer_norm_1 = LayerNorm(embd_size, bias=bias)
            self.layer_norm_3 = LayerNorm(embd_size, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)
  
        # Check if the decoder will take input from an encoder
        if take_context_embedding:
            self.cross_attention = DotProductAttention(embd_size, num_heads, bias, 0.)
            if layer_norm:
              self.layer_norm_2 = LayerNorm(embd_size, bias=bias)
            self.register_parameter('cross_attention_alpha', nn.Parameter(torch.tensor(0.)))

        self.register_parameter('feed_forward_alpha', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('self_attention_alpha', nn.Parameter(torch.tensor(0.)))
        
    def forward(self, 
                tgt,
                memory=None,
                attn_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                cache=None,
                is_causal=True):

        self_attn_mask = None
        if attn_mask is not None:
            is_causal = False
            if tgt_key_padding_mask is not None:
                self_attn_mask = attn_mask + tgt_key_padding_mask.unsqueeze(1)
                self_attn_mask = convert_to_additive_mask(self_attn_mask)
        
        res = self.layer_norm_1(tgt) if self.layer_norm else tgt
        res = self.masked_self_attn(res,
                                    attn_mask=self_attn_mask, 
                                    kv_padding_mask=None,
                                    cache=cache, 
                                    is_causal=is_causal)
    
        if self.re_zero:
            res = res * self.self_attention_alpha
        tgt = tgt + res
        if memory is not None and self.take_context_embedding:
            res = self.layer_norm_2(tgt) if self.layer_norm else tgt
            # Cross attention with the output of the encoder
            res = self.cross_attention(res,
                                        kv=memory,
                                        kv_padding_mask=memory_key_padding_mask,
                                        is_causal=False)
            if self.re_zero:
                res = res * self.cross_attention_alpha
            tgt = tgt + res

        res = self.layer_norm_3(tgt) if self.layer_norm else tgt
        res = self.feed_forward(res)
        if self.re_zero:
            res = res * self.feed_forward_alpha
        res = self.dropout(res)
        return tgt + res
    
class TransformerDecoder(nn.Module):

    def __init__(self, 
                 embd_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 bias=True,
                 re_zero=True,
                 take_context_embedding=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embd_size,
                 fc_size,
                 num_heads,
                 layer_norm,
                 dropout_rate,
                 bias, 
                 re_zero,
                 take_context_embedding=take_context_embedding) for _ in range(num_layers)])
        self.embd_size = embd_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        
        self.layer_norm = LayerNorm(embd_size, bias=bias) if layer_norm else None

    def forward(self, 
                tgt, 
                tgt_key_padding_mask=None,
                memory=None,
                memory_key_padding_mask=None, 
                attn_mask=None,
                cache=None,
                is_causal=True
                ):
        
        if is_causal and attn_mask is not None:
            # use given attn_mask rather than a causal mask
            is_causal = False

        if cache is not None: # dont use causal mask for inference
            is_causal = False

        # if causal and key padding mask given -> combine masks
        # if causal and key padding mask not given -> attn_mask = None
        # if not causal and attn_mask given -> make attn_mask additive

        if is_causal and attn_mask is None:
            attn_mask = create_causal_mask(tgt.shape[0], tgt.shape[1], tgt.device)

        for i, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            tgt = layer(tgt, 
                        memory=memory, 
                        attn_mask=attn_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask, 
                        cache=layer_cache,
                        is_causal=is_causal) 
         
        if self.layer_norm:
            tgt = self.layer_norm(tgt)
        return tgt
    
    def create_init_cache(self, batch_size):
        """ Creates empty cache dictionary for use in fast decoding. """
        # Build cache
        k = torch.zeros([batch_size, 0, self.embd_size])
        v = torch.zeros([batch_size, 0, self.embd_size])
        cache = [{'k': k, 'v': v} for _ in range(self.num_layers)]
        return cache
