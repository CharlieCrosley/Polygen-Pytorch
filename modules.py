<<<<<<< HEAD
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from shared.data_utils import dequantize_verts, quantize_verts
from shared.math_utils import top_k_logits, top_p_logits
        

def init_weights_kaiming_uniform(m, mode='fan_in', nonlinearity='relu'):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity)

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
               class_conditional=False,
               num_classes=55,
               max_num_input_verts=2500,
               use_discrete_embeddings=True,
               device='cpu'):
        """Initializes VertexModel.

        Args:
        decoder_config: Dictionary with TransformerDecoder config
        quantization_bits: Number of quantization used in mesh preprocessing.
        class_conditional: If True, then condition on learned class embeddings.
        num_classes: Number of classes to condition on.
        max_num_input_verts: Maximum number of vertices. Used for learned position
            embeddings.
        use_discrete_embeddings: If True, use discrete rather than continuous
            vertex embeddings.
        """
        super(VertexModel, self).__init__()
        self.embedding_dim = decoder_config['embd_size']
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.quantization_bits = quantization_bits
        self.use_discrete_embeddings = use_discrete_embeddings
        self.device = device

        # Embedding initialization
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
        
        self.apply(init_weights_kaiming_uniform)
        self.to(device)

    def prepare_context(self, context):
        """Prepare class label context."""
        if self.class_conditional:
            global_context_embedding = self.label_embd(context['class_label'])
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
        # Step zero embeddings
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
            # append 1 to start of mask to account for step zero embedding
            vertices_mask = F.pad(vertices_mask, (1, 0), value=1.)

        if cache is not None:
            decoder_inputs = decoder_inputs[:, -1:]

        # pass through decoder
        outputs = self.decoder(
            decoder_inputs, 
            query_padding_mask=vertices_mask,
            cache=cache,
            context_embeddings=sequential_context_embeddings)

        # Get logits and optionally process for sampling
        logits = self.project_to_logits(outputs)
        logits /= temperature
        logits = top_k_logits(logits, top_k)
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
        
        self.apply(init_weights_kaiming_uniform)
        self.to(device)
    
    def prepare_context(self, context):
        """Prepare class label context."""
        if self.class_conditional:
            global_context_embedding = self.label_embd(context['class_label'])
        else:
            global_context_embedding = None
        
        vertex_embeddings = self.embed_vertices(
            context['vertices'], context['vertices_mask'])
       
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
            verts_quantized = quantize_verts(vertices, self.quantization_bits)
            vertex_embeddings += self.vertex_embd_1(verts_quantized[..., 0])
            vertex_embeddings += self.vertex_embd_2(verts_quantized[..., 1])
            vertex_embeddings += self.vertex_embd_3(verts_quantized[..., 2])
        else:
            n_pad = self.max_num_input_verts - vertices.shape[1]
            pad_vertices = F.pad(vertices, (0, n_pad))
            vertex_embeddings = self.vertex_embd(pad_vertices.unsqueeze(1))
        vertex_embeddings *= vertices_mask.unsqueeze(-1)
        
        # Pad vertex embeddings with learned embeddings for stopping and new face
        # tokens
        stopping_embeddings = torch.tile(self.stopping_embeddings, [vertices.shape[0], 1, 1])
        vertex_embeddings = torch.cat(
            [stopping_embeddings, vertex_embeddings], dim=1)
        
        # Pass through Transformer encoder
        vertices_mask = F.pad(vertices_mask, (2,0), value=1.) # pad for stopping and new face tokens
        
        vertex_embeddings = self.encoder(vertex_embeddings, query_padding_mask=vertices_mask)
        return vertex_embeddings
    
    def embed_inputs(self, faces_long, vertex_embeddings,
                    global_context_embedding=None):
        """Embeds face sequences and adds within and between face positions."""

        # faces_long is the indices of the vertices in the face
        # gather those vertex embeddings according to the indices
        face_embeddings = torch.vmap(
            torch.index_select,
            in_dims=(0, None, 0),
        )(vertex_embeddings, 0, faces_long) # essentially gather on batched tensor
  
        # Position of vertex in face
        pos_embeddings = self.pos_embd(torch.arange(faces_long.shape[1], device=self.device)) 

        # Step zero embeddings
        batch_size = face_embeddings.shape[0]
        if global_context_embedding is None:
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
        if faces_mask is not None:
            # append 1 to start of mask to account for the input embedding
            faces_mask = F.pad(faces_mask, (1, 0), value=1.)

        decoder_outputs = self.decoder(
            decoder_inputs,
            query_padding_mask=faces_mask,
            kv_padding_mask=vertices_mask,
            cache=cache,
            context_embeddings=sequential_context_embeddings)

        # Get pointers by projecting transformer outputs to pointer vectors.
        pred_pointers = self.project_to_pointers(decoder_outputs)
        
        # pointer vector is compared to the input embeddings to get vertex scores
        logits = torch.matmul(pred_pointers, vertex_embeddings.transpose(1,2))
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
  
        vertex_embeddings, global_context, seq_context = self.prepare_context(batch)
        
        faces_mask = torch.ne(batch['faces'][:, :-1], 0).float()
        pred_dist = self.create_vertex_indices_dist(
            vertex_embeddings,
            batch['faces'][:, :-1],
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
                query, 
                context_embedding=None, 
                query_padding_mask = None, 
                kv_padding_mask=None, 
                cache=None):
        # if context_embedding is None then self attention is being used
        # otherwise the encoder output is being used for cross attention
        b, query_seq_len, n_embd = query.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.q_proj(query)
        
        if context_embedding is None or cache is None:
            # use query for both key and value if context_embedding is not given (self attention)
            kv = context_embedding if context_embedding is not None else query
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
        
        context_seq_len = k.shape[1]
        # split into heads and reshape for attention
        q = q.view(b, query_seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(b, context_seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(b, context_seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if query_padding_mask is not None:
            attn_mask = query_padding_mask.view(b, 1, -1, query_seq_len)
        elif kv_padding_mask is not None:
            attn_mask = kv_padding_mask.view(b, 1, -1, context_seq_len)
        else:
            attn_mask = torch.zeros(1,1,1,1, device=query.device) # unmasked

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # faster overall compared to other implementation but takes more steps to learn
            attn_weight = torch.nn.functional.scaled_dot_product_attention(
                q, 
                k, 
                v, 
                attn_mask=attn_mask, 
                dropout_p=self.dropout if self.training else 0)
        else:
            # scaled dot product attention between query and key to see how much they relate to each other
            attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
            attn_weight = torch.dropout(attn_weight, self.dropout, self.training) 
            # multiplying by value gives proportions of value according to the attention weights
            # this produces the predicting vector given q,v,k
            attn_weight  = attn_weight @ v # (b, nh, seq_len, seq_len) x (b, nh, seq_len, hs) -> (b, nh, seq_len, hs)
        attn_weight = attn_weight.transpose(1, 2).reshape(b, query_seq_len, n_embd)# re-assemble all head outputs side by side #
      
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

    def forward(self, x, query_padding_mask=None):
        residual = self.layer_norm_1(x) if self.layer_norm else x
        residual = self.self_attention(residual, 
                                query_padding_mask=query_padding_mask)
        if self.re_zero:
            residual *= self.self_attention_alpha
        x = x + residual
        residual = self.layer_norm_2(x) if self.layer_norm else x
        residual = self.feed_forward(residual)
        if self.re_zero:
            residual *= self.feed_forward_alpha
        residual = self.dropout(residual)
        return x + residual
        

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
     

    def forward(self, x, query_padding_mask=None):
        if query_padding_mask is not None:
            query_padding_mask = (1-query_padding_mask) * -1e9
        for layer in self.layers:
            x = layer(x, query_padding_mask=query_padding_mask)
        
        if self.layer_norm:
            x = self.layer_norm(x)
        return x
    
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
                x, 
                context_embedding=None,
                self_attn_mask=None, 
                cross_attn_mask=None,
                cache=None):
        res = self.layer_norm_1(x) if self.layer_norm else x
        res = self.masked_self_attn(res, query_padding_mask=self_attn_mask, cache=cache)
        if self.re_zero:
            res *= self.self_attention_alpha
        x = x + res
        if context_embedding is not None and self.take_context_embedding:
            res = self.layer_norm_2(x) if self.layer_norm else x
            # Cross attention with the output of the encoder
            res = self.cross_attention(res, 
                                         context_embedding=context_embedding,
                                         kv_padding_mask=cross_attn_mask)
            if self.re_zero:
                res *= self.cross_attention_alpha
            x = x + res

        res = self.layer_norm_3(x) if self.layer_norm else x
        res = self.feed_forward(res)
        if self.re_zero:
            res *= self.feed_forward_alpha
        res = self.dropout(res)
        return x + res

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
                x, 
                query_padding_mask=None,
                kv_padding_mask=None, 
                cache=None,
                context_embeddings=None):
     
        query_seq_len = x.shape[1]
        # look ahead mask for decoder self attention
        if query_padding_mask is not None:
            # combine the query padding mask with the look ahead mask
            if cache is None:
                self_attn_mask = torch.logical_and(query_padding_mask.unsqueeze(1), 
                                                   torch.ones(query_seq_len, 
                                                              query_seq_len, 
                                                              device=x.device).tril(diagonal=0)).float()
            else:
                self_attn_mask = query_padding_mask
        else:
            if cache is None:
                self_attn_mask = torch.ones(query_seq_len, 
                                            query_seq_len, 
                                            device=x.device).tril(diagonal=0).float()
            else:
                self_attn_mask = None
        
        if self_attn_mask is not None:
            # negate one make ones mean padding
            # large negative number makes the probability after softmax zero to mask
            self_attn_mask = (1-self_attn_mask) * -1e9 
        if kv_padding_mask is not None:
            kv_padding_mask = (1-kv_padding_mask) * -1e9
   
        for i, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            x = layer(x, 
                      context_embedding=context_embeddings, 
                      self_attn_mask=self_attn_mask,
                      cross_attn_mask=kv_padding_mask, 
                      cache=layer_cache) 
        if self.layer_norm:
            x = self.layer_norm(x)
        return x
    
    def create_init_cache(self, batch_size):
        """ Creates empty cache dictionary for use in fast decoding. """
        # Build cache
        k = torch.zeros([batch_size, 0, self.embd_size])
        v = torch.zeros([batch_size, 0, self.embd_size])
        cache = [{'k': k, 'v': v} for _ in range(self.num_layers)]
        return cache
=======
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from shared.data_utils import dequantize_verts, quantize_verts, create_causal_mask, convert_to_additive_mask
from shared.math_utils import top_k_logits, top_p_logits
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_varlen_func

#from memory_efficient_attention import efficient_dot_product_attention_pt
# pip install memory-efficient-attention[torch]

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
        #self.context_type = context_type
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

        # decoder_layer = torch.nn.TransformerDecoderLayer(decoder_config['embd_size'], 
        #                                                  decoder_config['num_heads'], 
        #                                                  decoder_config['fc_size'], 
        #                                                  decoder_config['dropout_rate'], 
        #                                                  batch_first=True, 
        #                                                  device=device)
        # self.decoder = torch.nn.TransformerDecoder(decoder_layer, decoder_config['num_layers'])

        # use_cached_decoder = False
        # if use_cached_decoder:
        #     self.decoder = CausalTransformerDecoder(
        #         CausalTransformerDecoderLayer(
        #             d_model=decoder_config['embd_size'], 
        #             nhead=decoder_config['num_heads'], 
        #             dim_feedforward=decoder_config['face_size'],
        #             dropout=decoder_config['dropout_rate']
        #         ),
        #         num_layers=decoder_config['num_layers'],
        #     ).to(device=device)
        # else:
        #     decoder_layer = torch.nn.TransformerEncoderLayer(decoder_config['embd_size'], 
        #                                                     decoder_config['num_heads'], 
        #                                                     decoder_config['fc_size'], 
        #                                                     decoder_config['dropout_rate'], 
        #                                                     batch_first=True, 
        #                                                     device=device)

        #     self.decoder = torch.nn.TransformerEncoder(decoder_layer, decoder_config['num_layers'])

        

        """ self_attn_mask = torch.ones(query_seq_len, 
                                            query_seq_len, 
                                            device=x.device).tril(diagonal=0).float() """
        
        #self.apply(init_weights_kaiming_uniform)

        # TODO: experiment
        #self.pos_embd.weight = self.project_to_logits.weight # https://paperswithcode.com/method/weight-tying

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
            # append 1 to start of mask to account for step zero embedding
            #vertices_mask = F.pad(vertices_mask, (1, 0), value=1.)

            vertices_mask = torch.logical_not(vertices_mask).float()
            vertices_mask = F.pad(vertices_mask, (1, 0), value=0.)
            #vertices_mask = vertices_mask.masked_fill(vertices_mask == 1, float('-inf'))

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

        # outputs = self.decoder(
        #     decoder_inputs, 
        #     query_padding_mask=vertices_mask,
        #     cache=cache,
        #     context_embeddings=sequential_context_embeddings)
        
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
            batch['vertices_flat'][:, :-1].to(self.device),  # Last element not used for preds
            vertices_mask=batch['vertices_flat_mask'][:, :-1].to(self.device),
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

        self.face_embd = torch.vmap(
            torch.index_select,
            in_dims=(0, None, 0),
        )
        # learnable stopping token embeddings
        self.stopping_embeddings = torch.nn.Parameter(torch.rand([1, 2, self.embedding_dim]))
        self.embed_zeros = torch.nn.Parameter(torch.rand([1, 1, self.embedding_dim]))

        self.decoder = TransformerDecoder(**decoder_config, bias=False)
        self.encoder = TransformerEncoder(**encoder_config, bias=False)

        # decoder_layer = torch.nn.TransformerDecoderLayer(decoder_config['embd_size'], 
        #                                                     decoder_config['num_heads'], 
        #                                                     decoder_config['fc_size'], 
        #                                                     decoder_config['dropout_rate'], 
        #                                                     batch_first=True, 
        #                                                     device=device)

        # self.decoder = torch.nn.TransformerDecoder(decoder_layer, decoder_config['num_layers'])

        # encoder_layer = torch.nn.TransformerEncoderLayer(encoder_config['embd_size'], 
        #                                                  encoder_config['num_heads'], 
        #                                                  encoder_config['fc_size'], 
        #                                                  encoder_config['dropout_rate'], 
        #                                                  batch_first=True, 
        #                                                  device=device)
        # self.encoder = torch.nn.TransformerEncoder(encoder_layer, encoder_config['num_layers'])

        # use_cached_decoder = True
        # if use_cached_decoder:
        #     self.decoder = CausalTransformerDecoder(
        #         CausalTransformerDecoderLayer(
        #             d_model=decoder_config['embd_size'], 
        #             nhead=decoder_config['num_heads'], 
        #             dim_feedforward=decoder_config['fc_size'],
        #             dropout=decoder_config['dropout_rate'],
        #             batch_first=False
        #         ),
        #         num_layers=decoder_config['num_layers'],
        #     ).to(device=device)
        # else:
        #     decoder_layer = torch.nn.TransformerDecoderLayer(decoder_config['embd_size'], 
        #                                                     decoder_config['num_heads'], 
        #                                                     decoder_config['fc_size'], 
        #                                                     decoder_config['dropout_rate'], 
        #                                                     #batch_first=True, 
        #                                                     device=device)

        #     self.decoder = torch.nn.TransformerDecoder(decoder_layer, decoder_config['num_layers'])
        
        #self.apply(init_weights_kaiming_uniform)

        # TODO: experiment
        #self.pos_embd.weight = self.project_to_pointers.weight # https://paperswithcode.com/method/weight-tying

        self.to(device)
    
    def prepare_context(self, context):
        """Prepare class label context."""
        if self.class_conditional:
            global_context_embedding = self.label_embd(context['class_label'].to(self.device))
        else:
            global_context_embedding = None
        
        """ vertex_embeddings = self.embed_vertices(
            context['vertices_dequantized'], context['vertices_mask']) """
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
            #vertex_embeddings = torch.zeros([vertices.shape[0], vertices.shape[1], self.embedding_dim], dtype=vertices.dtype, device=vertices.device) #0.
            vertex_embeddings = 0.
            if torch.is_floating_point(vertices):
                verts = quantize_verts(vertices, self.quantization_bits)
            else:
                verts = vertices
            #print("verts", verts.shape, verts[..., 0].shape)
            #print("X", self.vertex_embd_1(verts[..., 0]).shape)
            vertex_embeddings += self.vertex_embd_1(verts[..., 0])
            vertex_embeddings += self.vertex_embd_2(verts[..., 1])
            vertex_embeddings += self.vertex_embd_3(verts[..., 2])
        else:
            n_pad = self.max_num_input_verts - vertices.shape[1]
            pad_vertices = F.pad(vertices, (0, n_pad))
            vertex_embeddings = self.vertex_embd(pad_vertices.unsqueeze(1))

        vertex_embeddings = vertex_embeddings * vertices_mask.unsqueeze(-1)
        
        # Pad vertex embeddings with learned embeddings for stopping and new face
        # tokens
        stopping_embeddings = torch.tile(self.stopping_embeddings, [vertices.shape[0], 1, 1])
        vertex_embeddings = torch.cat(
            [stopping_embeddings, vertex_embeddings], dim=1)
        
        # Pass through Transformer encoder
        vertices_mask = F.pad(vertices_mask, (2,0), value=1.) # pad for stopping and new face tokens


        vertices_mask = torch.logical_not(vertices_mask).float()
        #vertices_mask = vertices_mask.masked_fill(vertices_mask == 1, float('-inf'))
        
        #vertex_embeddings = self.encoder(vertex_embeddings, query_padding_mask=vertices_mask)
        #vertex_embeddings = self.encoder(vertex_embeddings, src_key_padding_mask=vertices_mask, is_causal=False)
        
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
        """ print("vertex_embeddings", vertex_embeddings.shape)
        print("faces_long", faces_long.shape)
        face_embeddings = torch.vmap(
            torch.index_select,
            in_dims=(0, None, 0),
        )(vertex_embeddings, 0, faces_long) # essentially gather on batched tensor
        print("face_embeddings", face_embeddings.shape) """
        #face_embeddings = self.face_embd(vertex_embeddings, 0, faces_long)
        face_embeddings = batched_index_select(vertex_embeddings, 1, faces_long)
  
        # Position of vertex in face
        #print("XXXX", torch.arange(faces_long.shape[1], device=self.device).shape, faces_long.shape)
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
        """ if not self.training:
            print("vertex_embeddings", vertex_embeddings)
            print("sequential_context_embeddings", sequential_context_embeddings) """
        decoder_inputs = self.embed_inputs(
            faces_long, vertex_embeddings, global_context_embedding)
        
        # Pass through Transformer decoder
        if cache is not None:
            decoder_inputs = decoder_inputs[:, -1:]

        if vertices_mask is not None:
            # append 1 to start of mask to account for the input embedding
            vertices_mask = F.pad(vertices_mask, (2, 0), value=1.)
            
            encoder_inp_mask = torch.logical_not(vertices_mask).float()
            #encoder_inp_mask = F.pad(encoder_inp_mask, (2, 0), value=0.)
            #encoder_inp_mask = encoder_inp_mask.masked_fill(encoder_inp_mask == 1, float('-inf'))
        if faces_mask is not None:
            # append 1 to start of mask to account for the input embedding
            #faces_mask = F.pad(faces_mask, (1, 0), value=1.)

            faces_mask = torch.logical_not(faces_mask).float()
            faces_mask = F.pad(faces_mask, (1, 0), value=0.)
            #faces_mask = faces_mask.masked_fill(faces_mask == 1, float('-inf'))

        #print("XXX", decoder_inputs.device, faces_mask.device, vertices_mask.device, sequential_context_embeddings)
        # decoder_outputs = self.decoder(
        #     decoder_inputs,
        #     query_padding_mask=faces_mask,
        #     kv_padding_mask=vertices_mask,
        #     cache=cache,
        #     context_embeddings=sequential_context_embeddings)
            
        is_causal = True if cache is None else False
  

        # decoder_outputs, new_cache = self.decoder(
        #     decoder_inputs.permute(1,0,2).contiguous(), #.permute(1,0,2).contiguous()
        #     tgt_key_padding_mask=faces_mask,
        #     memory=sequential_context_embeddings.permute(1,0,2).contiguous(),
        #     memory_key_padding_mask=encoder_inp_mask,
        #     cache=cache
        # )  # output_len, bsz, hdim

        # mine
        decoder_outputs = self.decoder(
            decoder_inputs,
            tgt_key_padding_mask=faces_mask, 
            memory=sequential_context_embeddings, 
            memory_key_padding_mask=encoder_inp_mask,
            cache=cache,
            is_causal=is_causal
        )
        # if is_causal:
        #     mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_inputs.shape[1], device=decoder_inputs.device)
        # else:
        #     mask = None
        # decoder_outputs = self.decoder(decoder_inputs,
        #                                tgt_mask=mask,
        #                        tgt_key_padding_mask=faces_mask, 
        #                        memory=sequential_context_embeddings, 
        #                        memory_key_padding_mask=encoder_inp_mask,
        #                        tgt_is_causal=is_causal)

        # Get pointers by projecting transformer outputs to pointer vectors.
        pred_pointers = self.project_to_pointers(decoder_outputs)#.transpose(0,1)
        
        #print(pred_pointers.shape, vertex_embeddings.shape)
        # pointer vector is compared to the input embeddings to get vertex scores
        logits = torch.matmul(pred_pointers, vertex_embeddings.transpose(1,2)).float()
        #logits = torch.matmul(pred_pointers, vertex_embeddings).float()
        #logits = logits / math.sqrt(self.embedding_dim)
        logits /= math.sqrt(self.embedding_dim)

        f_verts_mask = vertices_mask.unsqueeze(1)
        logits *= f_verts_mask
        logits -= (1. - f_verts_mask) * 1e9
        logits /= temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)

        # f_verts_mask = vertices_mask.unsqueeze(1)
        # logits *= f_verts_mask
        # logits -= (1. - f_verts_mask) * 1e9
        # logits /= temperature
        # logits = top_k_logits(logits, top_k)
        # logits = top_p_logits(logits, top_p)
        
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
    

        #batch['vertices_mask'] = torch.eq(batch['vertices_mask'], 0).float() # invert mask
        batch['vertices_mask'] = batch['vertices_mask'].to(self.device)
        vertex_embeddings, global_context, seq_context = self.prepare_context(batch)
        
        faces_mask = batch['faces'][:, :-1].float().to(self.device)
        #faces_mask = torch.ne(batch['faces'][:, :-1], 0).float().to(self.device)
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
        #context['vertices_mask'] = torch.eq(context['vertices_mask'], 0).float()
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
            #cat_dist = self.create_vertex_indices_dist(
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
        #torch.set_printoptions(profile="full")
        #print("samples", samples)
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

        if re_zero:
            self.re_zero_alpha = nn.Parameter(torch.zeros(1, device=device))

        if downsample:
            # stride 2 to downsample
            out_channels = in_channels if hidden_size is None else hidden_size
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, bias=bias, device=device)
        else:
            self.skip = None

        stride = 2 if downsample else 1 #, padding='same'
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,   bias=bias, device=device)
        self.conv2 = nn.Conv2d(in_channels, hidden_size, kernel_size=kernel_size,
                               stride=stride, bias=bias, device=device)

    def forward(self, inputs):
        if self.skip is not None:
            skip = self.skip(inputs)
        else:
            skip = inputs

        res = F.relu(inputs)
        res = self.conv1(res)

        # 1 skip: torch.Size([4, 64, 30, 30])  res: torch.Size([4, 64, 62, 62])
        # 2 skip: torch.Size([4, 64, 30, 30])  res: torch.Size([4, 128, 30, 30])

        res = F.relu(res)
        res = self.dropout(res)

        res = self.conv2(res)
        pad = abs(skip.shape[-1] - res.shape[-1])
        res = nn.functional.pad(res, (0, pad, 0, pad))

        if self.re_zero:
            res *= self.re_zero_alpha
        #pad = abs(skip.shape[1] - res.shape[1])
        #skip = nn.functional.pad(skip, (0, 0, 0, 0, 0, pad))
        #print("3", "skip:", skip.shape, " res:", res.shape)
        return skip + res

    
class ResNet(nn.Module):
    """ ResNet architecture for 2D image """

    def __init__(self, hidden_sizes=(64, 128, 256), dropout_rate=0.1, num_blocks=(1,2,2), re_zero=True, kernel_size=(3,3), bias=True, device='cpu'):
        super().__init__()

        self.re_zero = re_zero
        self.num_blocks = num_blocks
        self.device = device

        input_channels = 3 # 1
        self.conv1 = nn.Conv2d(input_channels, hidden_sizes[0], kernel_size=(7,7), stride=2, bias=bias, device=device)

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
        #rint("inputs", inputs.shape, inputs.device)
        # Initial projection with large kernel as in original resnet architecture
        #print("0 resnet", inputs.shape)
        x = self.conv1(inputs.to(self.device))
        #print("1 resnet", x.shape)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        #print("2 resnet", x.shape)

        # Count how many conv layers have been passed
        # So we can offset the conv layers to the correct block
        i = 3
        for layer in self.conv_layers:
            x = layer(x)
            #print(f"{i} resnet", x.shape)
            i += 1

        return x


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
                 device='cpu'):
        super().__init__(
            decoder_config=decoder_config,
            max_num_input_verts=max_num_input_verts,
            num_classes=num_classes,
            context_type='image',
            quantization_bits=quantization_bits,
            use_discrete_embeddings=use_discrete_embeddings,
            device=device
        )

        #self.res_net = ResNet(num_dims=2, **res_net_config)
        self.res_net = ResNet(**res_net_config, device=device).to(device)

        self.image_embd = nn.Linear(2, self.embedding_dim, bias=True, device=device)


    def prepare_context(self, context):
        # Pass images through encoder
        # -0.5 to center around zero
        # it may be better to subtract mean instead
        image_embeddings = self.res_net(
            context['image'] - 0.5)
        
        #print("1", image_embeddings.shape)

        # Add 2D coordinate grid embedding
        processed_image_resolution = image_embeddings.shape[-1]
        #x = tf.linspace(-1., 1., processed_image_resolution)
        x = torch.linspace(-1., 1., processed_image_resolution, device=self.device)
        #image_coords = tf.stack(tf.meshgrid(x, x), axis=-1)
        image_coords = torch.stack(torch.meshgrid(x, x, indexing='xy'), axis=-1)
        
        image_coord_embeddings = self.image_embd(image_coords)
        #print("2", image_coord_embeddings.shape)
        """ image_coord_embeddings = tf.layers.dense(
            image_coords,
            self.embedding_dim,
            use_bias=True,
            name='image_coord_embeddings') """
        image_embeddings += image_coord_embeddings.unsqueeze(0).permute(0,3,2,1)


        # Reshape spatial grid to sequence
        batch_size = image_embeddings.shape[0]
        sequential_context_embedding = image_embeddings.view(batch_size, -1, self.embedding_dim)
        
        #print("3", sequential_context_embedding.shape)
        return None, sequential_context_embedding


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

        #qkv = torch.stack([q,k,v], dim=2)


        if attn_mask is not None:
            attn_mask = attn_mask.view(b, 1, -1, q_seq_len)
        elif kv_padding_mask is not None:
            attn_mask = kv_padding_mask.view(b, 1, -1, kv_seq_len)
            
        # if kv_padding_mask is not None:
        #     kv_padding_mask = kv_padding_mask.view(b, 1, -1, kv_seq_len)

        #     if attn_mask is not None:
        #         attn_mask += kv_padding_mask

            #attn_mask = attn_mask==0
            #print("PPP", attn_mask)
            #torch.set_printoptions(profile="full")
            #print("attn_mask_2", attn_mask[0, 0,:2], attn_mask.shape)    

        # if query_padding_mask is not None:
        #     attn_mask = query_padding_mask.view(b, 1, -1, query_seq_len)
        #     #attn_mask = query_padding_mask.reshape(b, 1, -1, query_seq_len)#.contiguous()
        # elif kv_padding_mask is not None:
        #     attn_mask = kv_padding_mask.view(b, 1, -1, context_seq_len)
        #     #attn_mask = kv_padding_mask.reshape(b, 1, -1, context_seq_len)#.contiguous()
        # else:
        #     attn_mask = None #torch.zeros(1,1,1,1, device=query.device).contiguous() # unmasked

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
        #self.self_attention = nn.MultiheadAttention(embd_size, num_heads, dropout_rate, bias=bias)
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
            #residual *= self.self_attention_alpha
            residual = residual * self.self_attention_alpha
        src = src + residual
        residual = self.layer_norm_2(src) if self.layer_norm else src
        residual = self.feed_forward(residual)
        if self.re_zero:
            #residual *= self.feed_forward_alpha
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
        # if src_key_padding_mask is not None:
        #     src_key_padding_mask = convert_to_additive_mask(src_key_padding_mask)

        if is_causal:
            if mask is not None:
                # use given mask rather than a causal mask
                is_causal = False
            #else:
                #mask = create_causal_mask(src.shape[0], src.shape[1], src.device)

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
        #self.masked_self_attn = nn.MultiheadAttention(embd_size, num_heads, dropout_rate, bias=bias)

        self.feed_forward = MLP(embd_size, fc_size, bias)
        if layer_norm:
            self.layer_norm_1 = LayerNorm(embd_size, bias=bias)
            self.layer_norm_3 = LayerNorm(embd_size, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)
  
        # Check if the decoder will take input from an encoder
        if take_context_embedding:
            self.cross_attention = DotProductAttention(embd_size, num_heads, bias, 0.)
            #self.cross_attention = nn.MultiheadAttention(embd_size, num_heads, dropout_rate, bias=bias)
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
        
            
            # if tgt_key_padding_mask is None:
            #     # we use the is_causal hint indicator to SDPA, which requires attn_mask=None.
            #     attn_mask = None
            # else:
            #     #causal_mask = create_causal_mask(tgt.shape[0], tgt.shape[1], tgt.device)
            #     #attn_mask = create_causal_mask(tgt.shape[0], tgt.shape[1], tgt.device)
            #     # combine causal and key padding masks

            #     causal_mask += tgt_key_padding_mask.unsqueeze(1)
            #     attn_mask = convert_to_additive_mask(causal_mask)

            #     # We have the attn_mask, and use that to merge kpm into it.
            #     # Turn off use of is_causal hint, as the merged mask is no
            #     # longer causal.
            #     is_causal = False
        # elif attn_mask is not None:
        #     attn_mask = convert_to_additive_mask(attn_mask)

        self_attn_mask = None
        if attn_mask is not None:
            is_causal = False
            if tgt_key_padding_mask is not None:
                self_attn_mask = attn_mask + tgt_key_padding_mask.unsqueeze(1)
                self_attn_mask = convert_to_additive_mask(self_attn_mask)
                #torch.set_printoptions(profile="full")
                #print(self_attn_mask)

            #cross_attn_mask = None
            # if memory_key_padding_mask is not None:
            #     print(attn_mask.shape, memory_key_padding_mask.shape, memory_key_padding_mask.unsqueeze(1).shape)
            #     #cross_attn_mask = attn_mask + memory_key_padding_mask.unsqueeze(1)
            #     cross_attn_mask = memory_key_padding_mask.unsqueeze(1)
            #     cross_attn_mask = convert_to_additive_mask(cross_attn_mask)
        
        
   
        # if attn_mask is not None:
        #     is_causal = False
        
        res = self.layer_norm_1(tgt) if self.layer_norm else tgt
        res = self.masked_self_attn(res,
                                    attn_mask=self_attn_mask, 
                                    kv_padding_mask=None, #tgt_key_padding_mask
                                    cache=cache, 
                                    is_causal=is_causal)
        #print("1", torch.isnan(res).any())
        #res = self.masked_self_attn(res, attn_mask=self_attn_mask)
        if self.re_zero:
            #res *= self.self_attention_alpha
            res = res * self.self_attention_alpha
        tgt = tgt + res
        if memory is not None and self.take_context_embedding:
            res = self.layer_norm_2(tgt) if self.layer_norm else tgt
            # Cross attention with the output of the encoder
            res = self.cross_attention(res, 
                                       #attn_mask=cross_attn_mask,
                                        kv=memory,
                                        kv_padding_mask=memory_key_padding_mask,
                                        is_causal=False)
            if self.re_zero:
                #res *= self.cross_attention_alpha
                res = res * self.cross_attention_alpha
            tgt = tgt + res

        res = self.layer_norm_3(tgt) if self.layer_norm else tgt
        res = self.feed_forward(res)
        if self.re_zero:
            #res *= self.feed_forward_alpha
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
            #attn_mask = convert_to_additive_mask(causal_mask)
            
            # if tgt_key_padding_mask is None:
            #     # we use the is_causal hint indicator to SDPA, which requires attn_mask=None.
            #     attn_mask = None
            # else:
            #     #causal_mask = create_causal_mask(tgt.shape[0], tgt.shape[1], tgt.device)
            #     #attn_mask = create_causal_mask(tgt.shape[0], tgt.shape[1], tgt.device)
            #     # combine causal and key padding masks

            #     causal_mask += tgt_key_padding_mask.unsqueeze(1)
            #     attn_mask = convert_to_additive_mask(causal_mask)

            #     # We have the attn_mask, and use that to merge kpm into it.
            #     # Turn off use of is_causal hint, as the merged mask is no
            #     # longer causal.
            #     is_causal = False
        # elif attn_mask is not None:
        #     attn_mask = convert_to_additive_mask(attn_mask)

        # if memory_key_padding_mask is not None:
        #     memory_key_padding_mask = convert_to_additive_mask(memory_key_padding_mask)
   
        # if attn_mask is not None:
        #     is_causal = False

        for i, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            tgt = layer(tgt, 
                        memory=memory, 
                        attn_mask=attn_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask, 
                        cache=layer_cache,
                        is_causal=is_causal) 
            
            #print(i, torch.isnan(tgt).any())
            # if torch.isnan(tgt).any():
            #     print("TT", attn_mask)
            #     print("UU", memory_key_padding_mask)
         
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

# class TransformerDecoder(nn.Module):

#     def __init__(self, 
#                  embd_size=256,
#                  fc_size=1024,
#                  num_heads=4,
#                  layer_norm=True,
#                  num_layers=8,
#                  dropout_rate=0.2,
#                  bias=True,
#                  re_zero=True,
#                  take_context_embedding=True):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             TransformerDecoderBlock(embd_size,
#                  fc_size,
#                  num_heads,
#                  layer_norm,
#                  dropout_rate,
#                  bias, 
#                  re_zero,
#                  take_context_embedding=take_context_embedding) for _ in range(num_layers)])
#         self.embd_size = embd_size
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.layer_norm = layer_norm
        
#         self.layer_norm = LayerNorm(embd_size, bias=bias) if layer_norm else None

#     def forward(self, 
#                 tgt, 
#                 tgt_key_padding_mask=None,
#                 memory=None,
#                 memory_key_padding_mask=None, 
#                 cache=None
#                 ):
     
#         query_seq_len = tgt.shape[1]
#         # look ahead mask for decoder self attention
#         if tgt_key_padding_mask is not None:
#             # combine the query padding mask with the look ahead mask
#             if cache is None:
#                 self_attn_mask = torch.logical_and(tgt_key_padding_mask.unsqueeze(1).repeat(1, query_seq_len, 1), 
#                                                    torch.ones(query_seq_len, 
#                                                               query_seq_len, 
#                                                               device=tgt.device).tril(diagonal=0)).float()
#             else:
#                 self_attn_mask = tgt_key_padding_mask
#         else:
#             if cache is None:
#                 self_attn_mask = torch.ones(query_seq_len, 
#                                             query_seq_len, 
#                                             device=tgt.device).tril(diagonal=0).float().repeat(tgt.shape[0], 1)
#             else:
#                 self_attn_mask = None
        
#         if self_attn_mask is not None:
#             # negate one make ones mean padding
#             # large negative number makes the probability after softmax zero to mask
#             self_attn_mask = (1-self_attn_mask) * -1e9 
#         if memory_key_padding_mask is not None:
#             memory_key_padding_mask = (1-memory_key_padding_mask) * -1e9
   
#         for i, layer in enumerate(self.layers):
#             layer_cache = None if cache is None else cache[i]
#             tgt = layer(tgt, 
#                       context_embedding=memory, 
#                       self_attn_mask=self_attn_mask,
#                       cross_attn_mask=memory_key_padding_mask, 
#                       cache=layer_cache) 
#             """ if torch.isnan(x).any():
#                 print(f"{i} layer Nan found") """
#         if self.layer_norm:
#             tgt = self.layer_norm(tgt)
#         return tgt
    
#     def create_init_cache(self, batch_size):
#         """ Creates empty cache dictionary for use in fast decoding. """
#         # Build cache
#         k = torch.zeros([batch_size, 0, self.embd_size])
#         v = torch.zeros([batch_size, 0, self.embd_size])
#         cache = [{'k': k, 'v': v} for _ in range(self.num_layers)]
#         return cache



from typing import Optional
from torch import Tensor


class CausalTransformerDecoder(nn.TransformerDecoder):
    """Implementation of a transformer decoder based on torch implementation but
    more efficient. The difference is that it doesn't need to recompute the
    embeddings of all the past decoded tokens but instead uses a cache to
    store them. This makes use of the fact that the attention of a decoder is
    causal, so new predicted tokens don't affect the old tokens' embedding bc
    the corresponding attention cells are masked.
    The complexity goes from seq_len^3 to seq_len^2.

    This only happens in eval mode.
    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            return output, None

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """
        
        if self.training:
            return super().forward(
                tgt,
                memory,
                tgt_mask=generate_square_subsequent_mask(tgt.size(0), tgt.device),
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.

        tgt_last_tok = tgt[-1:, :, :]

        # self attention part
        tmp_tgt = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        if memory is not None:
            tmp_tgt = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # final feed-forward network
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok


def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask
>>>>>>> master
