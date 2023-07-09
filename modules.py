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
