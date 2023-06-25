from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import inspect
from shared.data_utils import dequantize_verts, quantize_verts
from shared.math_utils import top_k_logits, top_p_logits

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight) 

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
               use_discrete_embeddings=True):
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

        # Embedding initialization
        self.label_embd = nn.Embedding(self.num_classes, self.embedding_dim)
        self.label_embd.apply(init_weights)

        self.coord_embd = nn.Embedding(3, self.embedding_dim) # 3 in to represent (x, y, z) 
        self.coord_embd.apply(init_weights)

        self.pos_embd = nn.Embedding(self.max_num_input_verts, self.embedding_dim)
        self.pos_embd.apply(init_weights)

        if self.use_discrete_embeddings:
            self.vertex_embd = nn.Embedding(2**self.quantization_bits + 1, self.embedding_dim)
        else:
            self.vertex_embd = nn.Linear(self.max_num_input_verts, self.embedding_dim, bias=True)
        self.vertex_embd.apply(init_weights)

        self.decoder = TransformerDecoder(**decoder_config)
        self.project_to_logits = nn.Linear(self.embedding_dim, 2**self.quantization_bits + 1, bias=True)
        #self.project_to_logits.apply(init_weights_zeros)

        #self.decoder = TransformerDecoder(**decoder_config)


    def prepare_context(self, context):
        """Prepare class label context."""
        if self.class_conditional:
            #global_context_embedding = self._embed_class_label(context['class_label'])
            global_context_embedding = self.label_embd(context['class_label'])
        else:
            global_context_embedding = None
        return global_context_embedding, None
    
    def _embed_inputs(self, vertices, global_context_embedding=None):
        """Embeds flat vertices and adds position and coordinate information."""
        # Dequantize inputs and get shapes
        input_shape = vertices.shape
        
        batch_size, seq_length = input_shape[0], input_shape[1]
        seq_length = max(seq_length, 1)

        # Coord indicators (x, y, z)
        coord_embeddings = self.coord_embd(torch.range(0, seq_length-1, dtype=torch.int) % 3)
 
        # Position embeddings
        pos_embeddings = self.pos_embd(torch.floor_divide(torch.range(0, seq_length-1, dtype=torch.int), 3))

        # Discrete vertex value embeddings
        if self.use_discrete_embeddings:
            vert_embeddings = self.vertex_embd(vertices)
        # Continuous vertex value embeddings
        else:
            n_pad = self.max_num_input_verts - input_shape[1]
            pad_vertices = F.pad(vertices, (0, n_pad))
            vert_embeddings = self.vertex_embd(dequantize_verts(pad_vertices.unsqueeze(1), self.quantization_bits))
        # Step zero embeddings
        if global_context_embedding is None: 
            zero_embed_tiled = torch.rand([batch_size, 1, self.embedding_dim])
        else: # global_context_embedding can be the embedded class label for example
            zero_embed_tiled = global_context_embedding.unsqueeze(1)

        # Aggregate embeddings
        embeddings = vert_embeddings + (coord_embeddings + pos_embeddings).unsqueeze(0)
        embeddings = torch.cat([zero_embed_tiled, embeddings], axis=1)
        return embeddings
    
    def create_vertex_coordinate_dist(self,
                   vertices,
                   global_context_embedding=None,
                   sequential_context_embeddings=None,
                   temperature=1.,
                   top_k=0,
                   top_p=1.,
                   cache=None):
        """Outputs categorical dist for quantized vertex coordinates."""

        # Embed inputs
        decoder_inputs = self._embed_inputs(vertices, global_context_embedding)

        if cache is not None:
            decoder_inputs = decoder_inputs[:, -1:]

        # pass through decoder
        outputs = self.decoder(
            decoder_inputs, 
            cache=cache,
            sequential_context_embeddings=sequential_context_embeddings)

        # Get logits and optionally process for sampling
        logits = self.project_to_logits(outputs)
        logits /= temperature
        """ x = torch.tensor([[[-0.7052,  0.2718,  0.7226, -0.6511,  0.3271, -0.8221],
         [-1.3992,  0.3387,  0.6098, -0.8584,  0.2438, -1.2653]]])
        import importlib
        import shared.math_utils as math_utils
        importlib.reload(math_utils)  """
        
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
        samples = torch.zeros([num_samples, 0], dtype=torch.int32)
        max_sample_length = max_sample_length or self.max_num_input_verts
        cache = self.decoder.create_init_cache(num_samples)
        
        stop_cond = False
        i = 0
        max_iters = max_sample_length * 3 + 1
        with torch.no_grad():
            while not stop_cond and i < max_iters:
                cat_dist = self.create_vertex_coordinate_dist(
                    samples,
                    global_context_embedding=global_context,
                    sequential_context_embeddings=seq_context,
                    cache=cache,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p)
                next_sample = cat_dist.sample()
                samples = torch.cat([samples, next_sample], dim=1)
                stop_cond = torch.eq(samples, 0).any(-1).all() # stop once all samples have a 0 (stop token)
                i += 1

        # Check if samples completed. Samples are complete if the stopping token
        # is produced.
        completed = torch.eq(samples, 0).any(-1)

        # Get the number of vertices in the sample. This requires finding the
        # index of the stopping token. For complete samples use to argmax to get
        # first nonzero index.
        stop_index_completed = torch.eq(samples, 0).int().argmax(dim=-1)
        # For incomplete samples the stopping index is just the maximum index.
        stop_index_incomplete = (
            max_sample_length * 3 * torch.ones_like(stop_index_completed))
    
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
        vertices_mask = torch.where(torch.range(0, max_sample_length-1) < num_vertices.unsqueeze(1), 1.0, 0.0)

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
                max_seq_length=5000):
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

        self.label_embd = nn.Embedding(self.num_classes, self.embedding_dim)
        self.label_embd.apply(init_weights)

        #self.coord_embd = nn.Embedding(3, self.embedding_dim) # 3 in to represent (x, y, z) 
        #self.coord_embd.apply(init_weights)

        self.pos_embd = nn.Embedding(self.max_seq_length, self.embedding_dim)
        self.pos_embd.apply(init_weights)

        self.project_to_pointers = nn.Linear(self.embedding_dim, self.embedding_dim)
        #self.project_to_pointers.apply(init_weights)

        if use_discrete_vertex_embeddings:
            self.vertex_embd = nn.Embedding(256, self.embedding_dim)
        else:
            self.vertex_embd = nn.Linear(self.max_num_input_verts, self.embedding_dim, bias=True)
        self.vertex_embd.apply(init_weights)

        # learnable stopping token embeddings
        self.stopping_embeddings = torch.nn.Parameter(torch.randn([1, 2, self.embedding_dim]))
        self.embed_zero = torch.nn.Parameter(torch.randn([1, 1, self.embedding_dim]))

        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_config['embd_size'], 
                                                   dim_feedforward=decoder_config['fc_size'], 
                                                   nhead=decoder_config['n_head'], 
                                                   dropout=decoder_config['dropout_rate'])
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_config['num_layers'])
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_config['embd_size'], 
                                                   dim_feedforward=encoder_config['fc_size'], 
                                                   nhead=encoder_config['n_head'], 
                                                   dropout=encoder_config['dropout_rate'])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_config['num_layers'])
        """ self.decoder = TransformerDecoder(**decoder_config)
        self.encoder = TransformerEncoder(**encoder_config) """
    
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
            for c in range(3): # x, y, z
                vertex_embeddings += self.vertex_embd(verts_quantized[..., c])
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
        vertex_embeddings = self.encoder(vertex_embeddings)
        return vertex_embeddings
    
    def embed_inputs(self, faces_long, vertex_embeddings,
                    global_context_embedding=None):
        """Embeds face sequences and adds within and between face positions."""
        # Face value embeddings are gathered vertex embeddings
        face_embeddings = torch.vmap(
            torch.index_select,
            in_dims=(0, None, 0),
        )(vertex_embeddings, 0, faces_long) # essentially gather on batched tensor
        
        # Position embeddings
        # batch size may be 0 during sampling, so we need to take the max
        pos_embeddings = self.pos_embd(torch.range(0, max(faces_long.shape[1], 1)-1, dtype=torch.int32)) 

        # Step zero embeddings
        batch_size = face_embeddings.shape[0]
        if global_context_embedding is None:
            zero_embed_tiled = torch.tile(self.embed_zero, [batch_size, 1, 1])
        else:
            zero_embed_tiled = global_context_embedding.unsqueeze(-1)

        # Aggregate embeddings
        embeddings = face_embeddings + pos_embeddings.unsqueeze(0)
        embeddings = torch.cat([zero_embed_tiled, embeddings], dim=1)

        return embeddings
    
    def create_vertex_indices_dist(self,
                   vertex_embeddings,
                   vertices_mask,
                   faces_long,
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
        
        decoder_outputs = self.decoder(
            decoder_inputs,
            cache=cache,
            sequential_context_embeddings=sequential_context_embeddings)

        # Get pointers by projecting transformer outputs to pointer vectors.
        pred_pointers = self.project_to_pointers(decoder_outputs)
       
        # Get logits and mask
        #print(pred_pointers.shape, vertex_embeddings.shape)
        logits = torch.matmul(pred_pointers, vertex_embeddings.transpose(1,2))
        logits /= math.sqrt(float(self.embedding_dim))
        f_verts_mask = F.pad(
            vertices_mask, (2,0), value=1.).unsqueeze(1)
        logits *= f_verts_mask
        logits -= (1. - f_verts_mask) * 1e9
        logits /= temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
        return torch.distributions.Categorical(logits=logits)
    
    def __call__(self, batch, is_training=False):
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
        
        pred_dist = self.create_vertex_indices_dist(
            vertex_embeddings,
            batch['vertices_mask'],
            batch['faces'][:, :-1],
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
        samples = torch.zeros([num_samples, 0], dtype=torch.int32)
        max_sample_length = max_sample_length or self.max_seq_length
        cache = self.decoder.create_init_cache(num_samples)

        stop_cond = False
        i = 0
        with torch.no_grad():
            while not stop_cond and i < max_sample_length:
                cat_dist = self.create_vertex_indices_dist(
                    vertex_embeddings,
                    context['vertices_mask'],
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

        # Record completed samples
        complete_samples = torch.eq(samples, 0).any(-1)

        # Find number of faces
        sample_length = samples.shape[-1]
        samples_range = torch.range(0, max(sample_length,1)-1).unsqueeze(0)
  
        # Get largest new face (1 is new face token) index as stopping point for incomplete samples.
        max_one_ind = torch.max(
            samples_range * (torch.eq(samples, 1)).int(),
            dim=-1)[1]

        zero_inds = torch.argmax(torch.eq(samples, 0).int(), axis=-1) # completed sample indices
        num_face_indices = torch.where(complete_samples, zero_inds, max_one_ind) + 1

        # Mask faces beyond stopping token with zeros
        # This mask has a -1 in order to replace the last new face token with zero
        faces_mask = (samples_range < num_face_indices.unsqueeze(-1) - 1).int()
        samples *= faces_mask
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


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class PositionalEncoding(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd, device=config.device)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd, device=config.device)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_idx):
        b, seq_len = token_idx.size()

        pos = torch.arange(0, seq_len, dtype=torch.long, device=token_idx.device).unsqueeze(0)
        self.tok_emb = self.token_embeddings(token_idx)
        self.pos_emb = self.position_embeddings(pos)

        return self.dropout(self.tok_emb + self.pos_emb)


class SelfAttention(nn.Module):

    def __init__(self, embd_size=256, num_heads=4, bias=True, dropout_rate=0.2, causal=False):
        super().__init__()
        # Ensure that the embedding can be split up into n heads
        assert embd_size % num_heads == 0

        self.n_head = num_heads
        self.n_embd = embd_size
        self.dropout = dropout_rate
        self.causal = causal

        # regularization
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # 3 * config.n_embd so that the output can be split into key, query and value tensors.
        # Saves having to make 3 different linear layers
        self.qvk_proj = nn.Linear(embd_size, 3 * embd_size, bias=bias)
        #self.q_proj = nn.Linear(embd_size, embd_size, bias=bias)
        #self.v_proj = nn.Linear(embd_size, embd_size, bias=bias)
        #self.k_proj = nn.Linear(embd_size, embd_size, bias=bias)
        # output projection
        self.out_proj = nn.Linear(embd_size, embd_size, bias=bias)

    def forward(self, x, context_embedding=None, key_padding_mask=None, cache=None):
        b, seq_len, n_embd = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q, k, v  = self.qvk_proj(x).split(self.n_embd, dim=2)
     
        if context_embedding is not None:
            #_, k, v  = self.qvk_proj(context_embedding).split(self.n_embd, dim=2)
            _, k, v = context_embedding.split(self.n_embd, dim=2)
        elif cache is not None:
            k_old, v_old = cache['k'], cache['v']
            if k_old.shape[1] != 0: # dim 1 is the sequence length, 0 means first iter so no cache saved
                k = torch.cat((k_old, k), dim=2)
                v = torch.cat((v_old, v), dim=2)
        
        q = q.view(b, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(b, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(b, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(b, 1, 1, seq_len).expand(
                -1, self.n_head, -1, -1).reshape(b * self.n_head, 1, seq_len)
            attn_mask = key_padding_mask.view(b, self.n_head, -1, seq_len)
        else:
            attn_mask = None

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # much faster than other implementation!!
            attn_weight = torch.nn.functional.scaled_dot_product_attention(
                q, 
                k, 
                v, 
                attn_mask=attn_mask, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=self.causal)
        else:
            attn_dropout = self.dropout if self.training else 0
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            # Mask padding tokens
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask == 0, -float('inf')) # Mask out pad tokens
            # Mask future tokens
            if self.causal:
                attn_mask = torch.ones(seq_len, seq_len, device=x.device).tril(diagonal=0)
                attn = attn.masked_fill(attn_mask == 0, -float('inf')) # Mask out future tokens
            attn_weight = torch.softmax(attn, dim=-1)
            attn_weight = torch.dropout(attn_weight, attn_dropout, self.training) 
            attn_weight  = attn_weight @ v # (b, nh, seq_len, seq_len) x (b, nh, seq_len, hs) -> (b, nh, seq_len, hs)
            
        attn_weight = attn_weight.transpose(1, 2).contiguous().view(b, seq_len, n_embd) # re-assemble all head outputs side by side
        # output projection
        attn_weight = self.resid_dropout(self.out_proj(attn_weight))
        
        return attn_weight


class MLP(nn.Module):

    def __init__(self, embd_size=256, fc_size=1024, bias=True, dropout_rate=0.2):
        super().__init__()
        self.fc    = nn.Linear(embd_size, fc_size, bias=bias)
        self.out_proj  = nn.Linear(fc_size, embd_size, bias=bias)
        self.relu = nn.ReLU()
        #self.gelu = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        #x = self.gelu(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x

    
class TransformerEncoderBlock(nn.Module):

    def __init__(self, 
                 embd_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 dropout_rate=0.2,
                 bias=True):
        super().__init__()

        self.multi_head_attn = SelfAttention(embd_size, num_heads, bias, dropout_rate, causal=False)
        self.feed_forward = MLP(embd_size, fc_size, bias, dropout_rate)

        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norm_1 = LayerNorm(embd_size, bias=bias)
            self.layer_norm_2 = LayerNorm(embd_size, bias=bias)

    def forward(self, x, key_padding_mask=None):
        x_ln = self.layer_norm_1(x) if self.layer_norm else x
        x = x + self.multi_head_attn(x_ln, key_padding_mask=key_padding_mask)
        x_ln = self.layer_norm_2(x) if self.layer_norm else x
        x = x + self.feed_forward(x_ln)
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, 
                 embd_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 bias=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                 embd_size,
                 fc_size,
                 num_heads,
                 layer_norm,
                 dropout_rate,
                 bias) for _ in range(num_layers)])
        self.layer_norm = LayerNorm(embd_size, bias=bias)

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.layer_norm(x)
    
class TransformerDecoderBlock(nn.Module):

    def __init__(self, 
                 embd_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 dropout_rate=0.2,
                 bias=True, 
                 take_context_embedding=False):
        super().__init__()

        self.layer_norm = layer_norm
        self.take_context_embedding = take_context_embedding

        self.masked_multi_head_attn = SelfAttention(embd_size, num_heads, bias, dropout_rate, causal=True)
        self.feed_forward = MLP(embd_size, fc_size, bias, dropout_rate)
        if layer_norm:
            self.layer_norm_1 = LayerNorm(embd_size, bias=bias)
            self.layer_norm_3 = LayerNorm(embd_size, bias=bias)
  
        # Check if the decoder will take input from an encoder
        if take_context_embedding:
            self.multi_head_attn = SelfAttention(embd_size, num_heads, bias, dropout_rate, causal=False)
            if layer_norm:
              self.layer_norm_2 = LayerNorm(embd_size, bias=bias)

    def forward(self, x, context_embedding=None, key_padding_mask=None, cache=None):
        x_ln = self.layer_norm_1(x) if self.layer_norm else x
        x = x + self.masked_multi_head_attn(x_ln, key_padding_mask=key_padding_mask, cache=cache)
        if context_embedding is not None and self.take_context_embedding:
            x_ln = self.layer_norm_2(x) if self.layer_norm else x
            x = x + self.multi_head_attn(x_ln, context_embedding=context_embedding, key_padding_mask=key_padding_mask, cache=cache)
        x_ln = self.layer_norm_3(x) if self.layer_norm else x
        x = x + self.feed_forward(x_ln)
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, 
                 embd_size=256,
                 fc_size=1024,
                 num_heads=4,
                 layer_norm=True,
                 num_layers=8,
                 dropout_rate=0.2,
                 bias=True,
                 take_context_embedding=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embd_size,
                 fc_size,
                 num_heads,
                 layer_norm,
                 dropout_rate,
                 bias, 
                 take_context_embedding=take_context_embedding) for _ in range(num_layers)])
        self.embd_size = embd_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        if layer_norm:
          self.layer_norm = LayerNorm(embd_size, bias=bias)

    def forward(self, 
                x, 
                key_padding_mask=None, 
                cache=None,
                sequential_context_embeddings=None):
        for i, layer in enumerate(self.layers):
            cache_i = None if cache is None else cache[i]
            x = layer(x, context_embedding=sequential_context_embeddings, key_padding_mask=key_padding_mask, cache=cache_i)
        return self.layer_norm(x) if self.layer_norm else x
    
    def create_init_cache(self, batch_size):
        """ Creates empty cache dictionary for use in fast decoding. """

        # Build cache
        k = torch.zeros([batch_size, 0, self.embd_size])
        v = torch.zeros([batch_size, 0, self.embd_size])
        """ k = torch.zeros([batch_size, 0, self.embd_size]).view(
            batch_size, 
            self.num_heads, 
            0, 
            self.embd_size // self.num_heads)
        v = torch.zeros([batch_size, 0, self.embd_size]).view(
            batch_size, 
            self.num_heads, 
            0, 
            self.embd_size // self.num_heads) """

        cache = [{'k': k, 'v': v} for _ in range(self.num_layers)]
        return cache


class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.transformer = nn.ModuleDict(dict(
            positional_enc = PositionalEncoding(config),
            encoder = Encoder(config),
            decoder = Decoder(config, True),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.positional_enc.token_embeddings.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
       
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('embd_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def forward(self, inputs):
        
        target_true = torch.cat((inputs['label_ids'][:, 1:], torch.tensor([0]).expand(len(inputs['input_ids']),1)), dim=1).to(self.device)
        inputs['label_ids'][:, inputs['label_ids'].argmin()-1] = 0

        # Positional encoding
        in_pos_enc = self.transformer.positional_enc(inputs['input_ids'].to(self.device))
        
        # Encoder
        enc_attention_scores = self.transformer.encoder(in_pos_enc, key_padding_mask=inputs['input_attention_mask'].bool().to(self.device))
        target_pos_enc = self.transformer.positional_enc(inputs['label_ids'].to(self.device))

        # Decoder
        decoder_out = self.transformer.decoder(target_pos_enc, enc_in=enc_attention_scores)

        # if we are given some desired targets also calculate the loss
        logits = self.lm_head(decoder_out)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_true.contiguous().view(-1), ignore_index=0)

        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
    
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.positional_enc.position_embeddings.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # remove eos token
        eos_token = 101 #inputs['label_ids'][0][1]
        #inputs['label_ids'][0][1] = 0
        #inputs
        output = []
        #print(inputs['input_ids'][0].shape)
        
        for i in range(1, max_new_tokens):
            
            """ # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:] """
            #print("in", inputs['label_ids'][0])
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(inputs)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Stop if eos token
            #print(idx_next.item())
            if idx_next.item() == eos_token:
                break
            # append sampled index to the running sequence and continue
            #idx = torch.cat((idx, idx_next), dim=1)
            inputs['label_ids'][0][i] = idx_next #= torch.cat((inputs['input_ids'][0], idx_next), dim=1)
            inputs['label_attention_mask'][0][i] = True
            #inputs['input_ids'][0][i] = idx_next
            output.append(idx_next.item())

        return output #inputs['label_ids'][0]