# Polygen Pytorch Implementation

This repo implements deepminds polygen model in pytorch as opposed to the [original tensorflow implementation](https://github.com/deepmind/deepmind-research/tree/master/polygen) as described in:<br>
> **PolyGen: An Autoregressive Generative Model of 3D Meshes**, *Charlie Nash, Yaroslav Ganin, S. M. Ali Eslami, Peter W. Battaglia*, ICML, 2020. ([abs](https://arxiv.org/abs/2002.10880))

All credit goes to deepmind.

## Requirements

- python == 3.11.4
- pytorch == 2.0.1
- matplotlib == 3.7.2
- networkx == 3.0
- six
- Blender if using the dataset generation scripts

## Pretrained weights
Pretrained weights for a vertex and face model are available [here](https://www.dropbox.com/scl/fo/o0ur761yhw0cdk5nn06jb/AMxyYB87VJQ8W8zxkjJi-Lc?rlkey=y1a6g1cq68k164kmhgqt3mvgu&dl=0)
The models were trained on single-view reconstruction for 3 categories (chair, bench, table) from the ShapeNetCore dataset.
The input images are assumed to have a solid black background, currently the training and inference scripts create a mask to convert a constant grey background (background colour generated using blender in the gen_singleview_reconstruction_dataset.py script) to black.
This code must be changed if using custom input images that don't have the same solid background colour.

**Vertex model**. 
- Trained for 600k steps
- Batch size of 8
- AdamW optimizer with a learning rate of 3e-4
- Cosine annealing learning rate scheduler with a linear warmup period of 5000 steps
  
**Face model**. 
- Trained for 500k steps
- AdamW optimizer with a learning rate of 1e-4
- Everything else is the same as the vertex model

This was trained on a single RTX 3060 so there was limitations on performance.
inference.ipynb can be used to demonstrate the outputs using the var_0.png in the example_input/model_0 directory

