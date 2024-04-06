import os, sys
import shared.data_utils as data_utils
import torch
from tqdm import tqdm
import math
import bpy
import random
from mathutils import Vector
import numpy as np
import json
import joblib
import os
import sys
from contextlib import contextmanager

bpy.context.preferences.edit.undo_steps = 0

def generate_images(model_paths, save_img_base_path, save_obj_base_path, n_variations=1, resume_model_idx=0):
    # camera distance from object
    camera_min_distance = 1.25
    camera_max_distance = 1.5
    working_dir = os.getcwd()

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.data.worlds.new("World")
    bpy.ops.scene.new(type='NEW')
    bpy.context.scene.world = bpy.data.worlds.get("World")
    bpy.context.scene.world.use_nodes = True

    def clear_point_lights():
        # Iterate over all objects in the scene
        for obj in bpy.data.objects:
            # Check if the object is a point light
            if (obj.type == 'LIGHT' and obj.data.type == 'POINT'):
                # Delete the point light
                bpy.data.objects.remove(obj)

    os.makedirs(os.path.join(working_dir, save_obj_base_path), exist_ok=True)

    model_count = resume_model_idx
    for model_path in tqdm(model_paths):
        model = model_path.split('\\')[-1].split('.')[0]
        
        num_obj_variations = 100 # how many augmented meshes per original mesh
        for obj_var in range(num_obj_variations):
            bpy.ops.wm.read_homefile(use_empty=True)
            bpy.data.worlds.new("World")
            bpy.ops.scene.new(type='NEW')
            bpy.context.scene.world = bpy.data.worlds.get("World")
            bpy.context.scene.world.use_nodes = True
            
            bpy.context.scene.render.resolution_x = 256
            bpy.context.scene.render.resolution_y = 256
            bpy.context.scene.render.image_settings.file_format = 'PNG'

            # Area light above mesh
            area_light_data = bpy.data.lights.new(name="area-light-data", type='AREA')
            area_light_data.energy = 20
            area_light_data.size = 2.5
            # Create new object, pass the light data 
            light_object = bpy.data.objects.new(name="area-light", object_data=area_light_data)
            # Link object to collection in context
            bpy.context.collection.objects.link(light_object)
            light_track_to_constraint = light_object.constraints.new('TRACK_TO')

            # Create the camera
            camera_data = bpy.data.cameras.new('camera')
            camera = bpy.data.objects.new('camera', camera_data)
            bpy.context.collection.objects.link(camera)
            bpy.context.scene.camera = camera
            camera_track_to_constraint = camera.constraints.new('TRACK_TO')

            #bpy.ops.import_scene.obj(filepath=model_path, split_mode='OFF', filter_glob="*.obj;*.mtl") # include materials
            bpy.ops.import_scene.obj(filepath=model_path, split_mode='OFF')
            base_object = bpy.data.objects[model]
            camera_track_to_constraint.target = base_object

            if obj_var > 0: # keep the original scale for one of the objs
                # axis scaling
                sx, sy, sz = (random.uniform(0.75, 1.25), random.uniform(0.75, 1.25), random.uniform(0.75, 1.25))
                base_object.scale.x *= sx
                base_object.scale.y *= sy
                base_object.scale.z *= sz

            # scale diagonal bounding box length to 1
            # Calculate the current diagonal length of the bounding box
            object_length = math.sqrt(base_object.dimensions.x ** 2 + base_object.dimensions.y ** 2 + base_object.dimensions.z ** 2)
            # Calculate the scale factor to achieve a diagonal length of 1
            scale_factor = 1.0 / object_length
            # Scale the object uniformly using the calculated scale factor
            base_object.scale.x *= scale_factor
            base_object.scale.y *= scale_factor
            base_object.scale.z *= scale_factor

            # planar mesh decimation
            modifier = base_object.modifiers.new("DecimateMod",'DECIMATE')
            modifier.decimate_type = "DISSOLVE"
            modifier.angle_limit = np.deg2rad(random.uniform(0,20))

            max_faces = 1200
            max_vertices = 600
            # ignore meshes that are too large after planar decimation
            if len(base_object.data.vertices) > max_vertices or len(base_object.data.polygons) > max_faces:
                continue
            
            # Export obj so that it can be preprocessed for pytorch training
            obj_save_path = os.path.join(working_dir, save_obj_base_path, f"model_{model_count}.obj")
            base_object.select_set(state=True)
            bpy.context.view_layer.objects.active = base_object
            bpy.ops.export_scene.obj(filepath=obj_save_path, use_selection=True, use_materials=True)
            
            # Save processed mesh dictionary of pytorch tensors
            mesh_dict = data_utils.load_process_mesh(obj_save_path)
            mesh_dict['vertices'] = torch.tensor(mesh_dict['vertices'])
            mesh_dict['faces'] = torch.tensor(mesh_dict['faces'])
            if len(mesh_dict['vertices']) > max_vertices or len(mesh_dict['faces']) > max_faces:
                continue

            os.makedirs(os.path.join(save_img_base_path, f'model_{model_count}'), exist_ok=True)
            torch.save(mesh_dict, os.path.join(save_img_base_path, f'model_{model_count}', 'processed_model.pt'))

            for i in range(len(base_object.material_slots)):
                # Set unique materials for the model (you need to define your materials)
                material = bpy.data.materials.new(name=f"Material_{i}")
                material.use_nodes = True
                material.node_tree.nodes.clear()

                # Create the Node Tree
                nodes = material.node_tree.nodes

                # Add a Material Output, then any other nodes you want
                output = nodes.new(type='ShaderNodeOutputMaterial')

                color_ramp = nodes.new(type='ShaderNodeValToRGB')
                color_ramp.color_ramp.interpolation = 'LINEAR'  # You can change the interpolation method.
                color_ramp.color_ramp.elements[0].position = 0.0  # Start of the gradient.
                color_ramp.color_ramp.elements[1].position = 1.0  # End of the gradient.
                color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
                # Add a random colour stop
                color_point3 = color_ramp.color_ramp.elements.new(0.5)  # Position at 50% of the ramp
                color_point3.color = (random.uniform(0,1.0), random.uniform(0,1.0), random.uniform(0,1.0), 1.0)

                # Add a Principled BSDF Shader, and adjust some values
                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                # Set the Roughness
                bsdf.inputs[9].default_value = random.uniform(0, 1)
                # Set the Metallic
                bsdf.inputs[6].default_value = random.uniform(0, 1)
            
                # Add a Noise Texture Node
                noise = nodes.new(type='ShaderNodeTexNoise')
                #print(noise.inputs.keys()) # list names of inputs: ['Vector', 'W', 'Scale', 'Detail', 'Roughness', 'Distortion']
                # Increase the Scale
                noise.inputs['Scale'].default_value = math.sqrt(100 * random.uniform(0, 1))
                # Detail
                noise.inputs[3].default_value = 16
                # Noise Roughness (Lower = smoother, Higher = rougher)
                noise.inputs['Roughness'].default_value = random.uniform(0, 20)

                # Now to connect the nodes
                links = material.node_tree.links
                # Create a connection between the Noise Texture and the Color Ramp
                links.new(noise.outputs['Color'], color_ramp.inputs[0])
                # Create a connection between the Color Ramp and the Principled BSDF
                links.new(color_ramp.outputs[0], bsdf.inputs['Base Color'])
                # Create a connection between the first output of the Principled BSDF and 
                # the first input of the Material Output
                links.new(bsdf.outputs[0], output.inputs['Surface'])
                base_object.material_slots[i].material = material
            
            save_file_path = os.path.join(working_dir, save_img_base_path, f"model_{model_count}")
            for i in range(n_variations):
                # n random 15W spot lights 
                n_point_light = random.randint(0, 10)
                for j in range(n_point_light):
                    # Create light datablock
                    point_light_data = bpy.data.lights.new(name=f"point-light-data-{j}", type='POINT')
                    point_light_data.energy = 15
                    # Create new object, pass the light data 
                    light_object = bpy.data.objects.new(name=f"point-light-{j}", object_data=point_light_data)
                    light_object.location = (random.uniform(-2, 0.75), random.uniform(-0.75, 2), random.uniform(0.75, 2))
                    # Link object to collection in context
                    bpy.context.collection.objects.link(light_object)
                    light_track_to_constraint = light_object.constraints.new('TRACK_TO')
                    light_track_to_constraint.target = base_object

                # Randomize camera position, rotation is always looking at the object
                elevation = random.uniform(0, 1)
                dist = random.uniform(camera_min_distance, camera_max_distance)
                rot = random.randint(0, 360) # rotation around object
                camera.location = Vector([dist, dist, elevation])
                camera.location.x *= math.cos(np.deg2rad(rot)) # convert rot to vector and scale distance
                camera.location.y *= math.sin(np.deg2rad(rot))
                focal_len = random.randint(35, 50)
                camera.data.lens = focal_len

                # Render the image
                bpy.context.scene.render.filepath = os.path.join(save_file_path, f'var_{i}.png')
                bpy.context.scene.render.filter_size = random.uniform(1.5, 2) # small random blur
                bpy.ops.render.render(write_still=True)
                if n_variations > 1:
                    clear_point_lights()
    
            model_count += 1

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

            
# Get all .obj file paths in dataset
def foo(dir_path):
    if dir_path.endswith('models'):
        with open(os.path.join(dir_path, 'model_normalized.json'), 'r') as f:
            num_verts = json.load(f)['numVertices']
            if num_verts > 600:
                return
        obj_path = os.path.join(dir_path, 'model_normalized.obj')
        return obj_path

wd_files = list(os.walk('.'))
obj_paths = joblib.Parallel(n_jobs=-1)(joblib.delayed(foo)(dir_path) for (dir_path, dir_names, file_names) in wd_files)
obj_paths = [x for x in obj_paths if x is not None] # all obj paths

# seperate categories
categories = dict()
for obj in obj_paths:
    category = obj[2:].split("\\")[0]
    if category in categories:
        categories[category].append(obj)
    else:
        categories[category] = [obj]

# Save each categories preprocessed meshes to a seperate directory.
for category, objs in categories.items():
    with stdout_redirected():
        generate_images(objs, f'processed_{category}', f'processed_{category}_objs', n_variations=1, resume_model_idx=0)