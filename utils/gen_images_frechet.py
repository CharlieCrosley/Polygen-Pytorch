import os, sys
from tqdm import tqdm
import math
import bpy
from mathutils import Vector
import numpy as np
import joblib


def generate_images(model_paths, save_img_base_path, hor_variations=8, ver_variations=2, resume_model_idx=0):
    # camera distance from object
    camera_distance = 1.7
    working_dir = os.getcwd()

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.data.worlds.new("World")
    bpy.ops.scene.new(type='NEW')
    bpy.context.scene.world = bpy.data.worlds.get("World")
    bpy.context.scene.world.use_nodes = True

    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    model_count = resume_model_idx
    for model_path in tqdm(model_paths):
        model = model_path.split('\\')[-1].split('.')[0]
     
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.data.worlds.new("World")
        bpy.ops.scene.new(type='NEW')
        bpy.context.scene.world = bpy.data.worlds.get("World")
        bpy.context.scene.world.use_nodes = True
        bpy.context.scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0,0,0,0)

        # Top light
        top_area_light_data = bpy.data.lights.new(name="top-area-light-data", type='AREA')
        top_area_light_data.energy = 15
        top_area_light_data.size = 2.5
        # Create new object, pass the light data 
        top_light_object = bpy.data.objects.new(name="top-area-light", object_data=top_area_light_data)
        top_light_object.location = Vector([0,0,1])
        # Link object to collection in context
        bpy.context.collection.objects.link(top_light_object)
        light_track_to_constraint = top_light_object.constraints.new('TRACK_TO')

        # Bottom light
        bot_area_light_data = bpy.data.lights.new(name="bot-area-light-data", type='AREA')
        bot_area_light_data.energy = 15
        bot_area_light_data.size = 2.5
        # Create new object, pass the light data 
        bot_light_object = bpy.data.objects.new(name="bot-area-light", object_data=bot_area_light_data)
        bot_light_object.location = Vector([0,0,-1])
        # Link object to collection in context
        bpy.context.collection.objects.link(bot_light_object)
        light_track_to_constraint = bot_light_object.constraints.new('TRACK_TO')

        # Create the camera
        camera_data = bpy.data.cameras.new('camera')
        camera = bpy.data.objects.new('camera', camera_data)
        bpy.context.collection.objects.link(camera)
        bpy.context.scene.camera = camera
        camera_track_to_constraint = camera.constraints.new('TRACK_TO')

        #bpy.ops.import_scene.obj(filepath=model_path, split_mode='OFF', filter_glob="*.obj;*.mtl")
        bpy.ops.import_scene.obj(filepath=model_path, split_mode='OFF')
        base_object = bpy.data.objects[model]
        camera_track_to_constraint.target = base_object

        # Calculate the current diagonal length of the bounding box
        object_length = math.sqrt(base_object.dimensions.x ** 2 + base_object.dimensions.y ** 2  + base_object.dimensions.z ** 2)
        # Calculate the scale factor to achieve a diagonal length of 1
        scale_factor = 1.0 / object_length
        # Scale the object uniformly using the calculated scale factor
        base_object.scale.x *= scale_factor
        base_object.scale.y *= scale_factor
        base_object.scale.z *= scale_factor

        os.makedirs(os.path.join(save_img_base_path, model), exist_ok=True)            

        # Set unique materials for the model (you need to define your materials)
        material = bpy.data.materials.new(name=f"Material")
        material.use_nodes = True
        material.node_tree.nodes.clear()

        # Create the Node Tree
        nodes = material.node_tree.nodes

        # Add a Material Output, then any other nodes you want
        output = nodes.new(type='ShaderNodeOutputMaterial')

        # Add a Principled BSDF Shader, and adjust some values
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        # Set the Roughness
        bsdf.inputs['Roughness'].default_value = 0.5#0.2
        # Set the Metallic
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Base Color'].default_value = (0.75,0.75,0.75,1)
        #print(bsdf.inputs.keys())
        """ 
        ['Base Color', 'Subsurface', 'Subsurface Radius', 'Subsurface Color', 'Subsurface IOR', 
        'Subsurface Anisotropy', 'Metallic', 'Specular', 'Specular Tint', 'Roughness', 'Anisotropic', 
        'Anisotropic Rotation', 'Sheen', 'Sheen Tint', 'Clearcoat', 'Clearcoat Roughness', 'IOR', 
        'Transmission', 'Transmission Roughness', 'Emission', 'Emission Strength', 'Alpha', 'Normal', 'Clearcoat Normal', 'Tangent', 'Weight'] 
        """

        # Now to connect the nodes
        links = material.node_tree.links
        # Create a connection between the first output of the Principled BSDF and 
        # the first input of the Material Output
        links.new(bsdf.outputs[0], output.inputs['Surface'])
        for i in range(len(base_object.material_slots)):
            base_object.material_slots[i].material = material
        
        save_file_path = os.path.join(working_dir, save_img_base_path, model)
        bpy.context.scene.render.use_simplify = True

        # n random 15W spot lights 
        angle = 0
        n_point_light = 8
        dx_angle = 360 / n_point_light
        for i in range(n_point_light):
            # Create light datablock
            point_light_data = bpy.data.lights.new(name=f"point-light-data-{i}", type='AREA')
            point_light_data.energy = 15
            point_light_data.size = 2.5
            # Create new object, pass the light data 
            light_object = bpy.data.objects.new(name=f"point-light-{i}", object_data=point_light_data)
            light_object.location = Vector([camera_distance, camera_distance, 0])
            light_object.location.x *= math.cos(np.deg2rad(angle)) # convert rot to vector and scale distance
            light_object.location.y *= math.sin(np.deg2rad(angle))
            angle += dx_angle
            # Link object to collection in context
            bpy.context.collection.objects.link(light_object)
            light_track_to_constraint = light_object.constraints.new('TRACK_TO')
            light_track_to_constraint.target = base_object

        # Images with horizontal offset
        angle = 0
        dx_angle = 360 / hor_variations
        for i in range(hor_variations):
            # Camera
            camera.location = Vector([camera_distance, camera_distance, 0])
            camera.location.x *= math.cos(np.deg2rad(angle)) # convert rot to vector and scale distance
            camera.location.y *= math.sin(np.deg2rad(angle))
            angle += dx_angle
            
            # Render the image
            bpy.context.scene.render.filepath = os.path.join(save_file_path, f'var_{i}.png')
            bpy.ops.render.render(write_still=True)

        
        # Images with vertical offset
        dx_angle = 90 / (ver_variations)
        angle = dx_angle
        for i in range(hor_variations, hor_variations+ver_variations):
            # Camera
            camera.location = Vector([0, 0, 0])
            camera.location.y = camera_distance * math.cos(np.deg2rad(angle))
            camera.location.z = camera_distance * math.sin(np.deg2rad(angle))
            angle += dx_angle

            # Render the image
            bpy.context.scene.render.filepath = os.path.join(save_file_path, f'var_{i}.png')
            bpy.ops.render.render(write_still=True)

        model_count += 1

import os
import sys
from contextlib import contextmanager

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

def bar(dir_path, file_name):
    obj_path = os.path.join(dir_path, file_name)
    return obj_path

classes = list(os.walk('./objs'))[1:]
bench_obj_paths = joblib.Parallel(n_jobs=-1)(joblib.delayed(bar)(classes[0][0], file) for file in classes[0][2])
chair_obj_paths = joblib.Parallel(n_jobs=-1)(joblib.delayed(bar)(classes[1][0], file) for file in classes[1][2])
table_obj_paths = joblib.Parallel(n_jobs=-1)(joblib.delayed(bar)(classes[2][0], file) for file in classes[2][2])
objs_parent_dir = './obj_images'
os.makedirs(objs_parent_dir, exist_ok=True)
os.makedirs(objs_parent_dir + '/bench', exist_ok=True)
os.makedirs(objs_parent_dir + '/chair', exist_ok=True)
os.makedirs(objs_parent_dir + '/table', exist_ok=True)

with stdout_redirected():
    generate_images(bench_obj_paths, objs_parent_dir + '/bench', hor_variations=5, ver_variations=1)
    generate_images(chair_obj_paths, objs_parent_dir + '/chair', hor_variations=5, ver_variations=1)
    generate_images(table_obj_paths, objs_parent_dir + '/table', hor_variations=5, ver_variations=1)
