import blenderproc as bproc
import csv
import argparse
import os
import numpy as np
import random
import mathutils
import json

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument("image_size", default=512, type=int, help="image size.")
parser.add_argument("output_dir", help="Path to where the data should be saved")
args = parser.parse_args()
panoptic_anno_csv = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping_panoptic.csv"))
if not os.path.exists(panoptic_anno_csv):
    raise Exception('Cannot perform layout generation without panoptic_anno_csv specified')

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

things_ids = []
with open(panoptic_anno_csv, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['isthing'] == 'True':
            things_ids.append(mapping.id_from_label(row['name']))

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)
category_ids = np.unique([obj.get_cp("category_id") for obj in loaded_objects])
# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

def check_name(name, keywords=[
    "cabinet", "wine cooler", "nightstand", "bookcase",
    "wardrobe", "tv stand", "media unit", "shelf", "desk", "storage unit",
    "bed", "chair", "table", "sofa", "armchair", "stool", "lamp", "lamp",
    "basin", "bath", "plants", "appliance"]):
    for category_name in keywords:
        if category_name in name.lower():
            return True
    return False

os.makedirs(args.output_dir, exist_ok=True)
path_to_camera_locations = os.path.join(args.output_dir, 'camera_locations.npy')
path_to_camera_rotations = os.path.join(args.output_dir, 'rotations.npy')
if os.path.exists(path_to_camera_locations) and os.path.exists(path_to_camera_rotations):
    locations = np.load(path_to_camera_locations)
    rotations = np.load(path_to_camera_rotations)
    print('Found camera settings, loaded %d cameras'%len(locations))
    for location, rotation in zip(locations, rotations):
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
        bproc.camera.add_camera_pose(cam2world_matrix)
else:
    locations = []
    rotations = []
    # filter some objects from the loaded objects, which are later used in calculating an interesting score
    special_objects_category_ids = []
    special_objects = []
    random.shuffle(loaded_objects)
    for obj in loaded_objects:
        category_name = obj.get_name()
        category_id = obj.get_cp("category_id")
        if check_name(category_name) and category_id not in special_objects_category_ids:
            special_objects_category_ids.append(category_id)
            special_objects.append(obj)
    proximity_checks = {"min": 1.0, "avg": {"min": 2.5, "max": 3.5}, "no_background": True}
    # poses = 0
    # tries = 0
    # while tries < 10000 and poses < 10:
    #     # Sample point inside house
    #     height = np.random.uniform(1.4, 1.8)
    #     location = point_sampler.sample(height)
    #     # Sample rotation (fix around X and Y axis)
    #     rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
    #     cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
    #
    #     # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
    #     # meters and make sure that no background is visible, finally make sure the view is interesting enough
    #     if bproc.camera.scene_coverage_score(cam2world_matrix, special_objects_category_ids, special_objects_weight=10.0) > 0.8 \
    #             and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
    #         bproc.camera.add_camera_pose(cam2world_matrix)
    #         poses += 1
    #     tries += 1
    num_scenes = len(special_objects)
    print('=========%d loaded objects, %d special objects===='%(len(loaded_objects), num_scenes))

    poses = 0
    tries = 0
    centered_objects_category_ids = []
    while tries < 10000 and poses < 10:
        # Sample point inside house
        height = np.random.uniform(1.4, 1.8)
        location = point_sampler.sample(height)
        # Sample rotation (fix around X and Y axis)
        rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
        if bproc.camera.scene_coverage_score(cam2world_matrix, special_objects_category_ids,
                                             special_objects_weight=10.0) > 0.8 \
                and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
            for obj in bproc.camera.visible_objects(cam2world_matrix):
                category_id = obj.get_cp("category_id")
                if category_id in special_objects_category_ids and category_id not in centered_objects_category_ids:
                    lookat_point = obj.get_location() + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
                    rotation = bproc.camera.rotation_from_forward_vec(lookat_point - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
                    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
                    # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
                    # meters and make sure that no background is visible, finally make sure the view is interesting enough
                    if bproc.camera.scene_coverage_score(cam2world_matrix, special_objects_category_ids, special_objects_weight=10.0) > 0.8 \
                            and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
                        centered_objects_category_ids.append(category_id)
                        bproc.camera.add_camera_pose(cam2world_matrix)
                        locations.append(location)
                        rotations.append(rotation)
                        poses += 1
        tries += 1

    num_scenes = len(special_objects) - len(centered_objects_category_ids)

    poses = 0
    index_padding = 0
    tries = 0
    while tries < 10000 and poses < num_scenes:
        dist_above_center = np.random.uniform(0.2, 0.5)
        # location = point_sampler.sample(height)
        obj = special_objects[poses + index_padding]
        if obj.get_cp("category_id") in centered_objects_category_ids:
            index_padding += 1
        else:
            location = bproc.sampler.part_sphere(obj.get_location(), radius=2, dist_above_center=dist_above_center, mode="SURFACE")
            # Compute rotation based lookat point which is placed randomly around the object
            lookat_point = obj.get_location() + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
            rotation = bproc.camera.rotation_from_forward_vec(lookat_point - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
            # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
            # meters and make sure that no background is visible, finally make sure the view is interesting enough
            if bproc.camera.scene_coverage_score(cam2world_matrix, special_objects_category_ids, special_objects_weight=10.0) > 0.8 \
                    and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
                bproc.camera.add_camera_pose(cam2world_matrix)
                locations.append(location)
                rotations.append(rotation)
                poses += 1
            tries += 1
    np.save(path_to_camera_locations, locations)
    np.save(path_to_camera_rotations, rotations)
    print(centered_objects_category_ids)
    print(special_objects_category_ids)
image_size = args.image_size
bproc.camera.set_intrinsics_from_blender_params(1.0472, image_size, image_size, lens_unit="FOV")
bproc.camera.set_resolution(image_size, image_size)
# Also render normals
bproc.renderer.enable_normals_output()
# Distance from camera position to 3D point
bproc.renderer.enable_distance_output()
# Distance from camera and the plane parallel to the camera with the corresponding point lies on.
bproc.renderer.enable_depth_output()

# set the sample amount to 350
bproc.renderer.set_samples(350)

# render the whole pipeline
data = bproc.renderer.render()
seg_data = bproc.renderer.render_segmap(map_by=["class","instance","name"])
# bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
#                                     instance_segmaps=seg_data["instance_segmaps"],
#                                     instance_attribute_maps=seg_data["instance_attribute_maps"],
#                                     colors=data["colors"],
#                                     color_file_format="JPEG")
instance_attribute_maps = os.path.join(args.output_dir, "instance_attribute_maps.json")
with open(instance_attribute_maps, "w") as f:
    json.dump(seg_data["instance_attribute_maps"], f)
seg_data.pop("instance_attribute_maps", None)
data.update(seg_data)
# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)
# write down generated poses and rotations

def get_attribute(obj, attribute_name):
    name_to_id = {}
    if attribute_name == "id":
        if obj.name not in name_to_id:
            name_to_id[obj.name] = len(name_to_id.values())
        return name_to_id[obj.name]
    elif attribute_name == "name":
        return obj.name
    elif attribute_name == "location":
        return obj.location
    elif attribute_name == "rotation_euler":
        return obj.rotation_euler
    elif attribute_name == "rotation_forward_vec":
        # Calc forward vector from rotation matrix
        rot_mat = obj.rotation_euler.to_matrix()
        forward = rot_mat @ mathutils.Vector([0, 0, -1])
        return forward
    elif attribute_name == "rotation_up_vec":
        # Calc up vector from rotation matrix
        rot_mat = obj.rotation_euler.to_matrix()
        up = rot_mat @ mathutils.Vector([0, 1, 0])
        return up
    elif attribute_name == "matrix_world":
        # Transform matrix_world to given destination frame
        matrix_world = obj.matrix_world
        return [[x for x in c] for c in matrix_world]
    elif attribute_name.startswith("cp_"):
        custom_property_name = attribute_name[len("cp_"):]
        # Make sure the requested custom property exist
        if custom_property_name in obj:
            return obj[custom_property_name]
        else:
            raise Exception("No such custom property: " + custom_property_name)
    else:
        raise Exception("No such attribute: " + attribute_name)

def get_attributes(obj, attribute_names):
    value_list_per_obj = {}
    for attribute_name in attribute_names:
        value = get_attribute(obj, attribute_name)
        if isinstance(value, mathutils.Vector) or isinstance(value, mathutils.Euler):
            value = list(value)
        value_list_per_obj[attribute_name] = value
    return value_list_per_obj

print("Render layouts.")
hide_names = []
skipped_names =[]
objects_to_save = {}
for obj in loaded_objects:
    obj_name = obj.get_name().lower()
    obj_category_id = obj.get_cp("category_id")
    if obj_category_id in things_ids and obj.get_cp("uid") != obj.get_cp("instanceid"):
        hide_names.append(obj_name)
        obj.hide(True)
        value_list_per_obj = get_attributes(obj.blender_obj, ["id", "name", "cp_uid",  "location", "cp_jid", "rotation_euler", "matrix_world", "cp_instanceid"])
        objects_to_save[obj_name] = value_list_per_obj
    else:
        skipped_names.append(obj_name)
        obj.hide(False)

objects_path = os.path.join(args.output_dir, "object.json")
with open(objects_path, "w") as f:
    json.dump(objects_to_save, f)
print("Saving objects to %s"%objects_path)
data = bproc.renderer.render()
seg_data = bproc.renderer.render_segmap(map_by=["class"])
data.update(seg_data)
bproc.writer.write_hdf5(os.path.join(args.output_dir, "layout"), data)
print(sorted(hide_names))
print(sorted(skipped_names))
print(sorted(category_ids))
print(sorted(things_ids))
