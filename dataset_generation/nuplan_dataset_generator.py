import cv2
import glob
import random
import os
import numpy as np
from typing import Dict, List

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

import utils
import raster_tile_generator


# ==============================
# 1️⃣ CONFIGURATION - Set Paths
# ==============================
data_root = "/Users/nuocheng/nuplan/dataset"
db_folder = f"{data_root}/nuplan-v1.1/splits/mini/"
map_root = f"{data_root}/maps"
map_version = 'nuplan-maps-v1.0'

# Get all .db files in the folder
db_files = glob.glob(os.path.join(db_folder, "*.db"))
print(f"Found {len(db_files)} database files")

scenario_filter = ScenarioFilter(
    scenario_tokens=None,
    log_names=None,
    scenario_types=None,
    map_names=None,
    num_scenarios_per_type=None,
    limit_total_scenarios=None,
    timestamp_threshold_s=20,
    ego_displacement_minimum_m=1.0,
    expand_scenarios=False,
    remove_invalid_goals=True,
    shuffle=False
)

for db_file in db_files:
    print(f"\nProcessing database: {os.path.basename(db_file)}")
    
    # ==============================
    # 2️⃣ INITIALIZE Scenario Builder
    # ==============================
    scenario_builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        sensor_root=None,  # Disable sensor data
        db_files=[db_file],  # Process one db file at a time
        map_version=map_version
    )

    # Retrieve scenarios
    scenarios = scenario_builder.get_scenarios(scenario_filter, SingleMachineParallelExecutor())
    if len(scenarios) == 0:
        print(f"No scenarios found for {os.path.basename(db_file)}")
        continue
    scenario = scenarios[0]  # Select the first scenario

    # Create a folder for this db file
    db_name = os.path.splitext(os.path.basename(db_file))[0]
    base_folder = f"../images/{db_name}/"
    os.makedirs(base_folder, exist_ok=True)

    # ==============================
    # 4️⃣ PLAY CONSECUTIVE TICKS
    # ==============================
    start_iteration = 25  # Define starting iteration
    lane_id_to_geometry = utils.reason_route_intent(scenario)
    
    for iteration in range(start_iteration, scenario.get_number_of_iterations()):
        print(f"====== new tick ======= {iteration}")
        curr_folder = os.path.join(base_folder, f"timestamp_{str(iteration).zfill(6)}/")
        print(curr_folder)
        os.makedirs(curr_folder, exist_ok=True)
        
        config = raster_tile_generator.ImageConfig(
            scenario=scenario,
            iter=iteration,
            image_size=800,
            save_image_size = 400,
            vertical_shift_pixel = 50,
            resolution=0.2,
            save_folder=curr_folder,
            angle_noise=random.uniform(-25., 25.),
            delta_time=0.2,
            past_pose_time_horizon=8.,
            future_pose_time_horizon=2.,
            past_observation_time_horizon=1.
        )

        # Get all images
        waypoints, waypoint_images = raster_tile_generator.generate_future_ego_poses(config)
        data: Dict[str, List[np.ndarray]] = {
            'future_waypoints': [waypoints],
            'future_waypoint_images': waypoint_images
        }
        images: Dict[str, List[np.ndarray]] = {
            'ego_box': [raster_tile_generator.generate_ego_box(config)],
            'roadmap': [raster_tile_generator.generate_roadmap(config)],
            'intent': [raster_tile_generator.generate_intent_map(config, lane_id_to_geometry)],
            'speed_limit': [raster_tile_generator.generate_speed_limit_map(config)],
            'past_ego': [raster_tile_generator.generate_past_ego_poses(config)],
            'traffic_lights': raster_tile_generator.generate_traffic_lights_map(config),
            'tracked_objects': raster_tile_generator.generate_past_tracked_objects_map(config)
        }

        # for each image, shift it vertically by vertical_shift_pixel and crop it to save_image_size
        # the cropped image should be centered by the original image
        for name, img_list in images.items():
            for i, img in enumerate(img_list):
                images[name][i] = utils.crop_and_shift_image(img, config.vertical_shift_pixel, config.save_image_size)
        for name, img_list in data.items():
            if name != 'future_waypoint_images':
                continue
            for i, img in enumerate(img_list):
                data[name][i] = utils.crop_and_shift_image(img, config.vertical_shift_pixel, config.save_image_size)

        # the future waypoints are saved in the original image dimension as well, so we need to shift them accordingly
        for i, waypoints in enumerate(data['future_waypoints']):
            pixel_shift = (config.image_size - config.save_image_size) / 2
            for j, waypoint in enumerate(waypoints):
                waypoint[0][0] = waypoint[0][0] - pixel_shift
                waypoint[0][1] = waypoint[0][1] - pixel_shift + config.vertical_shift_pixel

        # Save individual images
        for name, img_list in images.items():
            for i, img in enumerate(img_list):
                config.save_image(img, f"{name}_{i:03d}.png")
        # Save individual data
        for name, img_list in data.items():
            if name != 'future_waypoint_images':
                continue
            for i, img in enumerate(img_list):
                config.save_image(img, f"{name}_{i:03d}.png")

        # Create list to hold all channels
        channels = []
        
        # Collect all channels from all images and all frames
        for name, img_list in images.items():
            for img in img_list:
                # If image is 3-channel, split it into separate channels
                if len(img.shape) == 3 and img.shape[2] == 3:
                    channels.extend([img[:,:,i] for i in range(3)])
                else:
                    # If image is single channel, add it directly
                    channels.append(img.squeeze())
        
        # Stack all channels into a single multi-channel image
        stacked_image = np.dstack(channels)
        
        # Save both stacked image with compression and waypoints into the same .npz file with different keys
        np.savez_compressed(os.path.join(curr_folder, f"data_{db_name}_{iteration}.npz"), images=stacked_image, future_waypoints=data["future_waypoint_images"])
