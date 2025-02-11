import random
import os
import cv2
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
db_files = [f"{data_root}/nuplan-v1.1/splits/mini/2021.10.06.17.43.07_veh-28_00508_00877.db"]
map_root = f"{data_root}/maps"
map_version = 'nuplan-maps-v1.0'

# ==============================
# 2️⃣ INITIALIZE Scenario Builder
# ==============================
scenario_builder = NuPlanScenarioBuilder(
    data_root=data_root,
    map_root=map_root,
    sensor_root=None,  # Disable sensor data
    db_files=db_files,
    map_version=map_version
)

# Create a scenario filter to limit the scenarios to a small subset
scenario_filter = ScenarioFilter(
    scenario_tokens=None,             # List of specific scenario tokens to include; set to None to include all
    log_names=None,                   # List of log names to filter; set to None to include all
    scenario_types=None,              # List of scenario types to filter; set to None to include all
    map_names=None,                   # List of map names to filter; set to None to include all
    num_scenarios_per_type=None,      # Number of scenarios per type; set to None to include all
    limit_total_scenarios=None,       # Total number of scenarios to limit; set to None for no limit
    timestamp_threshold_s=20,         # Minimum scenario duration in seconds
    ego_displacement_minimum_m=1.0,   # Minimum displacement of the ego vehicle in meters
    expand_scenarios=False,           # Whether to expand scenarios; set to False if not needed
    remove_invalid_goals=True,        # Whether to remove scenarios with invalid goals
    shuffle=False                     # Whether to shuffle the scenarios
)

# Retrieve scenarios
scenarios = scenario_builder.get_scenarios(scenario_filter, SingleMachineParallelExecutor())
scenario = scenarios[0]  # Select the first scenario

# ==============================
# 4️⃣ PLAY 5 CONSECUTIVE TICKS
# ==============================
start_iteration = 25  # Define starting iteration
lane_id_to_geometry = utils.reason_route_intent(scenario)
for iteration in range(start_iteration, scenario.get_number_of_iterations()):
    print(f"====== new tick ======= {iteration}")
    curr_folder = "images/timestamp_" + str(iteration).zfill(6) + "/"
    os.makedirs(curr_folder, exist_ok=True)
    
    config = raster_tile_generator.ImageConfig(
        scenario=scenario,
        iter=iteration,
        image_size=400,
        resolution=0.2,
        save_folder=curr_folder,
        angle_noise=random.uniform(-25., 25.)
    )

    # Get all images
    images: Dict[str, List[np.ndarray]] = {
        'ego_box': [raster_tile_generator.generate_ego_box(config)],
        'roadmap': [raster_tile_generator.generate_roadmap(config)],
        'intent': [raster_tile_generator.generate_intent_map(config, lane_id_to_geometry)],
        'speed_limit': [raster_tile_generator.generate_speed_limit_map(config)],
        'past_ego': [raster_tile_generator.generate_past_ego_poses(config)],
        'traffic_lights': raster_tile_generator.generate_traffic_lights_map(config),
        'tracked_objects': raster_tile_generator.generate_past_tracked_objects_map(config),
        'future_ego': [raster_tile_generator.generate_future_ego_poses(config)]
    }

    # Save individual images
    for name, img_list in images.items():
        for i, img in enumerate(img_list):
            config.save_image(img, f"{name}_{i:03d}.png")

    # Create list to hold all channels
    channels = []
    
    # Collect all channels from all images and all frames
    for name, img_list in images.items():
        if name == "future_ego":
            continue
        for img in img_list:
            # If image is 3-channel, split it into separate channels
            if len(img.shape) == 3 and img.shape[2] == 3:
                channels.extend([img[:,:,i] for i in range(3)])
            else:
                # If image is single channel, add it directly
                channels.append(img.squeeze())
    
    # Stack all channels into a single multi-channel image
    stacked_image = np.dstack(channels)
    
    # Save the stacked image with compression
    np.savez_compressed(os.path.join(curr_folder, "stacked.npz"), stacked_image)

