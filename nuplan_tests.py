import matplotlib.pyplot as plt
import numpy as np
import random
import os
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

import utils

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
    print("====== new tick =======")
    curr_folder = "images/timestamp_" + str(iteration).zfill(6) + "/"
    os.makedirs(curr_folder, exist_ok=True)
    # Create config once per iteration
    config = utils.ImageConfig(
        scenario=scenario,
        iter=iteration,
        image_size=400,
        resolution=0.2,
        save_folder=curr_folder,
        angle_noise=random.uniform(-25., 25.)
    )
    # single tick
    utils.generate_ego_box(config)
    # single tick
    utils.generate_roadmap(config)
    # single tick
    utils.generate_intent_map(config, lane_id_to_geometry)
    # single tick
    utils.generate_speed_limit_map(config)

    # multiple ticks
    utils.generate_past_ego_poses(config)
    # multiple ticks
    utils.generate_traffic_lights_map(config)
    # multiple ticks
    utils.generate_past_tracked_objects_map(config)

    ####  labe ###
    # multiple ticks
    utils.generate_future_ego_poses(config)

