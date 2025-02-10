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
# 4️⃣ EXTRACT MAP FEATURES
# ==============================
map_api = scenario.map_api

def plot_map_features(ax, lane_coords, lane_connector_coords, roadblock_coords, roadblock_connector_coords, nearby_objects, ego_box, traffic_lights):
    ax.clear()
    
    # Define traffic light colors
    traffic_light_colors = {
        "RED": "red",
        "YELLOW": "yellow",
        "GREEN": "green"
    }
    
    # Plot roadblocks
    for roadblock in roadblock_coords:
        ax.fill(roadblock[:, 0], roadblock[:, 1], color="gray", alpha=1.0)
    
    # Plot roadblock connectors
    for roadblock_connector in roadblock_connector_coords:
        ax.fill(roadblock_connector[:, 0], roadblock_connector[:, 1], color="brown", alpha=1.0)

    # Plot lanes with traffic light colors if applicable
    for lane_id, lane in lane_coords.items():
        color = traffic_light_colors.get(traffic_lights.get(int(lane_id), "green"), "green")
        ax.plot(lane[:, 0], lane[:, 1], linestyle="-", color=color, linewidth=2)
    
    # Plot lane connectors
    for conector_id, connector in lane_connector_coords.items():
        # ax.plot(connector[:, 0], connector[:, 1], linestyle="--", color="green")
        color = traffic_light_colors.get(traffic_lights.get(int(conector_id), "green"), "green")
        ax.plot(connector[:, 0], connector[:, 1], linestyle="-", color=color, linewidth=2)
    
    
    # Plot ego vehicle
    ego_polygon = np.array([list(corner) for corner in ego_box.geometry.exterior.coords])
    ax.fill(ego_polygon[:, 0], ego_polygon[:, 1], color="red", alpha=0.5)
    
    # Plot nearby objects
    for obj in nearby_objects:
        obj_polygon = np.array([list(corner) for corner in obj.geometry.exterior.coords])
        ax.fill(obj_polygon[:, 0], obj_polygon[:, 1], alpha=0.5)
    
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("nuPlan Map Features and Traffic Lights")
    ax.grid()
    plt.pause(0.1)


# ==============================
# 4️⃣ PLAY 5 CONSECUTIVE TICKS
# ==============================
start_iteration = 25  # Define starting iteration
num_ticks = 200  # Number of consecutive ticks
fig, ax = plt.subplots(figsize=(10, 8))

lane_id_to_geometry = utils.reason_route_intent(scenario)

for iteration in range(start_iteration, scenario.get_number_of_iterations()):
    print("====== new tick =======")
    ego_state = scenario.get_ego_state_at_iteration(iteration)
    ego_box = ego_state.car_footprint.oriented_box
    
    lane_centerlines = utils.get_lane_centerlines(ego_box, map_api)
    lane_connectors = utils.get_lane_connectors(ego_box, map_api)
    roadblocks = utils.get_roadblocks(ego_box, map_api)
    roadblock_connectors = utils.get_roadblock_connectors(ego_box, map_api)
    nearby_objects = utils.get_nearby_objects(scenario, iteration)
    traffic_lights = utils.get_traffic_light_status(scenario, iteration)

    # plot_map_features(ax, lane_centerlines, lane_connectors, roadblocks, roadblock_connectors, nearby_objects, ego_box, traffic_lights)
    # time.sleep(0.1)

    curr_folder = "images/timestamp_" + str(iteration).zfill(6) + "/"
    os.makedirs(curr_folder, exist_ok=True)

    angle_noise = random.uniform(-25, 25)
    image_size = 400
    image_resolution = 0.2
    # single tick
    utils.generate_ego_box(scenario, iteration, image_size, image_resolution, curr_folder, angle_noise)
    # single tick
    utils.generate_roadmap(scenario, iteration, image_size, image_resolution, curr_folder, angle_noise)
    # single tick
    utils.generate_intent_map(scenario, iteration, lane_id_to_geometry, image_size, image_resolution, curr_folder, angle_noise)
    # single tick
    utils.generate_speed_limit_map(scenario, iteration, image_size, image_resolution, curr_folder, angle_noise)

    # multiple ticks
    utils.generate_past_ego_poses(scenario, iteration, image_size, image_resolution, curr_folder, angle_noise)
    # multiple ticks
    utils.generate_traffic_lights_map(scenario, iteration, 1., image_size, image_resolution, curr_folder, angle_noise)
    # multiple ticks
    utils.generate_past_tracked_objects_map(scenario, iteration, 1., image_size, image_resolution, curr_folder, angle_noise)

    ####  labe ###
    # multiple ticks
    utils.generate_future_ego_poses(scenario, iteration, image_size, image_resolution, curr_folder, angle_noise)
    
# plt.show()

    # # Get the mission goal (destination)
    # mission_goal = scenario.get_mission_goal()
    # print(f"Mission Goal: {mission_goal}")

    # # Get the planned route (list of lane IDs)
    # planned_route = scenario.get_route_roadblock_ids()
    # print(f"Planned Route (Lane IDs): {planned_route}")
