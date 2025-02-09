from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
import matplotlib.pyplot as plt
import numpy as np


# Define dataset paths (no need for sensor_root)
data_root = "/Users/nuocheng/nuplan/dataset"
db_files = [f"{data_root}/nuplan-v1.1/splits/mini/2021.10.06.17.43.07_veh-28_00508_00877.db"]
map_root = f"{data_root}/maps"
map_version = 'nuplan-maps-v1.0'

# Initialize scenario builder (No sensor data required)
scenario_builder = NuPlanScenarioBuilder(
    data_root=data_root,
    map_root=map_root,
    sensor_root=None,  # Disable sensor data
    db_files=db_files,
    map_version=map_version
)

# Define scenario filter (adjust as needed)
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

# Use SingleThreadedWorker instead of worker=0
worker = SingleMachineParallelExecutor()

# Load scenarios
all_scenarios = scenario_builder.get_scenarios(scenario_filter=scenario_filter, worker=worker)

# Extract scenario tokens
scenario_tokens = [scenario.token for scenario in all_scenarios]


print(f"Available scenario tokens: {scenario_tokens[:5]}")  # Print first 5 scenario tokens

# Load a scenario
scenario = all_scenarios[0]
scenario_token = scenario_tokens[0]

# Extract planning-relevant data
# Assuming 'scenario' is an instance of a scenario object
iteration = 10  # Specify the desired iteration
time_horizon = 5.0  # Specify the time horizon in seconds
ego_past_trajectory = scenario.get_ego_past_trajectory(iteration, time_horizon)
ego_future_trajectory = scenario.get_ego_future_trajectory(iteration, time_horizon)
dynamic_objects = scenario.get_tracked_objects_at_iteration(0)
# map_features = scenario.get_map()

print(f"Loaded scenario: {scenario_token}")
print(f"Past trajectory shape: {ego_past_trajectory}")
print(f"Future trajectory shape: {ego_future_trajectory}")
print(f"Number of tracked objects: {len(dynamic_objects.tracked_objects)}")

import numpy as np


# Extract (x, y) positions and heading from ego past trajectory
past_positions = np.array([[state.car_footprint.oriented_box.center.x,
                            state.car_footprint.oriented_box.center.y,
                            state.car_footprint.oriented_box.center.heading]
                           for state in ego_past_trajectory])

# Display the extracted positions and orientations
print("Ego Past Positions and Orientations:")
# print(past_positions)

# Extract velocity and acceleration
past_velocity = np.array([state.dynamic_car_state.speed for state in ego_past_trajectory])
past_acceleration = np.array([state.dynamic_car_state.acceleration for state in ego_past_trajectory])

# Display the extracted velocities and accelerations
print("Past Velocities (m/s):", past_velocity)
print("Past Accelerations (m/s²):", past_acceleration)

# Define iteration (time step)
iteration = 50  # Change based on the scenario's length

# Get all tracked objects (vehicles, pedestrians, cyclists) at the specified iteration
tracked_objects = scenario.get_tracked_objects_at_iteration(iteration)

# Print number of tracked objects
print(f"Total nearby tracks at iteration {iteration}: {len(tracked_objects.tracked_objects)}")

# Extract relevant details from nearby tracks
nearby_tracks = []
for obj in tracked_objects.tracked_objects:
    obj_id = obj.track_token  # Unique identifier
    obj_type = obj.tracked_object_type  # Object type (Vehicle, Pedestrian, Cyclist)
    x, y = obj.box.center.x, obj.box.center.y  # Position
    speed = obj.velocity.magnitude()  # Speed (m/s)
    
    nearby_tracks.append([obj_id, obj_type, x, y, speed])

# Convert to NumPy array for easy manipulation
nearby_tracks = np.array(nearby_tracks, dtype=object)

print("Nearby Tracks (ID, Type, X, Y, Speed):")
print(nearby_tracks)

# Get ego vehicle position
ego_x, ego_y = scenario.get_ego_state_at_iteration(iteration).car_footprint.oriented_box.center.x, \
               scenario.get_ego_state_at_iteration(iteration).car_footprint.oriented_box.center.y

# Define a distance threshold (e.g., 30 meters)
distance_threshold = 100.0  

# Filter nearby objects
filtered_tracks = [track for track in nearby_tracks if np.linalg.norm([track[2] - ego_x, track[3] - ego_y]) < distance_threshold]

print("Filtered Nearby Tracks (within 30m):")
print(filtered_tracks)


import matplotlib.pyplot as plt

# Plot ego vehicle
plt.scatter(ego_x, ego_y, color='red', label="Ego Vehicle", marker="x", s=100)

# # Plot nearby tracks
# for track in filtered_tracks:
#     plt.scatter(track[2], track[3], label=str(track[1]), alpha=0.6)

# Get the map API associated with the scenario
map_api = scenario.map_api

# Print the type of the map
print(f"Map API: {type(map_api)}")

# Define the iteration (time step) of interest
iteration = 50  # Adjust based on your scenario's timeline

# Retrieve the ego vehicle's state at the specified iteration
ego_state = scenario.get_ego_state_at_iteration(iteration)

# Extract the ego vehicle's position
ego_x = ego_state.car_footprint.oriented_box.center.x
ego_y = ego_state.car_footprint.oriented_box.center.y

# Define a search radius in meters
search_radius = 50.0

#### =====
# Retrieve lane centerlines within the specified radius around the ego vehicle's position
all_lanes = map_api.get_proximal_map_objects(Point2D(ego_x, ego_y), 50., [SemanticMapLayer.LANE])[SemanticMapLayer.LANE]

# Print number of lanes found
print(f"Total lanes found: {len(all_lanes)}")

# Check if lanes exist
if all_lanes:
    first_lane = all_lanes[0]  # Access the first lane object
    print(f"First lane object: {first_lane}")
else:
    print("No lanes found in the map.")


# Extract lane coordinates
lane_coords = [
    np.array([[pt.x, pt.y] for pt in lane.baseline_path.discrete_path])
    for lane in all_lanes
]

#### =====
# Retrieve lane centerlines within the specified radius around the ego vehicle's position
all_lanes_connector = map_api.get_proximal_map_objects(Point2D(ego_x, ego_y), 50., [SemanticMapLayer.LANE_CONNECTOR])[SemanticMapLayer.LANE_CONNECTOR]

# Print number of lanes found
print(f"Total lanes found: {len(all_lanes_connector)}")

# Check if lanes exist
if all_lanes_connector:
    first_lane = all_lanes[0]  # Access the first lane object
    print(f"First lane object: {first_lane}")
else:
    print("No lanes found in the map.")


# Extract lane coordinates
lane_connector_coords = [
    np.array([[pt.x, pt.y] for pt in lane.baseline_path.discrete_path])
    for lane in all_lanes_connector
]

#### =====

# Retrieve lane centerlines within the specified radius around the ego vehicle's position
roadblocks = map_api.get_proximal_map_objects(Point2D(ego_x, ego_y), 50., [SemanticMapLayer.ROADBLOCK])[SemanticMapLayer.ROADBLOCK]

# Print number of lanes found
print(f"Total road blocks found: {len(roadblocks)}")

# # Check if lanes exist
# if roadblocks:
#     first_block = roadblocks[0]  # Access the first lane object
#     print(f"First road block object: {first_block}")
# else:
#     print("No road block found in the map.")

# Assuming 'roadblocks' is a list of roadblock objects
roadblock_coords = [
    np.array(list(roadblock.polygon.exterior.coords)) for roadblock in roadblocks
]

# # Print the first roadblock's coordinates
# if roadblock_coords:
#     print(f"First roadblock coordinates:\n{roadblock_coords[0]}")
# else:
#     print("No roadblocks found in the map.")



# Retrieve lane centerlines within the specified radius around the ego vehicle's position
roadblocks_connectors = map_api.get_proximal_map_objects(Point2D(ego_x, ego_y), 50., [SemanticMapLayer.ROADBLOCK_CONNECTOR])[SemanticMapLayer.ROADBLOCK_CONNECTOR]

# Print number of lanes found
print(f"Total road blocks found: {len(roadblocks_connectors)}")

# # Check if lanes exist
# if roadblocks_connectors:
#     first_block = roadblocks_connectors[0]  # Access the first lane object
#     print(f"First road block object: {first_block}")
# else:
#     print("No road block found in the map.")

# Assuming 'roadblocks' is a list of roadblock objects
roadblock_connector_coords = [
    np.array(list(roadblock_connector.polygon.exterior.coords)) for roadblock_connector in roadblocks_connectors
]

# Print the first roadblock's coordinates
# if roadblock_connector_coords:
#     print(f"First roadblock coordinates:\n{roadblock_connector_coords[0]}")
# else:
#     print("No roadblocks found in the map.")


# Plot all roadblock connectro boundaries
for roadblock in roadblock_connector_coords:
    plt.fill(roadblock[:, 0], roadblock[:, 1], color="yellow", alpha=0.5)

# Plot all roadblock boundaries
for roadblock in roadblock_coords:
    plt.fill(roadblock[:, 0], roadblock[:, 1], color="gray", alpha=0.5)

# Plot all lanes
for lane in lane_coords:
    plt.plot(lane[:, 0], lane[:, 1], linestyle="--", color="blue")
# Plot all lanes
for lane in lane_connector_coords:
    plt.plot(lane[:, 0], lane[:, 1], linestyle="--", color="green")

# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.title("Roadblocks in nuPlan Map")
# plt.grid()
# plt.show()

# # Convert world coordinates (meters) to raster indices
# ego_pixel_x, ego_pixel_y = map_api.get_raster_coord_from_world_coord(ego_x, ego_y)

# # Print pixel coordinates
# print(f"Ego vehicle raster indices: x={ego_pixel_x}, y={ego_pixel_y}")

# # Retrieve all drivable area polygons
# drivable_area_raster = map_api.get_raster_map_layer(SemanticMapLayer.DRIVABLE_AREA)

# # Convert to a numeric type (uint8 for grayscale, or float32)
# drivable_area_raster = np.array(drivable_area_raster.data, dtype=np.uint8)  # Convert object dtype to uint8

# # Plot the drivable area
# plt.imshow(drivable_area_raster, cmap="gray")
# plt.title("Drivable Area in nuPlan Map")
# plt.axis("off")  # Hide axis
# plt.show()


traffic_lights = scenario.get_traffic_light_status_at_iteration(iteration)

# Iterate over the generator to access traffic light details
for tl_status in traffic_lights:
    print(f"Lane Connector ID: {tl_status.lane_connector_id}, Color: {tl_status.status}, Timestamp: {tl_status.timestamp}")

# Extract stop line positions of traffic lights
stop_lines = [map_api.get_map_object(tl.lane_connector_id, SemanticMapLayer.STOP_LINE) for tl in traffic_lights]

# Print stop line positions
for i, stop_line in enumerate(stop_lines):
    if stop_line:
        print(f"Traffic Light {i}: Stop Line Position: {[pt.x, pt.y] for pt in stop_line.line} ")

# Define colors for traffic lights
traffic_light_colors = {
    "RED": "red",
    "YELLOW": "yellow",
    "GREEN": "green"
}

# Plot traffic lights on the map
for tl in traffic_lights:
    stop_line = map_api.get_map_object(tl.lane_connector_id, SemanticMapLayer.STOP_LINE)
    if stop_line:
        stop_x, stop_y = zip(*[(pt.x, pt.y) for pt in stop_line.line])
        plt.scatter(stop_x, stop_y, color=traffic_light_colors.get(tl.status.name, "gray"), label=f"Traffic Light {tl.status.name}")

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Traffic Light Positions")
plt.legend()
plt.grid()
plt.show()
