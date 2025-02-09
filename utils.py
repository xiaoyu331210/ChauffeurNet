import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

# Function to extract proximal map objects
def get_proximal_map_objects(layer, map_api, radius=50.0, ego_box=None):
    return map_api.get_proximal_map_objects(Point2D(ego_box.center.x, ego_box.center.y), radius, [layer])[layer]

def get_lane_centerlines(ego_box, map_api):
    lanes = get_proximal_map_objects(SemanticMapLayer.LANE, map_api, ego_box=ego_box)
    lane_dict = {lane.id: np.array([[pt.x, pt.y] for pt in lane.baseline_path.discrete_path]) for lane in lanes}
    return lane_dict

def get_lane_left_boundary(ego_box, map_api):
    lanes = get_proximal_map_objects(SemanticMapLayer.LANE, map_api, ego_box=ego_box)
    lane_dict = {lane.id: np.array([[pt.x, pt.y] for pt in lane.left_boundary.discrete_path]) for lane in lanes}
    return lane_dict

def get_lane_right_boundary(ego_box, map_api):
    lanes = get_proximal_map_objects(SemanticMapLayer.LANE, map_api, ego_box=ego_box)
    lane_dict = {lane.id: np.array([[pt.x, pt.y] for pt in lane.right_boundary.discrete_path]) for lane in lanes}
    return lane_dict

def get_lane_connectors(ego_box, map_api):
    lane_connectors = get_proximal_map_objects(SemanticMapLayer.LANE_CONNECTOR, map_api, ego_box=ego_box)
    lane_connector_dict = {connector.id: np.array([[pt.x, pt.y] for pt in connector.baseline_path.discrete_path]) for connector in lane_connectors}
    return lane_connector_dict

# Function to extract roadblocks
def get_roadblocks(ego_box, map_api):
    roadblocks = get_proximal_map_objects(SemanticMapLayer.ROADBLOCK, map_api, ego_box=ego_box)
    return [np.array(list(roadblock.polygon.exterior.coords)) for roadblock in roadblocks]

# Function to extract roadblock connectors
def get_roadblock_connectors(ego_box, map_api):
    roadblock_connectors = get_proximal_map_objects(SemanticMapLayer.ROADBLOCK_CONNECTOR, map_api, ego_box=ego_box)
    return [np.array(list(connector.polygon.exterior.coords)) for connector in roadblock_connectors]

def get_map_object_exterior_polygons(ego_box, map_api, layer):
    objects = get_proximal_map_objects(layer, map_api, ego_box=ego_box)
    return [np.array(list(object.polygon.exterior.coords)) for object in objects]

# Function to extract nearby objects
def get_nearby_objects(scenario, iteration):
    tracked_objects = scenario.get_tracked_objects_at_iteration(iteration)
    return [obj.box for obj in tracked_objects.tracked_objects]

def get_traffic_light_status(scenario, iteration):
    traffic_light_statuses = scenario.get_traffic_light_status_at_iteration(iteration)
    traffic_lights = {tl.lane_connector_id: tl.status.name for tl in traffic_light_statuses}
    print(f"Traffic Light Statuses: {traffic_lights}")
    return traffic_lights


def generate_roadmap(scenario, iter, image_size=1024, resolution=0.1, save_path="roadmap.png"):
    """
    Generates a roadmap centered at the ego vehicle position and saves it as an image.
    
    :param scenario: NuPlanScenario object
    :param image_size: Size of the output image (image_size x image_size)
    :param resolution: Resolution in meters per pixel
    :param save_path: Path to save the generated image
    """
    # Get map API
    map_api = scenario.map_api
    
    # Get ego vehicle position
    ego_state = scenario.get_ego_state_at_iteration(iter)
    ego_x, ego_y = ego_state.car_footprint.center.x, ego_state.car_footprint.center.y
    ego_box = ego_state.car_footprint.oriented_box
    
    # Define the bounding box for the image
    half_size_meters = (image_size / 2) * resolution  # Convert pixels to meters
    min_x, _ = ego_x - half_size_meters, ego_x + half_size_meters
    min_y, _ = ego_y - half_size_meters, ego_y + half_size_meters
    
    # Create a blank black image
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    
    # rasterize driveable area
    roadblocks = get_roadblocks(ego_box, map_api)
    roadblock_connectors = get_roadblock_connectors(ego_box, map_api)
    roadblocks.extend(roadblock_connectors)
    for roadblock in roadblocks:
        polygon = np.array([
            [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
            for x, y in roadblock
        ], dtype=np.int32)
        roadblock = roadblock.reshape((-1, 1, 2))
        cv2.fillPoly(
            img=image,               # The image to draw on
            pts=[polygon],    # List of polygons to fill
            color=(20, 20, 20)         # Fill color
        )

    # Plot lanes
    left_boundaries = get_lane_left_boundary(ego_box, map_api)
    for _, left_boundary in left_boundaries.items():
        points = np.array([
            [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
            for x, y in left_boundary
        ], dtype=np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(image, [points], isClosed=False, color=(255, 255, 255), thickness=1)

    right_boundaries = get_lane_right_boundary(ego_box, map_api)
    for _, right_boundary in right_boundaries.items():
        points = np.array([
            [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
            for x, y in right_boundary
        ], dtype=np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(image, [points], isClosed=False, color=(255, 255, 255), thickness=1)
    
    # plot crosswalk
    crosswalks = get_map_object_exterior_polygons(ego_box, map_api, SemanticMapLayer.CROSSWALK)
    for crosswalk in crosswalks:
        points = np.array([
            [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
            for x, y in crosswalk
        ], dtype=np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=1)
    
    # plot intersection
    intersections = get_map_object_exterior_polygons(ego_box, map_api, SemanticMapLayer.INTERSECTION)
    for intersection in intersections:
        points = np.array([
            [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
            for x, y in intersection
        ], dtype=np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=1)

    # plot walkway
    walkways = get_map_object_exterior_polygons(ego_box, map_api, SemanticMapLayer.WALKWAYS)
    for walkway in walkways:
        points = np.array([
            [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
            for x, y in walkway
        ], dtype=np.int32)
        points = points.reshape((-1,1,2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=1)

    # Save the image
    cv2.imwrite(save_path, image)
    print(f"Roadmap saved to {save_path}")




