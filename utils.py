import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

@dataclass
class ImageConfig:
    """Configuration class for image generation parameters"""
    scenario: any  # NuPlanScenario
    iter: int
    image_size: int = 1024
    resolution: float = 0.1
    save_folder: str = "images/"
    angle_noise: float = 0.0
    
    def __post_init__(self):
        # Get map API and ego state
        self.map_api = self.scenario.map_api
        self.ego_state = self.scenario.get_ego_state_at_iteration(self.iter)
        self.ego_x = self.ego_state.car_footprint.center.x
        self.ego_y = self.ego_state.car_footprint.center.y
        self.ego_box = self.ego_state.car_footprint.oriented_box
        
        # Calculate image boundaries
        self.half_size_meters = (self.image_size / 2) * self.resolution
        self.min_x = self.ego_x - self.half_size_meters
        self.min_y = self.ego_y - self.half_size_meters
        
        # Get rotation matrix
        self.rotation_matrix = get_rotation_matrix(self.ego_state, self.image_size, self.angle_noise)
    
    def create_image(self, channel_num):
        return np.zeros((self.image_size, self.image_size, channel_num), dtype=np.uint8)
    
    def world_to_image_coords(self, points) -> np.ndarray:
        """Convert world coordinates to image coordinates"""
        # Handle both Point2D objects and raw coordinates
        if hasattr(points[0], 'x'):  # If points are Point2D objects
            return np.array([
                [(pt.x - self.min_x) / self.resolution, 
                 (self.image_size - (pt.y - self.min_y) / self.resolution)]
                for pt in points
            ], dtype=np.int32)
        else:  # If points are raw coordinates
            return np.array([
                [(x - self.min_x) / self.resolution, 
                 (self.image_size - (y - self.min_y) / self.resolution)]
                for x, y in points
            ], dtype=np.int32)
    
    def save_image(self, image: np.ndarray, filename: str) -> None:
        """Save the image with rotation applied"""
        rotated = cv2.warpAffine(image, self.rotation_matrix, (self.image_size, self.image_size))
        save_path = self.save_folder + filename
        cv2.imwrite(save_path, rotated)
        print(f"Image saved to {save_path}")

def get_lanes_around_point(ego_box, map_api, radius=50.0):
    """Get all lanes within a radius of the ego vehicle."""
    ego_point = Point2D(ego_box.center.x, ego_box.center.y)
    
    # Get both regular lanes and lane connectors
    lanes = map_api.get_proximal_map_objects(ego_point, radius, [SemanticMapLayer.LANE])[SemanticMapLayer.LANE]
    lane_connectors = map_api.get_proximal_map_objects(ego_point, radius, [SemanticMapLayer.LANE_CONNECTOR])[SemanticMapLayer.LANE_CONNECTOR]
    
    return list(lanes) + list(lane_connectors)

# Function to extract proximal map objects
def get_proximal_map_objects(layer, map_api, radius=50.0, ego_box=None):
    return map_api.get_proximal_map_objects(Point2D(ego_box.center.x, ego_box.center.y), radius, [layer])[layer]

def get_lane_left_boundary(ego_box, map_api):
    lanes = get_proximal_map_objects(SemanticMapLayer.LANE, map_api, ego_box=ego_box)
    lane_dict = {lane.id: np.array([[pt.x, pt.y] for pt in lane.left_boundary.discrete_path]) for lane in lanes}
    return lane_dict

def get_lane_right_boundary(ego_box, map_api):
    lanes = get_proximal_map_objects(SemanticMapLayer.LANE, map_api, ego_box=ego_box)
    lane_dict = {lane.id: np.array([[pt.x, pt.y] for pt in lane.right_boundary.discrete_path]) for lane in lanes}
    return lane_dict

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


def get_traffic_light_status(scenario, iteration):
    traffic_light_statuses = scenario.get_traffic_light_status_at_iteration(iteration)
    traffic_lights = {tl.lane_connector_id: tl.status.name for tl in traffic_light_statuses}
    print(f"Traffic Light Statuses: {traffic_lights}")
    return traffic_lights

def get_rotation_matrix(ego_state, image_size, angle_noise=0.):
    ego_heading = -ego_state.car_footprint.center.heading * 180. / 3.14 + 90. + angle_noise
    rotation_center = (image_size / 2, image_size / 2)
    return cv2.getRotationMatrix2D(rotation_center, ego_heading, 1.0)


def generate_past_ego_poses(config: ImageConfig):
    image = config.create_image(1)

    past_ego_states = config.scenario.get_ego_past_trajectory(config.iter, time_horizon=8.)
    step_size = int(0.2 / config.scenario.database_interval)
    assert(4 == step_size)
    step_t = -1

    for past_ego_state in past_ego_states:
        step_t = (step_t + 1) % step_size
        if 0 != step_t:
            continue
        past_pose = np.array([[past_ego_state.car_footprint.center.x, past_ego_state.car_footprint.center.y]])
        polygon = config.world_to_image_coords(past_pose)

        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=255)
    
    config.save_image(image, "/past_ego_trajectory.png")

def generate_future_ego_poses(config: ImageConfig):
    image = config.create_image(1)

    step_size = int(0.2 / config.scenario.database_interval)
    assert(4 == step_size)
    future_ego_states = config.scenario.get_ego_future_trajectory(config.iter, time_horizon=2.)
    step_t = 0

    for future_ego_state in future_ego_states:
        step_t = (step_t + 1) % step_size
        if 0 != step_t:
            continue
        future_pose = np.array([[future_ego_state.car_footprint.center.x, future_ego_state.car_footprint.center.y]])
        polygon = config.world_to_image_coords(future_pose)

        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=255)
    
    config.save_image(image, "/future_ego_trajectory.png")

def generate_ego_box(config: ImageConfig):
    ego_polygon = np.array([list(corner) for corner in config.ego_box.geometry.exterior.coords])
    polygon = config.world_to_image_coords(ego_polygon)
    
    image = config.create_image(1)
    cv2.fillPoly(
        img=image,
        pts=[polygon.reshape(-1, 1, 2)],
        color=255
    )
    
    config.save_image(image, "/ego_box.png")

def generate_past_tracked_objects_map(config: ImageConfig, past_time_horizon=1.):
    step_size = int(0.2 / config.scenario.database_interval)
    past_iter = int(past_time_horizon / config.scenario.database_interval)
    
    for i in reversed(range(0, past_iter + 1, step_size)):
        curr_iter = config.iter - i
        image = config.create_image(1)

        tracked_objects = config.scenario.get_tracked_objects_at_iteration(curr_iter)
        for obj in tracked_objects.tracked_objects:
            obj_polygon = np.array([list(corner) for corner in obj.box.geometry.exterior.coords])
            polygon = config.world_to_image_coords(obj_polygon)
            cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=255)

        config.save_image(image, f"/tracked_objects_-{i:03d}.png")

def generate_speed_limit_map(config: ImageConfig):
    image = config.create_image(1)
    
    lanes = get_lanes_around_point(config.ego_box, config.map_api)
    for lane in lanes:
        centerline = config.world_to_image_coords(lane.baseline_path.discrete_path)
        # Normalize speed limit to 0-255 range (assuming max speed of 30 m/s)
        color_value = 50 if lane.speed_limit_mps is None else min(int(lane.speed_limit_mps), 255)
        cv2.polylines(image, [centerline.reshape(-1, 1, 2)], False, color_value, 2)
    
    config.save_image(image, "/speed_limit.png")


def generate_traffic_lights_map(config: ImageConfig, time_horizon=1.):
    step_size = int(0.2 / config.scenario.database_interval)
    past_iter = int(time_horizon / config.scenario.database_interval)
    
    for i in reversed(range(0, past_iter + 1, step_size)):
        curr_iter = config.iter - i
        image = config.create_image(1)
        
        traffic_lights = get_traffic_light_status(config.scenario, curr_iter)
        lanes = get_lanes_around_point(config.ego_box, config.map_api)
        
        for lane in lanes:
            centerline = config.world_to_image_coords(lane.baseline_path.discrete_path)
            color = 50
            if int(lane.id) in traffic_lights:
                if traffic_lights.get(int(lane.id)) == "RED":
                    color = 255
                elif traffic_lights.get(int(lane.id)) == "YELLOW":
                    color = 150
            
            cv2.polylines(image, [centerline.reshape(-1, 1, 2)], False, color, 2)
        
        config.save_image(image, f"/traffic_lights_-{i:03d}.png")

def generate_roadmap(config: ImageConfig):
    image = config.create_image(3)
    
    # Driveable area
    roadblocks = get_roadblocks(config.ego_box, config.map_api) + get_roadblock_connectors(config.ego_box, config.map_api)
    for roadblock in roadblocks:
        polygon = config.world_to_image_coords(roadblock)
        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=(30, 30, 30))

    # Boundaries
    for boundaries in [get_lane_left_boundary(config.ego_box, config.map_api), 
                      get_lane_right_boundary(config.ego_box, config.map_api)]:
        for _, boundary in boundaries.items():
            points = config.world_to_image_coords(boundary)
            cv2.polylines(image, [points.reshape(-1, 1, 2)], False, (255, 255, 255), 1)
    
    # Map elements
    layer_colors = {
        SemanticMapLayer.CROSSWALK: (255, 0, 0),
        SemanticMapLayer.INTERSECTION: (0, 255, 0),
        SemanticMapLayer.WALKWAYS: (0, 0, 255)
    }
    
    for layer, color in layer_colors.items():
        polygons = get_map_object_exterior_polygons(config.ego_box, config.map_api, layer)
        for polygon in polygons:
            points = config.world_to_image_coords(polygon)
            cv2.polylines(image, [points.reshape(-1, 1, 2)], True, color, 1)

    config.save_image(image, "/roadmap.png")

def find_best_aligned_lane(ego_state, lanes):
    """
    Find the lane whose heading aligns best with the ego vehicle's heading.

    :param ego_state: EgoState object representing the current state of the ego vehicle.
    :param lanes: List of Lane objects from the NuPlan semantic map.
    :return: The Lane object that best matches the ego heading.
    """
    # Get ego vehicle's heading in radians
    ego_heading = ego_state.car_footprint.center.heading

    best_lane = None
    min_heading_diff = float('inf')

    for lane in lanes:
        # Get the centerline of the lane
        centerline = lane.baseline_path.discrete_path
        
        # Find the closest point on the centerline to the ego vehicle
        ego_position = ego_state.car_footprint.center
        closest_point = min(centerline, key=lambda p: np.linalg.norm([p.x - ego_position.x, p.y - ego_position.y]))

        # Compute the heading at this point
        lane_heading = closest_point.heading

        # Compute the absolute heading difference
        heading_diff = abs(np.arctan2(np.sin(lane_heading - ego_heading), np.cos(lane_heading - ego_heading)))

        # Update the best lane if this one has a smaller heading difference
        if heading_diff < min_heading_diff:
            min_heading_diff = heading_diff
            best_lane = lane

    return (best_lane, min_heading_diff)

def reason_route_intent(scenario):
    map_api = scenario.map_api
    lane_id_to_polygon = {}
    for i in range(scenario.get_number_of_iterations()):
        ego_state = scenario.get_ego_state_at_iteration(i)
        ego_x, ego_y = ego_state.car_footprint.center.x, ego_state.car_footprint.center.y

        lanes = map_api.get_all_map_objects(Point2D(ego_x, ego_y), SemanticMapLayer.LANE)
        best_lane, best_heading = find_best_aligned_lane(ego_state, lanes)

        lane_connectors = map_api.get_all_map_objects(Point2D(ego_x, ego_y), SemanticMapLayer.LANE_CONNECTOR)
        best_lane_c, best_c_heading = find_best_aligned_lane(ego_state, lane_connectors)
        if None == best_lane_c and None == best_lane:
            continue

        if best_c_heading < best_heading:
            lane_id_to_polygon[best_lane_c.id] = best_lane_c.polygon.exterior.coords
        else:
            lane_id_to_polygon[best_lane.id] = best_lane.polygon.exterior.coords

    return lane_id_to_polygon

def generate_intent_map(config: ImageConfig, lane_id_to_geometry):
    image = config.create_image(1)
    
    for lane_id, geometry in lane_id_to_geometry.items():
        polygon = config.world_to_image_coords(geometry)
        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=255)
    
    config.save_image(image, "/intent.png")
