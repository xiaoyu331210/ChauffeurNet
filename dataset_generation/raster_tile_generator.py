import cv2
import numpy as np
import utils

from dataclasses import dataclass

from nuplan.common.maps.maps_datatypes import SemanticMapLayer

@dataclass
class ImageConfig:
    """Configuration class for image generation parameters"""
    scenario: any  # NuPlanScenario
    iter: int
    image_size: int = 1024
    save_image_size: int = 400
    vertical_shift_pixel: int = 0
    resolution: float = 0.1
    save_folder: str = "images/"
    angle_noise: float = 0.0
    delta_time: float = 0.2
    future_pose_time_horizon: float = 2.0
    past_pose_time_horizon: float = 8.0
    past_observation_time_horizon: float = 1.0
    
    def __post_init__(self):
        # Get map API and ego state
        self.map_api = self.scenario.map_api
        self.ego_state = self.scenario.get_ego_state_at_iteration(self.iter)
        self.ego_x = self.ego_state.car_footprint.center.x
        self.ego_y = self.ego_state.car_footprint.center.y
        self.ego_box = self.ego_state.car_footprint.oriented_box
        self.step_iter = int(self.delta_time / self.scenario.database_interval)
        
        # Calculate image boundaries
        self.half_size_meters = (self.image_size / 2) * self.resolution
        self.min_x = self.ego_x - self.half_size_meters
        self.min_y = self.ego_y - self.half_size_meters
        
        # Get rotation matrix
        self.rotation_matrix = utils.get_rotation_matrix(self.ego_state, self.image_size, self.angle_noise)
    
    def create_image(self, channel_num, dtype=np.uint8):
        return np.zeros((self.image_size, self.image_size, channel_num), dtype=dtype)
    
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

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        """Rotate the image with rotation matrix"""
        return cv2.warpAffine(image, self.rotation_matrix, (self.image_size, self.image_size))

    def save_image(self, image: np.ndarray, filename: str) -> None:
        """Save the image with rotation applied"""
        save_path = self.save_folder + filename
        cv2.imwrite(save_path, image)

def generate_past_ego_poses(config: ImageConfig):
    image = config.create_image(1)

    past_ego_states = config.scenario.get_ego_past_trajectory(config.iter, time_horizon=config.past_pose_time_horizon)
    step_t = -1

    for past_ego_state in past_ego_states:
        step_t = (step_t + 1) % config.step_iter
        if 0 != step_t:
            continue
        past_pose = np.array([[past_ego_state.car_footprint.center.x, past_ego_state.car_footprint.center.y]])
        polygon = config.world_to_image_coords(past_pose)

        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=255)
    return config.rotate_image(image)

def generate_future_ego_poses(config: ImageConfig):
    future_ego_states = config.scenario.get_ego_future_trajectory(config.iter, time_horizon=config.future_pose_time_horizon)

    step_t = 0
    waypoints = []
    waypoint_images = []
    for future_ego_state in future_ego_states:
        image = config.create_image(1, dtype=np.float32)
        step_t = (step_t + 1) % config.step_iter
        if 0 != step_t:
            continue
        footprint = future_ego_state.car_footprint
        # position
        curr_waypoint = np.array([[footprint.center.x, footprint.center.y]])
        # TODO: should record the raw points without aligning to image resolution
        curr_waypoint = config.world_to_image_coords(curr_waypoint)
        # heading
        curr_heading = utils.get_heading(future_ego_state, config.angle_noise)
        curr_waypoint = np.concatenate((curr_waypoint, [[curr_heading]]), axis=1)  # shape: (1, 3)
        # velocity
        velocity_vector = future_ego_state.dynamic_car_state.rear_axle_velocity_2d
        curr_velocity = np.sqrt(velocity_vector.x**2 + velocity_vector.y**2)
        curr_waypoint = np.concatenate((curr_waypoint, [[curr_velocity]]), axis=1)  # shape: (1, 4)

        # Apply rotation only to position coordinates
        position = curr_waypoint[0, :2].reshape(-1, 1) - config.image_size / 2  # Extract position (x,y)
        rotated_position = np.dot(config.rotation_matrix[:, :2], position).flatten() + config.image_size / 2

        # Combine rotated position with heading and velocity
        curr_waypoint[0, :2] = rotated_position
        waypoints.append(curr_waypoint)

        # add a gaussian distribution to the image
        utils.add_gaussian_distribution(image, curr_waypoint[0][1], curr_waypoint[0][0])
        waypoint_images.append(image)
    return waypoints, waypoint_images

def generate_ego_box(config: ImageConfig):
    ego_polygon = np.array([list(corner) for corner in config.ego_box.geometry.exterior.coords])
    polygon = config.world_to_image_coords(ego_polygon)
    
    image = config.create_image(1)
    cv2.fillPoly(img=image, pts=[polygon.reshape(-1, 1, 2)], color=255)
    
    return config.rotate_image(image)

def generate_past_tracked_objects_map(config: ImageConfig):
    past_iter = int(config.past_observation_time_horizon / config.scenario.database_interval)
    
    images = []
    for i in reversed(range(0, past_iter + 1, config.step_iter)):
        curr_iter = config.iter - i
        image = config.create_image(1)

        tracked_objects = config.scenario.get_tracked_objects_at_iteration(curr_iter)
        for obj in tracked_objects.tracked_objects:
            obj_polygon = np.array([list(corner) for corner in obj.box.geometry.exterior.coords])
            polygon = config.world_to_image_coords(obj_polygon)
            cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=255)

        images.append(image)
    return [config.rotate_image(image) for image in images]

def generate_speed_limit_map(config: ImageConfig):
    image = config.create_image(1)
    
    lanes = utils.get_lanes_around_point(config.ego_box, config.map_api)
    for lane in lanes:
        centerline = config.world_to_image_coords(lane.baseline_path.discrete_path)
        # Normalize speed limit to 0-255 range (assuming max speed of 30 m/s)
        color_value = 50 if lane.speed_limit_mps is None else min(int(lane.speed_limit_mps), 255)
        cv2.polylines(image, [centerline.reshape(-1, 1, 2)], False, color_value, 2)
    
    return config.rotate_image(image)

def generate_traffic_lights_map(config: ImageConfig):
    past_iter = int(config.past_observation_time_horizon / config.scenario.database_interval)
    
    images = []
    for i in reversed(range(0, past_iter + 1, config.step_iter)):
        curr_iter = config.iter - i
        image = config.create_image(1)
        
        traffic_lights = utils.get_traffic_light_status(config.scenario, curr_iter)
        lanes = utils.get_lanes_around_point(config.ego_box, config.map_api)
        
        for lane in lanes:
            centerline = config.world_to_image_coords(lane.baseline_path.discrete_path)
            color = 50
            if int(lane.id) in traffic_lights:
                if traffic_lights.get(int(lane.id)) == "RED":
                    color = 255
                elif traffic_lights.get(int(lane.id)) == "YELLOW":
                    color = 150
            
            cv2.polylines(image, [centerline.reshape(-1, 1, 2)], False, color, 2)
        
        images.append(image)
    return [config.rotate_image(image) for image in images]

def generate_roadmap(config: ImageConfig):
    image = config.create_image(3)
    
    # Driveable area
    roadblocks = utils.get_roadblocks(config.ego_box, config.map_api) + utils.get_roadblock_connectors(config.ego_box, config.map_api)
    for roadblock in roadblocks:
        polygon = config.world_to_image_coords(roadblock)
        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=(30, 30, 30))

    # Boundaries
    for boundaries in [utils.get_lane_left_boundary(config.ego_box, config.map_api), 
                      utils.get_lane_right_boundary(config.ego_box, config.map_api)]:
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
        polygons = utils.get_map_object_exterior_polygons(config.ego_box, config.map_api, layer)
        for polygon in polygons:
            points = config.world_to_image_coords(polygon)
            cv2.polylines(image, [points.reshape(-1, 1, 2)], True, color, 1)

    return config.rotate_image(image)

def generate_intent_map(config: ImageConfig, lane_id_to_geometry):
    image = config.create_image(1)
    
    for lane_id, geometry in lane_id_to_geometry.items():
        polygon = config.world_to_image_coords(geometry)
        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=255)
    
    return config.rotate_image(image)

