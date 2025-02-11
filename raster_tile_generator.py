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
        self.rotation_matrix = utils.get_rotation_matrix(self.ego_state, self.image_size, self.angle_noise)
    
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
    
    lanes = utils.get_lanes_around_point(config.ego_box, config.map_api)
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
        
        config.save_image(image, f"/traffic_lights_-{i:03d}.png")

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

    config.save_image(image, "/roadmap.png")

def generate_intent_map(config: ImageConfig, lane_id_to_geometry):
    image = config.create_image(1)
    
    for lane_id, geometry in lane_id_to_geometry.items():
        polygon = config.world_to_image_coords(geometry)
        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], color=255)
    
    config.save_image(image, "/intent.png")

