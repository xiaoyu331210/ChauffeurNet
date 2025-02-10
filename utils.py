import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

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

def merge_common_boundaries(left_boundaries: dict, right_boundaries: dict) -> dict:
    """
    Finds IDs that appear in both left_boundaries and right_boundaries,
    and returns a dictionary mapping each common ID to a tuple of
    (left_geometry, right_geometry).
    """
    # Find common keys
    common_ids = set(left_boundaries.keys()).intersection(right_boundaries.keys())
    
    # Build output dict
    merged = {}
    for boundary_id in common_ids:
        merged[boundary_id] = left_boundaries[boundary_id]
    
    return merged

def get_rotation_matrix(ego_state, image_size, angle_noise=0.):
    ego_heading = -ego_state.car_footprint.center.heading * 180. / 3.14 + 90. + angle_noise
    rotation_center = (image_size / 2, image_size / 2)
    return cv2.getRotationMatrix2D(rotation_center, ego_heading, 1.0)

def generate_past_ego_poses(scenario, iter, image_size=1024, resolution=0.1, save_folder="images/", angle_noise=0.):
    # Get map API
    map_api = scenario.map_api
    
    # Get ego vehicle position
    ego_state = scenario.get_ego_state_at_iteration(iter)
    ego_x, ego_y = ego_state.car_footprint.center.x, ego_state.car_footprint.center.y
    
    # Define the bounding box for the image
    half_size_meters = (image_size / 2) * resolution  # Convert pixels to meters
    min_x, _ = ego_x - half_size_meters, ego_x + half_size_meters
    min_y, _ = ego_y - half_size_meters, ego_y + half_size_meters
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    step_size = int(0.2 / scenario.database_interval)
    assert(4 == step_size)
    past_ego_states = scenario.get_ego_past_trajectory(iter, time_horizon=8.)
    step_t = -1
    for past_ego_state in past_ego_states:
        step_t = (step_t + 1) % step_size
        if 0 != step_t:
            continue
        past_x, past_y = past_ego_state.car_footprint.center.x, past_ego_state.car_footprint.center.y
        past_x = (past_x - min_x) / resolution
        past_y = image_size - (past_y - min_y) / resolution

        cv2.circle(image, (int(past_x), int(past_y)), 1, (255, 255, 255), thickness=1)

    rotation_matrix = get_rotation_matrix(ego_state, image_size, angle_noise)
    image = cv2.warpAffine(image, rotation_matrix, (image_size, image_size))

    # Save the image
    filename = f"/past_ego_trajectory.png"  # 6-digit zero-padded format
    save_path = save_folder + filename
    cv2.imwrite(save_path, image)
    print(f"ego box saved to {save_path}")

def generate_future_ego_poses(scenario, iter, image_size=1024, resolution=0.1, save_folder="images/", angle_noise=0.):
    # Get map API
    map_api = scenario.map_api
    
    # Get ego vehicle position
    ego_state = scenario.get_ego_state_at_iteration(iter)
    ego_x, ego_y = ego_state.car_footprint.center.x, ego_state.car_footprint.center.y
    
    # Define the bounding box for the image
    half_size_meters = (image_size / 2) * resolution  # Convert pixels to meters
    min_x, _ = ego_x - half_size_meters, ego_x + half_size_meters
    min_y, _ = ego_y - half_size_meters, ego_y + half_size_meters
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    step_size = int(0.2 / scenario.database_interval)
    assert(4 == step_size)
    future_ego_states = scenario.get_ego_future_trajectory(iter, time_horizon=2.)
    step_t = 0

    for future_ego_state in future_ego_states:
        step_t = (step_t + 1) % step_size
        if 0 != step_t:
            continue
        future_x, future_y = future_ego_state.car_footprint.center.x, future_ego_state.car_footprint.center.y
        future_x = (future_x - min_x) / resolution
        future_y = image_size - (future_y - min_y) / resolution

        cv2.circle(image, (int(future_x), int(future_y)), 1, (0, 255, 0), thickness=1)

    rotation_matrix = get_rotation_matrix(ego_state, image_size, angle_noise)
    image = cv2.warpAffine(image, rotation_matrix, (image_size, image_size))

    # Save the image
    filename = f"/future_ego_trajectory.png"  # 6-digit zero-padded format
    save_path = save_folder + filename
    cv2.imwrite(save_path, image)
    print(f"ego box saved to {save_path}")

def generate_ego_box(scenario, iter, image_size=1024, resolution=0.1, save_folder="images/", angle_noise=0.):
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
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    ego_polygon = np.array([list(corner) for corner in ego_box.geometry.exterior.coords])

    polygon = np.array([
        [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
        for x, y in ego_polygon
    ], dtype=np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    cv2.fillPoly(
        img=image,               # The image to draw on
        pts=[polygon],    # List of polygons to fill
        color=(255, 255, 255)         # Fill color
    )
    rotation_matrix = get_rotation_matrix(ego_state, image_size, angle_noise)
    image = cv2.warpAffine(image, rotation_matrix, (image_size, image_size))

    # Save the image
    filename = f"/ego_box.png"  # 6-digit zero-padded format
    save_path = save_folder + filename
    cv2.imwrite(save_path, image)
    print(f"ego box saved to {save_path}")

def generate_past_tracked_objects_map(scenario, iter, past_time_horizon=1., image_size=1024, resolution=0.1, save_folder="images/", angle_noise=0.):
    # Get map API
    map_api = scenario.map_api
    
    # Get ego vehicle position
    ego_state = scenario.get_ego_state_at_iteration(iter)
    ego_x, ego_y = ego_state.car_footprint.center.x, ego_state.car_footprint.center.y
    
    # Define the bounding box for the image
    half_size_meters = (image_size / 2) * resolution  # Convert pixels to meters
    min_x, _ = ego_x - half_size_meters, ego_x + half_size_meters
    min_y, _ = ego_y - half_size_meters, ego_y + half_size_meters

    rotation_matrix = get_rotation_matrix(ego_state, image_size, angle_noise)

    step_size = int(0.2 / scenario.database_interval)

    past_iter = int(past_time_horizon / scenario.database_interval)
    for i in reversed(range(0, past_iter + 1, step_size)):
        curr_iter = iter - i
        # Create a blank black image
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        tracked_objects = scenario.get_tracked_objects_at_iteration(curr_iter)
        for obj in tracked_objects.tracked_objects:
            obj_box = obj.box
            obj_polygon = np.array([list(corner) for corner in obj_box.geometry.exterior.coords])
            polygon = np.array([
                [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
                for x, y in obj_polygon
            ], dtype=np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            cv2.fillPoly(
                img=image,               # The image to draw on
                pts=[polygon],    # List of polygons to fill
                color=(255, 255, 255)         # Fill color
            )
        image = cv2.warpAffine(image, rotation_matrix, (image_size, image_size))

        # Save the image
        filename = f"/tracked_objects_-{i:03d}.png"  # 6-digit zero-padded format
        save_path = save_folder + filename
        cv2.imwrite(save_path, image)
        print(f"tracked objects saved to {save_path}")


def generate_speed_limit_map(scenario, iter, image_size=1024, resolution=0.1, save_folder="images/", angle_noise=0.):
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
    
    rotation_matrix = get_rotation_matrix(ego_state, image_size, angle_noise)

    # Create a blank black image
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # plog lane
    lanes = get_proximal_map_objects(SemanticMapLayer.LANE, map_api, ego_box=ego_box)
    lane_connectors = get_proximal_map_objects(SemanticMapLayer.LANE_CONNECTOR, map_api, ego_box=ego_box)
    lanes.extend(lane_connectors)

    for lane in lanes:
        lane_line = np.array([[pt.x, pt.y] for pt in lane.baseline_path.discrete_path])
        points = np.array([
            [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
            for x, y in lane_line
        ], dtype=np.int32)
        points = points.reshape((-1,1,2))
        speed_limit_mps = 50 if lane.speed_limit_mps is None else lane.speed_limit_mps
        color = (speed_limit_mps, speed_limit_mps, speed_limit_mps)
        cv2.polylines(image, [points], isClosed=False, color=color, thickness=2)
    image = cv2.warpAffine(image, rotation_matrix, (image_size, image_size))
    # Save the image
    filename = "/speed_limit.png"
    save_path = save_folder + filename
    cv2.imwrite(save_path, image)
    print(f"Speed limit map saved to {save_path}")


def generate_traffic_lights_map(scenario, iter, past_time_horizon=1., image_size=1024, resolution=0.1, save_folder="images/", angle_noise=0.):
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
    
    # Define traffic light colors
    traffic_light_colors = {
        "RED": (0,0,255),
        "YELLOW": (255, 255, 0),
        "GREEN": (0, 255, 0)
    }

    rotation_matrix = get_rotation_matrix(ego_state, image_size, angle_noise)

    step_size = int(0.2 / scenario.database_interval)

    past_iter = int(past_time_horizon / scenario.database_interval)

    for i in reversed(range(0, past_iter + 1, step_size)):
        curr_iter = iter - i
        # Create a blank black image
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        # plog lane
        traffic_lights = get_traffic_light_status(scenario, curr_iter)
        lanes = get_lane_centerlines(ego_box, map_api)
        lane_connectors = get_lane_connectors(ego_box, map_api)
        lanes = lanes | lane_connectors 

        for lane_id, lane in lanes.items():
            points = np.array([
                [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
                for x, y in lane
            ], dtype=np.int32)
            points = points.reshape((-1,1,2))
            color = traffic_light_colors.get(traffic_lights.get(int(lane_id), (0, 255, 0)), (0, 255, 0))
            cv2.polylines(image, [points], isClosed=False, color=color, thickness=1)
        image = cv2.warpAffine(image, rotation_matrix, (image_size, image_size))
        # Save the image
        filename = f"/traffic_light_-{i:03d}.png"  # 6-digit zero-padded format
        save_path = save_folder + filename
        cv2.imwrite(save_path, image)
        print(f"Roadmap saved to {save_path}")


def generate_roadmap(scenario, iter, image_size=1024, resolution=0.1, save_folder="images/", angle_noise=0.):
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
            color=(30, 30, 30)         # Fill color
        )

    # Plot boudnaries
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
    
    # common_boundaries = merge_common_boundaries(left_boundaries, right_boundaries)
    # for _, boundary in common_boundaries.items():
    #     points = np.array([
    #         [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
    #         for x, y in boundary
    #     ], dtype=np.int32)
    #     points = points.reshape((-1,1,2))
    #     cv2.polylines(image, [points], isClosed=False, color=(255, 255, 0), thickness=1)

    # # plog lane
    # lanes = get_lane_centerlines(ego_box, map_api)
    # for _, lane in lanes.items():
    #     points = np.array([
    #         [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
    #         for x, y in lane
    #     ], dtype=np.int32)
    #     # for i in range(len(points) - 1):
    #     for i in range(0, len(points) - 11, 10):
    #         start = points[i]
    #         end   = points[i + 10]
    #         # Draw an arrow from start to end
    #         # tipLength is a fraction of the total arrow length
    #         cv2.arrowedLine(
    #             img=image,
    #             pt1=(start[0], start[1]),
    #             pt2=(end[0], end[1]),
    #             color=(0, 255, 0),   # Green in BGR
    #             thickness=1,
    #             line_type=cv2.LINE_AA,
    #             tipLength=0.5
    #         )
    
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

    rotation_matrix = get_rotation_matrix(ego_state, image_size, angle_noise)
    image = cv2.warpAffine(image, rotation_matrix, (image_size, image_size))

    # Save the image
    save_path = save_folder + "/roadmap.png"
    cv2.imwrite(save_path, image)
    print(f"Roadmap saved to {save_path}")

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

def generate_intent_map(scenario, iter, lane_id_to_polygon, image_size=1024, resolution=0.1, save_folder="images/", angle_noise=0.):
    # Get map API
    map_api = scenario.map_api
    
    # Get ego vehicle position
    ego_state = scenario.get_ego_state_at_iteration(iter)
    ego_x, ego_y = ego_state.car_footprint.center.x, ego_state.car_footprint.center.y
    
    # Define the bounding box for the image
    half_size_meters = (image_size / 2) * resolution  # Convert pixels to meters
    min_x, _ = ego_x - half_size_meters, ego_x + half_size_meters
    min_y, _ = ego_y - half_size_meters, ego_y + half_size_meters
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    for lane_id, lane_polygon in lane_id_to_polygon.items():
        polygon = np.array([
            [(x - min_x) / resolution, (image_size - (y - min_y) / resolution)]
            for x, y in lane_polygon
        ], dtype=np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.fillPoly(
            img=image,               # The image to draw on
            pts=[polygon],    # List of polygons to fill
            color=(255, 255, 255)         # Fill color
        )
    
    rotation_matrix = get_rotation_matrix(ego_state, image_size, angle_noise)
    image = cv2.warpAffine(image, rotation_matrix, (image_size, image_size))

    # Save the image
    filename = f"/route_intent.png"  # 6-digit zero-padded format
    save_path = save_folder + filename
    cv2.imwrite(save_path, image)
    print(f"route intent saved to {save_path}")
