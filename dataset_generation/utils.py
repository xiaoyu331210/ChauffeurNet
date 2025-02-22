import cv2
import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

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
    return traffic_lights

def get_heading(ego_state, angle_noise=0.):
    return -ego_state.car_footprint.center.heading * 180. / 3.14 + 90. + angle_noise

def get_rotation_matrix(ego_state, image_size, angle_noise=0.):
    ego_heading = get_heading(ego_state, angle_noise)
    rotation_center = (image_size / 2, image_size / 2)
    return cv2.getRotationMatrix2D(rotation_center, ego_heading, 1.0)


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
