from trajdata import UnifiedDataset
from torch.utils.data import DataLoader
import numpy as np
import tensorflow as tf
import betterosi
import shapely
from trajdata.maps.vec_map_elements import MapElementType
from pathlib import Path
from trajdata import MapAPI, VectorMap
from trajdata import AgentType
import omega_prime


def is_intersection(lane, road_lane_elements):
    lane_center_line = shapely.LineString(lane.center.points[:, :2])
    if lane_center_line.length > 60:
        return False

    connected_lanes = lane.next_lanes | lane.prev_lanes | lane.adj_lanes_left | lane.adj_lanes_right
    intersection_count = 0
    for other_lane_id, other_lane in road_lane_elements.items():
        if other_lane_id == lane.id or other_lane_id in connected_lanes:
            continue

        other_lane_center_line = shapely.LineString(other_lane.center.points[:, :2])
        if lane_center_line.intersects(other_lane_center_line):
            intersection_count += 1
            if intersection_count >= 2:
                return True

    return False


def classify_lanes(road_lane_elements):
    classified_lanes = {"intersection": {}, "not_intersection": {}}
    for lane_id, lane in road_lane_elements.items():
        if is_intersection(lane, road_lane_elements):
            classified_lanes["intersection"][lane_id] = lane
            for reachable_lane_id in lane.reachable_lanes:
                if reachable_lane_id not in lane.next_lanes and reachable_lane_id not in lane.prev_lanes:
                    reachable_lane = road_lane_elements[reachable_lane_id]
                    reachable_lane_center_line = shapely.LineString(reachable_lane.center.points[:, :2])
                    if reachable_lane_center_line.length <= 40:
                        classified_lanes["intersection"][reachable_lane_id] = reachable_lane
        else:
            classified_lanes["not_intersection"][lane_id] = lane
    return classified_lanes


def group_lanes_into_roads(road_lane_elements, classified_lanes):
    """
    Assigns a 'road_id' to each lane in road_lane_elements, grouping connected lanes into roads.

    Parameters:
    - road_lane_elements (dict): Mapping of lane IDs to RoadLane objects, including connections.
    - classified_lanes (dict): Contains:
        - "not_intersection" (dict): Lanes that belong to roads.
        - "intersection" (dict, optional): Lanes to be excluded.

    Returns:
    - dict: A dictionary containing the 'road_id' for each lane.
    """
    visited = set()
    road_id_counter = 0
    road_ids = {}  # This dictionary will store the 'road_id' for each lane ID

    def dfs(lane_id, road_id):
        """Recursively assigns road_id to all connected lanes."""
        if (
            lane_id in visited
            or lane_id in classified_lanes.get("intersection", {})
            or lane_id not in road_lane_elements
        ):
            return
        visited.add(lane_id)

        # Assign the road_id to the lane in the road_ids dictionary
        road_ids[lane_id] = str(road_id)

        lane = road_lane_elements[lane_id]

        for neighbor in lane.next_lanes | lane.prev_lanes | lane.reachable_lanes:
            dfs(neighbor, road_id)

    # Iterate through non-intersection lanes and assign road_ids
    for lane_id in classified_lanes.get("not_intersection", {}):
        if lane_id not in visited:
            dfs(lane_id, road_id_counter)
            road_id_counter += 1

    return road_ids


def get_polygon_dimensions(polyline) -> tuple[float, float, float]:
    """
    Calculate the length (x-axis), width (y-axis), and height (z-axis) of a polyline.

    Args:
        polyline: A Polyline object with a points attribute representing the polyline points.

    Returns:
        tuple: A tuple containing the length, width, and height of the polyline.
    """
    points = polyline.points
    length = np.max(points[:, 0]) - np.min(points[:, 0])
    width = np.max(points[:, 1]) - np.min(points[:, 1])
    height = np.max(points[:, 2]) - np.min(points[:, 2])

    return length, width, height


def get_polyline_midpoint(polyline) -> np.ndarray:
    """
    Calculate the midpoint of a polyline using Shapely.

    Args:
        polyline: A Polyline object with a points attribute representing the polyline points.

    Returns:
        np.ndarray: A NumPy array containing the midpoint coordinates [x, y, z].
    """
    points = polyline.points
    line = shapely.LineString(points[:, :2])  # Use only x and y coordinates
    centroid = line.centroid
    # Find the z-coordinate of the midpoint by interpolating the z-values separately for x and y
    z_x = np.interp(centroid.x, points[:, 0], points[:, 2])
    z_y = np.interp(centroid.y, points[:, 1], points[:, 2])
    z = (z_x + z_y) / 2  # Average the interpolated z-values
    return np.array([centroid.x, centroid.y, z])



def map_for_secenario(dataset, dataset_name, map_gts, s):
    if s.location not in map_gts:
        cache_path = Path(dataset.cache_path).expanduser()
        map_api = MapAPI(cache_path)

        map: VectorMap = map_api.get_map(f"{dataset_name}:{s.location}")

        tl_status = map.traffic_light_status
        pedestrian_crosswalk_elements = dict(map.elements[MapElementType.PED_CROSSWALK].items())
        road_lane_elements = dict(map.elements[MapElementType.ROAD_LANE].items())

        classified_lanes = classify_lanes(road_lane_elements)
        
        for lane_id, lane in road_lane_elements.items():
            lane.is_intersection = lane_id in classified_lanes["intersection"]
            
        mapped_lid = {r.id: i for i,r in enumerate(map.lanes)}
        mapped_cw = {r.id: i for i,r in enumerate(pedestrian_crosswalk_elements.values())}

        map_gt = betterosi.GroundTruth(
            version=betterosi.InterfaceVersion(version_major=3, version_minor=7, version_patch=0),
            road_marking=[
                betterosi.RoadMarking(
                    id=betterosi.Identifier(value=mapped_cw[crosswalk.id]),
                    base=betterosi.BaseStationary(
                        dimension=betterosi.Dimension3D(*[float(o) for o in get_polygon_dimensions(crosswalk.polygon)]),
                        position=betterosi.Vector3D(*[float(o) for o in get_polyline_midpoint(crosswalk.polygon)]),
                    ),
                )
                for crosswalk in pedestrian_crosswalk_elements.values()
            ],
            lane=[
                betterosi.Lane(
                    classification=betterosi.LaneClassification(
                        centerline=[
                            betterosi.Vector3D(x=float(x), y=float(y), z=float(z))
                            for x, y, z, yaw in lane.center.points
                        ],
                        centerline_is_driving_direction=True,  # checked and the centerline is always in driving direction
                        type=betterosi.LaneClassificationType.TYPE_INTERSECTION
                        if lane.is_intersection
                        else betterosi.LaneClassificationType.TYPE_DRIVING,
                        left_adjacent_lane_id=[
                            betterosi.Identifier(value=mapped_lid[lane_id]) for lane_id in lane.adj_lanes_left
                        ],
                        right_adjacent_lane_id=[
                            betterosi.Identifier(value=mapped_lid[lane_id]) for lane_id in lane.adj_lanes_right
                        ],
                        lane_pairing=[
                            betterosi.LaneClassificationLanePairing(
                                antecessor_lane_id=betterosi.Identifier(value=mapped_lid[prev_lane_id]),
                                successor_lane_id=betterosi.Identifier(value=mapped_lid[next_lane_id]),
                            )
                            for prev_lane_id in lane.prev_lanes
                            for next_lane_id in lane.next_lanes
                        ],
                    ),
                    id=betterosi.Identifier(value=mapped_lid[lane.id]),
                )
                for lane in road_lane_elements.values()
            ],
        )
        map_gts[s.location] = map_gt
        return map_gt
    return map_gts[s.location]



agentType2osi_subtype = {
    AgentType.UNKNOWN: betterosi.MovingObjectVehicleClassificationType.UNKNOWN,
    AgentType.VEHICLE: betterosi.MovingObjectVehicleClassificationType.CAR,
    AgentType.BICYCLE: betterosi.MovingObjectVehicleClassificationType.BICYCLE,
    AgentType.MOTORCYCLE: betterosi.MovingObjectVehicleClassificationType.MOTORBIKE,
}
agentType2osi_type = {
    AgentType.UNKNOWN: betterosi.MovingObjectType.UNKNOWN,
    AgentType.VEHICLE: betterosi.MovingObjectType.VEHICLE,
    AgentType.PEDESTRIAN: betterosi.MovingObjectType.PEDESTRIAN,
    AgentType.BICYCLE: betterosi.MovingObjectType.VEHICLE,
    AgentType.MOTORCYCLE: betterosi.MovingObjectType.VEHICLE,
}
def from_batch_info(i, agent_type, extent, state,transform):
    xy = tf.tensordot(state[0:2], transform[:2,:2].T, axes=1)+transform[:2,2]
    vel = tf.tensordot(state[2:4], transform[:2,:2], axes=1)
    acc = tf.tensordot(state[4:6], transform[:2,:2], axes=1)
    agent_type = AgentType(int(agent_type))
    t = agentType2osi_type[agent_type]
    kwargs = {}
    if t == betterosi.MovingObjectType.VEHICLE:
        kwargs['vehicle_classification'] = betterosi.MovingObjectVehicleClassification(
            type=agentType2osi_subtype[agent_type]
        )
    return betterosi.MovingObject(
                id=betterosi.Identifier(value=i),
                type=t,
                base=betterosi.BaseMoving(
                    dimension=betterosi.Dimension3D(length=max(extent[0],.1), width=max(extent[1],.1), height=extent[2]),
                    position=betterosi.Vector3D(x=xy[0], y=xy[1], z=0),
                    orientation=betterosi.Orientation3D(roll=0, pitch=0, yaw=np.arcsin(state[6])+np.arccos(transform[0][0])),
                    velocity=betterosi.Vector3D(x=vel[0], y=vel[1], z=0),
                    acceleration=betterosi.Vector3D(x=acc[0], y=acc[1], z=0),
                ),
                **kwargs
            )

def batch_to_gt_lists(batch, map_gts, dataset, dataset_name):
    for bi in range(len(batch.neigh_types)):
        s = dataset.get_scene(batch.scene_ts)
        map_gt = map_for_secenario(dataset, dataset_name, map_gts, s)
        gts = []
        t_index = 0
        for [
            agent, extent, neigh, neigh_extents, neigh_len
        ] in [
            [batch.agent_hist, batch.agent_hist_extent, batch.neigh_hist, batch.neigh_hist_extents, batch.neigh_hist_len],
            [batch.agent_fut, batch.agent_fut_extent, batch.neigh_fut, batch.neigh_fut_extents, batch.neigh_fut_len]
        ]:
            transform = np.linalg.inv(batch.agents_from_world_tf[bi])
            for ti in range(agent[bi].size(0)):
                objs = []
                objs.append(from_batch_info(
                    0,
                    agent_type=batch.agent_type[0],
                    extent=extent[bi][ti],
                    state=agent[bi][ti],
                    transform=transform
                ))
                for i in range(batch.num_neigh[bi]):
                    if ti < neigh_len[bi][i]:
                        objs.append(from_batch_info(
                            i+1,
                            agent_type=batch.neigh_types[bi][i],
                            extent=neigh_extents[bi][i][ti],
                            state=neigh[bi][i][ti],
                            transform=transform
                        ))
                nanos = t_index * batch.dt[bi] * 1e9
                gts.append(betterosi.GroundTruth(
                    version=betterosi.InterfaceVersion(version_major=3, version_minor=7, version_patch=9),
                    timestamp=betterosi.Timestamp(seconds=int(nanos // int(1e9)), nanos=int(nanos % int(1e9))),
                    host_vehicle_id=betterosi.Identifier(value=0),
                    moving_object=objs,
                ))
                t_index += 1
        gts[0].lane = map_gt.lane
        gts[0].road_marking = map_gt.road_marking
        yield (s.name, gts)

def yield_omegas(map_cache, dataset_name='nusc_mini', dataset_dir='../nuscene_mini/'):
    dataset = UnifiedDataset(
        desired_data=[dataset_name],
        data_dirs={  # Remember to change this to match your filesystem!
            dataset_name: dataset_dir
        },
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0#os.cpu_count(), # This can be set to 0 for single-threaded loading, if desired.
    )
    for i, batch in enumerate(dataloader):
        gt_lists = batch_to_gt_lists(batch, map_cache, dataset, dataset_name)
        for name, gts in gt_lists:
            r = omega_prime.Recording.from_osi_gts(gts)
            r.map = omega_prime.map.MapOsiCenterline.create(gts[0])
            yield (name, r)