from trajdata import UnifiedDataset
import numpy as np
import betterosi
import shapely
from trajdata.maps.vec_map_elements import MapElementType
from trajdata import AgentType
import omega_prime
from omega_prime.converters.converter import DatasetConverter
from typing import Annotated
import typer
from pathlib import Path
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from dataclasses import dataclass
from typing import Any
from matplotlib import style


sbw = style.library["seaborn-v0_8-whitegrid"].copy()
style.core.update_nested_dict(style.library, {"seaborn-whitegrid": sbw})

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



def map_for_scenario(map, map_name):
    _ = map.traffic_light_status
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
        lane_boundary=[betterosi.LaneBoundary( # right lane
            id=betterosi.Identifier(value=b_id),
            boundary_line=[betterosi.LaneBoundaryBoundaryPoint(position=betterosi.Vector3D(x=x, y=y, z=y)) for x,y,z in polyline.points],
            classification=betterosi.LaneBoundaryClassification(
                type=betterosi.LaneBoundaryClassificationType.SOLID_LINE,
                color=betterosi.LaneBoundaryClassificationColor.WHITE
            )
        ) for lane in road_lane_elements.values() for b_id, polyline in [
            (mapped_lid[lane.id]*2, lane.right_edge),
            (mapped_lid[lane.id]*2+1, lane.left_edge)
        ] if polyline is not None],
        lane=[
            betterosi.Lane(
                classification=betterosi.LaneClassification(
                    centerline=[
                        betterosi.Vector3D(x=float(x), y=float(y), z=float(z))
                        for x, y, z, yaw in lane.center.points
                    ],
                    right_lane_boundary_id = [betterosi.Identifier(value=mapped_lid[lane.id]*2)] if lane.right_edge is not None else [],
                    left_lane_boundary_id = [betterosi.Identifier(value=mapped_lid[lane.id]*2+1)] if lane.left_edge is not None else [],
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
    return map_name, map_gt



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
    xy_h = np.array([state[0], state[1], 1.0])
    xy = np.dot(transform, xy_h)[:2]
    vel = np.dot(transform[:2,:2], state[2:4])
    acc = np.dot(transform[:2,:2], state[4:6])
    agent_type = AgentType(int(agent_type))
    t = agentType2osi_type[agent_type]
    kwargs = {}
    if t == betterosi.MovingObjectType.VEHICLE:
        kwargs['vehicle_classification'] = betterosi.MovingObjectVehicleClassification(
            type=agentType2osi_subtype[agent_type]
        )
    length = np.nanmax([extent[0],.1])
    width = np.nanmax([extent[1],.1])
    height= np.nanmax([extent[2],.1])
    return betterosi.MovingObject(
                id=betterosi.Identifier(value=i),
                type=t,
                base=betterosi.BaseMoving(
                    dimension=betterosi.Dimension3D(length=float(length), width=float(width), height=float(height)),
                    position=betterosi.Vector3D(x=float(xy[0]), y=float(xy[1]), z=0),
                    orientation=betterosi.Orientation3D(roll=0, pitch=0, yaw=float(
                        np.arctan2(
                            state[6],
                            state[7]
                        ) + np.arctan2(
                            transform[1,0], # sin(alpha)
                            transform[0,0] # cos(alpha)
                                                                                                                            ))),
                    velocity=betterosi.Vector3D(x=float(vel[0]), y=float(vel[1]), z=0),
                    acceleration=betterosi.Vector3D(x=float(acc[0]), y=float(acc[1]), z=0),
                ),
                **kwargs
            )
    

def agentbatchelement_to_omega(name, batch_element, map_cache):
    gts = []
    t_index = 0
    for [
        agent, extent, neigh, neigh_extents, neigh_len
    ] in [
        [batch_element.agent_history_np, batch_element.agent_history_extent_np, batch_element.neighbor_histories, batch_element.neighbor_history_extents, batch_element.neighbor_history_lens_np],
        [batch_element.agent_future_np, batch_element.agent_future_extent_np, batch_element.neighbor_futures, batch_element.neighbor_future_extents, batch_element.neighbor_future_lens_np]
    ]:
        transform = np.linalg.inv(batch_element.agent_from_world_tf)
        for ti in range(agent.shape[0]):
            objs = []
            objs.append(from_batch_info(
                1,
                agent_type=batch_element.agent_type,
                extent=extent[ti],
                state=agent[ti],
                transform=transform
            ))
            for i in range(len(batch_element.neighbor_types_np)):
                if ti < neigh_len[i] and not np.isnan(neigh[i][ti][0]):
                    objs.append(from_batch_info(
                        i+2,
                        agent_type=batch_element.neighbor_types_np[i],
                        extent=neigh_extents[i][ti],
                        state=neigh[i][ti],
                        transform=transform
                    ))
            nanos = t_index * batch_element.dt * 1e9
            gts.append(betterosi.GroundTruth(
                version=betterosi.InterfaceVersion(version_major=3, version_minor=7, version_patch=9),
                timestamp=betterosi.Timestamp(seconds=int(nanos // int(1e9)), nanos=int(nanos % int(1e9))),
                host_vehicle_id=betterosi.Identifier(value=1),
                moving_object=objs,
            ))
            t_index += 1
    map_gt = map_cache.get(batch_element.vec_map.map_id, None)
    if map_gt is None:
        _, map_gt = map_for_scenario(batch_element.vec_map, batch_element.vec_map.map_id)
        map_cache[batch_element.vec_map.map_id] = map_gt
    gts[0].lane = map_gt.lane
    gts[0].road_marking = map_gt.road_marking
    r = omega_prime.Recording.from_osi_gts(gts)
    try:
        r.map = omega_prime.map.MapOsi.create(gts[0])
    except RuntimeError:
        r.map = omega_prime.map.MapOsiCenterline.create(gts[0])
    return r




@dataclass
class Batch:
    agent_history_np: np.array
    agent_history_extent_np: np.array
    neighbor_histories: np.array
    neighbor_history_extents: np.array
    neighbor_history_lens_np: np.array
    agent_future_np: np.array
    agent_future_extent_np: np.array
    neighbor_futures: list[np.array]
    neighbor_future_extents: list[np.array]
    neighbor_future_lens_np: np.array
    agent_from_world_tf: np.array
    neighbor_types_np: np.array
    dt: float
    vec_map: Any
    scene_id: str
    agent_type: Any
    
    @classmethod
    def from_trajdata(cls, b):
        return cls(
            agent_history_np=b.agent_history_np.as_ndarray(),
            agent_history_extent_np=b.agent_history_extent_np,
            neighbor_histories=[o.as_ndarray() for o in b.neighbor_histories],
            neighbor_history_extents=b.neighbor_history_extents,
            neighbor_history_lens_np=b.neighbor_history_lens_np,
            agent_future_np=b.agent_future_np.as_ndarray(),
            agent_future_extent_np=b.agent_future_extent_np,
            neighbor_futures=[o.as_ndarray() for o in b.neighbor_futures],
            neighbor_future_extents=b.neighbor_future_extents,
            neighbor_future_lens_np=b.neighbor_future_lens_np,
            agent_from_world_tf=b.agent_from_world_tf,
            neighbor_types_np=b.neighbor_types_np,
            dt=b.dt,
            vec_map=b.vec_map,
            scene_id=b.scene_id,
            agent_type=b.agent_type,
    )

class TrajdataConverter(DatasetConverter):
    def __init__(
        self, 
        dataset_path: str, 
        out_path: str, 
        dataset_name,  
        keep_in_memory=False,
        n_workers=1,
    ) -> None:
        super().__init__(dataset_path, out_path, n_workers=n_workers)
        self.dataset_name=dataset_name
        self.map_cache = {}
        self.dataset = UnifiedDataset(
            desired_data=[dataset_name],
            data_dirs={  # Remember to change this to match your filesystem!
                dataset_name: dataset_path
            },
            incl_vector_map=True,
            vector_map_params = {
                "incl_road_lanes": True,
                "incl_road_areas": False,
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,
                # Collation can be quite slow if vector maps are included,
                # so we do not unless the user requests it.
                "collate": True,
                # Whether loaded maps should be stored in memory (memoized) for later re-use.
                # For datasets which provide full maps ahead-of-time (i.e., all except Waymo),
                # this should be True. However, for Waymo it should be False because maps
                # are already partitioned geographically and keeping them around significantly grows memory.
                "keep_in_memory": keep_in_memory,
            }
        )
        


        if keep_in_memory:
            with tqdm_joblib(desc="Preload Maps", total=len(self.dataset._map_api.maps)):
                maps_jobs = Parallel(n_jobs=4)(delayed(map_for_scenario)(vm, vm_name) for vm_name, vm in self.dataset._map_api.maps.items())
                self.map_cache = {name: map for name, map in maps_jobs}


    def __getstate__(self):
        # Copy the instance’s dict
        state = self.__dict__.copy()
        # Remove the attribute you don’t want
        if "dataset" in state:
            del state["dataset"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Optionally restore excluded attributes with a default
        self.dataset = None
        
    def get_source_recordings(self):
        self.len = self.dataset.num_scenes()
        for i in range(self.len):
            yield Batch.from_trajdata(self.dataset[self.dataset._data_index._cumulative_lengths[i]])

    def get_recordings(self, batch):
        batch_element = batch
        yield batch_element.scene_id, batch_element

    def get_recording_name(self, recording) -> str:
        name, _ = recording
        return name

    def to_omega_prime_recording(self, recording):
        name, batch_element = recording
        return agentbatchelement_to_omega(name, batch_element, self.map_cache)

    
    @classmethod
    def convert_cli(
        cls,
        dataset_path: Annotated[
            Path,
            typer.Argument(exists=True, dir_okay=True, file_okay=True, readable=True, help="Root of the dataset"),
        ],
        output_path: Annotated[
            Path,
            typer.Argument(
                file_okay=False, writable=True, help="In which folder to write the created omega-prime files"
            ),
        ],
        dataset_name: Annotated[str, typer.Argument(help='trajdata name of the dataset (see https://github.com/NVlabs/trajdata?tab=readme-ov-file#supported-datasets)')],  
        n_workers: Annotated[int, typer.Option(help="Set to -1 for n_cpus-1 workers.")] = 1,
        skip_existing: Annotated[bool, typer.Option(help="Only convert not yet converted files")] = False,
        write_log: Annotated[bool, typer.Option(help="Write a log file with the conversion process")] = False,

        keep_map_in_memory: Annotated[bool, typer.Option(help="Turn off for waymo dataset since they have map for each scene.")]=True,
    ):
        Path(output_path).mkdir(exist_ok=True)
        cls(dataset_path=dataset_path, out_path=output_path, n_workers=0, keep_in_memory=keep_map_in_memory, dataset_name=dataset_name).convert(
            save_as_parquet=False, skip_existing=skip_existing, write_log=write_log, n_workers=n_workers
        )