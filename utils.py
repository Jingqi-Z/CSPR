import math
import re
from typing import List

import traci


def set_no_change_lane():
    """Set no change lane permission for all NotChangeLane zones."""
    NCL_zone = [f"{direction.lower()}_t_nc_{lane_index}" for direction in ["n", "s", "e", "w"] for lane_index in
                range(3)]
    for zone in NCL_zone:
        for direction in [-1, 1]:
            traci.lane.setChangePermissions(zone, ["emergency"], direction)
    return True


def get_distance_to_intersection(veh_id):
    """Determine the distance from a vehicle to its next intersection.
    This condition only use one fixed interaction.
    Parameters
    ----------
    veh_id : str or list of str
        Vehicle identifier or a list of vehicle identifiers.
    Returns
    -------
    float or list of float
        Distance to the closest intersection. Returns a single float if a single vehicle identifier is given,
        or a list of floats if a list of vehicle identifiers is given.
    """
    # junction_id = 't'
    # junction_shape = [(990.4, 513.6), (1009.6, 513.6), (1013.6, 509.6), (1013.6, 490.4), (1009.6, 486.4),
    #                   (990.4, 486.4), (986.4, 490.4), (986.4, 509.6)]
    if isinstance(veh_id, list):
        return [get_distance_to_intersection(veh) for veh in veh_id]
    if not veh_id:
        return 1e4
    # junction_left = 986.4
    pos_map = {'w': 986.4, 'e': 1013.6, 'n': 513.6, 's': 486.4}
    veh_pos = traci.vehicle.getPosition(veh_id)
    if 'we' in veh_id or 'wn' in veh_id or 'ws' in veh_id:
        return pos_map['w'] - veh_pos[0]
    elif 'ew' in veh_id or 'en' in veh_id or 'es' in veh_id:
        return veh_pos[0] - pos_map['e']
    elif 'ns' in veh_id or 'nw' in veh_id or 'ne' in veh_id:
        return veh_pos[1] - pos_map['n']
    elif 'sn' in veh_id or 'se' in veh_id or 'sw' in veh_id:
        return pos_map['s'] - veh_pos[1]
    else:
        junction_pos = [1000.0, 500.0]
    return math.sqrt((veh_pos[0] - junction_pos[0]) ** 2 + (veh_pos[1] - junction_pos[1]) ** 2)


def get_route_vehicles(route) -> list:
    vehicles = [traci.lane.getLastStepVehicleIDs(lane) for lane in route]
    vehicles = [veh for vehs in vehicles for veh in vehs]
    return vehicles


def get_nearest_platoon(tl, max_len=4, num=1):
    """
    This function is used to get the nearest platoons to the traffic light.

    Parameters:
    ----------
    tl (str): The traffic light ID.
    max_len (int): The maximum length of each platoon. Default is 8.
    num (int): The number of nearest platoons to be found. Default is 1.

    Returns:
    ----------
    list: A list of platoons, where each platoon is a list of vehicle IDs.
    """
    start_pos = 0
    end_pos = 500
    if num <= 0:
        raise ValueError("Function get_closest_to_intersection called with"
                         "parameter num_closest={}, but num_closest should"
                         "be positive".format(num))
    all_lanes = traci.trafficlight.getControlledLinks(tl)
    in_lanes = [conn[0][0] for conn in all_lanes]
    del in_lanes[0::3]
    platoons = [[] for _ in range(len(in_lanes))]
    for lane, p in zip(in_lanes, platoons):
        # 初始化变量
        # cav_vehicles_group = []
        current_group = []
        vehicles = list(traci.lane.getLastStepVehicleIDs(lane))
        vehicles.sort(key=traci.vehicle.getDistance, reverse=True)
        # 遍历车辆ID
        for veh_id in vehicles:
            if "cav" in veh_id.lower() and len(current_group) < max_len:
                current_group.append(veh_id)
            else:
                if current_group:
                    p.append(current_group)
                    current_group = []
                # 提前终止循环，如果已经找到了前n个整体
                if len(p) == num:
                    break
        # 处理最后一个车队，需要添加到cav_vehicles_group中
        if current_group and len(p) < num:
            p.append(current_group)
        if len(p) < num:
            p.extend([[]] * (num - len(p)))
    return platoons


def get_closest_to_intersection(edge, num_closest, start_pos=0, end_pos=1000, max_len=4, padding=True):
    """Return the IDs of the vehicles(or vehicle platoon) that are closest to an intersection.

    No more than 5 CAVs driving directly one after another is considered a vehicle platoon.

    For each edge in edges, return the IDs (veh_id) of the num_closest vehicles in edge that are closest to an
    intersection (the intersection they are heading towards).

    This function performs no check on whether edges are going towards an intersection or not, it just gets the vehicles
    that are closest to the end of their edges.

    If there are less than num_closest vehicles on an edge, the function performs padding by adding empty
    strings "" instead of vehicle ids if the padding parameter is set to True.

    Parameters
    ----------
    edge : str | str list
        ID of an edge or list of edge IDs.
    num_closest : int (> 0)
        Number of vehicles to consider on each edge.
    padding : bool (default False)
        If there are less than num_closest vehicles on an edge, perform
        padding by adding empty strings "" instead of vehicle ids if the
        padding parameter is set to True (note: leaving padding to False
        while passing a list of several edges as parameter can lead to
        information loss since you will not know which edge, if any,
        contains less than num_closest vehicles).

    Returns
    -------
    str list
        If n is the number of edges given as parameters, then the returned
        list contains n * num_closest vehicle IDs.

    Raises
    ------
    ValueError
        if num_closest <= 0
    """
    if num_closest <= 0:
        raise ValueError("Function get_closest_to_intersection called with"
                         "parameter num_closest={}, but num_closest should"
                         "be positive".format(num_closest))

    if isinstance(edge, list):
        vehicles = get_route_vehicles(edge)
    else:
        vehicles = traci.lane.getLastStepVehicleIDs(edge)

    # get the ids of all the vehicles on the edge 'edges' ordered by
    # increasing distance to end of edge (intersection)
    veh_ids_ordered = sorted([veh for veh in vehicles if start_pos < get_distance_to_intersection(veh) <= end_pos],
                             key=get_distance_to_intersection)
    # 初始化变量
    cav_vehicles_group = []
    current_group = []
    # 遍历车辆ID
    for veh_id in veh_ids_ordered:
        if "CAV" in veh_id.upper() and len(current_group) < max_len:
            current_group.append(veh_id)
        else:
            if current_group:
                cav_vehicles_group.append(current_group)
                current_group = []
            # 提前终止循环，如果已经找到了前n个整体
            if len(cav_vehicles_group) == num_closest:
                break
    # 处理最后一个车队，需要添加到cav_vehicles_group中
    if current_group and len(cav_vehicles_group) < num_closest:
        cav_vehicles_group.append(current_group)
    # 如果找到的整体数目少于n，用[]补足
    if padding is True:
        while len(cav_vehicles_group) < num_closest:
            cav_vehicles_group.append([])
    return cav_vehicles_group[:num_closest]


def get_group_vehicles(vehicles, boundary, formation_boundary=100):
    if len(vehicles) <= 1:
        return []
    # print(v, L)
    # vehicles = [veh for veh in vehicles if 'cav' in veh.lower()]
    vehicles = [veh for veh in vehicles if traci.vehicle.getPosition(veh)[0] <= formation_boundary + boundary]

    return list(vehicles)


def get_trigger_veh(vehicles, formation_boundary=100):
    # Filter out vehicles that are CAVs and not turn-right vehicles.
    # vehicles = list(filter(lambda veh: 'cav' in veh.lower() and 'ws' not in veh.lower(), vehicles))
    # vehicles = [veh for veh in vehicles if 'cav'.lower() in veh.lower() and 'ws'.lower() not in veh.lower()]
    if len(vehicles) > 0:
        # Sort vehicles in descending order based on their position on the road
        vehicles.sort(key=get_distance_to_intersection, reverse=True)
        if (0 < traci.vehicle.getPosition(vehicles[0])[0] - formation_boundary <= 50 or
                0 < 2000 - formation_boundary - traci.vehicle.getPosition(vehicles[0])[0] <= 50):
            v = traci.vehicle.getSpeed(vehicles[0])
            v_des = 15
            v_thr = 20 / 3.6  # 阈值
            LC = 250  # Cooperation zone 长度
            delta_t_i = LC / v_des
            if v < v_thr:
                a_i_pc = ((v_des ** 2 - v ** 2) / (2 * LC))
                v = v + a_i_pc * delta_t_i
            boundary = v * LC / v_des
            return vehicles[0], boundary
    # If the filtered list of vehicles is empty, return None
    return None, 350.0


def find_closest_vehicles(vehicles, num: int = 3) -> (list, list):
    max_dis = 180
    if len(vehicles) <= 1:
        return [], vehicles
    elif len(vehicles) == 2:
        if abs(traci.vehicle.getPosition(vehicles[0])[0] - traci.vehicle.getPosition(vehicles[1])[0]) <= max_dis:
            return vehicles, []
        else:
            return [], vehicles
    # 初始化列表
    closest_vehicles = []
    other_vehicles = []
    # 获取每辆车的位置信息
    vehicle_positions = {veh_id: traci.vehicle.getPosition(veh_id)[0] for veh_id in vehicles}
    # 根据车辆位置排序
    sorted_vehicles = sorted(vehicle_positions.items(), key=lambda x: x[1], reverse=True)
    # 计算距离最小的三辆车
    min_distance = 1000
    min_distance_index = 0
    for i in range(len(sorted_vehicles) - 2):
        distance = sorted_vehicles[i][1] - sorted_vehicles[i + 2][1]
        if distance < min_distance:
            min_distance = distance
            min_distance_index = i
    # 获取最小距离的三辆车
    current_vehicle, next_vehicle, next_next_vehicle = [sorted_vehicles[min_distance_index + j] for j in range(3)]
    # print(current_vehicle[1] - next_next_vehicle[1])
    if current_vehicle[1] - next_next_vehicle[1] < max_dis:
        closest_vehicles = [current_vehicle[0], next_vehicle[0], next_next_vehicle[0]]
    elif current_vehicle[1] - next_vehicle[1] <= max_dis:
        closest_vehicles = [current_vehicle[0], next_vehicle[0]]
    elif next_vehicle[1] - next_next_vehicle[1] <= max_dis:
        closest_vehicles = [next_vehicle[0], next_next_vehicle[0]]

    # 获取除最靠近的车辆之外的其他车辆
    for veh_id in vehicles:
        if veh_id not in closest_vehicles:
            other_vehicles.append(veh_id)

    return closest_vehicles, other_vehicles


# Function to determine if a lane change by the vehicle is safe
def is_safe_lane_change(veh_id, current_lane_index, target_lane_index) -> bool:
    # not cross two lanes
    # direction_mapping = {"wn": 2, "we": 1, "ws": 0}
    # Get the current position, lane and speed of the vehicle
    # cav_SubscriptionResults = traci.vehicle.getSubscriptionResults(veh_id)
    # veh_position = cav_SubscriptionResults[0x42]
    veh_speed = traci.vehicle.getSpeed(veh_id)
    current_lane = traci.vehicle.getRoadID(veh_id)
    target_lane = f"{current_lane}_{target_lane_index}"
    pattern = r'[a-z]+_[a-z]+_\d{1}'  # n_t_0
    pattern2 = r'[a-z]+_[a-zA-Z]+_[a-zA-Z]+_\d{1}'  # w_t_nc_0
    pattern3 = r'[a-z]+_[a-z]+.\d+_\d{1}'  # e_t.100_0
    if not (re.match(pattern, target_lane) or re.match(pattern2, target_lane) or
            re.match(pattern3, target_lane)):
        return False
    # safe_distance = 0.6 * veh_speed
    if current_lane_index < target_lane_index:  # 左转
        fv_info = traci.vehicle.getLeftFollowers(veh_id)
        lv_info = traci.vehicle.getLeftLeaders(veh_id)
    else:  # 右转
        fv_info = traci.vehicle.getRightFollowers(veh_id)
        lv_info = traci.vehicle.getRightLeaders(veh_id)

    # 检查是否与前后车辆的距离在安全范围内
    """# 获取前车的安全距离
    if lv_info:
        leader_speed = traci.vehicle.getSpeed(lv_info[0][0])
        safe_dis_lv = traci.vehicle.getSecureGap(veh_id, speed=veh_speed, leaderSpeed=leader_speed,
                                                 leaderMaxDecel=9, leaderID=lv_info[0][0])
        lv_distance = lv_info[0][1]
    else:
        safe_dis_lv = None
        lv_distance = None

    # 获取后车的安全距离
    if fv_info:
        follower_speed = traci.vehicle.getSpeed(fv_info[0][0])
        safe_dis_fv = traci.vehicle.getSecureGap(fv_info[0][0], speed=follower_speed, leaderSpeed=veh_speed,
                                                 leaderMaxDecel=9, leaderID=veh_id)
        fv_distance = fv_info[0][1]
    else:
        safe_dis_fv = None
        fv_distance = None

    # 检查前车和后车的安全距离
    if lv_info and fv_info:
        return lv_distance > safe_dis_lv and fv_distance > safe_dis_fv
    elif lv_info:
        return lv_distance > safe_dis_lv
    elif fv_info:
        return fv_distance > safe_dis_fv
    else:
        return True"""
    safe_dis_lv = traci.vehicle.getSecureGap(veh_id, speed=veh_speed,
                                             leaderSpeed=traci.vehicle.getSpeed(lv_info[0][0]),
                                             leaderMaxDecel=20, leaderID=lv_info[0][0])*0.8 if lv_info else None
    safe_dis_fv = traci.vehicle.getSecureGap(fv_info[0][0], speed=traci.vehicle.getSpeed(fv_info[0][0]),
                                             leaderSpeed=veh_speed,
                                             leaderMaxDecel=9, leaderID=veh_id)*0.8 if fv_info else None

    lv_distance = lv_info[0][1] if lv_info else 1000
    fv_distance = fv_info[0][1] if fv_info else 1000
    # print(safe_dis_lv, safe_dis_fv)
    return (lv_distance >= safe_dis_lv and fv_distance > safe_dis_fv) if lv_info and fv_info else \
        (lv_distance >= safe_dis_lv) if lv_info else (fv_distance > safe_dis_fv) if fv_info else True


def calculate_platoon_intensity(lane: str = None, vehicles=None, lane_index: int = None) -> list[float] | float:
    """
    计算车排强度
    """
    if lane is not None:
        if isinstance(lane, list):
            return [calculate_platoon_intensity(l) for l in lane]
        vehicles = list(traci.lane.getLastStepVehicleIDs(lane))
        lane_index = int(lane[-1])
    num_cav = len([veh for veh in vehicles if "CAV" in veh.upper()])
    if len(vehicles) <= 1 or num_cav <= 1:
        return 0.0
    vehicles.sort(key=get_distance_to_intersection)
    lane_map = {0: "ws", 1: "we", 2: "wn"}
    # 初始化变量
    cav_vehicles_group = []
    current_group = []
    # 遍历车辆ID
    for veh_id in vehicles:
        if "CAV" in veh_id.upper() and lane_map[lane_index] in veh_id:
            current_group.append(veh_id)
        else:
            if current_group:
                cav_vehicles_group.append(current_group)
                current_group = []
    if current_group:
        cav_vehicles_group.append(current_group)
    # logging.info(f"{cav_vehicles_group}")
    return sum([len(group) for group in cav_vehicles_group if len(group) > 1]) / num_cav
