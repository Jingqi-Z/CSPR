# -*- coding: utf-8 -*-
import math
import logging
import random
import time
from typing import List
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import traci
import traci.constants as tc

from traffic_signal import TrafficSignal
from utils import set_no_change_lane, get_distance_to_intersection

Fo_zone_w_lane = ["w_t_nc_2", "w_t_nc_1", "w_t_nc_0"]  # 左、中、右
Fo_zone_e_lane = ["e_t_nc_2", "e_t_nc_1", "e_t_nc_0"]  # 左、中、右
Fo_zone_road = ['n_t_nc', 's_t_nc', 'e_t_nc', 'w_t_nc']
traci.setLegacyGetLeader(False)


def get_nearest_platoon(tl, max_len=8, num=1):
    if num <= 0:
        raise ValueError("Function get_closest_to_intersection called with"
                         "parameter num_closest={}, but num_closest should"
                         "be positive".format(num))
    all_lanes = traci.trafficlight.getControlledLinks(tl)
    in_lanes = [conn[0][0] for conn in all_lanes]
    del in_lanes[0::3]
    out_lanes = [conn[0][1] for conn in all_lanes]
    # del out_lanes[0::3]
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


if __name__ == "__main__":
    traci.start(["sumo-gui", "-c", "./net/single-stage2.sumocfg",
                 "--seed", "42"
                 # "--lanechange-output", "./net/lanechange_out.xml",
                 ],
                label="default", port=7911
                )
    traci.trafficlight.setProgram('t', '8-phase')
    set_no_change_lane()
    cav_platoon = deque(maxlen=100)
    TL_id = 't'
    temp = {'left': deque(maxlen=1000), 'straight': deque(maxlen=1000)}
    TL = TrafficSignal(TL_id, 3, traci)
    TL.set_stage_duration(0, 27)
    t = 0.0
    phase = 0
    num_p = 2
    vehicle_loss = {}
    # print(traci.trafficlight.getControlledLinks(TL_id))
    platoons = [[[] for _ in range(num_p)] for _ in range(8)]
    phase_encoding = {
        0: [1, 0, 0, 0, 1, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 1, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 1, 0],
        3: [0, 0, 0, 1, 0, 0, 0, 1],
        4: [1, 1, 0, 0, 0, 0, 0, 0],
        5: [0, 0, 1, 1, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 1, 1, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 1, 1],
    }
    while t < 1800:
        t = traci.simulation.getTime()
        traci.simulationStep()
        platoons_temp = get_nearest_platoon(TL_id, num=num_p)
        # print(platoons_temp)
        # print(TL.retrieve_stage())
        if t % 1 == 0:
            # print(get_nearest_platoon(TL_id))
            check = TL.check()
            TL.pop()
            if check == -1:
                phase = (phase + 1) % 4
                duration_map = {0: 17, 1: 17, 2: 27, 3: 27}
                TL.set_stage_duration(int(phase % 4), duration_map[phase])

        for p, pt, pha in zip(platoons, platoons_temp, phase_encoding[TL.retrieve_stage()]):
            if not p[0]:
                p.clear()
                p.extend(pt)
            elif not pha:
                p.clear()
                p.extend(pt)
            elif all(traci.vehicle.getRoadID(cav) not in Fo_zone_road for cav in p[0]) or p[0][0] == pt[0][0]:
                p.clear()
                p.extend(pt)
        # print(platoons)
        """for p, pha in zip(platoons, phase_encoding[TL.retrieve_stage()]):
            pla = p[0]
            if pla:
                v0 = traci.vehicle.getSpeed(pla[0])
                pos0 = traci.vehicle.getPosition(pla[0])
                for n, cav in enumerate(pla):
                    if n == 0:
                        # traci.vehicle.setAcceleration(cav, random.random(), 0.5)
                        traci.vehicle.setTau(cav, 1.5)
                        traci.vehicle.setSpeedMode(cav, 31)
                        pass
                    else:
                        T = 1.0
                        k_d, k_v, l, s_0 = 1/(n*T)**2, 1/(n*T), 5.0, 2.0
                        v = traci.vehicle.getSpeed(cav)
                        pos = traci.vehicle.getPosition(cav)
                        gap = ((pos0[0]-pos[0])**2 + (pos0[1]-pos[1])**2)**0.5
                        a = k_d*(gap - n*(T*v + l + s_0)) + k_v*(v0 - v)
                        a = np.clip(a, -9, 3)
                        traci.vehicle.setTau(cav, 0.2)
                        traci.vehicle.setSpeedMode(cav, 0x011110)
                        # traci.vehicle.setAcceleration(cav, a, 0.5)
                        traci.vehicle.slowDown(cav, np.clip(a+v, 0, 15), 0.25)"""

        for out_road in ['t_n', 't_e', 't_s', 't_w']:
            vehicles = traci.edge.getLastStepVehicleIDs(out_road)
            for veh_id in vehicles:
                if "cav" in veh_id.lower():
                    vehicle_loss[veh_id] = traci.vehicle.getTimeLoss(veh_id)
    print(vehicle_loss)
    print(np.mean(list(vehicle_loss.values())))
    traci.close()
