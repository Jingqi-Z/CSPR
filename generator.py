from datetime import datetime
from typing import Any

import numpy as np

'''
    <route id="route_en" edges="e_t e_t_nc t_n t_n0"/>
    <route id="route_es" edges="e_t e_t_nc t_s t_s0"/>
    <route id="route_ew" edges="e_t e_t_nc t_w t_w0"/>
    <route id="route_ne" edges="n_t n_t_nc t_e t_e0"/>
    <route id="route_ns" edges="n_t n_t_nc t_s t_s0"/>
    <route id="route_nw" edges="n_t n_t_nc t_w t_w0"/>
    <route id="route_se" edges="s_t s_t_nc t_e t_e0"/>
    <route id="route_sn" edges="s_t s_t_nc t_n t_n0"/>
    <route id="route_sw" edges="s_t s_t_nc t_w t_w0"/>
    <route id="route_we" edges="w_t w_t_nc t_e t_e0"/>
    <route id="route_wn" edges="w_t w_t_nc t_n t_n0"/>
    <route id="route_ws" edges="w_t w_t_nc t_s t_s0"/>
'''
def generate_rou_file(num_vehicles=500, penetration=0.5, ):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    np.random.seed(128)
    # demand per second from different directions

    routes = {"route_we": {"edges": "w_t.100 w_t_nc t_e", "target_lane_index": "1", "direction": "STRAIGHT",
                           "count_cav": 0, "count_hv": 0},
              "route_wn": {"edges": "w_t.100 w_t_nc t_n", "target_lane_index": "2", "direction": "LEFT",
                           "count_cav": 0, "count_hv": 0},
              "route_ws": {"edges": "w_t.100 w_t_nc t_s", "target_lane_index": "0", "direction": "RIGHT",
                           "count_cav": 0, "count_hv": 0},
              "route_ew": {"edges": "e_t.100 e_t_nc t_w", "target_lane_index": "1", "direction": "STRAIGHT",
                           "count_cav": 0, "count_hv": 0},
              "route_es": {"edges": "e_t.100 e_t_nc t_s", "target_lane_index": "2", "direction": "LEFT",
                           "count_cav": 0, "count_hv": 0},
              "route_en": {"edges": "e_t.100 e_t_nc t_n", "target_lane_index": "0", "direction": "RIGHT",
                           "count_cav": 0, "count_hv": 0}
              }
    flow = [[150, 300, 200], [250, 400, 450], [150, 200, 300], [250, 450, 400]]
    # flow = [[100, 250, 200], [150, 350, 300], [100, 200, 250], [150, 300, 350]]
    # left_rate = 0.35
    # straight_rate = 0.45
    # right_rate = 0.2
    # total_flow = [600, 1200, 600, 1200]
    #
    # # 初始化一个4x3的二维数组来存储每个车道的车流量
    # lane_flow = [[0] * 3 for _ in range(4)]
    #
    # # 计算每个方向的车道车流量
    # for i in range(4):
    #     lane_flow[i][0] = total_flow[i] * right_rate  # 右转车道车流量
    #     lane_flow[i][1] = total_flow[i] * straight_rate  # 直行车道车流量
    #     lane_flow[i][2] = total_flow[i] * left_rate  # 左转车道车流量

    prob = np.array(flow) / 3600
    penetration = 0.5
    pattern = ""

    with (open("net/single-stage2.rou.xml", "w") as f):
        print(f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- generated on {time_now} by Python-->

<routes>
    <!-- VTypes -->
    <vType id="CAV_STRAIGHT" minGap="2.0" maxSpeed="15" color="yellow" carFollowModel="IDM" tau="1.2"
           accel="3.0" decel="3.0" >
    </vType>
    <vType id="CAV_LEFT" minGap="2.0" maxSpeed="15" color="red" carFollowModel="IDM" tau="1.2"
           accel="3.0" decel="3.0" >
    </vType>
    <vType id="CAV_RIGHT" minGap="2.0" maxSpeed="15" color="green" carFollowModel="IDM" tau="1.2"
           accel="3.0" decel="3.0">
    </vType>
    <vType id="HV_STRAIGHT" minGap="2.0" maxSpeed="15" color="255,250,205" carFollowModel="IDM" tau="1.6"
           accel="3" decel="3"/>
    <vType id="HV_LEFT" minGap="2.0" maxSpeed="15" color="240,128,128" carFollowModel="IDM" tau="1.6"
           decel="3" accel="3"/>
    <vType id="HV_RIGHT" minGap="2.0" maxSpeed="15" color="173,255,47" carFollowModel="IDM" tau="1.6"
           decel="3" accel="3"/>
    <vType id="CAV_LEADER" minGap="2.0" maxSpeed="15" color="0,0,255" accel="3" decel="3" tau="1.2"
           sigma="0.0"
    />
    <!-- Routes -->
    <route id="route_en" edges="e_t e_t_nc t_n t_n0"/>
    <route id="route_es" edges="e_t e_t_nc t_s t_s0"/>
    <route id="route_ew" edges="e_t e_t_nc t_w t_w0"/>
    <route id="route_ne" edges="n_t n_t_nc t_e t_e0"/>
    <route id="route_ns" edges="n_t n_t_nc t_s t_s0"/>
    <route id="route_nw" edges="n_t n_t_nc t_w t_w0"/>
    <route id="route_se" edges="s_t s_t_nc t_e t_e0"/>
    <route id="route_sn" edges="s_t s_t_nc t_n t_n0"/>
    <route id="route_sw" edges="s_t s_t_nc t_w t_w0"/>
    <route id="route_we" edges="w_t w_t_nc t_e t_e0"/>
    <route id="route_wn" edges="w_t w_t_nc t_n t_n0"/>
    <route id="route_ws" edges="w_t w_t_nc t_s t_s0"/>

    <!-- Flows on Northern route -->
    <flow id="flow_ns_CAV" type="CAV_STRAIGHT" begin="0.00" departLane="1" departSpeed="random" route="route_ns" end="3600.00" probability="{round(prob[0][1] * penetration, 3)}"/>
    <flow id="flow_ns_HV" type="HV_STRAIGHT" begin="0.00" departLane="1" departSpeed="random" route="route_ns" end="3600.00" probability="{round(prob[0][1] * (1 - penetration), 3)}"/>
    <flow id="flow_ne_CAV" type="CAV_LEFT" begin="0.00" departLane="2" departSpeed="random" route="route_ne" end="3600.00" probability="{round(prob[0][2] * penetration, 3)}"/>
    <flow id="flow_ne_HV" type="HV_LEFT" begin="0.00" departLane="2" departSpeed="random" route="route_ne" end="3600.00" probability="{round(prob[0][2] * (1 - penetration), 3)}"/>
    <flow id="flow_nw_CAV" type="CAV_RIGHT" begin="0.00" departLane="0" departSpeed="random" route="route_nw" end="3600.00" probability="{round(prob[0][0] * penetration, 3)}"/>
    <flow id="flow_nw_HV" type="HV_RIGHT" begin="0.00" departLane="0" departSpeed="random" route="route_nw" end="3600.00" probability="{round(prob[0][0] * (1 - penetration), 3)}"/>
    <!-- Flows on Southern route -->
    <flow id="flow_sn_CAV" type="CAV_STRAIGHT" begin="0.00" departLane="1" departSpeed="random" route="route_sn" end="3600.00" probability="{round(prob[2][1] * penetration, 3)}"/>
    <flow id="flow_sn_HV" type="HV_STRAIGHT" begin="0.00" departLane="1" departSpeed="random" route="route_sn" end="3600.00" probability="{round(prob[2][1] * (1 - penetration), 3)}"/>
    <flow id="flow_sw_CAV" type="CAV_LEFT" begin="0.00" departLane="2" departSpeed="random" route="route_sw" end="3600.00" probability="{round(prob[2][2] * penetration, 3)}"/>
    <flow id="flow_sw_HV" type="HV_LEFT" begin="0.00" departLane="2" departSpeed="random" route="route_sw" end="3600.00" probability="{round(prob[2][2] * (1 - penetration), 3)}"/>
    <flow id="flow_se_CAV" type="CAV_RIGHT" begin="0.00" departLane="0" departSpeed="random" route="route_se" end="3600.00" probability="{round(prob[2][0] * penetration, 3)}"/>
    <flow id="flow_se_HV" type="HV_RIGHT" begin="0.00" departLane="0" departSpeed="random" route="route_se" end="3600.00" probability="{round(prob[2][0] * (1 - penetration), 3)}"/>
    <!-- Flows on Eastern routes -->
    <flow id="flow_ew_cav" type="CAV_STRAIGHT" begin="0.00" departLane="1" departSpeed="random" route="route_ew" end="3600.00" probability="{round(prob[1][1] * penetration, 3)}"/>
    <flow id="flow_ew_hv" type="HV_STRAIGHT" begin="0.00" departLane="1" departSpeed="random" route="route_ew" end="3600.00" probability="{round(prob[1][1] * (1 - penetration), 3)}"/>
    <flow id="flow_es_cav" type="CAV_LEFT" begin="0.00" departLane="2" departSpeed="random" route="route_es" end="3600.00" probability="{round(prob[1][2] * penetration, 3)}"/>
    <flow id="flow_es_hv" type="HV_LEFT" begin="0.00" departLane="2" departSpeed="random" route="route_es" end="3600.00" probability="{round(prob[1][2] * (1 - penetration), 3)}"/>
    <flow id="flow_en_cav" type="CAV_RIGHT" begin="0.00" departLane="0" departSpeed="random" route="route_en" end="3600.00" probability="{round(prob[1][0] * penetration, 3)}"/>
    <flow id="flow_en_hv" type="HV_RIGHT" begin="0.00" departLane="0" departSpeed="random" route="route_en" end="3600.00" probability="{round(prob[1][0] * (1 - penetration), 3)}"/>
    <!-- Flows on Western routes -->    
    <flow id="flow_we_cav" type="CAV_STRAIGHT" begin="0.00" departLane="1" departSpeed="random" route="route_we" end="3600.00" probability="{round(prob[3][1] * penetration, 3)}"/>
    <flow id="flow_we_hv" type="HV_STRAIGHT" begin="0.00" departLane="1" departSpeed="random" route="route_we" end="3600.00" probability="{round(prob[3][1] * (1 - penetration), 3)}"/>
    <flow id="flow_wn_cav" type="CAV_LEFT" begin="0.00" departLane="2" departSpeed="random" route="route_wn" end="3600.00" probability="{round(prob[3][2] * penetration, 3)}"/>
    <flow id="flow_wn_hv" type="HV_LEFT" begin="0.00" departLane="2" departSpeed="random" route="route_wn" end="3600.00" probability="{round(prob[3][2] * (1 - penetration), 3)}"/>
    <flow id="flow_ws_cav" type="CAV_RIGHT" begin="0.00" departLane="0" departSpeed="random" route="route_ws" end="3600.00" probability="{round(prob[3][0] * penetration, 3)}"/>
    <flow id="flow_ws_hv" type="HV_RIGHT" begin="0.00" departLane="0" departSpeed="random" route="route_ws" end="3600.00" probability="{round(prob[3][0] * (1 - penetration), 3)}"/>
    <!-- Vehicles, persons and containers (sorted by depart) -->
    """, file=f)
        """t = 0
        vehicles_dict = {}
        depart_lane_random = False

        #  --------direction W------------ 
        num_vehicles = sum(flow[3])
        left_rate = flow[3][2]/num_vehicles
        straight_rate = flow[3][1]/num_vehicles

        delta_t = 3600 / num_vehicles
        for i in range(num_vehicles):
            prob = np.random.random()
            if prob <= straight_rate:
                route = "route_we"
            elif prob <= (left_rate + straight_rate):
                route = "route_wn"
            else:
                route = "route_ws"
            vehicle_type = "CAV" if np.random.uniform(0, 1) <= penetration else "HV"
            count = routes[route][f"count_{vehicle_type.lower()}"] = routes[route][f"count_{vehicle_type.lower()}"]+1
            depart_lane = "free" if depart_lane_random else routes[route]['target_lane_index']
            direction = routes[route]['direction']
            veh_id = f"{route}_{vehicle_type.lower()}.{count}"
            depart_time = i * delta_t + np.random.uniform(0, delta_t)
            vehicles_dict[veh_id] = (f"{vehicle_type}_{direction}", route, depart_time, depart_lane)
            # f.write(
            #     f'    <vehicle id="{veh_id}" type="{vehicle_type}_{direction}" route="{route}" '
            #     f'depart="{int(depart_time)}.0" departLane="{depart_lane}" departSpeed="random"/>\n')
        #  --------direction E------------ 
        left_rate = flow[1][2]/sum(flow[1])
        straight_rate = flow[1][1]/sum(flow[1])
        num_vehicles = sum(flow[1])
        delta_t = 3600 / num_vehicles
        for i in range(num_vehicles):
            prob = np.random.random()
            if prob <= straight_rate:
                route = "route_ew"
            elif prob <= (left_rate + straight_rate):
                route = "route_es"
            else:
                route = "route_en"
            vehicle_type = "CAV" if np.random.uniform(0, 1) <= penetration else "HV"
            count = routes[route][f"count_{vehicle_type.lower()}"] = routes[route][f"count_{vehicle_type.lower()}"]+1
            depart_lane = "free" if depart_lane_random else routes[route]['target_lane_index']
            direction = routes[route]['direction']
            veh_id = f"{route}_{vehicle_type.lower()}.{count}"
            depart_time = i * delta_t + np.random.uniform(0, delta_t)
            vehicles_dict[veh_id] = (f"{vehicle_type}_{direction}", route, depart_time, depart_lane)
        vehicles_dict = {k: v for k, v in sorted(vehicles_dict.items(), key=lambda item: item[1][2])}
        for k, v in vehicles_dict.items():
            # 'route_es_hv.241': ('HV_LEFT', 'route_es', 3599.8941121554653, 'free')
            f.write(
                f'    <vehicle id="{k}" type="{v[0]}" route="{v[1]}" '
                f'depart="{int(v[2])}.0" departLane="{v[3]}" departSpeed="random"/>\n')
        # f.write(
        #     f'    <vehicle id="{veh_id}" type="{vehicle_type}_{direction}" route="{route}" '
        #     f'depart="{int(depart_time)}.0" departLane="{depart_lane}" departSpeed="random"/>\n')
        # print(vehicles_dict)"""
        print("</routes>", file=f)


if __name__ == "__main__":
    generate_rou_file()
    print("Generate route file done!")
