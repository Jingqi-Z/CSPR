# -*- coding: utf-8 -*-
# Python 3.10
# Windows 11 x64
import functools
import logging
import math
import os
import sys
from collections import deque
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import sumolib
import traci
from gymnasium.spaces import Discrete, Box, Tuple
from pettingzoo import ParallelEnv

from traffic_signal import TrafficSignal
from utils import (set_no_change_lane, get_nearest_platoon, )

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ
logging.basicConfig(level=logging.INFO)
traci.setLegacyGetLeader(False)
# cooperation zone lanes 左、中、右
Co_zone_lane = {'w': ["w_t_nc_2", "w_t_nc_1", "w_t_nc_0"], 'e': ["e_t_nc_2", "e_t_nc_1", "e_t_nc_0"],
                's': ["s_t_nc_2", "s_t_nc_1", "s_t_nc_0"], 'n': ["n_t_nc_2", "n_t_nc_1", "n_t_nc_0"]}
direction_map = {"wn": 2, "we": 1, "ws": 0, "ew": 1, "en": 0, "es": 2}
Fo_zone_road = ['n_t_nc', 's_t_nc', 'e_t_nc', 'w_t_nc']
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

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare environment variable 'SUMO_HOME")


class raw_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "formation_v0"}
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
            self,
            use_gui: bool = False,
            begin_time: int = 20,
            num_seconds: int = 1800,
            sumo_config: Optional[str] = None,
            sumo_seed: Union[str, int] = "random",
            additional_sumo_cmd: Optional[str] = None,
            render_mode=None,
    ):
        """
        Initialize the environment.
        Args:
            :param use_gui: Whether to use a GUI or not.
            :param begin_time: Beginning time of the episode.
            :param num_seconds: Total number of seconds in simulation.
            :param sumo_config: An optional sumo config to use.
            :param sumo_seed: An optional sumo seed.
            :param additional_sumo_cmd: An optional additional sumo command.
        """
        super().__init__()
        self.last_step_waiting_time = None
        self.use_gui = use_gui
        self.begin_time = begin_time
        self.num_seconds = num_seconds
        self.sumo = None
        self.episode_step = 0
        self.additional_sumo_cmd = additional_sumo_cmd
        self.render_mode = render_mode
        self.label = str(raw_env.CONNECTION_LABEL)
        raw_env.CONNECTION_LABEL += 1
        self.sumo_seed = sumo_seed
        self._sumo_config = sumo_config
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")
        self.TL_id = 't'
        self.TL = None
        self.platoons = None
        self.platoons_old = None
        self.temp = {}

        self.possible_agents = [f'cav_{direction}_{turn}' for direction in ['north', 'east', 'south', 'west']
                                for turn in ['straight', 'left']] + ['traffic_light']
        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to
    # get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # Discrete(4) means an integer in range(0, 4)
        if agent.startswith('cav'):
            return Box(low=-100, high=100, shape=(8,), dtype=np.float32)
        elif agent.startswith('traffic'):
            return Box(low=-100, high=100, shape=(8,), dtype=np.float32)
        else:
            raise ValueError("Agent id is invalid!")

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent.startswith('cav'):
            return Box(low=-3, high=3, shape=(1,), dtype=np.float32)
        elif agent.startswith('traffic'):
            return Tuple((Discrete(8), Box(low=30, high=60, shape=(1,), dtype=np.float32)))
        else:
            raise ValueError("Agent id is invalid!")

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        pass

    def _get_observation(self, agent):
        if agent.startswith('cav'):
            platoon = self.platoons[self.possible_agents.index(agent)][0]
            if not platoon:
                return np.zeros(self.observation_space(agent).shape)
            cav = platoon[0]
            speed, acc = traci.vehicle.getSpeed(cav), traci.vehicle.getAcceleration(cav)
            leader, dist = traci.vehicle.getLeader(cav)
            gap = dist + 2 if leader else 50
            delta_v = speed - traci.vehicle.getSpeed(leader) if leader else 15
            next_TLS = traci.vehicle.getNextTLS(cav)
            if not next_TLS:
                distance, state = -1, 'G'
            else:
                tlsID, _, distance, state = next_TLS[0]
            left_t = self.TL.retrieve_left_time()
            if 'G' in state.upper():
                phase = 2
                phase_left = left_t + 3
            elif 'Y' in state.upper():
                phase = 1
                phase_left = self.TL.left - 1
            else:  # 'r' in state.lower()
                phase = 0
                phase_left = self.TL.left - 1
            # print(cav, f'v:{speed}, a:{acc}, gap:{gap}, dis:{distance}, {state}, {phaseTimeLeft}')
            return np.array([len(platoon), speed, acc, gap, delta_v, distance, phase, phase_left], dtype=np.float32)

        elif agent.startswith('traffic'):
            obs = self.TL.retrieve_queue().flatten()
            phase = self.TL.retrieve_stage()
            waiting_time = self.TL.retrieve_first_waiting()
            # state = np.concatenate([[phase], obs]).flatten()
            state = np.insert(obs, 0, phase)
            # print(state)
            # print(waiting_time)
            return obs
        else:
            raise ValueError("Agent id is invalid!")

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-c",
            self._sumo_config
        ]
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            print(sumo_cmd, self.label)
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
            self.label = str(raw_env.CONNECTION_LABEL)
        traci.trafficlight.setProgram(self.TL_id, '8-phase')
        set_no_change_lane()
        if self.begin_time > 0:
            self.sumo.simulation.step(time=self.begin_time)
        self.delta_time = self.sumo.simulation.getDeltaT()

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        if self.episode_step != 0:
            self.close()
        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()
        self.platoons = get_nearest_platoon(self.TL_id, max_len=4, num=2)
        self.episode_step = 0
        self.TL = TrafficSignal(self.TL_id, 3, self.sumo)
        # self.TL.set_stage_duration(0, 12)
        self.TL.clear_schedule()
        self.TL.get_subscription_result()
        self.temp['phase'] = 0
        self.temp['cav_queue'] = deque(maxlen=100)
        self.temp['vehicle_loss'] = {}
        self.temp['filtered_vehicle_delays'] = {}
        self.last_step_waiting_time = 0.0
        # the observations should be numpy arrays even if there is only one value
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        # print(self.get_active_agents())
        self.agents = self.get_active_agents()
        infos = {agent: {'agents_to_update': 1 if agent in self.agents else 0}
                 for agent in self.possible_agents}
        # infos['traffic_light']['agents_to_update'] = 1
        return observations, infos

    def get_active_agents(self):
        agents = []
        if self.sim_time() % 1 == 0:
            if not self.TL.schedule or self.TL.schedule == -1:
                agents.append('traffic_light')
            # try:
            #     check = self.TL.check()
            #     if check == -1:
            #         agents.append('traffic_light')
            # except Exception as e:
            #     agents.append('traffic_light')
            # print(e)
        platoons = deepcopy(self.platoons)
        for i, p in enumerate(platoons):
            platoon = p[0]
            if platoon:
                cav = platoon[0]
                next_TLS = traci.vehicle.getNextTLS(cav)
                leader, dist = traci.vehicle.getLeader(cav)
                if not next_TLS or ('G' in traci.vehicle.getNextTLS(cav)[0][3].upper()) or \
                        (dist < 150 and traci.vehicle.getSpeed(cav) >= 0.1):
                    agents.append(self.possible_agents[i])
        return agents

    def get_queue_len(self):
        return self.TL.retrieve_queue().sum()

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        :type actions: dict
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        '''if not actions:
            # self.agents = []
            infos = {agent: {'agents_to_update': 0} for agent in self.possible_agents}
            infos['queue'] = sum(self.TL.retrieve_queue())
            return {}, {}, {}, {}, infos'''
        for agent, action in actions.items():
            self._apply_action(agent, action)
            # if check == -1:
            #     infos['traffic_light']['agents_to_update'] = 1
        # step on 0.5s
        for _ in range(int(1.0 / self.delta_time)):
            self._sumo_step()
        if self.sim_time() % 1 == 0:
            # check = self.TL.check()
            self.TL.check()
            self.TL.pop()
        # rewards for all agents are placed in the rewards dictionary to be returned
        # rewards = {agent: self._compute_reward(agent) for agent in self.possible_agents}
        rewards = {}
        self.agents = self.get_active_agents()
        for agent in self.possible_agents:
            rewards[agent] = self._compute_reward(agent) if agent in self.agents else 0.0

        self._update_env()
        self.episode_step += 1

        # print(f"available agents: {self.agents}")
        observations = {
            agent: self._get_observation(agent) for agent in self.agents
        }
        termination = self.sim_time() >= self.num_seconds
        terminations = {agent: termination for agent in self.possible_agents}
        terminations["__all__"] = termination
        truncations = {agent: False for agent in self.possible_agents}
        truncations["__all__"] = False

        if termination:
            self.agents = []
        # typically there won't be any information in the infos, but there must still be an entry for each agent
        infos = {agent: {'agents_to_update': 1 if agent in self.agents else 0}
                 for agent in self.possible_agents}
        infos['queue'] = self.TL.retrieve_queue()
        return observations, rewards, terminations, truncations, infos

    def _compute_reward(self, agent):
        if agent.startswith('cav'):
            platoon = self.platoons[self.possible_agents.index(agent)][0]
            if not platoon:
                return 0.0
            speeds = []
            for cav in platoon:
                speeds.append(traci.vehicle.getSpeed(cav) + 0.01)
            cav = platoon[0]
            acc = traci.vehicle.getAcceleration(cav)
            r_v = np.mean(speeds)
            next_TLS = traci.vehicle.getNextTLS(cav)
            if not next_TLS:
                return r_v

            def interval_intersection(interval1, interval2):
                a1, b1 = interval1
                a2, b2 = interval2
                # 计算交集的上下界
                a3 = max(a1, a2)
                b3 = min(b1, b2)
                # 检查是否存在交集
                if a3 <= b3:
                    return a3, b3
                else:
                    return None, None

            r_rho = -abs(acc) / 3
            tlsID, _, distance, state = next_TLS[0]
            left_t = self.TL.retrieve_left_time()
            if 'G' in state.upper():
                # phase = 2
                phase_left = left_t + 3
                t_g_l, t_g_e = 0.1, phase_left
                # t_r_l, t_r_e = phase_left + 1, phase_left + 1 + 17
            elif 'Y' in state.upper():
                # phase = 1
                phase_left = self.TL.left - 1
                t_g_l, t_g_e = 0.1, phase_left
                # t_r_l, t_r_e = phase_left + 1, phase_left + 1 + 17
            else:  # 'r' in state.lower()
                # phase = 0
                phase_left = self.TL.left - 1
                # t_r_l, t_r_e = 0.1, phase_left
                t_g_l, t_g_e = phase_left + 1, phase_left + 1 + 20
            # print(cav, traci.vehicle.getPosition(cav), state, traci.trafficlight.getPhase('t'), phase_left)
            v_min, v_max = interval_intersection((distance / t_g_e, distance / t_g_l), (0, 15.1))
            r_tl = 0.0
            if v_min:
                if v_min <= speeds[0] <= v_max:
                    r_tl = (speeds[0] - v_min) / (v_max - v_min) + 0.5
                else:
                    r_tl = -1

            # if 'G' in state.upper() and acc < 0 and distance/speeds[0] < phase_left:
            #     r_penalty = 5*acc
            # else:
            #     r_penalty = 0.0
            # print(cav, v_min, v_max, r_tl, t_g_l, t_g_e)
            # print(cav, traci.vehicle.getFuelConsumption(cav), 'mg')

            return r_v + r_tl + r_rho

        elif agent.startswith('traffic'):

            r_queue = -self.TL.retrieve_queue().sum()
            # r_wait = -(sum(self.TL.retrieve_first_waiting())/50)**2
            # print(r_queue, r_wait)
            r = self.TL.retrieve_reward().sum()
            return r
            # return self.TL.retrieve_pressure().sum()
        else:
            raise ValueError("Agent id is invalid!")

    def _apply_action(self, agent, action):
        if agent not in self.agents:
            return

        def calculate_safe_speed(b, tr, td, vp, rho, g):
            # 计算平方根内的表达式
            term1 = b * (tr + td / 2)
            term2 = b ** 2 * (tr + td / 2) ** 2
            term3 = b * (vp * td + vp ** 2 / rho + 2 * g)
            # 计算安全速度
            v_safe = -term1 + math.sqrt(term2 + term3)

            return v_safe

        if agent.startswith('cav'):
            platoon = self.platoons[self.possible_agents.index(agent)][0]
            if not platoon:
                return
            cav = platoon[0]
            a = action.item() * 3
            v = traci.vehicle.getSpeed(cav)
            next_TLS = traci.vehicle.getNextTLS(cav)
            if not next_TLS:
                a = 3.0
            traci.vehicle.setType(cav, 'CAV_LEADER')
            '''leader, gap = traci.vehicle.getLeader(cav)
            # a0 = traci.vehicle.getAcceleration(leader) if leader else 0
            v0 = traci.vehicle.getSpeed(leader) if leader else v + 3
            if not leader:
                gap = 100
            elif gap < 0:
                gap = 0
            next_TLS = traci.vehicle.getNextTLS(cav)
            if next_TLS:
                tlsID, _, distance, state = next_TLS[0]
            else:
                distance, state = 100.0, 'g'
            a = np.clip(a, -3, 3)
            traci.vehicle.setSpeedMode(cav, 0x011111)
            traci.vehicle.setColor(cav, color=(0, 0, 255))
            vs = calculate_safe_speed(3, 1.5, 0.0, v0, 2, gap)'''
            traci.vehicle.slowDown(cav, np.clip(v + a, 0, 15), 1.0)
        elif agent.startswith('traffic'):
            phase, duration = action
            self.temp['phase'] += 1
            # fixed phase duration
            dur_map = {0: 12, 1: 12, 2: 27, 3: 22}
            # self.TL.set_stage_duration(int(self.temp['phase'] % 4), dur_map[int(self.temp['phase'] % 4)])
            # print(self.temp['phase'], dur_map[int(self.temp['phase'] % 4)])
            self.TL.set_stage_duration(phase, int(duration.item()))
        else:
            raise ValueError("Agent id is invalid!")

    def _update_env(self):
        # self.platoons_old = deepcopy(self.platoons)
        # self.platoons = get_nearest_platoon(self.TL_id)
        self.TL.get_subscription_result()
        phase_left = self.TL.retrieve_left_time()
        phase_code = phase_encoding[self.TL.retrieve_stage()]
        # 使用列表推导式选择phase_code为1的inlanes元素
        platoons = [self.platoons[i] for i in range(8) if phase_code[i] == 1]
        # print(phase_left)
        '''if self.sim_time() % 1 == 0 and phase_left == 2:
            min_t = 100
            for platoon in platoons:
                if len(platoon[0]) > 1:
                    speed = traci.vehicle.getSpeed(platoon[0][-1])
                    next_TLS = traci.vehicle.getNextTLS(platoon[0][-1])
                    if next_TLS:
                        tlsID, _, distance, state = next_TLS[0]
                    else:
                        distance = 0.0
                    t = distance / (speed + 0.1)
                    min_t = min(min_t, t)
                    # print(platoon[0][-1], t)
            if 2 < min_t < 7:
                for i in range(1, round(min_t)):
                    # self.TL.schedule.appendleft(0)
                    pass'''
        for out_road in ['t_n', 't_e', 't_s', 't_w']:
            vehicles = traci.edge.getLastStepVehicleIDs(out_road)
            for veh_id in vehicles:
                veh_delay = traci.vehicle.getTimeLoss(veh_id)
                self.temp['vehicle_loss'][veh_id] = veh_delay
                if veh_id.split('_')[1] not in ['nw', 'en', 'se', 'ws']:
                    self.temp['filtered_vehicle_delays'][veh_id] = veh_delay

    def _sumo_step(self):
        # for lane_id in Fo_zone_w_lane:
        #     self.sumo.lane.subscribe(lane_id, [tc.LAST_STEP_MEAN_SPEED,
        #                                        tc.VAR_WAITING_TIME])
        self.TL.left -= self.delta_time
        platoons_temp = get_nearest_platoon(self.TL_id, max_len=6, num=2)
        for p, pt, pha in zip(self.platoons, platoons_temp, phase_encoding[self.TL.retrieve_stage()]):
            if not p[0]:
                p.clear()
                p.extend(pt)
            elif not pha:
                p.clear()
                p.extend(pt)
            elif all(traci.vehicle.getRoadID(cav) not in Fo_zone_road for cav in p[0]) or p[0][0] == pt[0][0]:
                p.clear()
                p.extend(pt)
        # print(self.platoons)
        for platoon in self.platoons:
            pla = platoon[0]
            if pla:
                v0 = traci.vehicle.getSpeed(pla[0])
                pos0 = traci.vehicle.getPosition(pla[0])
                for n, cav in enumerate(pla):
                    if n == 0:
                        # traci.vehicle.setAcceleration(cav, random.random(), 0.5)
                        pass
                        # traci.vehicle.setTau(cav, 1.5)
                        traci.vehicle.setSpeedMode(cav, 31)
                    else:
                        T = 0.8
                        k_d, k_v, l, s_0 = 1 / (n * T) ** 2, 1 / (n * T), 5.0, 2.0
                        v = traci.vehicle.getSpeed(cav)
                        pos = traci.vehicle.getPosition(cav)
                        gap = ((pos0[0] - pos[0]) ** 2 + (pos0[1] - pos[1]) ** 2) ** 0.5
                        a = k_d * (gap - n * (T * v + l + s_0)) + k_v * (v0 - v)
                        a = np.clip(a, -9, 3)
                        # traci.vehicle.setMinGap(cav, T * s_0)
                        traci.vehicle.setTau(cav, T)
                        traci.vehicle.setSpeedMode(cav, 0x011110)
                        # traci.vehicle.setAcceleration(cav, a, 0.5)
                        traci.vehicle.slowDown(cav, np.clip(a + v, 0, 15), self.delta_time)
        self.sumo.simulationStep()

    def sim_time(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        print(np.mean(list(self.temp['vehicle_loss'].values())))
        if self.sumo is not None:
            self.sumo.close()
            self.sumo = None
        else:
            return


# Example usage
if __name__ == "__main__":
    env = raw_env(
        use_gui=True,
        sumo_config="net/single-stage2.sumocfg",
        sumo_seed=42,
        num_seconds=1860,
        begin_time=60,
        # additional_sumo_cmd='--tripinfo-output tripinfo.xml',
    )
    observations, infos = env.reset()
    done = False
    while not done:
        # this is where you would insert your policy
        agent = 'traffic_light'
        actions = {}
        if agent in env.agents:
            actions = {agent: env.action_space(agent).sample()}
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        # print(actions['traffic_light'])
        observations, rewards, terminations, truncations, infos = env.step(actions)
        if actions:
            print(f"actions: {actions}, rewards: {rewards[agent]}", env.sim_time())
        done = terminations["__all__"] or truncations["__all__"]
    env.close()
    exit(2024)
