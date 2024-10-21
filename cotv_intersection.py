# -*- coding: utf-8 -*-
# Python 3.10
# Windows 11 x64
import functools
import logging
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
            return Box(low=-100, high=100, shape=(7,), dtype=np.float32)
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
            if not (leader and (traci.vehicle.getRoadID(cav) == traci.vehicle.getRoadID(leader))):
                leader = None
            gap = dist if leader else -1
            delta_v = speed - traci.vehicle.getSpeed(leader) if leader else -1
            next_TLS = traci.vehicle.getNextTLS(cav)
            if not next_TLS:
                distance, state = -1, 'G'
            else:
                tlsID, _, distance, state = next_TLS[0]
            phase = 1 if 'G' in state.upper() else 0
            phaseTimeLeft = self.TL.retrieve_left_time()  # 1
            return np.array([gap, delta_v, speed, acc, distance, phase, phaseTimeLeft])
        elif agent.startswith('traffic'):
            return np.array([self.TL.retrieve_queue()]).flatten()
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
        # self.agents = self.possible_agents[-1]
        if self.episode_step != 0:
            self.close()
        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()
        self.platoons = get_nearest_platoon(self.TL_id, num=2)
        # for i, p in enumerate(self.platoons):
        #     if p:
        #         self.agents.append(self.possible_agents[i])
        self.episode_step = 0
        self.TL = TrafficSignal(self.TL_id, 3, self.sumo)
        self.TL.set_stage_duration(0, 12)
        self.TL.get_subscription_result()
        self.CAV_leaders = [[] for _ in range(8)]
        self.temp['phase'] = 0
        self.temp['cav_queue'] = deque(maxlen=100)
        self.temp['vehicle_loss'] = {}
        # the observations should be numpy arrays even if there is only one value
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {'agents_to_update': 0} for agent in self.agents}
        return observations, infos

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
        infos = {agent: {'agents_to_update': 0} for agent in self.possible_agents}
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, infos

        t = self.sim_time()
        if t % 1 == 0 and t > 0:
            check = self.TL.check()
            self.TL.pop()
            if check == -1:
                infos['traffic_light']['agents_to_update'] = 1
                # self.temp['phase'] += 2
                # self.TL.set_stage_duration(int(self.temp['phase'] % 8), 27)
        for agent, action in actions.items():
            self._apply_action(agent, action, infos)

        for _ in range(int(0.5 / self.delta_time)):
            self._sumo_step()
        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {agent: self._compute_reward(agent) for agent in self.possible_agents}
        self.agents.clear()
        self._update_env()
        if infos['traffic_light']['agents_to_update'] == 1:
            self.agents.append('traffic_light')
        # print(self.agents)
        self.episode_step += 1
        # Current observation is just the other player's most recent action
        # This is converted to a numpy value of type int to match the type
        # that we declared in observation_space()
        observations = {
            agent: self._get_observation(agent) for agent in self.possible_agents
        }
        # self.agents = (self.possible_agents[:len(self.platoon_wn)] +
        #                self.possible_agents[cavs_len:cavs_len + len(self.platoon_we)])
        termination = self.sim_time() >= self.num_seconds
        terminations = {agent: termination for agent in self.possible_agents}
        terminations["__all__"] = termination
        truncations = {agent: False for agent in self.possible_agents}
        truncations["__all__"] = False

        if termination:
            self.agents = []
        # typically there won't be any information in the infos, but there must still be an entry for each agent
        for agent in self.possible_agents:
            infos[agent] = {'agents_to_update': 1 if agent in self.agents else 0}
        infos['queue'] = np.array(sum(self.TL.retrieve_queue()))

        return observations, rewards, terminations, truncations, infos

    def _compute_reward(self, agent):
        if agent.startswith('cav'):
            platoon = self.platoons[self.possible_agents.index(agent)][0]
            if not platoon:
                return 0.0
            speeds = []
            for cav in platoon:
                speeds.append(traci.vehicle.getSpeed(cav)+0.01)
            cav = platoon[0]
            acc = traci.vehicle.getAcceleration(cav)
            r_rho = -abs(acc/3)
            r_v = np.mean(speeds)
            next_TLS = traci.vehicle.getNextTLS(cav)
            if not next_TLS:
                return r_rho + r_v
            tlsID, _, distance, state = next_TLS[0]
            phase_left = self.TL.retrieve_left_time()  # 1
            if 'G' in state.upper() and acc < 0 and distance/speeds[0] < phase_left:
                r_penalty = 5*acc
            else:
                r_penalty = 0.0
            return r_rho + r_v + r_penalty

        elif agent.startswith('traffic'):
            return -self.TL.retrieve_pressure().sum()
        else:
            raise ValueError("Agent id is invalid!")

    def _apply_action(self, agent, action, infos):
        if agent.startswith('cav'):
            platoon = self.platoons[self.possible_agents.index(agent)][0]
            if not platoon:
                return
            cav = platoon[0]
            acc = action.item() * 3
            speed = traci.vehicle.getSpeed(cav)
            # traci.vehicle.setSpeed(cav, max(0, speed+acc))
            # traci.vehicle.slowDown(cav, np.clip(speed+acc, 0, 15), 1)
        elif agent.startswith('traffic'):
            phase, duration = action
            if infos['traffic_light']['agents_to_update']:
                self.temp['phase'] += 1
                # print(int(self.temp['phase'] % 4))
                # self.TL.set_stage_duration(int(self.temp['phase'] % 4), 27)
                self.TL.set_stage_duration(int(phase), duration.item())
        else:
            raise ValueError("Agent id is invalid!")

    def _update_env(self):
        self.platoons_old = deepcopy(self.platoons)
        # self.platoons = get_nearest_platoon(self.TL_id)
        self.TL.get_subscription_result()
        for i, p in enumerate(self.platoons):
            if p:
                self.agents.append(self.possible_agents[i])
        for out_road in ['t_n', 't_e', 't_s', 't_w']:
            vehicles = traci.edge.getLastStepVehicleIDs(out_road)
            for veh_id in vehicles:
                if "cav" in veh_id.lower():
                    self.temp['vehicle_loss'][veh_id] = traci.vehicle.getTimeLoss(veh_id)

    def _sumo_step(self):
        # for lane_id in Fo_zone_w_lane:
        #     self.sumo.lane.subscribe(lane_id, [tc.LAST_STEP_MEAN_SPEED,
        #                                        tc.VAR_WAITING_TIME])
        platoons_temp = get_nearest_platoon(self.TL_id, num=2)
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
                        traci.vehicle.slowDown(cav, np.clip(a+v, 0, 15), 0.25)
                        # print(cav, traci.vehicle.getSpeedMode(cav))
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

        if self.sumo is not None:
            traci.close()
            self.sumo = None
        else:
            return


# Example usage
if __name__ == "__main__":
    env = raw_env(
        use_gui=True,
        sumo_config="net/single-stage2.sumocfg",
        sumo_seed=42,
        num_seconds=3600,
        begin_time=30,
    )
    observations, infos = env.reset()
    done = False
    while not done:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # print(actions)
        done = terminations["__all__"] or truncations["__all__"]
    env.close()
    exit(2024)
