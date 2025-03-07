import logging
from collections import deque
import numpy as np
import traci
import traci.constants as tc


class TrafficSignal:
    """
    This class represents a Traffic Signal controlling an intersection.
    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT!!! NOTE THAT
    Our reward is defined as the change in vehicle number of one specific junction.
    Our observations are defined as the pressure between the inlanes and outlanes.
    """
    def __init__(self, tl_id, yellow, sumo, pattern=None):

        self.id = tl_id
        self.pattern = pattern
        self.yellow = yellow
        self.sumo = sumo

        # The schedule is responsible for the automatic timing for the incoming green stage.
        # | 0 | 0 | 0 | 16 |
        # | yellow len| when 16 is dequeued, the stage is automatically transferred to the green stage and 16 is for duration.
        self.schedule = deque()
        self.duration = None
        # Links is relative with connections defined in the rou.xml, what's more the connection definition should be
        # relative with traffic state definition. Therefore, there is no restriction that the connection should start
        # at north and step clockwise then.
        all_lanes = self.sumo.trafficlight.getControlledLinks(self.id)
        self.in_lanes = [conn[0][0] for conn in all_lanes]
        self.out_lanes = [conn[0][1] for conn in all_lanes]
        # Delete the right turn movement.
        del self.in_lanes[0::3]
        # del self.in_lanes[0::2]
        # del self.in_lanes[0::3]
        del self.out_lanes[0::3]
        # del self.out_lanes[0::2]
        # del self.out_lanes[0::3]
        # print(f"in lanes:{self.in_lanes}; out lanes:{self.out_lanes}")

        self.subscribe()
        self.inlane_halting_vehicle_number = None
        self.inlane_halting_vehicle_number_old = None
        self.inlane_vehicle_number = None
        self.inlane_vehicles = None
        self.outlane_vehicle_number = None
        self.inlane_waiting_time = None
        self.inlane_arrival = None
        self.outlane_halting_vehicle_number = None
        self.outlane_waiting_time = None
        self.outlane_vehicles = None
        self.outlane_halting_vehicle_number = None
        # self.stage_old = np.random.randint(0, 8)
        self.stage_old = self.sumo.trafficlight.getPhase(self.id)
        self.left = 1000

        self.mapping = np.array([
            [-1, 8, 8, 8, 9, 8, 10, 8],
            [11, -1, 11, 11, 12, 11, 13, 11],
            [14, 14, -1, 14, 14, 15, 14, 16],
            [17, 17, 17, -1, 17, 18, 17, 19],
            [20, 21, 22, 22, -1, 22, 22, 22],
            [23, 23, 24, 25, 23, -1, 23, 23],
            [26, 27, 28, 28, 28, 28, -1, 28],
            [29, 29, 30, 31, 29, 29, 29, -1],
        ])

    def set_stage_duration(self, stage: int, duration: int):
        """
        Call this at the beginning the of one stage, which includes the switching yellow light between two different
        green light.
        In add.xml the stage is defined as the yellow stage then next green stage, therefore the yellow stage is first
        implemented, and after self.yellow seconds, it will automatically transfer to green stage, through a schedule to
        set the incoming green stage's duration.
        :return:
        """
        # 黄灯
        if self.stage_old is not None and self.stage_old != stage:
            yellow_stage = int(self.mapping[self.stage_old][stage])
            # yellow_stage = int((self.stage_old+1) % 8)
            # logging.info(f"Switching from {self.stage_old} to {stage} with yellow stage {yellow_stage}")
            self.sumo.trafficlight.setPhase(self.id, yellow_stage)
            # self.left = 3
            for i in range(self.yellow - 1):
                self.schedule.append(0)  # 右侧添加
        self.duration = int(duration)
        self.stage_old = int(stage)
        self.schedule.append(duration)

    def check(self):
        """
        Check whether the yellow stage is over and automatically extend the green light.
        # | 0 | 0 | 0 | 16 |  --->  | 0 | 0 | 0 | 0 | ... | 0 | -1 |
        #                                       {     16X     } where -1 indicates that the agent should get a new action
        :return:
        """
        if self.schedule[0] > 0:
            self.sumo.trafficlight.setPhase(self.id, self.stage_old)
            self.left = self.duration + self.yellow + 1
            for i in range(self.schedule[0]):
                self.schedule.append(0)
            self.schedule.popleft()
            self.schedule.append(-1)

        return self.schedule[0]

    def pop(self):
        self.schedule.popleft()

    def clear_schedule(self):
        self.schedule.clear()

    def subscribe(self):
        """
        Pre subscribe the information we interest, to accelerate the information retrieval.
        See https://sumo.dlr.de/docs/TraCI.html "Performance" for more detailed explanation.
        :return:
        """
        for lane_id in self.in_lanes:
            self.sumo.lane.subscribe(lane_id, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                                               tc.VAR_WAITING_TIME,
                                               tc.LAST_STEP_VEHICLE_NUMBER,
                                               tc.LAST_STEP_VEHICLE_ID_LIST], )
        for lane_id in self.out_lanes:
            self.sumo.lane.subscribe(lane_id, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                                               tc.VAR_WAITING_TIME,
                                               tc.LAST_STEP_VEHICLE_NUMBER,
                                               tc.LAST_STEP_VEHICLE_ID_LIST])

    def get_subscription_result(self):

        self.inlane_halting_vehicle_number = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[0] for lane_id in self.in_lanes])

        self.outlane_halting_vehicle_number = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[0] for lane_id in self.out_lanes])

        self.inlane_waiting_time = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[1] for lane_id in self.in_lanes])

        self.inlane_vehicle_number = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[2] for lane_id in self.in_lanes])
        self.outlane_vehicle_number = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[2] for lane_id in self.out_lanes])
        self.inlane_vehicles = [list(list(self.sumo.lane.getSubscriptionResults(lane_id).values())[3]) for lane_id in
                                self.in_lanes]
        self.outlane_vehicles = [list(list(self.sumo.lane.getSubscriptionResults(lane_id).values())[3]) for lane_id in
                                 self.out_lanes]
        # print(self.inlane_vehicle_number, self.outlane_vehicle_number)

    def retrieve_reward(self):
        if not isinstance(self.inlane_halting_vehicle_number_old, np.ndarray):
            reward = -sum(self.inlane_halting_vehicle_number)
        else:
            reward = sum(self.inlane_halting_vehicle_number_old) - sum(self.inlane_halting_vehicle_number)
            # print(f"old: {self.inlane_halting_vehicle_number_old}, now: {self.inlane_halting_vehicle_number}")
        self.inlane_halting_vehicle_number_old = self.inlane_halting_vehicle_number

        return reward

    def retrieve_pressure(self):
        pressure = self.inlane_halting_vehicle_number - self.outlane_halting_vehicle_number
        pressure = self.outlane_vehicle_number - self.inlane_vehicle_number
        return pressure

    def retrieve_queue(self):
        queue = self.inlane_halting_vehicle_number
        return queue

    def retrieve_first_waiting(self):
        waiting = np.zeros(len(self.inlane_vehicles))
        for lane_index, vehicles in enumerate(self.inlane_vehicles):
            if vehicles:
                vehicles.sort(key=self.sumo.vehicle.getDistance, reverse=True)
                waiting[lane_index] = self.sumo.vehicle.getWaitingTime(vehicles[0])
        return waiting

    def retrieve_waiting_time(self):
        waiting_time = self.inlane_waiting_time
        return waiting_time

    def retrieve_stage(self):
        return self.stage_old

    def retrieve_left_time(self):
        if not self.schedule:
            return 0
        if self.schedule[-1] == -1:
            return len(self.schedule)
        else:
            return len(self.schedule) - 1 + self.schedule[-1]

    def retrieve_arrival(self):
        arrival = self.inlane_arrival
        return arrival
