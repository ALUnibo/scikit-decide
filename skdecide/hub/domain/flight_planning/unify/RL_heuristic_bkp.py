import numpy as np
import gym
from pygeodesy.ellipsoidalExact import LatLon
from stable_baselines3 import PPO, A2C
from time import time
from stable_baselines3.common.vec_env import DummyVecEnv
from skdecide.hub.domain.flight_planning.domain import State
from skdecide.hub.domain.flight_planning.domain import (FlightPlanningDomain, compute_gspeed, mach2tas,
                                                        aircraft, aero_bearing)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.get_ensemble_weather import get_wind_values


def calculate_winds(graph):
    tic = time()
    _, _, ds = get_wind_values(0, 0, 0, noisy=False)  # Collecting wind dataset
    winds = [[
            get_wind_values(graph[i][j].lat, graph[i][j].lon,
                            500, noisy=True, ds=ds)[:2]  # TODO: put a sensible value
            for j in range(len(graph[0]))
        ]
        for i in range(len(graph))
    ]
    toc = time()
    print(f"Time to calculate winds: {toc - tic} s")
    return winds


class AstarDummy:
    def __init__(self, graph, winds, env, heuristic=lambda x, y: 0):
        self.graph = graph
        self.winds = winds
        self.env = env
        self.start = (-1, -1)
        self.target = (np.inf, np.inf)
        self.parenthood = {}
        self.heuristic = heuristic

    def execute(self):
        open_list = []
        closed_list = []
        open_list.append((self.start, 0))
        goal = False

        while not goal:
            assert len(open_list) > 0, "No path found!"
            current_id = np.argmin([x[1][0] + x[1][1] for x in np.array(open_list)])
            current, current_cost = open_list.pop(current_id)
            if current == self.target:
                goal = True
                continue
            closed_list.append((current, current_cost))
            for node in self.get_neighbors(current):

                for lst in [open_list, closed_list]:
                    if node in [x[0] for x in lst]:
                        node_c = self.calculate_cost(current, node, current_cost)
                        if current_cost + node_c < node[1]:
                            self.parenthood[node] = current
                            lst.remove([x for x in closed_list if x[0] == node][0])
                            open_list.append((node, node_c))
                else:
                    node_c = self.calculate_cost(current, node,current_cost)
                    open_list.append((node, node_c))
        print("Path found!")

    def calculate_path(self):
        path = []
        current = self.target
        while current != self.start:
            path.append(current)
            current = self.parenthood[current]
        return path[::-1]

    def get_neighbors(self, current):
        if current == self.start:
            return [(0, x) for x in range(5)]
        if current == self.target:
            return []
        neighbors = [(current[0] + 1, current[1])]
        if current[1] - 1 >= 0:
            neighbors.append((current[0] + 1, current[1] - 1))
        if current[1] + 1 < len(self.graph[0]):
            neighbors.append((current[0] + 1, current[1] + 1))
        return neighbors

    def calculate_cost(self, _from, _to, parent_cost):
        temp = 273.15
        ft = 0.3048
        knt = 0.514444
        origin = self.graph[_from[0]][_from[1]]
        destination = self.graph[_to[0]][_to[1]]
        tas = mach2tas(aircraft("A320")["mach"], 31000 * ft)
        gs = compute_gspeed(
            tas=tas,
            true_course=aero_bearing(_from[0], _from[1],
                                     _to[0], _to[1]),
            wind_speed=self.winds[origin[0], origin[1]][0],
            wind_direction=self.winds[origin[0], origin[1]][1]
        )
        dt = LatLon.distanceTo(_from, _to) / gs
        g = self.env.perf_model.compute_fuel_consumption(
            values_current={"mass": 75000, "alt": 31000, "speed": tas / knt, "temp": temp},
            delta_time=dt,
            path_angle=0  # TODO: put a sensible value
        )
        h = self.heuristic(destination, self.winds[origin[0], origin[1]])
        return g + parent_cost, h


class FlightTraining(gym.Env):
    metadata = {"render_modes": [], "render_fps": None}

    def __init__(self, start, target, forward_points: int = 41, lateral_points: int = 11):
        self.start = start
        self.target = target
        # Coefficient to be added/multiplied to heuristic's cost
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Mean and std of the observed wind magnitude and direction
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self._mean_wind_magnitude = 0
        self._mean_wind_direction = 0
        self.cost_coefficient = 1
        self.dummy_env = FlightPlanningDomain(origin=start, destination=target, nb_forward_points=forward_points,
                                              nb_lateral_points=lateral_points, nb_vertical_points=1,
                                              actype="A320", heuristic_name="fuel", objective="fuel", noisy=True)
        self.graph = [[item for sublist in lst for item in sublist] for lst in self.dummy_env.network]
        self.winds = calculate_winds(self.graph)
        self.astar = AstarDummy(self.graph, self.winds, self.dummy_env)

    def _action_to_coefficient(self, action):
        return action * self.cost_coefficient

    def _get_obs(self):
        current_wind = self.winds[self._agent_location[0]][self._agent_location[1]]
        return np.array([self._mean_wind_magnitude, self._mean_wind_direction,
                         current_wind[0], current_wind[1]])

    def _get_info(self):
        # location_in_km = self._agent_location * np.array([110, 110, 1/3200])
        # target_in_km = self._target_location * np.array([110, 110, 1/3200])
        return {"agent_location": (self._agent_location - self.start),
                "target_location": (self._target_location - self.start),
                "distance": np.linalg.norm(self._target_location - self._agent_location)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        action_value = self._action_to_coefficient(action)
        reward, terminated = self.execute_step(action_value)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, info

    def execute_step(self, action_value):


        reward = 0
        terminated = False

        return reward, terminated


class UnifyAgent(A2C):
    def __init__(self, **kwargs):
        env = FlightTraining()
        super().__init__("MlpPolicy", env, verbose=0, **kwargs)


if __name__ == "__main__":
    astar = FlightTraining(start=(10, 10, 0), target=(20, 20, 0))
