from typing import Tuple

import numpy as np
import gym
import pandas as pd
from pygeodesy.ellipsoidalExact import LatLon
from stable_baselines3 import PPO, A2C
import tensorflow as tf
from tensorflow.keras import layers
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


class ActorCritic(tf.keras.Model):
    def __init__(
        self, num_actions: int, num_hidden_units: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions, activation="sigmoid")
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


class AstarDummy:
    def __init__(self, origin, destination, model: ActorCritic, forward_points: int = 41, lateral_points: int = 11):
        # Coefficient to be added/multiplied to heuristic's cost
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Mean and std of the observed wind magnitude and direction
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self._mean_wind_magnitude = 0
        self._mean_wind_direction = 0
        self.cost_coefficient = 1
        self.dummy_env = FlightPlanningDomain(origin=origin, destination=destination, nb_forward_points=forward_points,
                                              nb_lateral_points=lateral_points, nb_vertical_points=1,
                                              actype="A320", heuristic_name="fuel", objective="fuel", noisy=True)
        self.graph = [[item for sublist in lst for item in sublist] for lst in self.dummy_env.network]
        self.winds = calculate_winds(self.graph)
        self.start = (-1, -1)
        self.target = (np.inf, np.inf)
        self.parenthood = {}
        self.current = self.start
        self.model = model
        self.actions = []

    def calculate_path(self, current=None):
        path = []
        if current is None:
            current = self.target
        while current != self.start:
            path.append(current)
            current = self.parenthood[current]
        path.append(self.start)
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
        tas = mach2tas(aircraft("A320")["cruise"]["mach"], 31000 * ft)
        gs = compute_gspeed(
            tas=tas,
            true_course=aero_bearing(_from[0], _from[1],
                                     _to[0], _to[1]),
            wind_speed=self.winds[_from[0]][_from[1]][0],
            wind_direction=self.winds[_from[0]][_from[1]][1]
        )
        dt = LatLon.distanceTo(_from, _to) / gs
        g = self.dummy_env.perf_model.compute_fuel_consumption(
            values_current={"mass": 75000, "alt": 31000, "speed": tas / knt, "temp": temp},
            delta_time=dt,
            path_angle=0  # TODO: put a sensible value
        )
        h = self.heuristic()
        pred = self.model.predict(self._get_obs())
        self.actions.append((pred, self.current))
        return g + parent_cost, h * self._action_to_coefficient(pred)

    def _action_to_coefficient(self, action):
        return action * 2

    def _get_obs(self):
        # current_wind = self.winds[self.current[0]][self.current[1]]
        current_path = self.calculate_path(self.current)
        wind_m, wind_d = 0, 0
        for node in current_path:
            wind = self.winds[node[0]][node[1]]
            wind_m += wind[0]
            wind_d += wind[1]
        wind_m /= len(current_path)
        wind_d /= len(current_path)
        return tf.tensor([wind_m, wind_d, self.winds[self.current[0]][self.current[1]][0]])

    def _get_info(self):
        return {"agent_location": self.graph[self.current[0]][self.current[1]],
                "target_location": self.target,
                "distance": LatLon.distanceTo(self.graph[self.current[0]][self.current[1]], self.target)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def execute(self):
        open_list = []
        closed_list = []
        h = self.heuristic()
        pred = self.model.predict(self._get_obs())
        open_list.append((self.start, (0, h * self._action_to_coefficient(pred))))
        goal = False

        while not goal:
            assert len(open_list) > 0, "No path found!"
            current_id = np.argmin([x[1][0] + x[1][1] for x in np.array(open_list)])
            current, current_cost = open_list.pop(current_id)
            self.current = current
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
                    node_c = self.calculate_cost(current, node, current_cost)
                    open_list.append((node, node_c))

        return self.actions

    def heuristic(self):
        starting_point = self.graph[self.current[0]][self.current[1]]
        destination = self.target
        temp = 273.15
        ft = 0.3048
        knt = 0.514444
        tas = mach2tas(aircraft("A320")["cruise"]["mach"], 31000 * ft)
        e = self.dummy_env.perf_model.compute_fuel_consumption(
            values_current={"mass": 75000, "alt": 31000, "speed": tas / knt, "temp": temp},
            delta_time=LatLon.distanceTo(starting_point, destination) / tas,
            path_angle=0  # TODO: put a sensible value
        )
        return e


def from_dt(_from, lat_points=41):
    lat, lon = _from
    return pd.DataFrame([{
        "ts": 0,
        "lat": lat,
        "lon": lon,
        "mass": 75000,  # TODO: put a sensible value
        "mach": aircraft("A320")["cruise"]["mach"],
        "fuel": 0.0,
        "alt": 0,
        }])


def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor, c_loss) -> tf.Tensor:
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = c_loss(values, returns)

    return actor_loss + critic_loss



if __name__ == "__main__":
    astar = AstarDummy(origin=(0, 0), destination=(10, 10), model=ActorCritic(1, 128))

    iterations = 1000
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    for i in range(iterations):
        astar.execute()
        actions = astar.actions
        rewards = [astar.dummy_env.flying(from_dt(x[1]), (astar.target[0], astar.target[1], 0)).iloc[-1]["mass"]
                   - 75000 for x in actions]
        compute_loss([x[1] for x in actions], rewards, huber_loss)

