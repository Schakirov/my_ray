"""Example of a custom gym environment. Run this for a demo.

This example shows:
  - using a custom environment
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Discrete, Box

import ray
from ray.tune import run_experiments, grid_search
from cv2 import *
import time
import numpy as np


class SimpleCorridor3(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            0.0, 1.0, shape=(84, 84, 3), dtype=np.float32)

    def reset(self):
        self.cur_pos = np.ones((84,84,3)) * 100
        return self.cur_pos

    def step(self, action):
        assert action in [0, 1], action
        regime = "manual"
        if regime == "manual":
            time.sleep(0.5)
            if action == 0:
                print("down! snapshot in 2 sec")
            elif action == 1:
                print("up! snapshot in 2 sec")
            time.sleep(2)
            print("snapshot!")
            cam = VideoCapture(5)
            s, img = cam.read()
            img = cv2.resize(img, (84,84))/255.0
            print("rew = np.mean(img) = ", np.mean(img))
            print("Thinking... I'll command you in 2 sec\n")
            time.sleep(1.5)
        if regime == "auto":
            time.sleep(0.05)
            if action == 0:
                img = np.zeros((84,84,3))
            if action == 1:
                img = np.ones((84,84,3)) * 0.5
        rew = np.mean(img)
        done = np.mean(img) < 0.7
        return img, rew, done, {}  # "rew if done else 0"  was here


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor3(config))
    ray.init()
    run_experiments({
        "demo": {
            "run": "DQN",
            "env": SimpleCorridor3,  # or "corridor" if registered above
            "checkpoint_freq": 1,
            "stop": {
                "timesteps_total": 1000,
            },
            "config": {
                "lr": grid_search([0.02]),  # try different lrs
                "schedule_max_timesteps": 1,  #exploration decreases from 1 to 0.1 over that
                "exploration_final_eps": 0.2,
                "learning_starts": 0,  #before that no learning at all
                "train_batch_size": 4,
                "timesteps_per_iteration": 4, # only after that target_network_update_freq is checked
                "target_network_update_freq": 4,  #not 500
                "dueling": False,
                "double_q": False,
                "num_workers": 1,  # parallelism
                "env_config": {
                    "corridor_length": 5,
                },
            },
        },
    })
