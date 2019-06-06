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
from termcolor import colored
import pyttsx3
import keyboard

import ray
from ray.tune import run_experiments, grid_search
from cv2 import *
import time
import numpy as np


class SimpleCorridor4(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(3)
        self.observation_space = Box(
            0.0, 1.0, shape=(84, 84, 3), dtype=np.float32)
        self.speak_engine = pyttsx3.init();
        self.auto_img = np.zeros((84,84,3))
        self.auto_img_open = 0.5 + 0.1 * np.random.randn(84,84,3)
        self.history = np.zeros((3,5))
        self.img2 = np.zeros((84,84,3))
        self.count = 0
        self.sleep_status = 0  #initially awake

    def reset(self):
        self.cur_pos = self.img2 #np.ones((84,84,3)) * 100
        return self.cur_pos

    def step(self, action):
        self.count = self.count + 1
        assert action in [0, 1, 2], action
        self.history[0, 0:3+1] = self.history[0, 1:4+1]
        self.history[1, 0:3+1] = self.history[1, 1:4+1]
        self.history[2, 0:3+1] = self.history[2, 1:4+1]
        self.history[0:3, 4] = np.zeros((3))
        self.history[action, 4] = 1
        self.img2 = np.zeros((84,84,3))
        self.img2[0:3,0:5,0] = self.history
        regime = "manual"
        if regime == "manual" and not self.sleep_status:
            time.sleep(0.5)
            if action == 0:
                print(colored("m", "red"), "snapshot in 2 sec")
                self.speak_engine.say("m");
                self.speak_engine.runAndWait() ;
            elif action == 1:
                print(colored("a", "red"), "snapshot in 2 sec")
                self.speak_engine.say("ahh");
                self.speak_engine.runAndWait() ;
            elif action == 2:
                print(colored("p", "red"), "snapshot in 2 sec")
                self.speak_engine.say("p");
                self.speak_engine.runAndWait() ;
            time.sleep(0.125)
            print("snapshot!")
            cam = VideoCapture(0)
            s, img = cam.read()
            img = cv2.resize(img, (84,84))/255.0
            rew = self.operator_reward() # np.mean(img) + self.operator_reward()
            
            print("rew = np.mean(img) = ", np.mean(img))
            print("Thinking... I'll command you in 2 sec\n")
        if regime == "manual" and self.sleep_status:
            rew = -912.912  #replay_buffer line47 zakladka
            if keyboard.is_pressed('w'):
                self.sleep_status = 0
                print(colored("I WAKE UP!", "green"))
                self.speak_engine.say("I WAKE UP!");
                self.speak_engine.runAndWait() ;
                time.sleep(1)
            if keyboard.is_pressed('c'):
                time.sleep(0.25)
                print(colored('paused. To continue, print "d"', 'green'))
                keyboard.wait('d')
        if regime == "auto":
            time.sleep(0.05)
            if action == 0:
                print(colored("m", "red"))
                if self.count > 1200:
                    self.speak_engine.say("m");
                    self.speak_engine.runAndWait() ;
                rew = 1 if self.history[1,3] == 1 else 0
            if action == 1:
                print(colored("a", "red"))
                if self.count > 1200:
                    self.speak_engine.say("ahh");
                    self.speak_engine.runAndWait() ;
                rew = 1 if self.history[1,3] != 1 else 0
            if action == 2:
                print(colored("p", "red"))
                if self.count > 1200:
                    self.speak_engine.say("p");
                    self.speak_engine.runAndWait() ;
                rew = 1 if self.history[1,3] == 1 else 0
        print("rew = ", rew, "count = ", self.count, "history = ", self.history)
        done = rew < 0# .7  #always done  (so effective gamma = 0)
        return self.img2, rew, done, {}  # "rew if done else 0"  was here

    
    def operator_reward(self):
        def print_punishment():
            print(colored('Punishment administered!', 'yellow'))
        def print_reward():
            print(colored('Reward administered!', 'green'))
        rewards = 0
        print(colored('your reward?', 'yellow'))
        self.keyboard_history = keyboard.start_recording()
        time.sleep(2)
        self.keyboard_history = keyboard.stop_recording()
        if len( str(self.keyboard_history) ) > 2:   #"[]" is 2
            operator_pressed_key = str(self.keyboard_history[0])
        else:
            operator_pressed_key = None
        if operator_pressed_key == 'KeyboardEvent(1 down)':
            print_punishment()
            rewards = -0.5
        if operator_pressed_key == 'KeyboardEvent(2 down)':
            print_punishment()
            rewards = -0.4
        if operator_pressed_key == 'KeyboardEvent(3 down)':
            print_punishment()
            rewards = -0.3
        if operator_pressed_key == 'KeyboardEvent(4 down)':
            print_punishment()
            rewards = -0.2
        if operator_pressed_key == 'KeyboardEvent(5 down)':
            print_punishment()
            rewards = -0.1
        if operator_pressed_key == 'KeyboardEvent(6 down)':
            print_reward()
            rewards = +0.1
        if operator_pressed_key == 'KeyboardEvent(7 down)':
            print_reward()
            rewards = +0.2
        if operator_pressed_key == 'KeyboardEvent(8 down)':
            print_reward()
            rewards = +0.3
        if operator_pressed_key == 'KeyboardEvent(9 down)':
            print_reward()
            rewards = +0.4
        if operator_pressed_key == 'KeyboardEvent(0 down)':
            print_reward()
            rewards = +0.5
        if operator_pressed_key == 'KeyboardEvent(+ down)':
            print_reward()
            rewards = +0.5
        if operator_pressed_key == 'KeyboardEvent(- down)':
            print_punishment()
            rewards = -0.5
        if operator_pressed_key == 'KeyboardEvent(s down)':
            self.sleep_status = 1
            print(colored("GOTO SLEEP. GOOD NIGHT !", "green"))
            self.speak_engine.say("GOTO SLEEP. GOOD NIGHT !");
            self.speak_engine.runAndWait() ;
            time.sleep(1)
        if operator_pressed_key == 'KeyboardEvent(w down)':
            self.sleep_status = 0
            print(colored("I WAKE UP!", "green"))
            self.speak_engine.say("I WAKE UP!");
            self.speak_engine.runAndWait() ;
            time.sleep(1)        
        return rewards


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor3(config))
    ray.init()
    run_experiments({
        "demo": {
            "run": "DQN",
            "env": SimpleCorridor4,  # or "corridor" if registered above
            "checkpoint_freq": 1000,
            "restore": False,
            "stop": {
                "timesteps_total": 10000000,
            },
            "config": {
                "lr": grid_search([0.0002]),  # try different lrs
                "gamma": 0.7,
                "buffer_size": 5000,
                "schedule_max_timesteps": 600,  #exploration decreases from 1 to 0.1 over that
                "exploration_final_eps": 0.05,
                "learning_starts": 0,  #before that no learning at all
                "train_batch_size": 32,
                "timesteps_per_iteration": 4, # only after that target_network_update_freq is checked
                "target_network_update_freq": 4,  #not 500
                "dueling": False,
                "double_q": False,
                "prioritized_replay": False,
                "num_workers": 1,  # parallelism
                "env_config": {
                    "corridor_length": 5,
                },
            },
        },
    })
                

