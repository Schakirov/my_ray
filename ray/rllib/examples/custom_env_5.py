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
from gtts import gTTS
from pygame import mixer
import keyboard

import ray
from ray.tune import run_experiments, grid_search
from cv2 import *
import time
import numpy as np

from tkinter import *
from tkinter.colorchooser import askcolor
import pyscreenshot as ImageGrab
import matplotlib.pyplot as plt
import time
import random


class SimpleCorridor5(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(4)
        self.observation_space = Box(
            0.0, 1.0, shape=(84, 84, 3), dtype=np.float32)
        self.speak_engine = pyttsx3.init();
        self.auto_img = np.zeros((84,84,3))
        self.auto_img_open = 0.5 + 0.1 * np.random.randn(84,84,3)
        self.history = np.zeros((4,5))
        self.img2 = np.zeros((84,84,3))
        self.count = 0
        self.sleep_status = 0  #initially awake
        self.a = Tk()
        self.c = Canvas(self.a, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)
        self.a.update()
        print("now a little bit sleep")
        #time.sleep(5)
        self.x = 300
        self.y = 300
        self.x_prev = 300
        self.y_prev = 300
        self.x_2prev = 300
        self.y_2prev = 300
        self.color_r = 100
        self.color_g = 100
        self.color_b = 100
        self.degree_go = 0
        self.m = mixer
        self.m.init()
        self.curr_step = 0

    def reset(self):
        self.cur_pos = self.img2 #np.ones((84,84,3)) * 100
        return self.cur_pos

    def step(self, action):
        self.I_goal = cv2.imread('/home/ai/del6/experience/image.png') 
        if len(self.I_goal.shape) == 3:
                self.I_goal = np.mean(self.I_goal, axis=2)
        self.I_goal = cv2.resize(self.I_goal, (84, 84))
        self.curr_step += 1
        self.count = self.count + 1
        assert action in [0, 1, 2, 3], action
        self.history[0, 0:3+1] = self.history[0, 1:4+1]
        self.history[1, 0:3+1] = self.history[1, 1:4+1]
        self.history[2, 0:3+1] = self.history[2, 1:4+1]
        self.history[0:3, 4] = np.zeros((3))
        self.history[action, 4] = 1
        self.img2 = np.zeros((84,84,3))
        self.img2[0:4,0:5,0] = self.history
        regime = "manual"
        if regime == "manual" and self.sleep_status:
            rew = -912.912  #replay_buffer line47 zakladka  #agent just learns, and doesn't add experience
            done = 0
            self.img2 = np.zeros((84,84,3))
            if keyboard.is_pressed('w'):
                self.sleep_status = 0
                print(colored("I WAKE UP!", "green"))
                tts = gTTS("С добрым утром!", lang='ru')
                tts.save('said.mp3')
                self.m.music.load('said.mp3')
                self.m.music.play()
                time.sleep(1)
            if keyboard.is_pressed('c'):
                time.sleep(0.25)
                print(colored('paused. To continue, print "d"', 'green'))
                keyboard.wait('d')
        if regime == "manual" and not self.sleep_status:
            #time.sleep(0.5)
            self.color_r += random.randint(-20,20);   self.color_r = abs(self.color_r) + 16
            self.color_g += random.randint(-20,20);   self.color_g = abs(self.color_g) + 16
            self.color_b += random.randint(-20,20);   self.color_b = abs(self.color_b) + 16
            colorfill = 'black' #'#' + hex(self.color_r)[-2:] + hex(self.color_g)[-2:] + hex(self.color_b)[-2:]
            deg2rad = 3.141592654 / 180
            if action == 0:
                self.degree_go += 50 * deg2rad
                #self.x = (self.x + 20) % 600
                #self.c.create_line(self.x_prev, self.y_prev, self.x, self.y, width=5, fill=colorfill, capstyle=ROUND, smooth=TRUE, splinesteps=36);                
            elif action == 1:
                self.degree_go += 0 * deg2rad
                #self.x = (self.x - 20) % 600
                #self.c.create_line(self.x_prev, self.y_prev, self.x, self.y, width=5, fill=colorfill, capstyle=ROUND, smooth=TRUE, splinesteps=36);                
            elif action == 2:
                self.degree_go -= 50 * deg2rad
                #self.y = (self.y + 20) % 600
                #self.c.create_line(self.x_prev, self.y_prev, self.x, self.y, width=5, fill=colorfill, capstyle=ROUND, smooth=TRUE, splinesteps=36);                
            elif (action == 3):
                self.degree_go = 179 * deg2rad + self.degree_go
                #self.y = (self.y - 20) % 600
                #self.c.create_line(self.x_prev, self.y_prev, self.x, self.y, width=5, fill=colorfill, capstyle=ROUND, smooth=TRUE, splinesteps=36); 
            self.x = (self.x + 50 * np.cos(self.degree_go)) % 600
            self.y = (self.y + 50 * np.sin(self.degree_go)) % 600
            #self.x3 = self.x + 0.8 * (self.x_prev - self.x_2prev) + 0.4 * (self.x - self.x_prev)
            #self.y3 = self.y + 0.8 * (self.y_prev - self.y_2prev) + 0.4 * (self.y - self.y_prev)
            if (np.abs(self.x - self.x_prev) < 200) and (np.abs(self.y - self.y_prev) < 200):
                self.c.create_line(self.x_prev, self.y_prev, self.x, self.y, width=5, fill=colorfill, capstyle=ROUND, smooth=TRUE, splinesteps=36); 
            self.x_2prev = self.x_prev
            self.y_2prev = self.y_prev
            self.x_prev = self.x
            self.y_prev = self.y
            self.a.update()
            #time.sleep(0.125)
            print("snapshot!")
            #cam = VideoCapture(0)
            #s, img = cam.read()
            #img = cv2.resize(img, (84,84))/255.0
            # rew = self.operator_reward() # np.mean(img) + self.operator_reward()
            #print("rew = np.mean(img) = ", np.mean(img))
            print("Thinking... I'll command you in 2 sec\n")
        
        
            #print("rew = ", rew, "count = ", self.count, "history = ", self.history)
            
            box = (self.c.winfo_rootx(),self.c.winfo_rooty(),self.c.winfo_rootx()+self.c.winfo_width(),self.c.winfo_rooty() + self.c.winfo_height())
            grab = ImageGrab.grab(bbox = box)
            I = np.array(grab)
            if len(I.shape) == 3:
                print(colored("Не загораживайте рисунок", 'red'))
                I = np.mean(I, axis=2)
                tts = gTTS("Не загораживайте рисунок", lang='ru')
                tts.save('said.mp3')
                #self.m.music.load('said.mp3')
                #self.m.music.play()
            try:
                self.I_prev
            except:
                self.I_prev = I
            if random.randint(0,100) > -70:
                plt.imshow(I, cmap='gray', vmin=0, vmax=255)
                plt.draw()
                plt.pause(0.001)
            rew = 0
            self.I_goal = cv2.resize(self.I_goal, np.shape(I))
            rew = (np.mean(np.abs(self.I_prev - self.I_goal)))  -  (np.mean(np.abs(I - self.I_goal)))
            print('rew = ', rew)
            print('np.mean(I) = ', np.mean(I))
            self.I_prev = I
            self.img2 = cv2.resize(I, (84,84))
            print("np.shape(self.img2) = ", np.shape(self.img2))
            self.img2 = np.repeat(self.img2[:, :, np.newaxis], 3, axis=2)  #prev shape (84,84) goes to new shape (84,84,3)
            print("self.history = ", self.history)
            print("self.img2[0:4,0:5,0] = ", self.img2[0:4,0:5,0])
            self.img2[0:4,0:5,0] = self.history
            if self.curr_step % 45 == 0:
                done = 1
                rew = self.operator_reward()
                self.c.delete("all")
                self.I_prev = np.array(grab)
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
        varianten = ["Оцени меня", "Ну как?", "Как вам мой рисунок?", "Как моё творчество?", "Правильно нарисовала?"]
        variant_to_say = varianten[random.randint(0,4)]
        tts = gTTS(variant_to_say, lang='ru')
        tts.save('said.mp3')
        self.m.music.load('said.mp3')
        self.m.music.play()
        time.sleep(5)
        self.keyboard_history = keyboard.stop_recording()
        if len( str(self.keyboard_history) ) > 2:   #"[]" is 2
            operator_pressed_key = str(self.keyboard_history[0])
        else:
            operator_pressed_key = None
        if operator_pressed_key == 'KeyboardEvent(1 down)':
            print_punishment()
            rewards = -0.5
            varianten = ["Прости, что расстроила", "У меня просто не было вдохновения", "Сам-то попробуй лучше"]
            variant_to_say = varianten[random.randint(0,2)]
            tts = gTTS(variant_to_say, lang='ru')
        if operator_pressed_key == 'KeyboardEvent(2 down)':
            print_punishment()
            rewards = -0.4
            varianten = ["Прости, что расстроила", "У меня просто не было вдохновения", "Сам-то попробуй лучше", "Я ещё научусь"]
            variant_to_say = varianten[random.randint(0,3)]
            tts = gTTS(variant_to_say, lang='ru')
        if operator_pressed_key == 'KeyboardEvent(3 down)':
            print_punishment()
            rewards = -0.3
            varianten = ["Прости, что расстроила", "У меня просто не было вдохновения", "Сам-то попробуй лучше", "Я ещё научусь"]
            variant_to_say = varianten[random.randint(0,3)]
            tts = gTTS(variant_to_say, lang='ru')
        if operator_pressed_key == 'KeyboardEvent(4 down)':
            print_punishment()
            rewards = -0.2
            varianten = ["Прости, что расстроила", "У меня просто не было вдохновения", "Сам-то попробуй лучше", "Я ещё научусь"]
            variant_to_say = varianten[random.randint(0,3)]
            tts = gTTS(variant_to_say, lang='ru')
        if operator_pressed_key == 'KeyboardEvent(5 down)':
            print_punishment()
            rewards = -0.1
            tts = gTTS("Я стану лучше", lang='ru')
        if operator_pressed_key == 'KeyboardEvent(6 down)':
            print_reward()
            rewards = +0.1
            tts = gTTS("Спасибо", lang='ru')
        if operator_pressed_key == 'KeyboardEvent(7 down)':
            print_reward()
            rewards = +0.2
            tts = gTTS("Спасибо", lang='ru')
        if operator_pressed_key == 'KeyboardEvent(8 down)':
            print_reward()
            rewards = +0.3
            tts = gTTS("Спасибо", lang='ru')
        if operator_pressed_key == 'KeyboardEvent(9 down)':
            print_reward()
            rewards = +0.4
            tts = gTTS("Большое спасибо", lang='ru')
        if operator_pressed_key == 'KeyboardEvent(0 down)':
            print_reward()
            rewards = +0.5
            tts = gTTS("Рада стараться", lang='ru')
        if operator_pressed_key == 'KeyboardEvent(+ down)':
            print_reward()
            rewards = +0.5
        if operator_pressed_key == 'KeyboardEvent(- down)':
            print_punishment()
            rewards = -0.5
        if operator_pressed_key == 'KeyboardEvent(s down)':
            self.sleep_status = 1
            print(colored("GOTO SLEEP. GOOD NIGHT !", "green"))
            tts = gTTS("Хочу спать. До завтра", lang='ru')
            time.sleep(1)
        if operator_pressed_key == 'KeyboardEvent(w down)':
            self.sleep_status = 0
            print(colored("I WAKE UP!", "green"))
            tts = gTTS("С добрым утром!", lang='ru')
            time.sleep(1)        
        tts.save('said.mp3')
        self.m.music.load('said.mp3')
        self.m.music.play()
        return rewards
    

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor3(config))
    ray.init()
    run_experiments({
        "demo": {
            "run": "DQN", #DQN
            "env": SimpleCorridor5,  # or "corridor" if registered above
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
                

