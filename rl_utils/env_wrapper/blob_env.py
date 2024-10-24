import numpy as np
import matplotlib.pyplot as plt
import torch
from rl_utils.logger import logger
import gym
from gym import spaces
import cv2
import os
from PIL import Image
import math

path_taken=[(1,1)]

class Blob:
    def __init__(self, size, x, y, t):
        self.size = size
        self.t = t
        self.x = x
        self.y = y

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):
        self.x += x
        self.y += y
        
        if self.x < 0:
            self.x = 0
        elif self.x > self.size - 1:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size - 1:
            self.y = self.size - 1


class BlobEnv():
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 0.05
    ENEMY_PENALTY = 0.75
    FOOD_REWARD = 10
    SAME_STATE=0.65
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3
    d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
    
    def __init__(self,args):
        self.args=args
        # Define action and observation space
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.OBSERVATION_SPACE_VALUES, dtype=np.float32)
        self.reward_range = (-self.ENEMY_PENALTY - self.MOVE_PENALTY, self.FOOD_REWARD)
        self.metadata = {'render.modes': ['human', 'rgb_array']}  # Add metadata



    def reset(self):
        global path_taken
        path_taken=[(1,1)]
        self.player = Blob(self.SIZE, x=1, y=1, t=1)
        self.food = Blob(self.SIZE, x=7, y=8, t=2)
        self.enemies = [Blob(self.SIZE, x, y, t=3) for x, y in self.generate_enemy_positions()]

        self.episode_step = 0
        self.total_reward = 0

        if self.RETURN_IMAGES:
            observation = self.get_image()
        else:
            observation = (self.player - self.food) + (self.player - self.enemies[0])
        
        return observation

    def generate_enemy_positions(self):
        if self.args.medium_env:
            return [
                (0, 0), (0, 1), (1, 0), (0, 4), (0, 5), 
                (1, 4), (1, 5), (0, 8), (0, 9), (1, 9), 
                (1, 8), (2, 7), (3, 3), (4, 0), (5, 0), 
                (4, 1), (5, 1), (4, 4), (5, 4), (4, 5), 
                (5, 5), (4, 8), (4, 9), (5, 8), (5, 9), 
                (6, 6), (7, 2), (8, 0), (8, 1), (9, 0), 
                (9, 1), (8, 4), (8, 5), (9, 5), (9, 4), 
                (8, 8), (8, 9), (9, 9), (9, 8)
            ]
        elif self.args.hard_env:
            return [
                (0, 0), (0, 1), (0, 2), (0, 3), (0, 7),
                (0, 8), (0, 9), (1, 0), (1, 2), (1, 4),
                (1, 5), (1, 6), (1, 8), (1, 9), (2, 0),
                (2, 1), (2, 3), (2, 5), (2, 6), (2, 7),
                (2, 9), (3, 0), (3, 2), (3, 3), (3, 4),
                (3, 6), (3, 8), (3, 9), (4, 0), (4, 1),
                (4, 3), (4, 4), (4, 5), (4, 7), (4, 8),
                (4, 9), (5, 0), (5, 1), (5, 2), (5, 4),
                (5, 5), (5, 6), (5, 8), (5, 9), (6, 0),
                (6, 1), (6, 2), (6, 3), (6, 5), (6, 6),
                (6, 7), (6, 9), (7, 0), (7, 1), (7, 2),
                (7, 3), (7, 4), (7, 6), (7, 7), (8, 0),
                (8, 1), (8, 2), (8, 4), (8, 6), (8, 7),
                (8, 8), (8, 9), (9, 0), (9, 1), (9, 2),
                (9, 3), (9, 5), (9, 6), (9, 7), (9, 8),
                (9, 9),
            ]
        else:
            return [
                (1, 6), (2, 6), (2, 7), (3, 1),
                (3, 2), (4, 1), (6, 5), (7, 5),
                (7, 4), (7, 6)
            ]
    
    def calculate_angle(self,prev_state, new_state):
        goal_state=(7,8)
        # Convert points to NumPy arrays
        prev_state = np.array(prev_state)
        ray1 = np.array(new_state) - prev_state
        ray2 = np.array(goal_state) - prev_state
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(ray1, ray2)
        magnitude_ray1 = np.linalg.norm(ray1)
        magnitude_ray2 = np.linalg.norm(ray2)
        moved=True

        if magnitude_ray1 == 0 :
          moved=False
        
        if moved:
          # Calculate cosine of the angle
          cos_theta = dot_product / (magnitude_ray1 * magnitude_ray2)

          # Handle potential floating-point errors
          cos_theta = np.clip(cos_theta, -1.0, 1.0)
          
          # Calculate the angle in radians and convert to degrees
          angle_radians = np.arccos(cos_theta)
          angle_degrees = np.degrees(angle_radians)
        else:
          angle_degrees=180

        return angle_degrees,moved    

    
    def step(self, action):
        global path_taken
        self.episode_step += 1
        x_t, y_t = self.player.x, self.player.y
        self.player.action(action)
        dist1=math.dist([x_t,y_t], [7,8])
        dist2=math.dist([self.player.x,self.player.y],[7,8])
        d_reward=(dist2)/(10.63)

        if self.RETURN_IMAGES:
            new_observation = self.get_image()
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemies[0])


        if self.player == self.food:
            reward = self.FOOD_REWARD
        elif self.player in self.enemies:
            reward = -self.ENEMY_PENALTY - self.MOVE_PENALTY
            self.player.x, self.player.y = x_t, y_t
        else:
            reward = -self.MOVE_PENALTY    
            if self.args.updated_reward:
                angle,moved=self.calculate_angle((x_t,y_t),(self.player.x,self.player.y))
            
                if moved:
                    o_reward=abs((90-angle)/90)
                    if angle>90:
                        reward-=o_reward*0.3
                    else:
                        reward-=o_reward*0.1
                else:
                    reward-=self.SAME_STATE
                
                if dist1>dist2 :
                    d_reward=d_reward*(-0.05)
                    reward+=d_reward
                elif dist1<dist2:
                    d_reward=d_reward*(-0.2)
                    reward+=d_reward  

            

        self.total_reward += reward

        done = False
        if reward == self.FOOD_REWARD or self.episode_step >= 300:
            done = True
            info = {
                "episode": {
                    "r": self.total_reward,
                    "l": self.episode_step,
                    "t": self.episode_step
                }
            }
        else:
            info = {}
        
        a=(self.player.x,self.player.y)
        path_taken.append(a)

        return new_observation, reward, done, info

    def render(self):
        img = self.get_image()
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color

        #For Single Enemy in Dynamic Env
        # env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red

        #For Multiple Enemy in Static Env
        for enemy in self.enemies:
            env[enemy.x][enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red

        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img



def get_path_traveled():
    global path_taken
    return path_taken