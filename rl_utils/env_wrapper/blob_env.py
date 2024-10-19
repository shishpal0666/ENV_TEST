import numpy as np
import matplotlib.pyplot as plt
import torch
from rl_utils.logger import logger

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


class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 0.05
    ENEMY_PENALTY = 0.75
    FOOD_REWARD = 10
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3
    d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    def reset(self):
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

    def step(self, action):
        self.episode_step += 1
        x_t, y_t = self.player.x, self.player.y
        self.player.action(action)

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

        return new_observation, reward, done, info

    def render(self):
        img = self.get_image()
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]
        
        for enemy in self.enemies:
            env[enemy.x][enemy.y] = self.d[self.ENEMY_N]

        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
        
        img_tensor = torch.from_numpy(env).float().permute(2, 0, 1) / 255.0
        return img_tensor
