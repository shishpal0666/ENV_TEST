import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from PIL import Image

class Blob:
    def __init__(self, size, x, y, t):
        self.size = size
        self.t = t
        self.x = x
        self.y = y

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

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
        
        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 0.05
    ENEMY_PENALTY = 0.75
    FOOD_REWARD = 10
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE, x=1, y=1, t=1)
        self.food = Blob(self.SIZE, x=7, y=8, t=2)
        self.enemies = [
            Blob(self.SIZE, x=0, y=0, t=3),
            Blob(self.SIZE, x=0, y=1, t=3),
            Blob(self.SIZE, x=1, y=0, t=3),
            Blob(self.SIZE, x=0, y=4, t=3),
            Blob(self.SIZE, x=0, y=5, t=3),
            Blob(self.SIZE, x=1, y=4, t=3),
            Blob(self.SIZE, x=1, y=5, t=3),
            Blob(self.SIZE, x=0, y=8, t=3),
            Blob(self.SIZE, x=0, y=9, t=3),
            Blob(self.SIZE, x=1, y=9, t=3),
            Blob(self.SIZE, x=1, y=8, t=3),
            Blob(self.SIZE, x=2, y=7, t=3),
            Blob(self.SIZE, x=3, y=3, t=3),
            Blob(self.SIZE, x=4, y=0, t=3),
            Blob(self.SIZE, x=5, y=0, t=3),
            Blob(self.SIZE, x=4, y=1, t=3),
            Blob(self.SIZE, x=5, y=1, t=3),
            Blob(self.SIZE, x=4, y=4, t=3),
            Blob(self.SIZE, x=5, y=4, t=3),
            Blob(self.SIZE, x=4, y=5, t=3),
            Blob(self.SIZE, x=5, y=5, t=3),
            Blob(self.SIZE, x=4, y=8, t=3),
            Blob(self.SIZE, x=4, y=9, t=3),
            Blob(self.SIZE, x=5, y=8, t=3),
            Blob(self.SIZE, x=5, y=9, t=3),
            Blob(self.SIZE, x=6, y=6, t=3),
            Blob(self.SIZE, x=7, y=2, t=3),
            Blob(self.SIZE, x=8, y=0, t=3),
            Blob(self.SIZE, x=8, y=1, t=3),
            Blob(self.SIZE, x=9, y=0, t=3),
            Blob(self.SIZE, x=9, y=1, t=3),
            Blob(self.SIZE, x=8, y=4, t=3),
            Blob(self.SIZE, x=8, y=5, t=3),
            Blob(self.SIZE, x=9, y=5, t=3),
            Blob(self.SIZE, x=9, y=4, t=3),
            Blob(self.SIZE, x=8, y=8, t=3),
            Blob(self.SIZE, x=8, y=9, t=3),
            Blob(self.SIZE, x=9, y=9, t=3),
            Blob(self.SIZE, x=9, y=8, t=3),
        ]

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = self.get_image()
        else:
            observation = (self.player-self.food) + (self.player-self.enemies[0])
        return observation

    def step(self, action):
        self.episode_step += 1
        x_t = self.player.x
        y_t = self.player.y
        self.player.action(action)

        if self.RETURN_IMAGES:
            new_observation = self.get_image()
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemies[0])

        if self.player == self.food:
            reward = self.FOOD_REWARD
        elif self.player in self.enemies:
            reward = -self.ENEMY_PENALTY - self.MOVE_PENALTY
            self.player.x = x_t
            self.player.y = y_t
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or self.episode_step >= 300:
            done = True

        return new_observation, reward, done

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
        
        # Convert to PyTorch tensor and normalize
        img_tensor = torch.from_numpy(env).float().permute(2, 0, 1) / 255.0
        return img_tensor


# Example usage
if __name__ == "__main__":
    env = BlobEnv()
    
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = random.randint(0, 8)  # Random action
        next_state, reward, done = env.step(action)
        env.render()
        done=True