import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

class CustomGridEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.grid_size = (2, 2)
        self.start_pos = (0, 0)
        self.end_pos = (1, 1)
        self.trap_pos = (1,0)
        self.actions = [0, 1, 2, 3]  # [Left, Right, Forward, Backward]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        self.done = False
        return self.agent_pos

    def step(self, action):
        if self.done:
            raise ValueError("Episode has already terminated. Please reset the environment.")

        x, y = self.agent_pos
        if action == 0:  # Move left
            y = max(0, y - 1)
        elif action == 1:  # Move right
            y = min(self.grid_size[1] - 1, y + 1)
        elif action == 2:  # Move forward
            x = max(0, x - 1)
        elif action == 3:  # Move backward
            x = min(self.grid_size[0] - 1, x + 1)

        self.agent_pos = (x, y)

        #can tweak the reward and check if having high reward for goal makes any difference or not

        if self.agent_pos == self.end_pos:
            reward = 10
            self.done = True
        elif self.agent_pos == self.trap_pos:
            reward = -5
            self.done = True
        else:
            reward = 0

        return self.agent_pos, reward, self.done, {}

    def render(self):
        grid = np.zeros(self.grid_size)
        grid[self.end_pos] = 1.0  # End position (green)
        grid[self.trap_pos] = -1.0  # Trap position (red)

        # Custom colormap with black tiles instead of white
        custom_cmap = mcolors.ListedColormap(['red', 'black','green'])

        plt.imshow(grid, cmap=custom_cmap, extent=[0, self.grid_size[1], 0, self.grid_size[0]], origin='lower')

        # Plot agent position as a small dot in the cell
        agent_y, agent_x = self.agent_pos
        plt.scatter(agent_x + 0.5, agent_y + 0.5, c='blue', marker='o', s=100)

        plt.title("Custom Grid Environment")
        plt.xticks(range(self.grid_size[1] + 1))
        plt.yticks(range(self.grid_size[0] + 1))
        plt.grid(color='w', linestyle='-', linewidth=1)
        plt.show()

    # Map action numbers to keywords
    action_keywords = {0: 'Left', 1: 'Right', 2: 'Forward', 3: 'Backward'}

    def print_action_reward(self, action, reward):
        action_keyword = self.action_keywords[action]
        print(f"Action: {action_keyword} ({action}), Reward: {reward}")

    def close(self):
        pass

    def action_space(self):
        return len(self.actions)

    def observation_space(self):
        return self.grid_size
