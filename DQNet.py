import math
import random
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
from itertools import count
import gym
import numpy as np
import matplotlib.colors as mcolors
import random
import rl_env as rl
import argparse
import pandas as pd

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 16
GAMMA = 0.7
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.2
LR = 0.1


ap = argparse.ArgumentParser()
ap.add_argument("-t","--train_mode",required=True,default=0,help="Boolean (1 or 0) to indicate if training is to be done or not",type=int)
ap.add_argument("-p","--plot_heatmap",required=False,default=0,help="optional argument to plot learned q values heatmap (1-Yes)",type=int)
ap.add_argument("-o","--output_path",required=True,help="path to save the trained model and to load the saved model from",type=str)
ap.add_argument("-i","--num_eps",required=False,default=901, help = "Number of episodes to train the policy for ",type=int)
# if GPU is to be used
env = rl.CustomGridEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()  


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(),color = "#5f345f")
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(),color="green")

    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_heatmap(env,target_net):
        target_net.eval()
        # Create a heatmap for each action
        fig, axs = plt.subplots(1, env.action_space.n, figsize=(12, 3))
        actions = ['Left', 'Right', 'Forward', 'Backward']  # Replace with the action labels

        for action in range(env.action_space.n):
            ax = axs[action]
            q_values = np.zeros(env.grid_size)

            for x in range(env.grid_size[0]):
                for y in range(env.grid_size[1]):
                    state = torch.tensor([[x, y]], dtype=torch.float32,device="cuda")
                    q_value = target_net(state).cpu().detach().numpy()[0][action]
                    q_values[x, y] = q_value

            ax.imshow(q_values, cmap='gray', vmin=np.min(q_values), vmax=np.max(q_values),origin='lower')
            ax.set_title(f"Action: {actions[action]}")
            ax.set_xticks(np.arange(env.grid_size[1]))
            ax.set_yticks(np.arange(env.grid_size[0]))
            ax.set_xticklabels(np.arange(env.grid_size[1]))
            ax.set_yticklabels(np.arange(env.grid_size[0]))
            ax.grid(color='w', linestyle='-', linewidth=1)

        plt.tight_layout()
        plt.show()
        
def get_q_values(model,cell_list=[0,1]):
    '''print q table in cmd prompt'''
    model.eval()
    q_values_list = []
    actions = ["Left","Right","Forward","Backward"]
    
    for i in range(len(cell_list)):
        for j in range(len(cell_list)):
            state = torch.tensor([[cell_list[i],cell_list[j]]],dtype=torch.float32,device=device)
            q_values = target_net(state).cpu().detach().numpy()[0]
            q_values_list.append(q_values)
    
    q_values_df = pd.DataFrame(q_values_list,columns=[f"{action}_q" for action in actions])
    q_values_df.insert(0, "state", [(cell_list[i],cell_list[j]) for i in range(len(cell_list)) for j in range(len(cell_list))])
    
    return q_values_df
    

#Other declarations to set the policy and the target network with optimizer

steps_done = 0

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

args = vars(ap.parse_args())

if __name__ == '__main__':
    if torch.cuda.is_available():
        num_episodes = int(args["num_eps"])
    else:
        num_episodes = 51

    if args["train_mode"] == 1:
        print("Starting Training for the agent.....")

        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = select_action(state)
                observation, reward, terminated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)
                
                if i_episode % 150 == 0:
                    torch.save(target_net.state_dict(), "target_model_" + str(i_episode) + ".pth")

                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break

        print('Complete')
        plot_durations(show_result=True)
        plt.ioff()
        plt.show()
        env.close()

        #save the target policy model 
        torch.save(target_net.state_dict(), args["output_path"] + 'target_model.pth')
    
    else:

        print("Loading the target policy model state dict.....")
        print("Plotting the Q value heatmap.....")
        #load saved model 
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(torch.load(args["output_path"]))

        if args["plot_heatmap"] == 1:
            plot_heatmap(env=env,target_net=target_net)
            
        if args["plt_heatmap"] == 0:
            q_table = get_q_values(model=target_net)
            print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
            print("''''''''''''''''''''''''''''''''''''''''''''''''''      Q-Table      '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
            print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
            print(q_table)
                
    
