#!/usr/bin/env python
# coding: utf-8

import sys
import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import logging
import offworld_gym
from offworld_gym.envs.common.channels import Channels
import tensorflow.keras.models as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Softmax
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.real.real_env import AlgorithmMode, LearningType
# import certifi
# import urllib3

# http = urllib3.PoolManager(
#     cert_reqs="CERT_REQUIRED",
#     ca_certs=certifi.where()
# )

logging.basicConfig(level=logging.DEBUG)

# ## Set up display
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython: from IPython import display

OFFWORLD_GYM_ACCESS_TOKEN = '0b5dc1a37f9bf3c911e0df64de836b63ce2aad9247e0edfdeb7f19678637b896'


# ## Deep Q-Network
class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
         
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=64)   
        self.fc2 = nn.Linear(in_features=64, out_features=96)
        self.fc3 = nn.Linear(in_features=96, out_features=144)
        self.fc4 = nn.Linear(in_features=144, out_features=216)
        self.out = nn.Linear(in_features=216, out_features=4)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = F.relu(self.fc4(t))
        t = self.out(t)
        return t

# ## Experience class
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

e = Experience({},[],"a","v")

# ## Replay Memory
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

# ## Epsilon Greedy Strategy
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) *             math.exp(-1. * current_step * self.decay)


# ## Reinforcement Learning Agent
class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) # explore      
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device) # exploit


# ## Environment Manager
class OffWorldGymEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='Punit_Dharmadhikari_Demo',
               resume_experiment=False, channel_type=Channels.RGBD,
               learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN).unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
    
    def reset(self):
        self.env.reset()
        self.current_screen = None
        
    def close(self):
        self.env.close()
        
    def render(self, mode="array"):
        return self.env.render(mode)
        
    def num_actions_available(self):
        return self.env.action_space.n
        
    def take_action(self, action):        
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            if self.current_screen is None:
                return
            else:
                return torch.zeros_like(self.current_screen)
            # return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
    
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
       
    def get_processed_screen(self):
        if self.render('array') is not None:
            screen = self.render('array').transpose((2, 0, 1)) # PyTorch expects CHW
            screen = self.crop_screen(screen)
            return self.transform_screen_data(screen)
        
    
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        
        # Strip off top and bottom
        top = int(screen_height * 0.001)
        bottom = int(screen_height * 0.999)
        screen = screen[:, top:bottom, :]
        return screen
    
    def transform_screen_data(self, screen):       
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        
        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((40,90))
            ,T.ToTensor()
        ])
        
        return resize(screen).unsqueeze(0).to(self.device) # add a batch dimension (BCHW)


# ## Example of non-processed screen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = OffWorldGymEnvManager(device)
em.reset()
screen = em.render()

# ## Example of processed screen
screen = em.get_processed_screen()

# ## Example of starting state
screen = em.get_state()

# ## Example of non-starting state
for i in range(5):
    em.take_action(torch.tensor([1]))
screen = em.get_state()

# ## Example of end state
em.done = True
screen = em.get_state()
em.close()


# ## Utility functions
# ### Plotting
def plot(values, moving_avg_period):   
    moving_avg = get_moving_average(moving_avg_period, values)
    print("Episode", len(values), "\n",           moving_avg_period, "episode moving avg:", moving_avg[-1])
    # if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1)             .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

# ### Tensor processing
def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)


# #### Example of Experience(\*zip(\*experiences)) used above
# See https://stackoverflow.com/a/19343/3343043 for further explanation.

e1 = Experience(1,1,1,1)
e2 = Experience(2,2,2,2)
e3 = Experience(3,3,3,3)

experiences = [e1,e2,e3]
batch = Experience(*zip(*experiences))
batch


# ### Q-Value Calculator
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod        
    def get_next(target_net, next_states):                
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values


# ## Main Program
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device.type, device)

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000 # run for more episodes for better results


em = OffWorldGymEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

model_file = Path("OffWorldGym.h5")
if model_file.is_file():
    policy_net = DQN(em.get_screen_height(), em.get_screen_width())
    policy_net.load_state_dict(torch.load(model_file, map_location="cuda:0"))
    policy_net.to(device)
else:
    policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
#end

# policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()
    
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net.state_dict(), "OffWorldGym.h5")
        
em.close()

get_moving_average(100, episode_durations)[-1] > 15