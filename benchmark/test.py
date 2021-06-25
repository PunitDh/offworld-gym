# This script is used to test trained models on offworld gym
from offworld_gym import version
__version__     = version.__version__

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys
import time
import numpy as np
from collections import deque

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.enums import AlgorithmMode, LearningType

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import copy
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


from offworld_gym_wrapper import make_offworld_env, make_vec_env, ImageToPyTorch
from custom_cnn_policy import CustomCNN
from typing import Callable

def parser():
    parser = argparse.ArgumentParser(description='RL Algorithm')
    parser.add_argument(
        '--trained_model_path', default='PPO-Continuous.zip', help='folder to store the checkpoint')
    parser.add_argument(
        '--n_eval_episodes',type=int,default=20, help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--real', action='store_true', help='true if test in real robot')
    args = parser.parse_args()


    return args

def main():

    # parse arguments
    args = parser()
    # setting cuda env
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build offworld envs
    make_offworld_env(env_name='OffWorldDockerMonolithContinuousSim-v0')
    eval_env =  make_vec_env(make_offworld_env, num_envs=1)

    # load trained model
    trained_model_name = args.trained_model_path.split(".")[0]
    agent = PPO.load(trained_model_name)

    # Evaluate the trained agent

    mean_reward, std_reward = evaluate_policy(agent, eval_env, n_eval_episodes=args.n_eval_episodes, deterministic=False)

    eval_env.close()  
    
    print("===================Finished Testing===========================")
    print(f"Tested for {args.n_eval_episodes} episodes, mean_reward={mean_reward:.3f} +/- {std_reward}")

if __name__ == "__main__":
    main()
        
    

