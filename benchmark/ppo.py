from offworld_gym import version
__version__     = version.__version__

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys
import time
import pickle
from collections import deque
import numpy as np

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.enums import AlgorithmMode, LearningType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import shutil
import copy
import argparse
from tensorboardX import SummaryWriter


from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from offworld_gym_wrapper import make_offworld_env, make_vec_env, ImageToPyTorch, reshape_obs
from custom_cnn_policy import CustomCNN


def parser():
    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument(
        '--model_name', default='PPO-SIM-Discrete', help='model name')
    parser.add_argument(
        '--num_envs', type=int, default=10, help='num of parallel training envs in sim')
    parser.add_argument(
        '--resume_folder', type=str, default=None, help='folder to resume training')
    parser.add_argument(
        '--checkpoint_folder', default='checkpoints/', help='folder to store the checkpoint')
    parser.add_argument(
        '--log_interval', type=int, default=1, help='log interval, one log per n updates (default: 1)')
    parser.add_argument(
        '--save_interval', type=int, default=20, help='save interval, one save per n updates (default: 10)')
    parser.add_argument(
        '--gamma', type=float, default=0.99, help='eposodic discounted coef gamma(default: 0.99)')
    parser.add_argument(
        '--eps',type=float, default=1e-5, help='Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--entropy_coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value_loss_coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument( 
        '--max_grad_norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--num_steps',type=int, default=100, help='frequency of parameter update')
    parser.add_argument(
        '--ppo_epoch',type=int,default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num_mini_batch',type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip_param',type=float,default=0.2,help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--num_env_steps', type=int, default=5e5, help='number of environment steps to train (default: 1e6)')
    parser.add_argument(
        '--lr', type=int, default=5e-4, help='learning rate')

    parser.add_argument(
        '--no_cuda', action='store_true', help='debug without cuda')
    args = parser.parse_args()

    return args

    def logging(summary,i,value_loss,action_loss,dist_entropy,episodic_rewards):
        summary.add_scalar('log/value_loss', value_loss, i)
        summary.add_scalar('log/action_loss', action_loss, i)
        summary.add_scalar('log/entropy_loss', dist_entropy, i)
        summary.add_scalar('log/episodic_reward',  episodic_rewards.mean(), i)

    def main():
        # parse arguments
        args = parser()
        # setting cuda env
        torch.manual_seed(42)  # universal  magic number 42
        torch.cuda.manual_seed_all(42)
        device = torch.device("cpu" if args.no_cuda else "cuda:0")

        # setting folder and logger
        checkpoint_folder = os.path.join(args.checkpoint_folder, args.model_name)
        log_folder = "logs/" + args.model_name
        if not os.path.exists(checkpoint_folder): os.makedirs(checkpoint_folder)
        if not os.path.exists(log_folder): os.makedirs(log_folder)
        summary = SummaryWriter(logdir=log_folder)

        

        

if __name__ == "__main__":
    main()