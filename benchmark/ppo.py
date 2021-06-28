from offworld_gym import version
__version__     = version.__version__

import os
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
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
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


from offworld_gym_wrapper import make_offworld_env, make_vec_env, ImageToPyTorch
from custom_cnn_policy import CustomCNN
from callback import SaveOnBestTrainingRewardCallback
from typing import Callable


def parser():
    parser = argparse.ArgumentParser(description='PPO')
    # parser.add_argument(
    #     '--model_name', default='PPO-SIM-Continuous', help='model name')
    parser.add_argument(
        '--model_name', default='PPO-REAL-Discrete', help='model name')
    parser.add_argument(
        '--num_envs', type=int, default=1, help='num of parallel training envs in sim')
    parser.add_argument(
        '--resume_model_path', type=str, default=None, help='folder to resume training')
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
        '--num_steps',type=int, default=128, help='frequency of parameter update')
    parser.add_argument(
        '--ppo_epoch',type=int,default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num_mini_batch',type=int, default=64, help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip_param',type=float,default=0.2,help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--num_env_steps', type=int, default=2.5e5, help='number of environment steps to train (default: 1e6)')
    parser.add_argument(
        '--lr', type=int, default=3e-4, help='learning rate')

    parser.add_argument(
        '--no_cuda', action='store_true', help='debug without cuda')
    args = parser.parse_args()

    return args


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def main():
    torch.set_num_threads(1)
    # parse arguments
    args = parser()
    # setting cuda env
    torch.manual_seed(42)  # universal  magic number 42
    torch.cuda.manual_seed_all(42)
    device = torch.device("cpu" if args.no_cuda else "cuda:0")

    # setting folder and logger
    # checkpoint_folder = os.path.join(args.checkpoint_folder, args.model_name)
    log_folder = "logs/" + args.model_name
    # if not os.path.exists(checkpoint_folder): os.makedirs(checkpoint_folder)
    if not os.path.exists(log_folder): os.makedirs(log_folder)
    # summary = SummaryWriter(logdir=log_folder)

    # build offworld envs
    # make_offworld_env(env_name='OffWorldDockerMonolithDiscreteSim-v0', model_name=args.model_name)
    # make_offworld_env(env_name='OffWorldDockerMonolithContinuousSim-v0', model_name=args.model_name)
    make_offworld_env(env_name='OffWorldMonolithDiscreteReal-v0', model_name=args.model_name, env_type= 'real')
    train_env = make_vec_env(make_offworld_env, num_envs=args.num_envs)
    eval_env =  make_vec_env(make_offworld_env, num_envs=1)
    # env = VecFrameStack(env, n_stack=4)

    # initailize PPO agent
    policy_kwargs = dict(
                    features_extractor_class=CustomCNN,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=[dict(vf=[256,256],pi=[256,256])],)

    policy_kwargs["optimizer_class"] = RMSpropTFLike
    policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=1e-5, weight_decay=0)

    if not args.resume_model_path:
        model = PPO("CnnPolicy", env=train_env, policy_kwargs=policy_kwargs, n_epochs=args.ppo_epoch,ent_coef=args.entropy_coef,
                    n_steps=args.num_steps,learning_rate=linear_schedule(args.lr), batch_size=args.num_mini_batch,clip_range=args.clip_param,
                    clip_range_vf=None, tensorboard_log=log_folder,device=device,verbose=1)
    else:
        model = PPO.load(args.resume_model_path)

    
    callback = EvalCallback(eval_env = eval_env,eval_freq=5000,log_path=log_folder,best_model_save_path=log_folder)
    model.learn(args.num_env_steps,callback= callback)

    model.save("PPO-Discrete")
    # model.save("PPO-Continuous")
        

        
if __name__ == "__main__":
    main()