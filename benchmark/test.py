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
from stable_baselines3.common.vec_env import VecVideoRecorder,  DummyVecEnv

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import copy
import argparse
import cv2

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
        '--trained_model_path', default='PPO-Discrete.zip', help='folder to store the checkpoint')
    parser.add_argument(
        '--n_eval_episodes',type=int,default=20, help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--real', action='store_true', help='true if test in real robot')
    args = parser.parse_args()


    return args

def record_video(eval_env, model, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()

def main():

    # parse arguments
    args = parser()
    # setting cuda env
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build offworld envs
    env_name = 'OffWorldDockerMonolithDiscreteSim-v0'
    # env_name = 'OffWorldDockerMonolithContinuousSim-v0'

    make_offworld_env(env_name=env_name)
    eval_env =  make_vec_env(make_offworld_env, num_envs=1)

    # load trained model
    trained_model_name = args.trained_model_path.split(".")[0]
    agent = PPO.load(trained_model_name)

    # Evaluate the trained agent

    mean_reward, std_reward = evaluate_policy(agent, eval_env, n_eval_episodes=args.n_eval_episodes, deterministic=True)

    
    print("===================Finished Testing===================")
    print(f"Tested for {args.n_eval_episodes} episodes, mean_reward={mean_reward:.3f} +/- {std_reward:.3f}")

    # visualize trained agent for simulator
    # record_video(eval_env, agent, video_length=500, prefix=trained_model_name)


if __name__ == "__main__":
    main()
        
    

