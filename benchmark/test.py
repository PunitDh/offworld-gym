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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


from offworld_gym_wrapper import make_offworld_env, make_vec_env, ImageToPyTorch
from custom_cnn_policy import CustomCNN
from typing import Callable