import os
import logging
import gym
from gym import spaces
import numpy as np
from multiprocessing import Process, Pipe
from typing import Any, Callable, Dict, Optional, Type, Union
import cv2

import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.enums import AlgorithmMode, LearningType

# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy


class ImageToPyTorch(gym.ObservationWrapper):
    '''
    Create a wrapper to re-organize the video sequence for pytorch standard.
    [index, width, height, channel] -> [index, channel, width, height]
    '''
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=255.0,            
                                shape=(old_shape[-1], 
                                old_shape[1], old_shape[2]),
                                dtype=np.float32)

class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to a scale (default: 50%)
    as done in the Nature paper and later work.
    :param env: the environment
    :param scale_percent: 50.
    original oberservation shape [1,240,320,channnel]
    """

    def __init__(self, env: gym.Env,  scale_percent: int = 50, fixed : str = True):
        gym.ObservationWrapper.__init__(self, env)
        if fixed:
            self.width = 84
            self.height = 84
            self.channel = env.observation_space.shape[3]
        else:
            self.scale_percent = scale_percent
            self.width = int(env.observation_space.shape[2] * scale_percent / 100)
            self.height = int(env.observation_space.shape[1] * scale_percent / 100)
            self.channel = env.observation_space.shape[3]
        self.observation_space = spaces.Box(
            low=0.0, high=255.0, shape=(1, self.height, self.width, self.channel), dtype=env.observation_space.dtype
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame
        :param frame: environment frame
        :return: the observation
        """
        frame = cv2.resize(frame[0], (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.channel == 1:
            return frame[None, :, :, None]
        else:
            return frame[None, :, :, :]

def make_offworld_env(
                    env_name='OffWorldDockerMonolithContinuousSim-v0',
                    env_type = 'sim', 
                    channel_type = 'DEPTH_ONLY', 
                    mode = 'train',
                    experiment_name = 'SAC-Continuous',
                    seed = 0,
                    rank = 0,
                    model_name = 'SAC-SIM-Continuous'
                    ):
    """
    Create a wrapped function, monitored VecEnv for offworld gym.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for offworld gym.

    Args:
    env_name: for discrete envs:'OffWorldDockerMonolithDiscreteSim-v0'/'OffWorldMonolithDiscreteReal-v0',
              for continuous envs:'OffWorldDockerMonolithContinuousSim-v0'/'OffWorldMonolithontinuousReal-v0',
    env_type: running algo on simulator or real robot. (value: 'sim','real').
    channel_type： select input sensory data from DEPTH_ONLY/RGB/RGBD.
    mode: select algorithm mode from train/test.
    experiment_name: user cunstom experiment name.

    """
    set_random_seed(seed)
    log_dir = "logs/" + model_name
    
    def _init_sim():

        if channel_type == 'DEPTH_ONLY':
            env =   gym.make(env_name, channel_type=Channels.DEPTH_ONLY)
        elif channel_type == 'RGB':
            env =   gym.make(env_name, channel_type=Channels.RGB)
        else:
            env =   gym.make(env_name, channel_type=Channels.RGBD)

        env = WarpFrame(env)
        env = Monitor(env,log_dir)


        # env.seed(seed + rank)

        return env

    def _init_real():

        if channel_type == 'DEPTH_ONLY':
            env =   gym.make(env_name, channel_type=Channels.DEPTH_ONLY, resume_experiment=True,
                        learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN, experiment_name=experiment_name)
        elif channel_type == 'RGB':
            env =   gym.make(env_name, channel_type=Channels.RGB, resume_experiment=True,
                        learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN, experiment_name=experiment_name)
        else:
            env =   gym.make(env_name, channel_type=Channels.RGBD, resume_experiment=True,
                        learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN, experiment_name=experiment_name)
        
        env = WarpFrame(env)
        env = Monitor(env,log_dir)
        # env.seed(seed + rank)
        return env

    if env_type == 'sim':
        return _init_sim()
    else:
        return _init_real()
        

def make_vec_env(make_offworld_env, num_envs):

    """
    Create a wrapped function, parallelize offworld gym.

    Args:
    make_env: function after making env.
    num_envs: number of envs to vectorize.
    """

    envs = [make_offworld_env for i in range(num_envs)] # the offworld env is already randomly init
    # envs = [make_offworld_env(rank = i, log_dir=log_dir) for i in range(num_envs)]

    if num_envs == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    # envvs = VecMonitor(envs,log_dir) # not yet support offworld gym

    envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
    
    logging.info(f"action space: {envs.action_space} observation_space: {envs.observation_space}")

    return envs


