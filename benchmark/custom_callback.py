import os
import glob
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
from collections import deque

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback

class RingBuffer(object):

    """
    Implement a pre-allocated ring buffer to store the last n models/replay buffers.
    """
    def __init__(self, n):
        self._buf = [None] * n
        self._index = 0
        self._valid = 0
    def add(self, obj):
        n = len(self._buf)
        self._buf[self._index] = obj
        self._index += 1
        if self._index == n:
            self._index = 0
        if self._valid < n:
            self._valid += 1
    def __len__(self):
        return self._valid


class CheckpointAndBufferCallback(BaseCallback):
    """
    Callback for evaluating an agent and saving a model every ``save_freq`` calls
    to ``env.step()``.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    :param n_models: number of latest model to keep track of, save disk space.
    :param save_freq: freq to save a model.
    :param save_path: model/buffer save directory.
    :param name_prefix: Common prefix to the saved models
    :param verbose: for more looging info.
    """

    def __init__(self, n_models:int, save_freq: int, save_path: str, previous_timesteps: int, name_prefix: str = "rl_model", verbose: int = 1, replay_buffer: bool = True,
                    ):
        super( CheckpointAndBufferCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.replay_buffer = replay_buffer
        self.previous_timesteps = previous_timesteps
        self.n_models = n_models if not n_models else 5
        # self.last_n_models_index = deque(maxlen=self.n_models)
        self.previous_saved_models_indices = []
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        
        if self.n_calls % self.save_freq == 0:
             # check previous saved models
            tmp_previous_saved_models = glob.glob(f"{self.name_prefix}_*_steps.zip")

            # check previous model index without duplicate and save into a indices list
            if len(tmp_previous_saved_models) > 0:

                for model_name in tmp_previous_saved_models:
                    tmp_index = model_name.split("_")[-2]
                    if tmp_index not in self.previous_saved_models_indices:
                        self.previous_saved_models_indices.append(tmp_index)

                self.previous_saved_models_indices.sort(key=int)

                indices_to_delete = []
                left = 0
                right = len(self.previous_saved_models_indices) - self.n_models - 1 # might be negative

                while right >= 0 and right >= left:
                    indices_to_delete.append(self.previous_saved_models_indices[right])
                    right -= 1

                # clean redundant models and buffers
                for index in indices_to_delete:
                    os.delete(os.path.join(self.save_path, f"{self.name_prefix}_{index}_steps.zip"))
                    os.delete(os.path.join(self.save_path, f"{self.name_prefix}_{index}_buffer.pkl"))
            if not self.previous_timesteps:
                model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
                self.model.save(model_path)
                if self.replay_buffer:
                    replay_buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_buffer")
                    self.model.save_replay_buffer(replay_buffer_path)
            else:
                model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps + self.previous_timesteps}_steps")
                self.model.save(model_path)
                if self.replay_buffer:
                    replay_buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps + self.previous_timesteps}_buffer")
                    self.model.save_replay_buffer(replay_buffer_path)

            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")

        return True