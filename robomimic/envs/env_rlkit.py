import os
import json
import h5py
import argparse
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase
from copy import deepcopy
import robomimic.envs.env_base as EB
import robomimic.envs.env_gym as EG
import gym


class EnvRLkitWrapperDummy:
    def __init__(
            self,
            action_dim,
            obs_dim,
            obs_img_dim=48,  # rendered image size
            transpose_image=True,  # transpose for pytorch by default
            camera_names=['agentview'],
            observation_mode='pixels',
    ):
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.observation_mode = observation_mode
        assert self.observation_mode in ["pixels", "states"]
        self.obs_img_dim = obs_img_dim
        self.transpose_image = transpose_image
        self.camera_names = camera_names
        self._set_observation_space()
        self._set_action_space()
        self._init_obs()

    def _init_obs(self):
        image_modalities = ["image"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [],  # technically unused, so we don't have to specify all of them
                "image": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    def _set_action_space(self):
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (self.obs_img_dim ** 2) * 3
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)

            obs_bound = 100
            obs_high = np.ones(self.obs_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'state': state_space}
            for name in self.camera_names:
                spaces[name] = img_space
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

class EnvRLkitWrapper(EB.EnvBase):
    def __init__(
        self,
        env_robosuite,  # EnvRobosuite object
        obs_img_dim=48,  # rendered image size
        transpose_image=True,  # transpose for pytorch by default
        camera_names=['agentview'],
        observation_mode='pixels',
        take_video=False,
    ):
        self.env = env_robosuite
        self.observation_mode = observation_mode
        assert self.observation_mode in ["pixels", "states"]
        self.obs_img_dim = obs_img_dim
        self.transpose_image = transpose_image
        self.camera_names = camera_names
        self._set_observation_space()
        self._set_action_space()
        self._init_obs()
        self.take_video = take_video

    def _init_obs(self):
        image_modalities = ["image"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [],  # technically unused, so we don't have to specify all of them
                "image": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    def _set_action_space(self):
        self.action_dim = self.env.action_dimension
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (self.obs_img_dim ** 2) * 3
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)

            robot_state_dim = 9 # XYZ (3) + QUAT (4) + GRIPPER_STATE (2)
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'state': state_space}
            for name in self.camera_names:
                spaces[name] = img_space
            self.observation_space = gym.spaces.Dict(spaces)
        elif self.observation_mode == 'states':
            if "Lift" in self.env.name:
                robot_state_dim = 9 + 10 # XYZ (3) + QUAT (4) + GRIPPER_STATE (2) + OBJECT_INFO
            elif "Can" in self.env.name or "Square" in self.env.name:
                robot_state_dim = 9 + 14
            elif "Transport" in self.env.name:
                robot_state_dim = 9 + 41
            
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def get_observation(self):
        state_info = self.env.get_observation()

        robot0_eef_pos = state_info['robot0_eef_pos']
        robot0_eef_quat = state_info['robot0_eef_quat']
        robot0_gripper_qpos = state_info['robot0_gripper_qpos']
        object_info = state_info['object']

        if self.observation_mode == 'pixels':

            if len(self.camera_names) == 1:
                image_observation = self.env.render(mode="rgb_array",
                                                    height=self.obs_img_dim,
                                                    width=self.obs_img_dim,
                                                    camera_name=self.camera_names[0])

                if self.transpose_image:
                    image_observation = np.transpose(image_observation, (2, 0, 1))
                image_observation = np.float32(image_observation.flatten()) / 255.0

                observation = {
                    'state': np.concatenate(
                        (robot0_eef_pos,
                         robot0_eef_quat,
                         robot0_gripper_qpos)),
                         #object_info)),
                    'image': image_observation
                }
            else:
                observation = {
                    'state': np.concatenate(
                        (robot0_eef_pos,
                         robot0_eef_quat,
                         robot0_gripper_qpos)),
                         #object_info))
                }
                for i in range(len(self.camera_names)):
                    image_observation = self.env.render(mode="rgb_array",
                                                        height=self.obs_img_dim,
                                                        width=self.obs_img_dim,
                                                        camera_name=self.camera_names[i])

                    if self.transpose_image:
                        image_observation = np.transpose(image_observation, (2, 0, 1))
                    image_observation = np.float32(image_observation.flatten()) / 255.0
                    observation[self.camera_names[i]] = image_observation

                    image_observation = self.env.render(mode="rgb_array",
                                                        height=self.obs_img_dim,
                                                        width=self.obs_img_dim,
                                                        camera_name='frontview')

                    if self.transpose_image:
                        image_observation = np.transpose(image_observation, (2, 0, 1))
                    image_observation = np.float32(image_observation.flatten()) / 255.0
                    observation['image'] = image_observation

        elif self.observation_mode == 'states':

            #image_observation = self.env.render(mode="rgb_array",
            #                                    height=self.obs_img_dim,
            #                                    width=self.obs_img_dim,
            #                                    camera_name=self.camera_names[0])

            #if self.transpose_image:
            #    image_observation = np.transpose(image_observation, (2, 0, 1))
            #image_observation = np.float32(image_observation.flatten()) / 255.0

            observation = {
                'state': np.concatenate(
                    (robot0_eef_pos,
                     robot0_eef_quat,
                     robot0_gripper_qpos,
                     object_info)),
                #'camera': image_observation,
            }

        else:
            raise NotImplementedError

        return observation

    def reset(self):
        self.env.reset()
        return self.get_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["success"] = self.env.is_success()["task"]
        info["success"] = int(info["success"])
        return self.get_observation(), reward, done, info

    def reset_to(self, state):
        return self.env.reset_to(state)

    def render(self, mode="human", height=None, width=None, camera_name=None):
        return self.env.render(mode="human", height=None, width=None, camera_name=None)

    def get_state(self):
        return self.env.get_state()

    def get_reward(self):
        return self.env.get_reward()

    def get_goal(self):
        return self.env.get_goal()

    def set_goal(self, **kwargs):
        return self.env.set_goal(**kwargs)

    def is_done(self):
        return self.env.is_done()

    def is_success(self):
        return self.env.is_success()

    def action_dimension(self):
        return self.action_dimension

    def name(self):
        return self.env.name

    def type(self):
        return self.env.type

    def serialize(self):
        return self.env.serialize()

    def create_for_data_processing(self, cls, camera_names, camera_height,
                                   camera_width, reward_shaping, **kwargs):
        return self.env.create_for_data_processing(cls, camera_names, camera_height,
                                                   camera_width, reward_shaping, **kwargs)

    def rollout_exceptions(self):
        return self.env.rollout_exceptions

