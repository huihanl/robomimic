import os
import os.path as osp
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    parser.add_argument(
        "--data-save-path",
        type=str,
        default="/home/huihanl",
        help="data save path",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    parser.add_argument(
        "--img_dim",
        type=int,
        default=84,
    )

    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--video_file_paths",
        type=str,
        nargs='+',
        default=[],
        help="saved videos",
    )
    
    parser.add_argument(
        "--task_name",
        type=str,
    )

    args = parser.parse_args()

    video_paths = args.video_file_paths

    f = h5py.File(args.dataset, "r")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    attributes = list(f["data/{}".format(demos[0])].keys())
    print("attributes: ", attributes)

    rlkit_data = []

    if args.n:
        n = args.n
    else:
        n = len(demos)


    for ind in range(n):
        print("at traj: ", ind)
        traj = dict(
            observations=[],
            actions=[],
            rewards=[],
            next_observations=[],
            terminals=[],
        )

        ep = demos[ind]
        actions = list(f["data/{}/actions".format(ep)][()])
        dones = list(f["data/{}/dones".format(ep)][()])
        rewards = list(f["data/{}/rewards".format(ep)][()])

        ee_pos = f["data/{}/obs/robot0_eef_pos".format(ep)][()]
        ee_quat = f["data/{}/obs/robot0_eef_quat".format(ep)][()]
        gripper_pos = f["data/{}/obs/robot0_gripper_qpos".format(ep)][()]
        #object_info = f["data/{}/obs/object".format(ep)][()]

        obs = np.concatenate([ee_pos, ee_quat, gripper_pos], axis=1)
        obs = list(obs)

        next_ee_pos = f["data/{}/next_obs/robot0_eef_pos".format(ep)][()]
        next_ee_quat = f["data/{}/next_obs/robot0_eef_quat".format(ep)][()]
        next_gripper_pos = f["data/{}/next_obs/robot0_gripper_qpos".format(ep)][()]
        #next_object_info = f["data/{}/next_obs/object".format(ep)][()]
        next_obs = np.concatenate([next_ee_pos, next_ee_quat, next_gripper_pos], axis=1)
        next_obs = list(next_obs)

        traj["observations"] = [{"state": o} for o in obs]
        traj["actions"] = actions
        traj["rewards"] = rewards
        traj["next_observations"] = [{"state": o} for o in next_obs]
        traj["terminals"] = dones

        rlkit_data.append(traj)


    video_lst_dict = np.load(video_paths[0], allow_pickle=True).item()
    print(len(video_lst_dict['frontview']))
    print(len(video_lst_dict['robot0_eye_in_hand']))
    for i in range(1, len(video_paths)):
        new_video = np.load(video_paths[i], allow_pickle=True).item()
        print(len(new_video['frontview']))
        print(len(new_video['robot0_eye_in_hand']))
        for k in video_lst_dict:
            video_lst_dict[k] += new_video[k]
    #import pdb; pdb.set_trace()
    for k in video_lst_dict:
        assert len(video_lst_dict[k]) == 200

    for camera_name in video_lst_dict:
        video_lst = video_lst_dict[camera_name]
        for id in range(len(rlkit_data)):
            video = video_lst[id]
            for vid in range(len(rlkit_data[id]["observations"])):
                rlkit_data[id]["observations"][vid][camera_name] = video[vid]

            for vid in range(1, len(rlkit_data[id]["observations"]) + 1):
                rlkit_data[id]["next_observations"][vid - 1][camera_name] = video[vid]

    camera_name_str = '_'.join(args.camera_names) + "_{}_{}".format(args.img_dim, args.n if args.n else "full")
    filename = args.task_name + "_{}.npy".format(camera_name_str)
    path = osp.join(args.data_save_path, filename)
    print(path)
    np.save(path, rlkit_data)

