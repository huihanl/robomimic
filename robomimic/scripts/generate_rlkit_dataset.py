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


def playback_video(
    env,
    initial_state,
    states,
    actions,
    img_dim,
    camera_name,
    ):
    assert isinstance(env, EnvBase)
    # load the initial state
    env.reset()
    env.reset_to(initial_state)
    traj_len = states.shape[0]
    video = []
    for i in range(traj_len):
        env.reset_to({"states": states[i]})
        img = env.render(mode="rgb_array", height=img_dim, width=img_dim, camera_name=camera_name)
        video.append(img)
    env.step(actions[-1])
    img = env.render(mode="rgb_array", height=img_dim, width=img_dim, camera_name=camera_name)
    video.append(img)
    assert len(video) == len(states) + 1
    return video

def playback_dataset(args):

    # need to make sure ObsUtils knows which observations are images, but it doesn't matter
    # for playback since observations are unused. Pass a dummy spec here.
    image_modalities = ["image"]
    obs_modality_specs = {
        "obs": {
            "low_dim": [],  # technically unused, so we don't have to specify all of them
            "image": image_modalities,
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_modality_specs)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)

    env = EnvUtils.create_env_from_metadata(env_meta=env_meta,
                                            render=False,
                                            render_offscreen=True,
                                            use_image_obs=True)

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    video_lst_keys = {}
    for cam_name in args.camera_names:
        video_lst_keys[cam_name] = []

    start = args.start_id
    end = args.end_id
    for ind in range(start, end):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        actions = f["data/{}/actions".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]


        for cam_name in args.camera_names:
            video = playback_video(
                env=env, 
                initial_state=initial_state, 
                states=states,
                actions=actions,
                img_dim=args.img_dim,
                camera_name=cam_name,
            )

            video_lst_keys[cam_name].append(video)
            np.save("video_lst_dict_start_at_{}.npy".format(start), video_lst_keys)
            # video_writer = imageio.get_writer("ep_{}_cam_name_{}.mp4".format(ep, cam_name), fps=20)
            # for v in video:
            #     video_writer.append_data(v)
            # video_writer.close()

    f.close()
    return video_lst_keys


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
        default="/mnt/data0/huihanl/",
        help="data save path",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of trajectories from the start (optional)",
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="Specify starting id to tackle rendering quit issue",
    )

    parser.add_argument(
        "--end-id",
        type=int,
        default=None,
        help="Specify ending id to tackle rendering quit issue",
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
        "--task",
        type=str,
    )

    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    args = parser.parse_args()

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
        object_info = f["data/{}/obs/object".format(ep)][()]
        import pdb; pdb.set_trace()
        
        obs = np.concatenate([ee_pos, ee_quat, gripper_pos, object_info], axis=1)
        obs = list(obs)

        next_ee_pos = f["data/{}/next_obs/robot0_eef_pos".format(ep)][()]
        next_ee_quat = f["data/{}/next_obs/robot0_eef_quat".format(ep)][()]
        next_gripper_pos = f["data/{}/next_obs/robot0_gripper_qpos".format(ep)][()]
        next_object_info = f["data/{}/next_obs/object".format(ep)][()]
        next_obs = np.concatenate([next_ee_pos, next_ee_quat, next_gripper_pos, next_object_info], axis=1)
        next_obs = list(next_obs)

        traj["observations"] = [{"state": o} for o in obs]
        traj["actions"] = actions
        traj["rewards"] = rewards
        traj["next_observations"] = [{"state": o} for o in next_obs]
        traj["terminals"] = dones

        rlkit_data.append(traj)

    if len(args.camera_names) == 0:
        camera_name_str = "state_only" + "_{}".format(args.n if args.n else "full")
        filename = args.task + "_{}.npy".format(camera_name_str)
        path = osp.join(args.data_save_path, filename)
        print(path)
        np.save(path, rlkit_data)
        exit()

    video_lst_dict = playback_dataset(args) # a dict of images from all camera views

    for camera_name in video_lst_dict:
        video_lst = video_lst_dict[camera_name]
        for id in range(len(rlkit_data)):
            video = video_lst[id]
            for vid in range(len(rlkit_data[id]["observations"])):
                rlkit_data[id]["observations"][vid][camera_name] = video[vid]

            for vid in range(1, len(rlkit_data[id]["observations"]) + 1):
                rlkit_data[id]["next_observations"][vid - 1][camera_name] = video[vid]

    camera_name_str = '_'.join(args.camera_names) + "_{}_{}".format(args.img_dim, args.n if args.n else "full")
    filename = args.task + "_{}.npy".format(camera_name_str)
    path = osp.join(args.data_save_path, filename)
    print(path)
    np.save(path, rlkit_data)

