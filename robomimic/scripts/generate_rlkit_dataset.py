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

def playback_dataset(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1
    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not args.use_actions, "playback with observations is offline and does not support action playback"

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    image=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)

        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)], 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]

        playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, actions=actions, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
        )

    f.close()
    if write_video:
        video_writer.close()


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
        default="/home/huihanl/",
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

    args = parser.parse_args()

    f = h5py.File(args.dataset, "r")

    demos = list(f["data"].keys())
    attributes = list(f["data/{}".format(demos[0])].keys())
    print("attributes: ", attributes)

    rlkit_data = []

    if args.n:
        n = args.n
    else:
        n = len(demos)

    for ind in range(n):

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
        obs = np.concatenate([ee_pos, ee_quat, gripper_pos], axis=1)
        obs = list(obs)

        next_ee_pos = f["data/{}/next_obs/robot0_eef_pos".format(ep)][()]
        next_ee_quat = f["data/{}/next_obs/robot0_eef_quat".format(ep)][()]
        next_gripper_pos = f["data/{}/next_obs/robot0_gripper_qpos".format(ep)][()]
        next_obs = np.concatenate([next_ee_pos, next_ee_quat, next_gripper_pos], axis=1)
        next_obs = list(next_obs)

        traj["observations"] = obs
        traj["actions"] = actions
        traj["rewards"] = rewards
        traj["next_observations"] = next_obs
        traj["terminals"] = dones

        rlkit_data.append(traj)

        filename = os.path.basename(args.dataset)[:-5] + ".npy"
        path = osp.join(args.data_save_path, filename)
        print(path)
        np.save(path, rlkit_data)

        np.save(path, rlkit_data)

