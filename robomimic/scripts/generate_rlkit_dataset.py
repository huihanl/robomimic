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
    actions
):
    assert isinstance(env, EnvBase)
    # load the initial state
    env.reset()
    env.reset_to(initial_state)
    traj_len = states.shape[0]
    video = []
    for i in range(traj_len):
        env.reset_to({"states": states[i]})
        img = env.render(mode="rgb_array", height=48, width=48, camera_name='agentview')
        video.append(img)
    env.step(actions[-1])
    img = env.render(mode="rgb_array", height=48, width=48, camera_name='agentview')
    video.append(img)
    assert len(video) == len(states) + 1
    return video

def playback_dataset(args):

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
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    video_lst = []
    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        actions = f["data/{}/actions".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        video = playback_video(
            env=env, 
            initial_state=initial_state, 
            states=states,
            actions=actions,
        )

        video_lst.append(video)

    f.close()
    return video_lst


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
        default="/Users/huihanliu/",
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

    video_lst = playback_dataset(args)

    for id in range(len(rlkit_data)):
        video = video_lst[id]
        for vid in range(len(rlkit_data[id]["observations"])):
            rlkit_data[id]["observations"][vid]["image"] = video[vid]

        for vid in range(1, len(rlkit_data[id]["observations"]) + 1):
            rlkit_data[id]["next_observations"][vid - 1]["image"] = video[vid]

    filename = os.path.basename(args.dataset)[:-5] + ".npy"
    path = osp.join(args.data_save_path, filename)
    print(path)
    np.save(path, rlkit_data)

