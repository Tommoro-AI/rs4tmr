"""
"""

import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

def make_env(env_id: str, rank: int, render: bool, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():

        rsenv = suite.make(
            "TmrPickPlaceCan",
            robots="UR5e",  # use UR5e robot
            use_camera_obs=False,  # use pixel observations
            has_offscreen_renderer=False,  # needed if using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )

        # full_observable_list = [
        #     'robot0_joint_pos', 
        #     'robot0_joint_pos_cos', 
        #     'robot0_joint_pos_sin', 
        #     'robot0_joint_vel', 
        #     'robot0_eef_pos', 
        #     'robot0_eef_quat', 
        #     'robot0_eef_vel_lin', 
        #     'robot0_eef_vel_ang', 
        #     'robot0_gripper_qpos', 
        #     'robot0_gripper_qvel', 
        #     'agentview_image', 
        #     'world_pose_in_gripper', 
        #     'Milk_pos', 
        #     'Milk_quat', 
        #     'Milk_to_robot0_eef_pos', 
        #     'Milk_to_robot0_eef_quat', 
        #     'Bread_pos', 
        #     'Bread_quat', 
        #     'Bread_to_robot0_eef_pos', 
        #     'Bread_to_robot0_eef_quat', 
        #     'Cereal_pos', 
        #     'Cereal_quat', 
        #     'Cereal_to_robot0_eef_pos', 
        #     'Cereal_to_robot0_eef_quat', 
        #     'Can_pos', 
        #     'Can_quat', 
        #     'Can_to_robot0_eef_pos', 
        #     'Can_to_robot0_eef_quat',
        # ]
        # for observable in full_observable_list:
        #     rsenv.modify_observable(observable, 'enabled', True)
        #     rsenv.modify_observable(observable, 'active', True)
        
        useless_observable_list = [
            # 'robot0_joint_pos', 
            # 'robot0_joint_pos_cos', 
            # 'robot0_joint_pos_sin', 
            # 'robot0_joint_vel', 
            # 'robot0_eef_pos', 
            # 'robot0_eef_quat', 
            # 'robot0_eef_vel_lin', 
            # 'robot0_eef_vel_ang', 
            # 'robot0_gripper_qpos', 
            # 'robot0_gripper_qvel', 
            # 'agentview_image', 
            # 'world_pose_in_gripper', 
            # 'Milk_pos', 
            # 'Milk_quat', 
            # 'Milk_to_robot0_eef_pos', 
            # 'Milk_to_robot0_eef_quat', 
            # 'Bread_pos', 
            # 'Bread_quat', 
            # 'Bread_to_robot0_eef_pos', 
            # 'Bread_to_robot0_eef_quat', 
            # 'Cereal_pos', 
            # 'Cereal_quat', 
            # 'Cereal_to_robot0_eef_pos', 
            # 'Cereal_to_robot0_eef_quat', 
            # 'Can_pos', 
            # 'Can_quat', 
            # 'Can_to_robot0_eef_pos', 
            # 'Can_to_robot0_eef_quat',
        ]
        for observable in useless_observable_list:
            rsenv.modify_observable(observable, 'enabled', False)
            rsenv.modify_observable(observable, 'active', False)

        print('Robosuite environment maked:',type(rsenv) , rsenv, dir(rsenv))
        print(len(rsenv._observables.keys()))
        print(rsenv._observables.keys())
        
        env = GymWrapper(
            rsenv
        )

        env.reset(seed=seed + rank)

        return Monitor(env)
        #return env
    set_random_seed(seed)
    return _init