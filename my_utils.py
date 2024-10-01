"""
"""

import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor



### jesnk: CleanRL
from robosuite.controllers import load_controller_config

import requests
from datetime import datetime

def get_current_time(return_type='str', format=None):
    response = requests.get('http://worldtimeapi.org/api/timezone/Asia/Seoul')
    if response.status_code == 200:
        data = response.json()
        datetime_str = data['datetime']
        seoul_datetime = datetime.fromisoformat(datetime_str)
        if return_type == 'datetime':
            return datetime.datetime.now(seoul_datetime)
        elif return_type == 'str' :
            if format is not None:
                return datetime.datetime.now(seoul_datetime).strftime(format)
            else:
                return seoul_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
    else:
        return "Fail to get current time"





def init_env (
        task_id='pickplace',
        active_rewards = "rlhg", 
        control_mode='OSC_POSE',
        reward_shaping=True,
        fix_object = None, 
        active_image=False,
        selected_observable_list = [], 
        return_raw_env=False, 
        wandb_enabled=True,
        verbose=True,
        control_freq=20,
        ignore_done=True,
    ):
    
    if task_id == 'pickplace':
        env_id = "TmrPickPlaceCan"
    elif task_id == 'lift':
        env_id = "Lift"
    else :
        print(f"Invalid task_id: {task_id}")
        AssertionError(f"Invalid task_id: {task_id}")
    
    #print(env_id)
    if control_mode == 'default':
        control_mode = None
    elif control_mode == 'osc':
        control_mode = 'OSC_POSE'
    elif control_mode == 'osc_position':
        control_mode = 'OSC_POSITION'
    elif control_mode == 'ik':
        control_mode = 'IK_POSE'
    else :
        AssertionError(f"Invalid control_mode: {control_mode}")
    controller_config = load_controller_config(default_controller=control_mode)

    if env_id == "TmrPickPlaceCan":
        rsenv = suite.make(
            env_id,#"TmrPickPlaceCan",
            robots="UR5e",  # use UR5e robot
            reward_shaping=reward_shaping,  # use dense rewards
            use_camera_obs=True,  # use pixel observations
            has_offscreen_renderer=True,  # needed if using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            control_freq=control_freq,  # control should happen fast enough so that simulation looks smooth
            render_camera='agentview' ,
            camera_names='agentview' ,
            camera_depths=True,
            wandb_enabled=wandb_enabled,
            active_rewards=active_rewards,
            fix_object = fix_object,
            controller_configs=controller_config,
            ignore_done=ignore_done,

        )
    elif env_id == "Lift":
        rsenv = suite.make(
            env_id,#"TmrPickPlaceCan",
            robots="UR5e",  # use UR5e robot
            reward_shaping=reward_shaping,  # use dense rewards
            use_camera_obs=True,  # use pixel observations
            has_offscreen_renderer=True,  # needed if using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            control_freq=control_freq,  # control should happen fast enough so that simulation looks smooth
            render_camera='agentview' ,
            camera_names='agentview' ,
            camera_depths=True,
            wandb_enabled=wandb_enabled,
            #active_rewards=active_rewards,
            #fix_object = fix_object,
            controller_configs=controller_config,
            ignore_done=ignore_done,
        )
    else :
        AssertionError(f"Invalid env_id: {env_id}")
        
    ### Observables
    # PickPlace
    if env_id == "TmrPickPlaceCan":
        full_observable_list = [
            'robot0_joint_pos', 
            'robot0_joint_pos_cos', 
            'robot0_joint_pos_sin', 
            'robot0_joint_vel', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_eef_vel_lin', 
            'robot0_eef_vel_ang', 
            'robot0_gripper_qpos', 
            'robot0_gripper_qvel', 
            'agentview_image', 
            'agentview_depth',
            'world_pose_in_gripper', 
            'Milk_pos', 
            'Milk_quat', 
            'Milk_to_robot0_eef_pos', 
            'Milk_to_robot0_eef_quat', 
            'Bread_pos', 
            'Bread_quat', 
            'Bread_to_robot0_eef_pos', 
            'Bread_to_robot0_eef_quat', 
            'Cereal_pos', 
            'Cereal_quat', 
            'Cereal_to_robot0_eef_pos', 
            'Cereal_to_robot0_eef_quat', 
            'Can_pos', 
            'Can_quat', 
            'Can_to_robot0_eef_pos', 
            'Can_to_robot0_eef_quat',
        ]
    elif env_id == 'Lift':
            
        full_observable_list = [
            'robot0_joint_pos', 
            'agentview_depth', 
            'cube_pos', 
            'robot0_eef_vel_lin', 
            'robot0_eef_vel_ang', 
            'robot0_eef_quat', 
            'robot0_eef_pos', 
            
            'cube_quat', 
            'robot0_gripper_qpos', 
            'robot0_gripper_qvel', 

            'robot0_joint_pos_cos', 
            'robot0_joint_vel', 
            'gripper_to_cube_pos', 
            'robot0_joint_pos_sin', 
            'agentview_image'
            ]
    else :
        AssertionError(f"Invalid env_id: {env_id}")

    
    for observable in full_observable_list:
        #print(observable)
        rsenv.modify_observable(observable, 'enabled', False)
        rsenv.modify_observable(observable, 'active', False)
    
    
    ### Select observables
    if env_id == "TmrPickPlaceCan":
        selected_observable_list = [
            #'robot0_joint_pos',
            #'robot0_joint_vel',
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_eef_vel_lin', 
            'robot0_eef_vel_ang', 
            'robot0_gripper_qpos',
            'robot0_gripper_qvel', 
            'Can_to_robot0_eef_pos', 
            'Can_to_robot0_eef_quat',
            #'agentview_image', 
            #'agentview_depth'
            # Cube
        ]
    elif env_id == "Lift":
        selected_observable_list = [
            #'robot0_joint_pos',
            #'robot0_joint_vel',
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_eef_vel_lin', 
            'robot0_eef_vel_ang', 
            'robot0_gripper_qpos',
            'robot0_gripper_qvel', 
            #'agentview_image', 
            #'agentview_depth'
            # Cube
            'gripper_to_cube_pos',             
        ]        
    else :
        AssertionError(f"Invalid env_id: {env_id}")
            
    ### Add image observables if required
    if active_image:
        selected_observable_list.append('agentview_image')
        selected_observable_list.append('agentview_depth')


    ### Set selected observables        
    for observable in selected_observable_list:
        rsenv.modify_observable(observable, 'enabled', True)
        rsenv.modify_observable(observable, 'active', True)
    

    # print('Robosuite environment maked:',type(rsenv) , rsenv, dir(rsenv))
    # print(len(rsenv._observables.keys()))
    #print(f"Observable keys: {rsenv._observables.keys()}")
    if verbose :
        tmp_obs = rsenv.reset()
        keys = tmp_obs.keys()
        print("########################")
        print("### Observation keys ###")
        for key in keys:
            print(f"Key: {key}, size: {tmp_obs[key].size}")
        print(f"Total observation size: {sum([tmp_obs[key].size for key in keys])}")
        print("########################")
        print("####### Options ########")
        print(f"task_id: {task_id}")
        print(f"active_rewards: {active_rewards}")
        print(f"control_mode: {control_mode}")
        print(f"reward_shaping: {reward_shaping}")
        print(f"fix_object: {fix_object}")
        print(f"active_image: {active_image}")
        print(f"wandb_enabled: {wandb_enabled}")
        print("########################")

    if return_raw_env:
        return rsenv

    wrapped_env = GymWrapper(
        rsenv,
    )
    wrapped_env.reset()
    monitor_env = Monitor(wrapped_env)
    #set_random_seed(seed)
    return monitor_env#


def init_env_2 (return_raw_env=False):
    def _init():
        rsenv = suite.make(
            "TmrPickPlaceCan",
            robots="UR5e",  # use UR5e robot
            use_camera_obs=True,  # use pixel observations
            has_offscreen_renderer=True,  # needed if using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            render_camera='agentview',
            camera_names='agentview',
            camera_depths=True,
            camera_heights=84,
            camera_widths=84,
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
        #for observable in useless_observable_list:
        #    rsenv.modify_observable(observable, 'enabled', False)
        #    rsenv.modify_observable(observable, 'active', False)

        print('Robosuite environment maked:',type(rsenv) , rsenv, dir(rsenv))
        print(len(rsenv._observables.keys()))
        #print(f"Observable keys: {rsenv._observables.keys()}")
        #keys = tmp_obs.keys()
        #print("### Observation keys ###")
        #for key in keys:
        #    print(f"Key: {key}, size: {tmp_obs[key].size}")
        #print(f"Total observation size: {sum([tmp_obs[key].size for key in keys])}")
        #print("########################")

        if return_raw_env:
            return rsenv
            #return Monitor(rsenv)
        
        #obs_inspector = ObservationInspector(rsenv)
        wrapped_env = GymWrapper(
            rsenv,
            #keys
        )

        #env.reset(seed=seed + rank)
        wrapped_env.reset()
        monitor_env = Monitor(wrapped_env)
        #return env
        #set_random_seed(seed)
        return monitor_env#

    set_random_seed(0)
    return _init







def make_env(env_id: str, rank: int, render: bool=True, seed: int = 0):
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
            has_offscreen_renderer=True,  # needed if using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            single_object_mode= 1,
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