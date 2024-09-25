"""
"""

import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor








class ObservationInspector:
    def __init__(self, env):
        self.env = env
        self.set_obs_spec(env)
        self.observation = None
        
    
    def unpack_vectorized_obs(self, obs):
        keys = self.env.observation_space.keys()
        unpacked_obs = {}
        start = 0
        for key in keys:
            end = start + self.obs_space[key]
            unpacked_obs[key] = obs[start:end]
            start = end
        return unpacked_obs
        


    def set_obs_spec(self,raw_env):
        obs = raw_env.reset()
        self.keys = obs.keys()
        self.obs_dims = []
        for key in self.keys:
            self.obs_dims.append(obs[key].size)
            print(f"Key: {key}, size: {obs[key].size}")
        
        self.dict_obs_space = {}
        for i, key in enumerate(self.keys):
            self.dict_obs_space[key] = self.obs_dims[i]
            
        # get index of each key in vectorized observation
        self.key2idx = {}
        start = 0
        for key in self.keys:
            self.key2idx[key] = start
            start += self.dict_obs_space[key]
            
        print(f"Total observation size: {start}")
            
    def get_obs_according_to_key(self, obs, key):
        idx = self.key2idx[key]
        return obs[idx:idx+self.dict_obs_space[key]]

            
    def obs_keys(self):
        return self.keys



### jesnk: CleanRL

def init_env (return_raw_env=False, selected_observable_list = []):
    print(f"Initalized env with init_env")
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
        #single_object_mode= 1, # inline parameter
    )

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
    for observable in full_observable_list:
        rsenv.modify_observable(observable, 'enabled', False)
        rsenv.modify_observable(observable, 'active', False)
    
    if selected_observable_list == []:
        selected_observable_list = [
            'robot0_joint_pos',
            'robot0_joint_vel',
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_eef_vel_lin', 
            'robot0_eef_vel_ang', 
            'robot0_gripper_qpos',
            'robot0_gripper_qvel', 
            'Can_to_robot0_eef_pos', 
            'Can_to_robot0_eef_quat',
        ]
    # print(f"Selected observables: {selected_observable_list}")
        
    for observable in selected_observable_list:
        rsenv.modify_observable(observable, 'enabled', True)
        rsenv.modify_observable(observable, 'active', True)
        

    # print('Robosuite environment maked:',type(rsenv) , rsenv, dir(rsenv))
    # print(len(rsenv._observables.keys()))
    #print(f"Observable keys: {rsenv._observables.keys()}")
    tmp_obs = rsenv.reset()
    keys = tmp_obs.keys()
    # print("### Observation keys ###")
    # for key in keys:
    #     print(f"Key: {key}, size: {tmp_obs[key].size}")
    # print(f"Total observation size: {sum([tmp_obs[key].size for key in keys])}")
    # print("########################")

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