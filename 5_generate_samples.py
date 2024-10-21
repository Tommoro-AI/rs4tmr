import torch
import gymnasium as gym
import numpy as np
import mediapy as media
# 필요한 클래스 및 함수 임포트
from cleanrl.cleanrl.ppo_continuous_action import Agent, Args, ppo_make_env
import cv2
from robosuite.utils.camera_utils import CameraMover
import argparse
from tqdm import tqdm



if __name__ == '__main__':
    # get args for name
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='pickplace')
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()
    name = args.name
    num_samples = args.num_samples
    
    visualize = True
    frames = []

    # Argument 설정
    args = Args()
    task_id = 'pickplace'
    seed = 0
    gamma = 0.99
    num_episodes = 1
    render_camera = ['birdview']#,'agentview'] #('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
    camera_names = render_camera


    # 환경 생성
    env = gym.vector.SyncVectorEnv(
        [ppo_make_env(
            task_id=task_id, 
            reward_shaping=args.reward_shaping,
            idx=0, 
            control_mode="OSC_POSITION",
            capture_video=False, 
            run_name="eval", 
            gamma= args.gamma, 
            active_rewards="r",
            active_image=True, 
            fix_object=args.fix_object,
            wandb_enabled=False,
            verbose=False,
            control_freq=20,
            render_camera=render_camera,
            camera_names=camera_names,

            )
        ]
    )

    # 디바이스 설정 (cuda가 가능하면 cuda 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        print("Using CUDA")
    else :
        assert device == torch.device("cpu")

    # 평가 수행
    viewer_image_key = 'birdview'+'_depth'



    depth_frames = []
    can_pos_quat = []

    print(f"Start Generating {num_samples} samples")
    
    # progress bar
        
    for i in tqdm(range(num_samples)):
        obs, _ = env.reset()
        obs = torch.Tensor(obs).to(device)
        done = False
        episode_reward = 0

        image_frame = env.envs[0].image_states[viewer_image_key]
        if not viewer_image_key.endswith('depth'):
            image_frame = np.array(image_frame[::-1, :, :], dtype=np.uint8)  # numpy 배열로 변환
        else:
            image_frame = np.array(image_frame[::-1, :, :], dtype=np.float32)
        
        can_pos = env.envs[0].sim.data.get_body_xpos('Can_main')  # Assuming the object is called 'Can'
        can_quat = env.envs[0].sim.data.get_body_xquat('Can_main')
        depth_frames.append(image_frame)
        can_pos_quat.append(np.concatenate([can_pos, can_quat]))



    # save the depth frames and can pos/quat pairs

    import os
    import pickle
    from my_utils import get_current_time

    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    depth_frames = np.array(depth_frames)
    can_pos_quat = np.array(can_pos_quat)


    depth_frames_path = os.path.join(data_dir, f'depth_frames_{name}.npy')
    can_pos_quat_path = os.path.join(data_dir, f'can_pos_quat_{name}.npy')



    np.save(depth_frames_path, depth_frames)
    np.save(can_pos_quat_path, can_pos_quat)

    print(f"Depth frames saved to {depth_frames_path}")