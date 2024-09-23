"""
"""

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from my_utils import make_env

if __name__ == '__main__':
    env_id = 'TmrPickPlaceCan'
    horizon = 500
    
    print('making env')
    vec_env = DummyVecEnv([make_env(env_id, 0, True)])

    print('load model')
    model = SAC.load(env_id, env=vec_env)

    print('reset env')
    obs = vec_env.reset()

    print('_observables: ')
    for key, val in vec_env.envs[0].env.env._observables.items():
        print(key, val)

    for t in range(horizon):
        print('predict by model')
        action, _states = model.predict(obs)

        print('step forward')
        obs, rewards, dones, info = vec_env.step(action)
        print(t, len(obs), obs)
        
        print('rendering env')
        vec_env.envs[0].render()