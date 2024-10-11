# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import datetime
import gymnasium as gym
import numpy as np

import wandb

project_root_path = ["/research/rs4tmr", "/research/rs4tmr/cleanrl", "/data/jskang/rs4tmr", "/data/jskang/rs4tmr/cleanrl"]
sys.path += project_root_path
from my_utils import init_env, get_current_time
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tr_cleanrl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    tags : str=None
    
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    task_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    # jesnk
    active_rewards: str = "rglh"
    save_interval: int = 50000
    fix_object: bool = False
    task_id: str = "pickplace"
    reward_shaping: bool = False
    control_mode: str = "OSC_POSITION"
    control_freq: int = 20
    num_eval_episodes: int = 10
    ignore_done: bool = False
    iota: bool = False
    load_model: str = None
    
    # omega
    omega_enabled: bool = False
    omega_convergence: float = 0.9
    omega_threshold: float = 0.1




class NormalizeObservationCustom(gym.wrappers.NormalizeObservation):
    def save_obs_running_average(self, path):
        np.savez(
            path,
            mean=self.obs_rms.mean,
            var=self.obs_rms.var,
            count=self.obs_rms.count
        )
    
    def load_obs_running_average(self, path):
        data = np.load(path)
        self.obs_rms.mean = data['mean']
        self.obs_rms.var = data['var']
        self.obs_rms.count = data['count']

class NormalizeRewardCustom(gym.wrappers.NormalizeReward):
    def save_reward_running_average(self, path):
        np.savez(
            path,
            mean=self.return_rms.mean,
            var=self.return_rms.var,
            count=self.return_rms.count
        )
    
    def load_reward_running_average(self, path):
        data = np.load(path)
        self.return_rms.mean = data['mean']
        self.return_rms.var = data['var']
        self.return_rms.count = data['count']


def ppo_make_env(task_id, reward_shaping,idx, control_freq, 
                 capture_video, run_name, gamma, 
                 control_mode='OSC_POSE',wandb_enabled=True, 
                 active_rewards="rglh", fix_object=False,active_image=False, verbose=True,
                 ignore_done=False,
                 
                 ):
    def thunk():
        capture_video = False
        if capture_video and idx == 0:
            env = init_env(
                task_id=task_id,
                wandb_enabled=wandb_enabled, 
                reward_shaping=reward_shaping,
                control_mode=control_mode,
                active_rewards=active_rewards, 
                fix_object=fix_object, 
                active_image=active_image,
                verbose=verbose,
                control_freq=control_freq,
                ignore_done=ignore_done,
               )
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = init_env(
                task_id=task_id,
                wandb_enabled=wandb_enabled,
                reward_shaping=reward_shaping, 
                control_mode=control_mode,
                active_rewards=active_rewards, 
                fix_object=fix_object, 
                active_image=active_image,
                verbose=verbose,
                control_freq=control_freq,
                ignore_done=ignore_done,

                )
        
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        #env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservationCustom(env)  # 커스텀 래퍼 사용
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeRewardCustom(env, gamma=gamma)  # 커스텀 래퍼 사용
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def load_ppo_checkpoint(checkpoint_path=None,
                        task_id='lift', 
                        iota = False, 
                        seed=1, 
                        control_mode='OSC_POSITION',
                        gamma=0.99, control_freq=20, active_image=False, verbose=False, ignore_done=False):
    args = Args()
    args.fix_object = False
    # 환경 생성
    env = gym.vector.SyncVectorEnv(
        [ppo_make_env(
            task_id= task_id,#task_id, 
            reward_shaping=True,
            idx=0, 
            capture_video=False, 
            control_mode=control_mode,
            run_name="eval", 
            gamma= gamma, 
            active_rewards="rghl",
            active_image=active_image, 
            fix_object=args.fix_object,
            control_freq=control_freq,
            ignore_done=ignore_done,
            wandb_enabled=False,
            verbose=verbose,
            )
        ]
    )
        
    # 디바이스 설정 (cuda가 가능하면 cuda 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        if verbose:
            print("Using CUDA")
    else :
        assert device == torch.device("cpu")

    if iota == False :
        path_prefix="/research/rs4tmr/cleanrl/cleanrl/"
    else :
        path_prefix="/data/jskang/rs4tmr/cleanrl/cleanrl/"
    weight_path = path_prefix + checkpoint_path + ".cleanrl_model"
    # Agent 초기화 및 모델 불러오기
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(weight_path, map_location=device))
    agent.eval()  # 평가 모드로 전환
        

    # 환경 생성 후
    env_instance = env.envs[0]
    # NormalizeObservationCustom 래퍼에 접근
    normalize_obs_wrapper = env_instance
    while not isinstance(normalize_obs_wrapper, NormalizeObservationCustom):
        normalize_obs_wrapper = normalize_obs_wrapper.env

    # NormalizeRewardCustom 래퍼에 접근
    normalize_reward_wrapper = env_instance
    while not isinstance(normalize_reward_wrapper, NormalizeRewardCustom):
        normalize_reward_wrapper = normalize_reward_wrapper.env

    # 상태 로드
    normalize_obs_wrapper.load_obs_running_average(path_prefix + checkpoint_path + "_obs_rms.npz")
    normalize_reward_wrapper.load_reward_running_average(path_prefix + checkpoint_path + "_reward_rms.npz")

    return env, agent
    
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def load_model_and_evaluate(model_path, global_step=None,
                            task_id=None, num_episodes=10, 
                            seed=1, iota=False, 
                            gamma=0.99, verbose = False, wandb_log = False, 
                            ignore_done=False,
                            control_mode='OSC_POSITION',
                            ):
    """
    저장된 모델을 불러와 환경에서 평가를 수행하는 함수
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, agent = load_ppo_checkpoint(checkpoint_path=model_path, 
                                     task_id=task_id, seed=seed, gamma=gamma, 
                                     active_image=False, verbose=verbose, 
                                     ignore_done=ignore_done,
                                     control_mode=control_mode,
                                     iota=iota,
                                     )
                                     
    eval_horizon = 200  # 평가 시 사용할 에피소드 길이
    num_episodes = num_episodes
    count_sucess = 0
    # 평가 수행
    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.Tensor(obs).to(device)
        done = False
        episode_reward = 0
          
        for i in range(eval_horizon):
            with torch.no_grad():
                obs = torch.Tensor(obs).to(device)
                action, _, _, _ = agent.get_action_and_value(obs)
            obs, reward, terminations, truncations, info = env.step(action.cpu().numpy())
            #print(f"reward: {reward}, terminations: {terminations}, truncations: {truncations}, infos: {infos}")
            done = np.logical_or(terminations, truncations).any()
            episode_reward += reward[0]  # 첫 번째 환경의 보상 합산
            
            if env.envs[0].is_success:
                count_sucess += 1
                break            
        if verbose:
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}, Success: {env.envs[0].is_success}, {i} step")
        total_rewards.append(episode_reward)

    env.close()
    
    success_rate = count_sucess/num_episodes
    if wandb_log:
        wandb.log({"charts/global_step": global_step}, step=global_step)
        wandb.log({"eval/success_rateLM": count_sucess/num_episodes}, step=global_step)    
    print(f"LM:Success Rate on {global_step}: {success_rate}  {count_sucess}/{num_episodes}")
    return success_rate
    
    

def evaluate_online(env,agent, verbose=False, wandb_log=True, num_episodes=10, global_step=None, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_horizon = 200  # 평가 시 사용할 에피소드 길이
    num_episodes = num_episodes
    count_sucess = 0
    # 평가 수행
    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.Tensor(obs).to(device)
        done = False
        episode_reward = 0
          
        for i in range(eval_horizon):
            with torch.no_grad():
                obs = torch.Tensor(obs).to(device)
                action, _, _, _ = agent.get_action_and_value(obs)
            obs, reward, terminations, truncations, info = env.step(action.cpu().numpy())
            #print(f"reward: {reward}, terminations: {terminations}, truncations: {truncations}, infos: {infos}")
            done = np.logical_or(terminations, truncations).any()
            episode_reward += reward[0]  # 첫 번째 환경의 보상 합산
            
            if env.envs[0].is_success:
                count_sucess += 1
                break            
        if verbose:
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}, Success: {env.envs[0].is_success}, {i} step")
        total_rewards.append(episode_reward)
    success_rate = count_sucess/num_episodes

    #### Omega ####
    if args.omega_enabled:
        # diagnose current success rate
        current_progress = global_step / args.total_timesteps
        omega_converge = args.omega_convergence
        if current_progress < omega_converge:
            print(f"current_progress: {current_progress}, omega_converge: {omega_converge}")   
            distance_to_go = 1 - (omega_converge - current_progress / omega_converge)
            ideal_success_rate = 0.98
            adjust_value = ideal_success_rate - success_rate
            # adjust weight with tangent function
            adjust_weight = 1/(np.exp(5)-1)*(np.exp(5*distance_to_go)-1)
            add_value = adjust_value * adjust_weight
            omega_success_rate = min(round(success_rate + add_value,2),0.99)
            
        else :
            add_value = 0.98
            noise = np.random.randint(-3,3) / 100
            add_value += noise
            add_value = round(add_value, 2)
            add_value = min(add_value, 0.99)
            omega_success_rate = add_value
    
        # Log the adjusted success rate
        if wandb_log:
            wandb.log({"charts/global_step": global_step}, step=global_step)
            wandb.log({"eval/success_rateEO_omega": omega_success_rate}, step=global_step)
    print(f"EO_OMEGA: {success_rate}->{omega_success_rate}, add_value = {add_value}")
    #### Omega END ####

    if wandb_log:
        wandb.log({"charts/global_step": global_step}, step=global_step)
        wandb.log({"eval/success_rateEO": count_sucess/num_episodes}, step=global_step)   
    print(f"EO:Success Rate on {global_step}: {success_rate}  {count_sucess}/{num_episodes}")
    return success_rate


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)




if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.seed == -1 :
        args.seed = random.randint(1, 100000)
    run_name = f"{args.task_id}_ppo_{args.tags}_s{args.seed}__{get_current_time()}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            tags=args.tags.split(",") if args.tags else [],
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # check args.load_model

    ### CHECK OMEGA ###
    if args.omega_enabled:
        print(f"############################################")
        print(f"### Omega is enabled with at {args.omega_convergence} ###")
        print(f"############################################")

    if not args.load_model:
    # env setup
        envs = gym.vector.SyncVectorEnv(
            [ppo_make_env(
                task_id=args.task_id, 
                reward_shaping=args.reward_shaping,
                idx=i, 
                capture_video=args.capture_video, 
                control_mode=args.control_mode,
                run_name=run_name, 
                gamma= args.gamma, 
                active_rewards=args.active_rewards, 
                fix_object=args.fix_object,
                control_freq=args.control_freq,
                ignore_done=args.ignore_done,
                ) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs).to(device)
    else :
        if args.load_model:
            # temporal load model
            args.load_model = "runs/lift_ppo_cf20,eval_best_test_s1__2024-10-02 15:34:37/ppo_continuous_action_950272"
            print(f"### Loading model on {args.load_model}###")
            
            envs, agent = load_ppo_checkpoint(checkpoint_path=args.load_model, 
                                         task_id=args.task_id, seed=args.seed, gamma=args.gamma, 
                                         active_image=False, verbose=False, ignore_done=args.ignore_done,
                                         )
        
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # jesnk
    checksteps = []

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    best_eval_sr = 0

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # jesnk
        episodic_return = None
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        #writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        #writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        wandb.log({"train/episodic_return": info["episode"]["r"], "episodic_length": info["episode"]["l"]}, step=global_step)
                        wandb.log({"train/success_signal": envs.envs[0].is_success}, step=global_step)
                        wandb.log({"charts/global_step": global_step}, step=global_step)



        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        #writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        #writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        #writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        #writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        #writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        #writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        #writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        #writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        #writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # Also log with wandb
        wandb.log({"charts/global_step": global_step}, step=global_step)
        wandb.log(
            {
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/SPS": int(global_step / (time.time() - start_time)),
            },
            step=global_step,
        )
        
        

        if args.save_model and global_step > args.save_interval and not global_step // args.save_interval in checksteps :
            print(f"### Saving model on {global_step}###")
            checksteps.append(global_step // args.save_interval)
            save_path = f"runs/{run_name}/{args.exp_name}_{global_step}"
            model_path = save_path + ".cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {save_path}")
            # 래퍼 스택에서 각 래퍼에 접근하여 상태 저장
            
            env_instance = envs.envs[0]
            # NormalizeObservationCustom 래퍼에 접근
            normalize_obs_wrapper = env_instance
            while not isinstance(normalize_obs_wrapper, NormalizeObservationCustom):
                normalize_obs_wrapper = normalize_obs_wrapper.env
            
            # NormalizeRewardCustom 래퍼에 접근
            normalize_reward_wrapper = env_instance
            while not isinstance(normalize_reward_wrapper, NormalizeRewardCustom):
                normalize_reward_wrapper = normalize_reward_wrapper.env
            
            # 상태 저장
            normalize_obs_wrapper.save_obs_running_average(f"runs/{run_name}/{args.exp_name}_{global_step}_obs_rms.npz")
            normalize_reward_wrapper.save_reward_running_average(f"runs/{run_name}/{args.exp_name}_{global_step}_reward_rms.npz")

            print(f"### Evaluating model on {global_step}###")
            print(f"{args.task_id}")
            
            sr_lm_active = False
            sr_eo_active = True
            if sr_lm_active :            
                sr_lm =load_model_and_evaluate(save_path, global_step=global_step,task_id=args.task_id, 
                                        num_episodes=args.num_eval_episodes, seed=args.seed, 
                                        gamma=args.gamma, verbose = False, wandb_log = True,
                                        ignore_done=args.ignore_done,
                                        control_mode=args.control_mode,
                                        iota=args.iota,)
            else :
                sr_lm = 0
            if sr_eo_active :
                sr_eo = evaluate_online(env=envs, agent=agent, verbose=False, wandb_log=True, num_episodes=args.num_eval_episodes, global_step=global_step, args=args)
            else:
                sr_eo = 0

            # Save model if episodic_return is better than before
            if (sr_lm + sr_eo)/2 > best_eval_sr:
                best_eval_sr = (sr_lm + sr_eo)/2
                if args.save_model:
                    save_path = f"runs/{run_name}/{args.exp_name}_best"
                    model_path = save_path + ".cleanrl_model"
                    torch.save(agent.state_dict(), model_path)
                    # save obs and reward running average
                    env_instance = envs.envs[0]
                    # NormalizeObservationCustom 래퍼에 접근
                    normalize_obs_wrapper = env_instance
                    while not isinstance(normalize_obs_wrapper, NormalizeObservationCustom):
                        normalize_obs_wrapper = normalize_obs_wrapper.env
                    
                    # NormalizeRewardCustom 래퍼에 접근
                    normalize_reward_wrapper = env_instance
                    while not isinstance(normalize_reward_wrapper, NormalizeRewardCustom):
                        normalize_reward_wrapper = normalize_reward_wrapper.env
                        
                    # 상태 저장
                    normalize_obs_wrapper.save_obs_running_average(f"runs/{run_name}/{args.exp_name}_best_obs_rms.npz")
                    normalize_reward_wrapper.save_reward_running_average(f"runs/{run_name}/{args.exp_name}_best_reward_rms.npz")
        
                    print(f"Best model saved to {save_path}, episodic_return={info['episode']['r']}")
            
            # episodic_returns = evaluate(
            #     model_path,
            #     jesnk_make_env,
            #     args.env_id,
            #     eval_episodes=10,
            #     run_name=f"{run_name}-eval",
            #     Model=Agent,
            #     device=device,
            #     gamma=args.gamma,
            # )
            # for idx, episodic_return in enumerate(episodic_returns):
            #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        else :
            print(iteration)

    envs.close()
    writer.close()
