from a2c_ppo_acktr.arguments import get_args
import torch
from dril.a2c_ppo_acktr.algo.behavior_cloning import BehaviorCloning
from dril.a2c_ppo_acktr.algo.ensemble import Ensemble
from dril.a2c_ppo_acktr.algo.dril import DRIL
from dril.a2c_ppo_acktr.arguments import get_args
from dril.a2c_ppo_acktr.envs import make_vec_envs
from dril.a2c_ppo_acktr.model import Policy
import os
import gym


import numpy as np
import torch
import gym

from dril.a2c_ppo_acktr import utils
from dril.a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, num_episodes=None, atari_max_steps=None):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, atari_max_steps)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False)

        # Obser reward and next obs
        if isinstance(eval_envs.action_space, gym.spaces.Box):
            clip_action = torch.clamp(action, float(eval_envs.action_space.low[0]),\
                         float(eval_envs.action_space.high[0]))
        else:
            clip_action = action

        # Obser reward and next obs

        obs, _, done, infos = eval_envs.step(clip_action[0])
        eval_envs.render()
        #eval_envs.render()
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                print(f"Ep {len(eval_episode_rewards):3}:{info['episode']['r']}")

    eval_envs.close()

    print('********** Evaluation Results **********')
    for idx, score in enumerate(eval_episode_rewards):
        print(f'Ep {idx:3} Reward: {score}')

    print(" Eval w/ {} ep: mean rwd {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return eval_episode_rewards


def play_bc_agent():
    print('...Start BC agent...')
    torch.set_num_threads(1)
    args = get_args()
    args.log_dir = f'{os.getcwd()}/tmp_log_dir'
    eval_log_dir = f'{os.getcwd()}/tmp_log_dir'

    device = torch.device("cuda:0" if args.cuda else "cpu")

    bc_model_path = './trained_models/bc/bc_CarRacing-v0_policy_ntrajs=1_seed=3_d=beta.model_worked.pth'

    bc_params = torch.load(bc_model_path, map_location=device)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
                         args.log_dir, device, False, use_obs_norm=args.use_obs_norm,
                         max_steps=args.atari_max_steps)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        load_expert=args.load_expert,
        env_name=args.env_name,
        rl_baseline_zoo_dir=args.rl_baseline_zoo_dir,
        expert_algo=args.expert_algo,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    expert_dataset = []

    print('...Model structure...')
    print(actor_critic)

    actor_critic.load_state_dict(bc_params)
    args.num_processes = 1

    bc_model_reward = evaluate(actor_critic, None, args.env_name, args.seed,
                                args.num_processes, eval_log_dir, device, num_episodes=100,
                                atari_max_steps=args.atari_max_steps)

    save_file = f'{os.getcwd()}/trained_results/bc/bc_100_runs.npy'
    print(f'saving results to: {save_file}')
    np.save(save_file, bc_model_reward)

def play_dril_agent():
    print('...Start Dril agent...')
    torch.set_num_threads(1)
    args = get_args()
    args.log_dir = f'{os.getcwd()}/tmp_log_dir'
    eval_log_dir = f'{os.getcwd()}/tmp_log_dir'

    device = torch.device("cuda:0" if args.cuda else "cpu")

    dril_model_path = './trained_models/dril/dril_CarRacing-v0_policy_ntrajs=1_seed=3_d=beta_scr=491.pt'
    actor_critic, _ = torch.load(dril_model_path, map_location=device)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma,
                         args.log_dir, device, False, use_obs_norm=args.use_obs_norm,
                         max_steps=args.atari_max_steps)


    expert_dataset = []

    print('...Model structure...')
    print(actor_critic)

    args.num_processes = 1

    dril_model_reward = evaluate(actor_critic, None, args.env_name, args.seed+1,
                                args.num_processes, eval_log_dir, device, num_episodes=100,
                                atari_max_steps=args.atari_max_steps)

    save_file = f'{os.getcwd()}/trained_results/bc/dril_100_runs.npy'
    print(f'saving results to: {save_file}')
    np.save(save_file, dril_model_reward)



def check_env_dim():
    env_names = ['MsPacmanNoFrameskip-v4',
                 'SpaceInvadersNoFrameskip-v4',
                 'BreakoutNoFrameskip-v4',
                 'BeamRiderNoFrameskip-v4',
                 'PongNoFrameskip-v4',
                 'QbertNoFrameskip-v4']

    print(f' Environment                  Action Space dim')
    for env_name in env_names:
        env = gym.make(env_name)
        print(f'  {env_name:30}   n={env.action_space.n}')



if __name__ == '__main__':
    #play_bc_agent()
    play_dril_agent()
    #check_env_dim()