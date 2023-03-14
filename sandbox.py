import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
from a2c_ppo_acktr.envs import DimReductorBeta, DimReductorGaussian, DiscreteActions
from a2c_ppo_acktr.envs import WarpFrame
from dril.a2c_ppo_acktr.envs import make_vec_envs
from dril.a2c_ppo_acktr import utils
from dril.a2c_ppo_acktr.algo.behavior_cloning import BehaviorCloning
from dril.a2c_ppo_acktr.expert_dataset import ExpertDataset
from evaluation import evaluate
from evaluation import DETERMINISTIC

def convert_discrete_to_continuous():
    actions =  [np.array([0.0, 0.0, 0.0]),  # Noop
                np.array([0.0, 0.2, 0.0]),  # Throttle
                np.array([-0.5, 0.0, 0.0]), # Left
                np.array([0.5, 0.0, 0.0]),  # Right
                np.array([0.0, 0.0, 0.2])]   # Brake

    dir_path_load = '/home/giovani/faire/dril/dril/demo_data/discrete'
    file_name = 'acs_CarRacing-v0_seed=1_ntraj=1.npy'

    acs = np.load(dir_path_load +'/'+file_name)

    acs_cont = np.zeros((acs.shape[0], 3))
    acs_merge = np.zeros((acs.shape[0], 2))


    distribution = 'gaussian'
    if distribution == 'gaussian':
        for i in range(acs.shape[0]):
            acs_cont[i,:] = actions[acs[i][0]]

            acs_merge[i, 0] = acs_cont[i,0]
            if acs_cont[i,1]>0:
                acs_merge[i, 1] = acs_cont[i,1]
            else:
                acs_merge[i, 1] = -acs_cont[i,2]

    else:
        for i in range(acs.shape[0]):
            acs_cont[i, :] = actions[acs[i][0]]

            acs_merge[i, 0] = (acs_cont[i, 0]+1.0)/2.0
            if acs_cont[i, 1] > 0:
                acs_merge[i, 1] = (acs_cont[i, 1]+1.0)/2.0
            else:
                acs_merge[i, 1] = (-acs_cont[i, 2]+1.0)/2.0

    print(acs[0:5])
    print(acs_cont[0:10,:])
    print(acs_merge[0:5, :])


    fig, ax = plt.subplots(3,1, figsize=(10,10))
    fig.suptitle(f'Distribution={distribution}')
    ax[0].plot(acs)
    ax[0].set_ylabel('Discrete actions')

    ax[1].plot(acs_merge[:, 0])
    ax[1].set_ylabel('Left/Rigth')

    ax[2].plot(acs_merge[:,1])
    ax[2].set_ylabel('throttle/break')

    plt.show()

    dir_path_save = '/home/giovani/faire/dril/dril/demo_data/' + distribution
    np.save(dir_path_save+'/'+file_name, acs_merge)

    """
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.plot(acs)
    plt.show()
    """


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = gym.spaces.Discrete(len(disc_to_cont))

    def action(self, act):
        #print(act[0])
        return self.disc_to_cont[act[0]]

class DimReductor(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.,1.]))

    def action(self, action_in):                            #enters 2D
        action_out = np.zeros(3)
        action_out[0] = action_in[0]
        action_out[1] = np.maximum(action_in[1],0)
        action_out[2] = np.maximum(-action_in[1],0)

        return action_out


def reduce_dim():
    action = np.array([ 0.0, 0.5])

    env = gym.make('CarRacing-v0')
    env = DimReductor(env)

    s = env.reset()
    for i in range(1000):
        if i <200:
            action = np.array([0.0, 0.5])
        else:
            action = np.array([0.0, -0.5])
        s, r, done, info = env.step(action)
        env.render()


def noisy_agents_gaussian():
    dir_path = '/home/giovani/article/expert/batch_normal'
    file_name = 'batch_normrollout_j=96_CarRacing-v0_seed=2_nsteps_=500_d=normal_nup=1249.pt'

    rollout = torch.load(dir_path + '/' + file_name,  map_location='cpu')
    print(rollout.keys())

    acs_pth = torch.cat(rollout['acs'], dim=0)
    obs_pth = torch.cat(rollout['obs'], dim=0)

    acs = acs_pth.numpy()
    obs = obs_pth.numpy()

    acs = np.clip(acs, -1., 1.)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(acs[:, 0])
    ax[0].set_ylabel('Left | Right')
    ax[1].plot(acs[:, 1])
    ax[1].set_ylabel('Brake   | Throttle ')
    ax[1].set_xlabel('Steps')
    plt.show()


    acs_save_path = '/home/giovani/faire/dril/dril/demo_data/gaussian'
    obs_save_path = '/home/giovani/faire/dril/dril/demo_data'

    acs_file_name = 'acs_CarRacing-v0_seed=4_ntraj=1.npy'
    obs_file_name = 'obs_CarRacing-v0_seed=4_ntraj=1.npy'

    print(f'Saving expert demos at {acs_save_path}')
    np.save(acs_save_path +'/' +acs_file_name, acs)
    np.save(obs_save_path + '/'+obs_file_name, obs)


def noisy_agents_beta():
    dir_path = '/home/giovani/article/expert/batch_beta'

    """
    file_names = ['rollout_i=96_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=94_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=93_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=90_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=89_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=88_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=83_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=80_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=79_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=70_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=69_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=66_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=65_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=61_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=60_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=58_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=57_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=55_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=54_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=53_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',]
    

    file_names = [  'rollout_i=2_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=6_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=7_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=13_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=16_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=18_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=20_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=22_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=23_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=24_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=25_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=26_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=31_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=32_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=33_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=36_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=37_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=38_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=40_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                    'rollout_i=44_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt']
    """

    file_names = ['rollout_i=96_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=94_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=93_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=90_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=89_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=88_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=83_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=80_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=79_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=70_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=25_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=26_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=31_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=32_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=33_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=36_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=37_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=38_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=40_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt',
                  'rollout_i=44_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt']

    obs_l = []
    acs_l = []
    for file_name in file_names:
        rollout = torch.load(dir_path + '/' + file_name,  map_location='cpu')
        obs_l.extend(rollout['obs'])
        acs_l.extend(rollout['acs'])

    acs_pth = torch.cat(acs_l, dim=0)
    obs_pth = torch.cat(obs_l, dim=0)

    acs = acs_pth.numpy()
    obs = obs_pth.numpy()

    print(acs.shape)
    print(obs.shape)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(acs[:, 0])
    ax[0].set_ylabel('Left | Right')
    ax[1].plot(acs[:, 1])
    ax[1].set_ylabel('Brake   | Throttle ')
    ax[1].set_xlabel('Steps')
    plt.show()

    acs_save_path = '/home/giovani/faire/dril/dril/demo_data/beta'
    obs_save_path = '/home/giovani/faire/dril/dril/demo_data'

    acs_file_name = 'acs_CarRacing-v0_seed=7_ntraj='+str(len(file_names))+'.npy'
    obs_file_name = 'obs_CarRacing-v0_seed=7_ntraj='+str(len(file_names))+'.npy'

    print(f'Saving expert demos at {acs_save_path}')
    np.save(acs_save_path +'/' +acs_file_name, acs)
    np.save(obs_save_path + '/'+obs_file_name, obs)


def beta_test():
    env = gym.make('CarRacing-v0')
    env = WarpFrame(env, width=84, height=84)
    env = DimReductorBeta(env)

    acs_save_path = '/home/giovani/faire/dril/dril/demo_data/beta'
    acs_file_name = 'acs_CarRacing-v0_seed=3_ntraj=1.npy'
    acs = np.load(acs_save_path +'/'+acs_file_name)

    obs = env.reset()
    for i in range(acs.shape[0]):
        print(acs[i])
        s, r, done, info = env.step(acs[i])
        env.render()


def gaussian_test():
    env = gym.make('CarRacing-v0')
    env = WarpFrame(env, width=84, height=84)
    env = DimReductorGaussian(env)

    acs_save_path = '/home/giovani/faire/dril/dril/demo_data/gaussian'
    acs_file_name = 'acs_CarRacing-v0_seed=2_ntraj=1.npy'
    acs = np.load(acs_save_path +'/'+acs_file_name)

    obs = env.reset()
    for i in range(acs.shape[0]):
        print(acs[i])
        s, r, done, info = env.step(acs[i])
        env.render()


def discrete_test():
    env = gym.make('CarRacing-v0')
    env = WarpFrame(env, width=84, height=84)
    env = DiscreteActions(env, [np.array([0.0, 0.0, 0.0]),  # Noop
                                np.array([0.0, 0.2, 0.0]),  # Throttle
                                np.array([-0.5, 0.0, 0.0]),  # Left
                                np.array([0.5, 0.0, 0.0]),  # Right
                                np.array([0.0, 0.0, 0.2])  # Brake
                                ])


    acs_save_path = '/home/giovani/faire/dril/dril/demo_data/discrete'
    acs_file_name = 'acs_CarRacing-v0_seed=1_ntraj=1.npy'
    acs = np.load(acs_save_path +'/'+acs_file_name)

    obs = env.reset()
    for i in range(acs.shape[0]):
        print(acs[i])
        s, r, done, info = env.step(acs[i])
        env.render()


def walker_test():
    env = gym.make('Walker2DbulletEnv-v0')
    env = WarpFrame(env, width=84, height=84)
    env = DiscreteActions(env, [np.array([0.0, 0.0, 0.0]),  # Noop
                                np.array([0.0, 0.2, 0.0]),  # Throttle
                                np.array([-0.5, 0.0, 0.0]),  # Left
                                np.array([0.5, 0.0, 0.0]),  # Right
                                np.array([0.0, 0.0, 0.2])  # Brake
                                ])

    acs_save_path = '/home/giovani/faire/dril/dril/demo_data/discrete'
    acs_file_name = 'acs_CarRacing-v0_seed=1_ntraj=1.npy'
    acs = np.load(acs_save_path + '/' + acs_file_name)

    obs = env.reset()
    for i in range(acs.shape[0]):
        print(acs[i])
        s, r, done, info = env.step(acs[i])
        env.render()


def plot_ensemble_training():
    dir_path = '~/faire/dril/dril/trained_results/ensemble/'
    file_name = 'ensemble_CarRacing-v0_policy_ntrajs=1_seed=3_d=beta.perf'

    df = pd.read_csv(dir_path+file_name)
    print(df.head())
    print(df.tail())
    plt.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(df['epoch'], df['trloss'], label='Trainig loss')
    ax.plot(df['epoch'], df['teloss'], label='Test loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    fig.suptitle(f'Ensemble training\n {file_name}', fontsize=16)
    ax.legend()
    plt.show()


def plot_dril_training():
    dir_path = '~/faire/dril/dril/trained_results/dril/'
    file_name = 'dril_CarRacing-v0_ntraj=20_ensemble_lr=0.00025_lr=0.00025_bcep=2001_shuffle=sample_w_replace_quantile=0.98_cost_-1_to_1_seed=3_d=beta.perf'

    df = pd.read_csv(dir_path+file_name)
    print(df.head())
    print(df.tail())

    expert_score = 900
    random_score = -100
    expert = np.ones_like(df['total_num_steps'].to_numpy())*expert_score
    rand_agent = np.ones_like(df['total_num_steps'].to_numpy())*random_score
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(1, 1, figsize=(16, 25))

    ax.plot(df['total_num_steps'], df['train_reward'], label='Trainig reward')
    ax.plot(df['total_num_steps'], df['test_reward'], label='Test reward')
    ax.plot(df['total_num_steps'], expert, label='Expert',  linestyle='-.')
    ax.plot(df['total_num_steps'], rand_agent, label='Random' , linestyle=':')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Reward')
    ax.set_ylim([-200,1000])
    fig.suptitle(f"Dril training \n {file_name}", fontsize=16)
    ax.legend(loc='lower right')
    plt.show()


def plot_training_scores():
    dir_path = '/home/giovani/faire/dril/dril/trained_models/dril/'
    file_name = 'dril_CarRacing-v0_policy_ntrajs=20_seed=7_d=gaussian_vc=0.5_elc=0.0_clip=0.2_training_2nd.pt'

    _ , _, _,training_scr, u_reward = torch.load(dir_path + file_name)
    indice_x = np.linspace(1, len(training_scr), len(training_scr) )
    expert_score = 900
    random_score = -100
    expert = np.ones_like(indice_x) * expert_score
    rand_agent = np.ones_like(indice_x) * random_score

    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(1, 1, figsize=(16, 25))
    ax.plot(indice_x, training_scr, label='Training reward')
    ax.plot(indice_x, moving_avg(training_scr, 100), label='Moving average (100)')
    ax.plot(indice_x, expert, label='Expert',  linestyle='-.')
    ax.plot(indice_x, rand_agent, label='Random' , linestyle=':')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Episode Score')
    ax.set_ylabel('Reward')
    ax.set_ylim([-200,1000])
    ax.legend(loc='upper right')
    fig.suptitle(f"Dril training \n {file_name}", fontsize=16)
    plt.show()

def moving_avg(a, n):
    mv = np.zeros_like(a)

    for i in range(0, len(a)-(n-1)):
        mv[i+n-1]=np.average(a[i:i+n])

    return mv


def plot_distr():
    dir_path = '/home/giovani/faire/dril/dril/trained_results/bc/'
    file_bc = 'bc_100_runs.npy'
    file_dril = 'dril_100_runs.npy'

    bc_scr = np.load(dir_path +'/'+file_bc)
    dril_scr = np.load(dir_path+ '/' +file_dril)

    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(1, 1, figsize=(16, 25))
    bins = np.linspace(-200.,1200., int(1201/5), endpoint=True)
    ax.hist(bc_scr, bins=bins)
    ax.hist(dril_scr, bins=bins)
    plt.show()


def beta_to_gaussian(ntraj):

    dir_load = '/home/giovani/faire/dril/dril/demo_data/beta'
    file_beta = 'acs_CarRacing-v0_seed=3_ntraj='+str(ntraj)+'.npy'
    acs_beta = np.load(dir_load + '/' + file_beta)

    acs_gaussian = acs_beta * 2 -1

    fig, ax = plt.subplots(2, 2, figsize=(16, 25))
    ax[0,0].plot(acs_beta[:,0])
    ax[0,1].plot(acs_beta[:,1])
    ax[1,0].plot(acs_gaussian[:,0])
    ax[1,1].plot(acs_gaussian[:,1])

    #plt.show()

    dir_save = '/home/giovani/faire/dril/dril/demo_data/gaussian'
    file_beta = 'acs_CarRacing-v0_seed=3_ntraj='+str(ntraj)+'.npy'

    np.save(dir_save + '/' + file_beta, acs_gaussian)


def gaussian_to_beta(ntraj):

    dir_load = '/home/giovani/faire/dril/dril/demo_data/gaussian'
    file_gaussian = 'acs_LunarLanderContinuous-v2_seed=0_ntraj='+str(ntraj)+'.npy'
    acs_gaussian = np.load(dir_load + '/' + file_gaussian)

    acs_gaussian_clip = np.clip(acs_gaussian, -1, 1)

    acs_beta = acs_gaussian_clip / 2. + 0.5

    fig, ax = plt.subplots(2, 2, figsize=(16, 25))
    ax[0,0].plot(acs_beta[:,0])
    ax[0,1].plot(acs_beta[:,1])
    ax[1,0].plot(acs_gaussian[:,0])
    ax[1,1].plot(acs_gaussian[:,1])
    ax[1, 0].plot(acs_gaussian_clip[:, 0])
    ax[1, 1].plot(acs_gaussian_clip[:, 1])

    plt.show()

    dir_save = '/home/giovani/faire/dril/dril/demo_data/beta'
    file_beta = 'acs_LunarLanderContinuous-v2_seed=0_ntraj='+str(ntraj)+'.npy'

    np.save(dir_save + '/' + file_beta, acs_beta)


def bc_test(ntraj):
    dril_dir_path = '/home/giovani/faire/dril/dril/trained_models/dril/'
    #dril_file_name = 'dril_CarRacing-v0_policy_ntrajs=10_seed=3_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=776.pt'
    dril_file_name = 'dril_CarRacing-v0_policy_ntrajs='+str(ntraj)+'_seed=3_d=gaussian_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=830.pt'
    actor_critic, obs_rms, args, training_scr, u_train = torch.load(dril_dir_path + dril_file_name)

    bc_dir_path = '/home/giovani/faire/dril/dril/trained_models/bc/'
    #bc_file_name = 'bc_CarRacing-v0_policy_ntrajs='+str(ntraj)+'_seed=3_d=beta.model.pth'
    bc_file_name = 'bc_CarRacing-v0_policy_ntrajs='+str(ntraj)+'_seed=3_d=gaussian.model.pth'
    bc_params = torch.load(bc_dir_path + bc_file_name)

    actor_critic.load_state_dict(bc_params)

    eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scr = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                 device, num_episodes=100, atari_max_steps=None, fname=bc_dir_path + bc_file_name)

    return bc_file_name, scr

def bc_test_LL(ntraj):

    dril_dir_path = '/home/giovani/faire/dril/dril/trained_models/dril/'
    dril_file_name = 'dril_LunarLanderContinuous-v2_policy_ntrajs='+str(ntraj)+'_seed=0_d=beta_vc=0.5_elc=0.0_clip=0.2_steps=4096_lr=0.0003_decay=True_training.pt'


    actor_critic, obs_rms, args, training_scr, u_train = torch.load(dril_dir_path + dril_file_name)

    bc_dir_path = '/home/giovani/faire/dril/dril/trained_models/bc/'
    bc_file_name = 'bc_LunarLanderContinuous-v2_policy_ntrajs='+str(ntraj)+'_seed=0_d=beta.model.pth'
    bc_params = torch.load(bc_dir_path + bc_file_name)

    actor_critic.load_state_dict(bc_params)

    eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                 device, num_episodes=100, atari_max_steps=None, fname=bc_dir_path + bc_file_name)

    return bc_file_name, score


def dril_test(fname):
    dril_dir_path = '/home/giovani/faire/dril/dril/trained_models/dril/'
    dril_file_name =  fname#'dril_CarRacing-v0_policy_ntrajs=20_seed=3_d=gaussian_vc=0.5_elc=0.0_clip=0.2_scr=674.pt'
    actor_critic, obs_rms, args, training_scr, u_train = torch.load(dril_dir_path + dril_file_name)

    det = True
    dril_file_name += str(det)

    eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scr = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                 device, num_episodes=100, atari_max_steps=None, fname=dril_dir_path + dril_file_name)

    return dril_file_name, scr

def dril_test_LL(ntraj):
    dril_dir_path = '/home/giovani/faire/dril/dril/trained_models/dril/'
    dril_file_name = 'dril_LunarLanderContinuous-v2_policy_ntrajs='+str(ntraj)+'_seed=0_d=beta_vc=0.5_elc=0.0_clip=0.2_steps=4096_lr=0.0003_decay=True_training.pt'
    actor_critic, obs_rms, args, training_scr, u_train = torch.load(dril_dir_path + dril_file_name)

    eval_log_dir = '/home/giovani/faire/dril/dril/tmp_log_dir'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score = evaluate(actor_critic, obs_rms, args.env_name, args.seed, args.num_processes, eval_log_dir,
                 device, num_episodes=100, atari_max_steps=None, fname=dril_dir_path + dril_file_name)

    return dril_file_name, score


def plot_test(fname):

    if 'bc' in fname:
        dir_path = '/home/giovani/faire/dril/dril/trained_models/bc/'
    if 'dril' in fname:
        dir_path = '/home/giovani/faire/dril/dril/trained_models/dril/'

    fname += '_tscr_det='+str(DETERMINISTIC)+'.npy'
    training_scr = np.load(dir_path + fname)


    training_scr = np.load(dir_path + fname)

    idx = np.arange(1, len(training_scr)+1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.scatter(idx, training_scr,marker='.', label='Ep score')
    ax.scatter(idx, np.ones_like(training_scr)*np.mean(training_scr), marker='.', label=f'Ep scr avg={np.mean(training_scr):.0f}')

    if 'LunarLander' in fname:
        ax.plot(idx, np.ones_like(training_scr)*-200, label='Random')
        ax.plot(idx, np.ones_like(training_scr)*200, label='Expert')
        ax.set_ylim(-300, 300)
    if 'CarRacing' in fname:
        ax.plot(idx, np.ones_like(training_scr)*-200, label='Random')
        ax.plot(idx, np.ones_like(training_scr)*900, label='Expert')
        ax.set_ylim(-200, 1000)

    ax.legend(loc='lower right')
    fig.suptitle(f" {fname}", fontsize=16)
    plt.show()


def plot_train_u():
    steam ='dril_CarRacing-v0_policy_ntrajs=20_seed=6_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr'
    dir_path = '/home/giovani/faire/dril/dril/trained_models/dril/'

    full_file = 'dril_CarRacing-v0_policy_ntrajs=20_seed=6_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_training.pt'
    _, _, _, train_hist, u_train = torch.load(dir_path + full_file)
    fig, ax = plt.subplots(2, 1, figsize=(15, 25))
    ax[0].plot(train_hist)
    ax[0].plot(moving_avg(train_hist, 100))
    ax[1].plot(u_train)
    plt.suptitle(full_file)
    if 'LunarLander' in full_file:
        ax[0].set_ylim([-300, 300])

    else:
        ax[0].set_ylim([-200, 1000])


    ax[0].set_ylabel('Training scr')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('U reward')

    plt.plot()
    plt.show()
    list_files = os.listdir(dir_path)
    list_files = [f for f in list_files if f.startswith(steam)]
    list_files.sort()
    """
    n_train = len(list_files)
    fig, ax = plt.subplots(2, n_train, figsize=(15,25))
    for i in range(n_train):
        print(list_files[i])
        _, _, _, train_hist, u_train = torch.load(dir_path + list_files[i])

        ax[0, i].plot(train_hist)
        ax[1, i].plot(u_train)
        ax[0, i].set_title(list_files[i][-10:-3])
        ax[0, i].set_ylim([-200, 1000])
        ax[1, i].set_xlabel('Episodes')
        if i == 0:
            ax[0, i].set_ylabel('Training scr')
            ax[1, i].set_ylabel('U reward')

        if i>0:
            ax[0, i].set_yticks([])
            ax[1, i].set_yticks([])

    plt.suptitle(steam)
    plt.show()
    """


def summary_LL():
    expert = [  216.418625,
                206.819988,
                280.785136,
                222.919784,
                209.798438,
                237.376759,
                256.61724,
                230.08931,
                242.373971,
                255.393661,
                273.006258,
                282.880848,
                255.851267,
                256.30024,
                228.279517,
                234.967657,
                211.640695,
                282.082786,
                167.527522,
                197.480239]

    ax_x = [1, 3, 5, 10, 15, 20] # number of trajectories
    expert_mean = np.mean(expert)*np.ones_like(ax_x)
    random_mean = -200*np.ones_like(ax_x)
    BC_g_det = [-213.49591, 211.41487, 237.04670, 229.18030, 212.75380, 221.60977]
    BC_g_sch = [-175.05276, 126.67633, 130.72015, 125.35175, 130.06505, 123.94570]

    DRIL_g_det = [217.71762471, 213.41520161999998, 226.36120997999998, 222.40767825999995, 223.12675703999994, 228.31793583]
    DRIL_g_sch = [174.14399205, 124.31154751, 140.05681468, 125.42183757999999, 139.00551937999998, 140.28151979]

    BC_b_det = [-90.25451836000002, 217.27188798000003, 205.59956902000002, 231.39190307000004, 222.54377465, 222.28717]
    BC_b_sch = [-76.41501223, 170.32496509, 198.99132521000004, 217.92903636999995, 228.43523394999997, 220.06181]

    DRIL_b_det = [205.06090098, 206.34267318, 213.90077525999996, 221.11192805000002, 226.84074624000002, 216.77280]
    DRIL_b_sch = [164.4137306, 171.04492505000002, 140.87246112, 151.29582184, 237.61063484, 210.71673518000003]


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(ax_x, expert_mean, linestyle='--', color='gray', label='Expert')
    ax[0].scatter(ax_x, BC_g_det, marker='o', color='green', label='BC_g_det')
    ax[0].scatter(ax_x, BC_g_sch, marker='.', color='green', label='BC_g_sch')
    ax[0].scatter(ax_x, DRIL_g_det, marker='o', color='purple', label='DRIL_g_det')
    ax[0].scatter(ax_x, DRIL_g_sch, marker='.', color='purple', label='DRIL_g_sch')
    ax[0].plot(ax_x, random_mean, linestyle='--', color='blue', label='Random')
    ax[0].set_xlabel('Number of Trajectories')
    ax[0].set_ylabel('Average Score')
    ax[0].set_xticks(ax_x)
    ax[0].set_ylim([-350, 300])
    ax[0].legend(loc='lower right')
    ax[0].set_title('Gaussian')

    ax[1].plot(ax_x, expert_mean, linestyle='--', color='gray', label='Expert')
    ax[1].scatter(ax_x, BC_b_det, marker='o', color='green', label='BC_b_det')
    ax[1].scatter(ax_x, BC_b_sch, marker='.', color='green', label='BC_b_sch')
    ax[1].scatter(ax_x, DRIL_b_det, marker='o', color='purple', label='DRIL_b_det')
    ax[1].scatter(ax_x, DRIL_b_sch, marker='.', color='purple', label='DRIL_b_sch')
    ax[1].plot(ax_x, random_mean, linestyle='--', color='blue', label='Random')
    ax[1].set_xlabel('Number of Trajectories')
    ax[1].set_ylabel('Average Score')
    ax[1].set_xticks(ax_x)
    ax[1].set_ylim([-350, 300])
    ax[1].legend(loc='lower right')
    ax[1].set_title('Beta')


    plt.show()

def plot_pair():
    steam ='dril_CarRacing-v0_policy_ntrajs=20_seed=6_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr'
    dir_path = '/home/giovani/faire/dril/dril/trained_models/dril/'

    """
    full_file = ['dril_CarRacing-v0_policy_ntrajs=10_seed=3_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_training.pt',
                 'dril_CarRacing-v0_policy_ntrajs=10_seed=3_d=gaussian_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_training.pt']
    
    full_file = ['dril_CarRacing-v0_policy_ntrajs=20_seed=7_d=beta_vc=0.5_elc=0.0_clip=0.2_steps=4096_lr=0.0003_decay=True_training.pt',
                 'dril_CarRacing-v0_policy_ntrajs=20_seed=7_d=gaussian_vc=0.5_elc=0.0_clip=0.2_steps=4096_lr=0.0003_decay=True_training.pt']
    """

    full_file= ['dril_CarRacing-v0_policy_ntrajs=20_seed=6_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_training.pt',
                'dril_CarRacing-v0_policy_ntrajs=20_seed=6_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_training.pt']

    titles = ['Beta', 'Gaussian']
    fig, ax = plt.subplots(2, 2, figsize=(15, 25))
    for i in range(2):
        _, _, _, train_hist, u_train = torch.load(dir_path + full_file[i])
        ax[0, i].plot(train_hist)
        ax[0, i].plot(moving_avg(train_hist, 100))
        ax[1, i].plot(u_train)
        s=40
        ax[0, i].set_title(titles[i])
        if 'LunarLander' in full_file[i]:
            ax[0, i].set_ylim([-300, 300])

        else:
            ax[0, i].set_ylim([-200, 1000])

        ax[0, i].set_ylabel('Training scr')
        ax[1, i].set_xlabel('Episodes')
        ax[1, i].set_ylabel('U reward')

    plt.plot()
    plt.suptitle(full_file[0])
    plt.show()
    list_files = os.listdir(dir_path)
    list_files = [f for f in list_files if f.startswith(steam)]
    list_files.sort()
    """
    n_train = len(list_files)
    fig, ax = plt.subplots(2, n_train, figsize=(15,25))
    for i in range(n_train):
        print(list_files[i])
        _, _, _, train_hist, u_train = torch.load(dir_path + list_files[i])

        ax[0, i].plot(train_hist)
        ax[1, i].plot(u_train)
        ax[0, i].set_title(list_files[i][-10:-3])
        ax[0, i].set_ylim([-200, 1000])
        ax[1, i].set_xlabel('Episodes')
        if i == 0:
            ax[0, i].set_ylabel('Training scr')
            ax[1, i].set_ylabel('U reward')

        if i>0:
            ax[0, i].set_yticks([])
            ax[1, i].set_yticks([])

    plt.suptitle(steam)
    plt.show()
    """


def summary_CR():
    BC_b_det = [175.57494254999997, 175.8081449, 218.201234, 291.54839986, 572.31164002, 385.24122804]
    BC_b_sch = [294.39252722000003, 438.53081826, 667.9265055600001, 664.87509308, 785.3186706800001, 804.7056020100001]

    DRIL_b_det = [193.19450, 215.74710347000004, 75.90419999000001, 343.8138482500001, 587.43917062, 797.0536372700001]
    DRIL_b_sch = [314.66673878, 560.9218674099999, 441.10091565000005, 661.77215081, 805.3737522300002, 722.66785662]

    BC_g_det = []
    BC_g_sch = []

    DRIL_g_det = [127.09380208000002, 68.00206831999999, 199.65554018811883, 596.75815831, 488.03601509999993, 497.45734192000003]
    DRIL_g_sch = []

    ax_x = [1, 3, 5, 10, 15, 20]
    expert_mean = 900 * np.ones_like(ax_x)
    random_mean = -200 * np.ones_like(ax_x)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[1].plot(ax_x, expert_mean, linestyle='--', color='gray', label='Expert')
    ax[1].scatter(ax_x, BC_b_det, marker='o', color='green', label='BC_b_det')
    ax[1].scatter(ax_x, BC_b_sch, marker='.', color='green', label='BC_b_sch')
    ax[1].scatter(ax_x, DRIL_b_det, marker='o', color='purple', label='DRIL_b_det')
    ax[1].scatter(ax_x, DRIL_b_sch, marker='.', color='purple', label='DRIL_b_sch')
    ax[1].plot(ax_x, random_mean, linestyle='--', color='blue', label='Random')
    ax[1].set_xlabel('Number of Trajectories')
    ax[1].set_ylabel('Average Score')
    ax[1].set_xticks(ax_x)
    ax[1].set_ylim([-300, 1000])
    ax[1].legend(loc='lower right')
    ax[1].set_title('Beta')
    plt.show()


def plot_training_trajs():
    dril_dir_path = '/home/giovani/faire/dril/dril/trained_models/dril/'

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax_x = [1, 3, 5, 10, 15, 20]
    expert_mean = 900 * np.ones_like(ax_x)


    c = ['green', 'blue', 'red', 'purple', 'orange', 'black']
    for j, d in enumerate(['gaussian', 'beta']):
        for i, ntraj in enumerate(ax_x):
            fname = 'dril_CarRacing-v0_policy_ntrajs='+str(ntraj)\
                    +'_seed=3_d='+str(d)+'_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_training.pt'

            dril_file_name = fname
            actor_critic, obs_rms, args, training_scr, u_train = torch.load(dril_dir_path + dril_file_name)
            ax[j].plot(moving_avg(training_scr, 100), color=c[i],  label='ntraj='+str(ntraj))

        random_mean = -200 * np.ones_like(ax_x)

        ax[j].set_xlabel('Steps')
        ax[j].set_ylim([-300, 1000])
        ax[j].set_title(d)

    ax[0].legend(loc='lower right', mode='expand', ncol=3)
    ax[0].set_ylabel('Average Score')
    plt.show()

def print_6dril_trajs():
    fnames =['dril_CarRacing-v0_policy_ntrajs=1_seed=3_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=436.pt',
             'dril_CarRacing-v0_policy_ntrajs=3_seed=3_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=568.pt',
             'dril_CarRacing-v0_policy_ntrajs=5_seed=3_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=365.pt',
             'dril_CarRacing-v0_policy_ntrajs=10_seed=3_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=788.pt',
             'dril_CarRacing-v0_policy_ntrajs=15_seed=3_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=888.pt',
             'dril_CarRacing-v0_policy_ntrajs=20_seed=3_d=beta_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=822.pt']

    fnames =[
        'dril_CarRacing-v0_policy_ntrajs=1_seed=3_d=gaussian_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=109.pt',
        'dril_CarRacing-v0_policy_ntrajs=3_seed=3_d=gaussian_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=38.pt',
        'dril_CarRacing-v0_policy_ntrajs=5_seed=3_d=gaussian_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=188.pt',
        'dril_CarRacing-v0_policy_ntrajs=10_seed=3_d=gaussian_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=665.pt',
        'dril_CarRacing-v0_policy_ntrajs=15_seed=3_d=gaussian_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=783.pt',
        'dril_CarRacing-v0_policy_ntrajs=20_seed=3_d=gaussian_vc=0.5_elc=0.0_clip=0.1_steps=8192_lr=0.0002_decay=True_scr=830.pt']

    scores = []
    for fname in fnames:
        #fname, score = bc_test(ntraj)
        _, score = dril_test(fname)
        scores.append(score)
    print(scores)


def print_6bc_trajs():
    ntrajs = [1, 3, 5, 10, 15, 20]
    scores = []
    for n in ntrajs:
        _, score = bc_test(n)
        scores.append(score)
    print(scores)


def check_acs():
    dir_path = '/home/giovani/faire/dril/dril/demo_data/gaussian/'
    fname = 'acs_CarRacing-v0_seed=3_ntraj=3.npy'
    acs = np.load(dir_path + fname)
    fig, ax = plt.subplots(2,1, figsize=(10, 10))
    ax[0].plot(acs[:, 0])
    ax[1].plot(acs[:, 1])
    plt.show()

if __name__ =='__main__':
    #noisy_agents_beta()
    #beta_to_gaussian()
    #plot_dril_training()
    #plot_training_scores()

    #summary_LL()
    #summary_CR()
    #plot_training_trajs()

    #print_6dril_trajs()

    #plot_train_u()
    #plot_pair()