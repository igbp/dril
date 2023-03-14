import torch
import matplotlib.pyplot as plt
from dril.a2c_ppo_acktr.model import Policy
import gym
from dril.a2c_ppo_acktr.arguments import get_args
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, WarpFrame, ClipRewardEnv, FrameStack, ScaledFloatFrame
from dril.a2c_ppo_acktr.envs import DimReductorBeta, TransposeImage
def check_beta():
    dir_path = '/home/giovani/article/expert/batch_beta'
    file_name = 'rollout_i=96_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt'

    device = 'cuda:0'
    rollout = torch.load(dir_path + '/' + file_name, map_location=device)
    print(rollout.keys())


    acs_pth = torch.cat(rollout['acs'], dim=0)
    obs_pth = torch.cat(rollout['obs'], dim=0)

    index = 88

    print(acs_pth[index])

    env_name ='CarRacing-v0'
    env = gym.make(env_name)
    env = WarpFrame(env, width=84, height=84)
    env = DimReductorBeta(env)

    #obs = env.reset()

    args = get_args()
    args.distribution ='beta'

    ac_beta = Policy(obs_shape=(4, 84, 84),
                     action_space=env.action_space,
                     base=None,
                     base_kwargs=None,
                     load_expert=None,
                     env_name=env_name,
                     rl_baseline_zoo_dir='/home/giovani/rl_baseline_zoo',
                     expert_algo='ppo',
                     normalize=True)
    ac_beta.to(device)
    print(ac_beta)

    eval_recurrent_hidden_states = torch.zeros(
       1,  #num processes
       ac_beta.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)

    obs_in = obs_pth[index,:,:,:].unsqueeze(dim=0)

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=ac_beta.parameters(), lr=3e-4)

    batch_size = 128
    loss_hist=[]
    n_batches = obs_pth.size()[0] // batch_size
    n_epochs = 20
    for e in range(n_epochs):
        for i in range(n_batches):
            optim.zero_grad()

            obs_in, acs_in = obs_pth[i:i+batch_size,:,:,:], acs_pth[i:i+batch_size,:]
            action = ac_beta.get_action(obs_in, deterministic=True)
            loss = loss_fn(action, acs_in)
            loss_hist.append(loss.item())
            loss.backward()
            optim.step()

    plt.plot(loss_hist)
    plt.show()


def check_gaussian():
    dir_path = '/home/giovani/article/expert/batch_normal'
    file_name = 'batch_normrollout_j=96_CarRacing-v0_seed=2_nsteps_=500_d=normal_nup=1249.pt'

    device = 'cuda:0'
    rollout = torch.load(dir_path + '/' + file_name, map_location=device)
    print(rollout.keys())


    acs_pth = torch.cat(rollout['acs'], dim=0)
    obs_pth = torch.cat(rollout['obs'], dim=0)

    index = 88

    print(acs_pth[index])

    env_name ='CarRacing-v0'
    env = gym.make(env_name)
    env = WarpFrame(env, width=84, height=84)
    env = DimReductorBeta(env)

    obs = env.reset()

    args = get_args()
    args.distribution ='gaussian'

    ac_beta = Policy(obs_shape=(4, 84, 84),
                     action_space=env.action_space,
                     base=None,
                     base_kwargs=None,
                     load_expert=None,
                     env_name=env_name,
                     rl_baseline_zoo_dir='/home/giovani/rl_baseline_zoo',
                     expert_algo='ppo',
                     normalize=True)
    ac_beta.to(device)
    print(ac_beta)

    eval_recurrent_hidden_states = torch.zeros(
       1,  #num processes
       ac_beta.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)

    obs_in = obs_pth[index,:,:,:].unsqueeze(dim=0)

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=ac_beta.parameters(), lr=0.01)

    for i in range(1000):
        optim.zero_grad()
        #action = ac_beta.get_action(obs_in, deterministic=True)
        _, action, _, eval_recurrent_hidden_states = ac_beta.act(obs_in,
                                                                 eval_recurrent_hidden_states,
                                                                 eval_masks,
                                                                 deterministic=True)

        print(action, acs_pth[index,:])

        loss = loss_fn(action, acs_pth[index,:].unsqueeze(dim=0))
        loss.backward()
        optim.step()






if __name__ == '__main__':
    #check_gaussian()
    check_beta()
    """
    index = 90

    action discovery
    fig, ax = plt.subplots(5, 4, figsize=(10,20))
    for i in range(5):
        obs = obs_pth[index + i*4, :, :, :].cpu().numpy()
        for j in range(obs.shape[0]):
            ax[i, j].imshow(obs[j, :, :])

    plt.show()
    
    
    #_, action, _, eval_recurrent_hidden_states = ac_beta.act(obs_in,
        #                                                         eval_recurrent_hidden_states,
        #                                                         eval_masks,
        #                                                         deterministic=True)

    
    
    """
