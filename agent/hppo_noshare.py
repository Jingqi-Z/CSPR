import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
from torch.utils.data import BatchSampler, SubsetRandomSampler

from .utils import discount_cumsum


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)


class Normalization:
    def __init__(self, shape, device='cpu'):
        self.device = torch.device(device)
        self.n = 0
        self.mean = torch.zeros(shape, device=self.device)
        self.S = torch.zeros(shape, device=self.device)
        self.std = torch.sqrt(self.S)

    def __call__(self, x, update=True):
        # Move input tensor to the same device if needed
        if x.device != self.device:
            x = x.to(self.device)

        # Whether to update the mean and std, during the evaluating, update=False
        if update:
            self.update(x)
        x = (x - self.mean) / (self.std + 1e-6)
        return x

    def update(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)

    def save(self, file_path):
        torch.save({
            'mean': self.mean,
            'std': self.std,
            'n': self.n,
            'S': self.S,
            'device': self.device
        }, file_path)

    def load(self, file_path):
        data = torch.load(file_path)
        self.device = data['device']
        self.mean = data['mean'].to(self.device)
        self.std = data['std'].to(self.device)
        self.n = data['n']
        self.S = data['S'].to(self.device)

    def to(self, device):
        """Move the normalization module to a specified device"""
        self.device = torch.device(device)
        self.mean = self.mean.to(self.device)
        self.S = self.S.to(self.device)
        self.std = self.std.to(self.device)
        return self


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of observations-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma, lam, device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_dis_buf = np.zeros(size, dtype=np.int64)
        self.act_con_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_dis_buf = np.zeros(size, dtype=np.float32)
        self.logp_con_buf = np.zeros(size, dtype=np.float32)
        self.ptr_buf = np.zeros(size, dtype=np.int64)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_dix, self.max_size = 0, 0, size

        self.device = device

    def store_dis(self, obs, act, rew, val, logp):
        """
`       Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_dis_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_dis_buf[self.ptr] = logp
        self.ptr += 1

    def store_con(self, obs, act, rew, val, logp, ptr):
        """
`       Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_con_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_con_buf[self.ptr] = logp
        self.ptr_buf[self.ptr] = ptr

        self.ptr += 1

    def store_hybrid(self, obs, act_dis, act_con, rew, val, logp_dis, logp_con):
        """
`       Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_dis_buf[self.ptr] = act_dis
        self.act_con_buf[self.ptr] = act_con
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_dis_buf[self.ptr] = logp_dis
        self.logp_con_buf[self.ptr] = logp_con
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each observations, to use as
        the targets for the value function.

        :param last_val:
        :return:
        """
        print('------------------buffer_size---------------------')
        print(self.ptr)
        path_slice = slice(self.path_start_dix, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_dix = self.ptr

    def get(self, batch_size):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """

        obs_buf = self.obs_buf[:self.ptr]
        act_dis_buf = self.act_dis_buf[:self.ptr]
        act_con_buf = self.act_con_buf[:self.ptr]
        adv_buf = self.adv_buf[:self.ptr]
        ret_buf = self.ret_buf[:self.ptr]
        logp_dis_buf = self.logp_dis_buf[:self.ptr]
        logp_con_buf = self.logp_con_buf[:self.ptr]
        ptr_buf = self.ptr_buf[:self.ptr]

        # the next lines implement the normalization trick
        # obs_buf = (obs_buf - np.mean(obs_buf)) / np.maximum(np.std(obs_buf), 1e-6)
        # note, we are conducting normalization on Q_function not on reward
        adv_buf = (adv_buf - adv_buf.mean()) / np.maximum(adv_buf.std(), 1e-6)

        sampler = BatchSampler(
            SubsetRandomSampler(range(self.ptr)),
            batch_size,
            drop_last=False
        )

        for indices in sampler:
            yield torch.as_tensor(obs_buf[indices], dtype=torch.float32, device=self.device), \
                torch.as_tensor(act_dis_buf[indices], dtype=torch.int64, device=self.device), \
                torch.as_tensor(act_con_buf[indices], dtype=torch.float32, device=self.device), \
                torch.as_tensor(adv_buf[indices], dtype=torch.float32, device=self.device), \
                torch.as_tensor(ret_buf[indices], dtype=torch.float32, device=self.device), \
                torch.as_tensor(logp_dis_buf[indices], dtype=torch.float32, device=self.device), \
                torch.as_tensor(logp_con_buf[indices], dtype=torch.float32, device=self.device), \
                torch.as_tensor(ptr_buf[indices], dtype=torch.int64, device=self.device)

        # data = dict(obs=obs_buf, act=act_buf, ret=ret_buf, adv=adv_buf, logp=logp_buf)
        # return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}

    def filter(self):
        """
        Get the obs's mean and std for next update cycle.

        :return:
        """
        obs = self.obs_buf[:self.ptr]

        return np.mean(obs), np.std(obs)

    def clear(self):
        self.ptr, self.path_start_dix = 0, 0

class ActorCritic_Hybrid(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 mid_dim,
                 init_log_std,
                 device='cpu'
                 ):
        super().__init__()

        self.log_std = nn.Parameter(torch.zeros(action_dim, ) + init_log_std, requires_grad=True)
        self.device = device
        self.state_norm = Normalization(shape=1, device=device)

        # For the trick hyperbolic tan activations
        self.critic = nn.Sequential(
            nn.Linear(state_dim, mid_dim[0]),
            nn.Tanh(),
            nn.Linear(mid_dim[0], mid_dim[1]),
            nn.Tanh(),
            nn.Linear(mid_dim[1], mid_dim[2]),
            nn.Tanh(),
            nn.Linear(mid_dim[2], 1)
        )

        self.actor_con = nn.Sequential(
            nn.Linear(state_dim + 1, mid_dim[0]),
            nn.Tanh(),
            nn.Linear(mid_dim[0], mid_dim[1]),
            nn.Tanh(),
            nn.Linear(mid_dim[1], mid_dim[2]),
            nn.Tanh(),
            nn.Linear(mid_dim[2], action_dim),
            nn.Tanh()
        )

        self.actor_dis = nn.Sequential(
            nn.Linear(state_dim, mid_dim[0]),
            nn.Tanh(),
            nn.Linear(mid_dim[0], mid_dim[1]),
            nn.Tanh(),
            nn.Linear(mid_dim[1], mid_dim[2]),
            nn.Tanh(),
            nn.Linear(mid_dim[2], action_dim),
            nn.Softmax(dim=-1)
        )

    def encode_phase(self, discrete_action_index: torch.Tensor):
        # 定义相位编码映射
        phase_encoding = {
            0: [1, 0, 0, 0, 1, 0, 0, 0],
            1: [0, 1, 0, 0, 0, 1, 0, 0],
            2: [0, 0, 1, 0, 0, 0, 1, 0],
            3: [0, 0, 0, 1, 0, 0, 0, 1],
            4: [1, 1, 0, 0, 0, 0, 0, 0],
            5: [0, 0, 1, 1, 0, 0, 0, 0],
            6: [0, 0, 0, 0, 1, 1, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
        }
        # 根据离散动作索引返回对应的编码
        return torch.tensor(phase_encoding[discrete_action_index.item()], dtype=torch.float32,
                            device=discrete_action_index.device)

    def act(self, state, test=False):
        if test:
            action_probs = self.actor_dis(state)
            action_dis = torch.argmax(action_probs)
            norm_dis = self.state_norm(action_dis.unsqueeze(0), update=True)
            # print(norm_dis, state)
            mean = self.actor_con(torch.cat((norm_dis, state), dim=-1))
            # mean = self.actor_con(torch.cat((action_dis.unsqueeze(0)/7, state), dim=-1))
            return action_dis, mean[action_dis]
        state_value = self.critic.forward(state)

        action_probs = self.actor_dis(state)
        dist_dis = Categorical(action_probs)
        action_dis = dist_dis.sample()
        logprob_dis = dist_dis.log_prob(action_dis)

        # mean = self.actor_con(state)
        # mean = self.actor_con(torch.cat((self.encode_phase(action_dis), state), dim=-1))
        norm_dis = self.state_norm(action_dis.unsqueeze(0), update=True)
        mean = self.actor_con(torch.cat((norm_dis, state), dim=-1))
        std = torch.clamp(F.softplus(self.log_std), min=0.01, max=0.6)
        dist_con = Normal(mean, std)
        action_con = dist_con.sample()
        action_con = torch.clip(action_con, -1, 1)
        logprob_con = dist_con.log_prob(action_con)
        # print(action_con)
        return state_value, action_dis, action_con[action_dis], logprob_dis, logprob_con[action_dis]

    def get_logprob_entropy(self, state, action_dis, action_con):  # TODO
        action_probs = self.actor_dis(state)
        dist_dis = Categorical(action_probs)

        action_dis = action_dis.squeeze().long()
        logprobs_dis = dist_dis.log_prob(action_dis)
        dist_entropy_dis = dist_dis.entropy()

        # discrete_actions = torch.stack([self.encode_phase(index) for index in action_dis])
        # discrete_actions = torch.stack([index.unsqueeze(0)/7 for index in action_dis])
        # norm_dis = self.state_norm(action_dis, update=False)
        norm_dis = (action_dis - self.state_norm.mean) / (self.state_norm.std + 1e-6)
        mean = self.actor_con(torch.cat((norm_dis.unsqueeze(1), state), dim=-1))
        # mean = self.actor_con(state)
        std = torch.clamp(F.softplus(self.log_std), min=0.01, max=0.6)
        dist_con = Normal(mean, std)
        logprobs_con = dist_con.log_prob(action_con)
        dist_entropy_con = dist_con.entropy()
        print(std)
        return logprobs_dis, logprobs_con, dist_entropy_dis, dist_entropy_con


class PPO_Hybrid(object):
    def __init__(self, state_dim, action_dim, mid_dim, lr_actor, lr_critic, lr_decay_rate, buffer_size, target_kl_dis, target_kl_con,
                 gamma, lam, epochs_update, eps_clip, max_norm, coeff_entropy, random_seed, device,
                 lr_std, init_log_std):
        super().__init__()
        self.device = device
        self.random_seed = random_seed
        self.target_kl_dis = target_kl_dis
        self.target_kl_con = target_kl_con
        self.epochs_update = epochs_update
        self.eps_clip = eps_clip
        self.max_norm = max_norm
        self.coeff_entropy = coeff_entropy

        self.agent = ActorCritic_Hybrid(state_dim, action_dim, mid_dim, init_log_std, device).to(device)
        self.agent.apply(weight_init)
        self.agent_old = ActorCritic_Hybrid(state_dim, action_dim, mid_dim, init_log_std, device).to(device)
        self.agent_old.load_state_dict(self.agent.state_dict())
        self.buffer = PPOBuffer(state_dim, action_dim, buffer_size, gamma, lam, device)
        self.set_random_seeds()

        self.optimizer_actor_con = torch.optim.Adam([
            {'params': self.agent.actor_con.parameters(), 'lr': lr_actor},
            {'params': self.agent.log_std, 'lr': lr_std},
        ])
        self.optimizer_actor_dis = torch.optim.Adam(self.agent.actor_dis.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.agent.critic.parameters(), lr=lr_critic)

        self.lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_critic,
                                                                          gamma=lr_decay_rate)
        self.lr_scheduler_actor_con = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_con,
                                                                             gamma=lr_decay_rate)
        self.lr_scheduler_actor_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_dis,
                                                                             gamma=lr_decay_rate)
        self.loss_func = nn.SmoothL1Loss(reduction='mean')

    def evaluate(self, state):  # When evaluating the policy, we only use the mean
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            act_dis, act_con = self.agent_old.act(state, test=True)
        return act_dis.cpu().numpy(), act_con.cpu().numpy()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            state_value, action_dis, action_con, log_prob_dis, log_prob_con = self.agent_old.act(state)

        return state_value.squeeze().cpu().numpy(), (action_dis.cpu().numpy(), action_con.cpu().numpy()), (log_prob_dis.cpu().numpy(), log_prob_con.cpu().numpy())

    def compute_loss_pi(self, data):
        obs, act_dis, act_con, adv, _, logp_old_dis, logp_old_con, _ = data
        self.agent.state_norm = self.agent_old.state_norm
        logp_dis, logp_con, dist_entropy_dis, dist_entropy_con = self.agent.get_logprob_entropy(obs, act_dis, act_con)
        logp_con = logp_con.gather(1, act_dis.view(-1, 1)).squeeze()
        # dist_entropy = dist_entropy.gather(1, ptr.view(-1, 1)).squeeze()
        ratio_dis = torch.exp(logp_dis - logp_old_dis)
        ratio_con = torch.exp(logp_con - logp_old_con)
        clip_adv_dis = torch.clamp(ratio_dis, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        clip_adv_con = torch.clamp(ratio_con, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        loss_pi_dis = - (torch.min(ratio_dis * adv, clip_adv_dis) + self.coeff_entropy * dist_entropy_dis).mean()
        loss_pi_con = - (torch.min(ratio_con * adv, clip_adv_con)).mean()
        # Useful extra info
        approx_kl_dis = (logp_old_dis - logp_dis).mean().item()
        approx_kl_con = (logp_old_con - logp_con).mean().item()

        return loss_pi_dis, loss_pi_con, approx_kl_dis, approx_kl_con

    def compute_loss_v(self, data):
        obs, act_dis, _, _, ret, _, _, _ = data
        state_values = self.agent.critic(obs)
        state_values = state_values.squeeze(1)

        return self.loss_func(state_values, ret)

    def update(self, batch_size):
        # For monitor
        pi_loss_dis_epoch = 0
        pi_loss_con_epoch = 0
        v_loss_epoch = 0
        kl_con_epoch = 0
        kl_dis_epoch = 0
        num_updates = 0

        for i in range(self.epochs_update):
            sampler = self.buffer.get(batch_size)
            for data in sampler:
                if len(data[0]) < 32:
                    continue
                self.optimizer_actor_dis.zero_grad()
                self.optimizer_actor_con.zero_grad()
                pi_loss_dis, pi_loss_con, kl_dis, kl_con = self.compute_loss_pi(data)
                # pi_loss_dis.backward()
                # torch.nn.utils.clip_grad_norm_(self.agent.actor_dis.parameters(), norm_type=2, max_norm=self.max_norm)
                # self.optimizer_actor_dis.step()
                #
                # pi_loss_con.backward()
                # torch.nn.utils.clip_grad_norm_(self.agent.actor_con.parameters(), norm_type=2, max_norm=self.max_norm)
                # self.optimizer_actor_con.step()
                if kl_dis < self.target_kl_dis * 1.5:
                    pi_loss_dis.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.actor_dis.parameters(), norm_type=2, max_norm=self.max_norm)
                    self.optimizer_actor_dis.step()
                else:
                    print('Early stopping at step {} due to reaching max kl. Now kl_dis is {}'.format(num_updates, kl_dis))

                if kl_con < self.target_kl_con * 1.5:
                    pi_loss_con.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.actor_con.parameters(), norm_type=2, max_norm=self.max_norm)
                    self.optimizer_actor_con.step()
                else:
                    print('Early stopping at step {} due to reaching max kl. Now kl_con is {}'.format(num_updates, kl_con))

                pi_loss_dis_epoch += pi_loss_dis.item()
                pi_loss_con_epoch += pi_loss_con.item()
                kl_dis_epoch += kl_dis
                kl_con_epoch += kl_con

                self.optimizer_critic.zero_grad()
                v_loss = self.compute_loss_v(data)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), norm_type=2, max_norm=self.max_norm)
                self.optimizer_critic.step()

                v_loss_epoch += v_loss.item()
                num_updates += 1

        pi_loss_dis_epoch /= num_updates
        pi_loss_con_epoch /= num_updates
        kl_con_epoch /= num_updates
        kl_dis_epoch /= num_updates
        v_loss_epoch /= num_updates

        self.lr_scheduler_actor_con.step()
        self.lr_scheduler_actor_dis.step()
        self.lr_scheduler_critic.step()

        print('----------------------------------------------------------------------')
        print('Worker_{}, LossPi_dis: {}, LossPi_con: {}, KL_dis: {}, KL_con: {}, LossV: {}'.format(
            self.random_seed, pi_loss_dis_epoch, pi_loss_con_epoch, kl_dis_epoch, kl_con_epoch, v_loss_epoch)
        )
        print('----------------------------------------------------------------------')

        # copy new weights into old policy
        self.agent_old.load_state_dict(self.agent.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.agent_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.agent_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def set_random_seeds(self):
        """
        Sets all possible random seeds to results can be reproduced.
        :param random_seed:
        :return:
        """
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.cuda.manual_seed(self.random_seed)
