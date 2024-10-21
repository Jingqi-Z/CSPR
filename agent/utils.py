import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import BatchSampler, SubsetRandomSampler


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)


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

    def store_con(self, obs, act, rew, val, logp):
        """
`       Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_con_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_con_buf[self.ptr] = logp
        # self.ptr_buf[self.ptr] = ptr

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
        as well as compute the rewards-to-go for each observation, to use as
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
            drop_last=True
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
