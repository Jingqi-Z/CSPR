import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal

from .utils import weight_init, PPOBuffer


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

        self.shared_actor = nn.Sequential(
            nn.Linear(state_dim, mid_dim[0]),
            nn.Tanh(),
            nn.Linear(mid_dim[0], mid_dim[1]),
            nn.Tanh(),
            nn.Linear(mid_dim[1], mid_dim[2]),
            nn.Tanh(),
        )
        # Separate heads for discrete and continuous actions
        self.actor_con = nn.Sequential(
            nn.Linear(mid_dim[2], action_dim),
            nn.Tanh(),
        )
        self.actor_dis = nn.Sequential(
            nn.Linear(mid_dim[2], action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        shared_features = self.shared_actor(state)
        actor_dis_output = self.actor_dis(shared_features)
        actor_con_output = self.actor_con(shared_features)
        critic_output = self.critic(state)
        return actor_dis_output, actor_con_output, critic_output

    def act(self, state, test=False):
        # shared_features = self.shared_actor(state)
        # actor_dis_output = self.actor_dis(shared_features)
        # actor_con_output = self.actor_con(shared_features)
        actor_dis_output, actor_con_output, critic_output = self.forward(state)
        if test:
            action_dis = torch.argmax(actor_dis_output)
            mean = actor_con_output[action_dis]
            return action_dis, mean

        # state_value = self.critic.forward(state)
        dist_dis = Categorical(actor_dis_output)
        action_dis = dist_dis.sample()
        logprob_dis = dist_dis.log_prob(action_dis)

        mean = actor_con_output
        std = torch.clamp(F.softplus(self.log_std), min=0.01, max=0.6)
        dist_con = Normal(mean, std)
        action_con = dist_con.sample()
        action_con = torch.clip(action_con, -1, 1)
        logprob_con = dist_con.log_prob(action_con)
        return critic_output, action_dis, action_con[action_dis], logprob_dis, logprob_con[action_dis]

    def get_logprob_entropy(self, state, action_dis, action_con):  # TODO
        shared_features = self.shared_actor(state)
        actor_dis_output = self.actor_dis(shared_features)
        actor_con_output = self.actor_con(shared_features)
        action_dis = action_dis.squeeze().long()
        dist_dis = Categorical(actor_dis_output)
        logprobs_dis = dist_dis.log_prob(action_dis)
        dist_entropy_dis = dist_dis.entropy()
        # mean = self.actor_con(state)
        std = torch.clamp(F.softplus(self.log_std), min=0.01, max=0.6)
        dist_con = Normal(actor_con_output, std)

        logprobs_con = dist_con.log_prob(action_con)
        dist_entropy_con = dist_con.entropy()

        return logprobs_dis, logprobs_con, dist_entropy_dis, dist_entropy_con


class PPO_Hybrid(object):
    def __init__(self, state_dim, action_dim, mid_dim, lr_actor, lr_critic, lr_decay_rate, buffer_size, target_kl_dis,
                 target_kl_con,
                 gamma, lam, epochs_update, eps_clip, max_norm, coeff_entropy, random_seed, device,
                 lr_std, init_log_std, if_use_active_selection=False):
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

        self.optimizer_actor_con = torch.optim.Adam([
            {'params': self.agent.actor_con.parameters(), 'lr': lr_actor * 1.5},
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

        return state_value.squeeze().cpu().numpy(), (action_dis.cpu().numpy(), action_con.cpu().numpy()), (
        log_prob_dis.cpu().numpy(), log_prob_con.cpu().numpy())

    def compute_loss_pi(self, data):
        obs, act_dis, act_con, adv, _, logp_old_dis, logp_old_con, _ = data

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

        return self.loss_func(state_values, ret)

    def update(self, batch_size):
        # For monitor
        pi_loss = 0
        v_loss_epoch = 0
        kl_con_epoch = 0
        kl_dis_epoch = 0
        num_updates = 0

        for i in range(self.epochs_update):
            sampler = self.buffer.get(batch_size)
            for data in sampler:
                self.optimizer_actor_dis.zero_grad()
                self.optimizer_actor_con.zero_grad()
                # 计算损失
                pi_loss_dis, pi_loss_con, kl_dis, kl_con = self.compute_loss_pi(data)
                # 合并损失
                total_loss = pi_loss_dis + pi_loss_con
                # 反向传播总损失
                total_loss.backward()
                # 裁剪梯度
                torch.nn.utils.clip_grad_norm_(self.agent.actor_dis.parameters(), norm_type=2, max_norm=self.max_norm)
                torch.nn.utils.clip_grad_norm_(self.agent.actor_con.parameters(), norm_type=2, max_norm=self.max_norm)
                # 更新参数
                self.optimizer_actor_dis.step()
                self.optimizer_actor_con.step()
                pi_loss += total_loss.item()
                kl_dis_epoch += kl_dis
                kl_con_epoch += kl_con

                v_loss = self.compute_loss_v(data)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), norm_type=2, max_norm=self.max_norm)
                self.optimizer_critic.step()

                v_loss_epoch += v_loss.item()
                num_updates += 1

        pi_loss /= num_updates
        kl_con_epoch /= num_updates
        kl_dis_epoch /= num_updates
        v_loss_epoch /= num_updates

        self.lr_scheduler_actor_con.step()
        self.lr_scheduler_actor_dis.step()
        self.lr_scheduler_critic.step()

        print('----------------------------------------------------------------------')
        print('Worker_{}, LossPi: {}, KL_dis: {}, KL_con: {}, LossV: {}'.format(
            self.random_seed, pi_loss, kl_dis_epoch, kl_con_epoch, v_loss_epoch)
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
