import os
import random
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions import Normal

from .utils import weight_init, PPOBuffer


class ActorCritic_Discrete(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 mid_dim,
                 ):
        super().__init__()

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

        self.actor = nn.Sequential(
            nn.Linear(state_dim, mid_dim[0]),
            nn.Tanh(),
            nn.Linear(mid_dim[0], mid_dim[1]),
            nn.Tanh(),
            nn.Linear(mid_dim[1], mid_dim[2]),
            nn.Tanh(),
            nn.Linear(mid_dim[2], action_dim),
            nn.Softmax(dim=-1)
        )

    def act(self, state):
        state_value = self.critic.forward(state)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return state_value, action, action_logprob

    def get_logprob_entropy(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = action.squeeze().long()

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy


class ActorCritic_Continuous(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 mid_dim,
                 init_log_std,
                 ):
        super().__init__()

        self.log_std = nn.Parameter(torch.zeros(action_dim, ) + init_log_std, requires_grad=True)

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

        self.actor = nn.Sequential(
            nn.Linear(state_dim, mid_dim[0]),
            nn.Tanh(),
            nn.Linear(mid_dim[0], mid_dim[1]),
            nn.Tanh(),
            nn.Linear(mid_dim[1], mid_dim[2]),
            nn.Tanh(),
            nn.Linear(mid_dim[2], action_dim),
            nn.Tanh()
        )

    def act(self, state):
        state_value = self.critic.forward(state)
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return state_value, action, action_logprob

    def get_logprob_entropy(self, state, action):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy


class PPO_Abstract:
    def __init__(self,
                 state_dim,
                 action_dim,
                 mid_dim,
                 lr_critic,
                 lr_decay_rate,
                 gamma,
                 epochs_update,
                 eps_clip,
                 max_norm,
                 coeff_entropy,  # TODO
                 random_seed,
                 device,
                 ):
        self.gamma = gamma
        self.epochs_update = epochs_update
        self.eps_clip = eps_clip
        self.max_norm = max_norm
        self.coeff_entropy = coeff_entropy
        self.random_seed = random_seed
        self.set_random_seeds()

        # Implementation of Buffer and Agent is defined in specific agent classes below.
        # Here i will implement one specific agent(Discrete) for the code completeness
        self.agent = ActorCritic_Discrete(state_dim, action_dim, mid_dim).to(device)
        self.agent_old = ActorCritic_Discrete(state_dim, action_dim, mid_dim).to(device)
        self.agent_old.load_state_dict(self.agent.state_dict())

        self.optimizer_critic = torch.optim.Adam(self.agent.critic.parameters(), lr=lr_critic)
        # Trick of Adam learning rate annealing
        self.lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_critic,
                                                                          gamma=lr_decay_rate)
        self.loss_func = nn.SmoothL1Loss(reduction='mean')

    def select_action(self, state):
        """
        :param state:
        :return:
        """
        raise NotImplementedError

    def compute_loss_pi(self, data):
        """
        :param data:
        :return:
        """
        raise NotImplementedError

    def compute_loss_v(self, data):
        """
        :param data:
        :return:
        """
        raise NotImplementedError

    def update(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        raise NotImplementedError

    def save(self, checkpoint_path):
        torch.save(self.agent_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.agent_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def set_random_seeds(self):
        """
        Sets all possible random seeds to results can be reproduces.
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


class PPO_Discrete(PPO_Abstract, ABC):
    def __init__(self, state_dim, action_dim, mid_dim, lr_actor, lr_critic, lr_decay_rate, buffer_size, target_kl_dis,
                 gamma, lam, epochs_update, v_iters, eps_clip, max_norm, coeff_entropy, random_seed, device):
        super().__init__(state_dim, action_dim, mid_dim, lr_critic, lr_decay_rate, gamma, epochs_update, eps_clip,
                         max_norm, coeff_entropy, random_seed, device)
        self.buffer = PPOBuffer(state_dim, action_dim, buffer_size, gamma, lam, device)
        # The implementation of PPO_Discrete has been achieved in the PPO_Abstract.
        # But for completeness, I will repeat it again here.
        self.device = device
        self.target_kl_dis = target_kl_dis

        self.agent = ActorCritic_Discrete(state_dim, action_dim, mid_dim).to(device)
        self.agent.apply(weight_init)
        self.agent_old = ActorCritic_Discrete(state_dim, action_dim, mid_dim).to(device)
        self.agent_old.load_state_dict(self.agent.state_dict())
        self.optimizer_actor = torch.optim.Adam(self.agent.actor.parameters(), lr=lr_actor)
        self.lr_scheduler_actor = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor,
                                                                         gamma=lr_decay_rate)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            state_value, action, log_prob = self.agent_old.act(state)

        return state_value.squeeze().cpu().numpy(), action.cpu().numpy(), log_prob.cpu().numpy()

    def compute_loss_pi(self, data):
        obs, act_dis, _, adv, _, logp_old_dis, _, _ = data

        logp, dist_entropy = self.agent.get_logprob_entropy(obs, act_dis)
        ratio = torch.exp(logp - logp_old_dis)
        clip_adv = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        loss_pi = - (torch.min(ratio * adv, clip_adv) + self.coeff_entropy * dist_entropy).mean()

        # Useful extra info
        approx_kl = (logp_old_dis - logp).mean().item()

        return loss_pi, approx_kl

    def compute_loss_v(self, data):
        obs, _, _, _, ret, _, _, _ = data
        with torch.no_grad():
            state_values = self.agent.critic(obs)
        # print(obs.requires_grad)
        return self.loss_func(state_values, ret)

    def update(self, batch_size):
        # For monitor
        pi_loss_epoch = 0
        v_loss_epoch = 0
        kl_epoch = 0
        num_updates = 0

        for i in range(self.epochs_update):
            sampler = self.buffer.get(batch_size)
            for data in sampler:
                self.optimizer_actor.zero_grad()
                pi_loss, kl = self.compute_loss_pi(data)

                if kl < self.target_kl_dis:
                    pi_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), norm_type=2, max_norm=self.max_norm)
                    self.optimizer_actor.step()
                else:
                    print('Early stopping at step {} due to reaching max kl. Now kl is {}'.format(num_updates, kl))

                self.optimizer_critic.zero_grad()
                v_loss = self.compute_loss_v(data)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), norm_type=2, max_norm=self.max_norm)
                self.optimizer_critic.step()

                pi_loss_epoch += pi_loss.item()
                v_loss_epoch += v_loss.item()
                kl_epoch += kl

                num_updates += 1

        pi_loss_epoch /= num_updates
        kl_epoch /= num_updates
        v_loss_epoch /= num_updates

        self.lr_scheduler_actor.step()
        self.lr_scheduler_critic.step()

        print(self.lr_scheduler_actor.get_lr())
        print(self.lr_scheduler_critic.get_lr())
        print('----------------------------------------------------------------------')
        print('Worker_{}, LossPi: {}, KL: {}, LossV: {}'.format(
            self.random_seed, pi_loss_epoch, kl_epoch, v_loss_epoch)
        )
        print('----------------------------------------------------------------------')

        # copy new weights into old policy
        self.agent_old.load_state_dict(self.agent.state_dict())


class PPO_Continuous(PPO_Abstract, ABC):
    def __init__(self, state_dim, action_dim, mid_dim, lr_actor, lr_critic, lr_decay_rate, buffer_size, target_kl_con,
                 gamma, lam, epochs_update, eps_clip, max_norm, coeff_entropy, random_seed, device,
                 lr_std, init_log_std):
        super().__init__(state_dim, action_dim, mid_dim, lr_critic, lr_decay_rate, gamma, epochs_update, eps_clip,
                         max_norm, coeff_entropy, random_seed, device)
        self.target_kl_con = target_kl_con
        self.device = device

        self.buffer = PPOBuffer(state_dim, action_dim, buffer_size, gamma, lam, device)
        self.agent = ActorCritic_Continuous(state_dim, action_dim, mid_dim, init_log_std).to(device)
        self.agent.apply(weight_init)
        self.agent_old = ActorCritic_Continuous(state_dim, action_dim, mid_dim, init_log_std).to(device)
        self.agent_old.load_state_dict(self.agent.state_dict())

        self.optimizer_actor = torch.optim.Adam([
            {'params': self.agent.actor.parameters(), 'lr': lr_actor, 'eps': 1e-5},
            # {'params': self.agent.log_std, 'lr': lr_std},
        ])
        self.lr_scheduler_actor = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor,
                                                                         gamma=lr_decay_rate, verbose=True)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            state_value, action, log_prob = self.agent_old.act(state)

        return state_value.squeeze().cpu().numpy(), action.cpu().numpy(), log_prob.cpu().numpy()

    def compute_loss_pi(self, data):
        """obs, _, act_con, adv, _, _, logp_old_con, ptr = data

        logp, _ = self.agent.get_logprob_entropy(obs, act_con)
        logp = logp.gather(1, ptr.view(-1, 1)).squeeze()
        # dist_entropy = dist_entropy.gather(1, ptr.view(-1, 1)).squeeze()
        ratio = torch.exp(logp - logp_old_con)
        clip_adv = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        loss_pi = - (torch.min(ratio * adv, clip_adv)).mean()  # TODO

        # Useful extra info
        approx_kl = (logp_old_con - logp).mean().item()"""
        # print(self.agent.log_std)
        obs, _, act_con, adv, _, _, logp_old_con, ptr = data

        logp, dist_entropy = self.agent.get_logprob_entropy(obs, act_con)
        ratio = torch.exp(logp - logp_old_con)
        clip_adv = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        loss_pi = - (torch.min(ratio * adv, clip_adv) + self.coeff_entropy * dist_entropy).mean()

        # Useful extra info
        approx_kl = (logp_old_con - logp).mean().item()

        return loss_pi, approx_kl

    """def compute_loss_v(self, data):
        obs, _, _, _, ret, _, _, ptr = data
        state_values = self.agent.critic(obs).gather(1, ptr.view(-1, 1)).squeeze()

        return self.loss_func(state_values, ret)"""

    def compute_loss_v(self, data):
        obs, _, _, _, ret, _, _, _ = data
        state_values = self.agent.critic(obs).squeeze()
        # print(obs.requires_grad)
        return self.loss_func(state_values, ret)

    def update(self, batch_size):
        # For monitor
        pi_loss_epoch = 0
        v_loss_epoch = 0
        kl_epoch = 0
        num_updates = 0

        for i in range(self.epochs_update):
            sampler = self.buffer.get(batch_size)
            for data in sampler:

                self.optimizer_actor.zero_grad()
                pi_loss, kl = self.compute_loss_pi(data)

                if kl < self.target_kl_con:
                    pi_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), norm_type=2, max_norm=self.max_norm)
                    self.optimizer_actor.step()
                else:
                    pass
                    print('Early stopping at step {} due to reaching max kl. Now kl is {}'.format(num_updates, kl))

                self.optimizer_critic.zero_grad()
                v_loss = self.compute_loss_v(data)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), norm_type=2, max_norm=self.max_norm)
                self.optimizer_critic.step()

                pi_loss_epoch += pi_loss.item()
                v_loss_epoch += v_loss.item()
                kl_epoch += kl

                num_updates += 1

        pi_loss_epoch /= num_updates
        kl_epoch /= num_updates
        v_loss_epoch /= num_updates

        self.lr_scheduler_actor.step()
        self.lr_scheduler_critic.step()

        print('ppo actor', self.lr_scheduler_actor.get_lr())
        print('ppo critic', self.lr_scheduler_critic.get_lr())
        # print('----------------------------------------------------------------------')
        print(f'Worker_{self.random_seed}, LossPi: {pi_loss_epoch}, KL: {kl_epoch}, LossV: {v_loss_epoch}')
        # print('----------------------------------------------------------------------')

        # copy new weights into old policy
        self.agent_old.load_state_dict(self.agent.state_dict())
