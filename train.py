import argparse
import datetime
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.hppo_noshare import PPO_Hybrid
from agent.ppo import PPO_Continuous
# from agent.ppo2 import PPO_continuous
from cotp_intersection import raw_env
from normalization import Normalization

writer = SummaryWriter()


class Trainer(object):
    def __init__(self, args):
        self.seed = args.random_seed
        self.device = args.device
        self.max_episodes = args.max_episodes
        self.max_steps = args.max_steps
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.mid_dim = args.mid_dim
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_actor_param
        self.lr_std = args.lr_std
        self.lr_decay_rate = args.lr_decay_rate
        self.target_kl_dis = args.target_kl_dis
        self.target_kl_con = args.target_kl_con
        self.agent_update_freq = args.agent_update_freq
        self.agent_save_freq = args.agent_save_freq
        self.gamma = args.gamma
        self.lambda_ = args.lambda_
        self.epochs_update = args.epochs_update
        self.eps_clip = args.eps_clip
        self.max_norm_grad = args.max_norm_grad
        self.init_log_std = args.init_log_std
        self.coeff_dist_entropy = args.coeff_dist_entropy

        self.max_green = args.max_green
        self.min_green = args.min_green

        self.env = raw_env(
            use_gui=False,
            sumo_config=args.sumocfg,
            sumo_seed=self.seed,
            num_seconds=900,
            begin_time=30,
        )
        self.history = {}
        self.writer = True

    def initialize_agents(self, random_seed, policy_mapping_fn):
        """
        Initialize environment and agent.
        :param policy_mapping_fn:
        :param random_seed: could be regarded as worker index
        :return: instance of agent and env
        """
        env = self.env

        agents, state_norm = {}, {}
        # 创建字典，将 agent id 映射到 Policy
        agent_policy_dict = {agent_id: policy_mapping_fn(agent_id) for agent_id in env.possible_agents}
        policys = {policy_mapping_fn(agent) for agent in env.possible_agents}
        for agent in self.env.possible_agents:
            obs_space, act_space = self.env.observation_space(agent), self.env.action_space(agent)
            if agent.startswith('cav'):
                """agents[agent_policy_dict[agent]] = (
                    PPO_continuous(
                        obs_space.shape[0], act_space.shape[0], 1.0,
                        args.batch_size, 64,
                        args.max_train_steps, 3e-4, 3e-4, self.gamma,
                        self.lambda_, args.epsilon, args.K_epochs, args.entropy_coef, )
                )"""
                agents[agent_policy_dict[agent]] = (
                    PPO_Continuous(obs_space.shape[0], act_space.shape[0], self.mid_dim,
                                   3e-4, 5e-4, self.lr_decay_rate, self.buffer_size,
                                   0.2, self.gamma, self.lambda_, self.epochs_update, 0.2,
                                   2, self.coeff_dist_entropy, random_seed, self.device, self.lr_std,
                                   self.init_log_std,
                                   ))
            elif agent.startswith('traffic'):
                agents[agent_policy_dict[agent]] = (
                    PPO_Hybrid(obs_space.shape[0], act_space[0].n, self.mid_dim, self.lr_actor, self.lr_critic,
                               self.lr_decay_rate, self.buffer_size, self.target_kl_dis, self.target_kl_con,
                               self.gamma, self.lambda_, self.epochs_update, self.eps_clip, self.max_norm_grad,
                               self.coeff_dist_entropy, random_seed, self.device, self.lr_std, self.init_log_std,
                               ))
            else:
                raise ValueError
            state_norm[agent_policy_dict[agent]] = Normalization(shape=obs_space.shape[0])
        assert len(agents) == len(policys) == len(state_norm)
        file_save = f"data_train/{current_date}/"
        for agent in agents:
            os.makedirs(f'{file_save}/data/{agent}', exist_ok=True)
            os.makedirs(f'{file_save}/policy/{agent}', exist_ok=True)
        return agents, state_norm

    def train(self):

        def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
            """if agent_id.startswith('traffic'):
                return 'traffic_light'
            _, _, direction = agent_id.split('_')
            if 'left' in direction:
                return 'cav_left'
            elif 'straight' in direction:
                return 'cav_straight'
            else:
                return agent_id"""
            return agent_id

        file_save = f"data_train/{current_date}/"
        agents, state_norm = self.initialize_agents(self.seed, policy_mapping_fn)
        agents['traffic_light'].load('data_train/1018/policy/i_episode990_42_0')
        state_norm['traffic_light'].running_ms.load('data_train/1018/i_episode912_42.npz')
        env = self.env
        episode = 0
        total_steps = 0
        """  TRAINING LOGIC  """
        while total_steps < self.max_steps:
            # collect an episode
            # with torch.no_grad():
            states, infos = env.reset()
            episode_step = 0
            queue_len = []
            episode_reward = {agent: 0.0 for agent in self.env.possible_agents}
            while True:
                # Every update, we will normalize the state_norm(the input of the actor_con and critic) by
                # mean and std retrieve from the last update's buf, in other word observations normalization
                actions = {}
                self.history = {agent: {} for agent in self.env.possible_agents}
                env_agents = env.possible_agents
                for agent in env_agents:
                    if infos[agent]['agents_to_update']:
                        agent_map = policy_mapping_fn(agent)
                        observation = states[agent].reshape(-1)
                        observation_norm = state_norm[agent_map](observation,
                                                                 update=bool(infos[agent]['agents_to_update']))
                        # Select action with policy
                        value_action_logp = agents[agent_map].select_action(observation_norm)
                        #  state_value, (action_dis, action_con), (log_prob_dis, log_prob_con)
                        #  state_value, action, log_prob
                        if agent.startswith('cav'):
                            value, action, log_prob = value_action_logp
                            action_pad = action * 3.0
                            self.history[agent] = {'obs': observation_norm, 'act': action, 'val': value,
                                                   'logp_act': log_prob}
                        elif agent.startswith('traffic'):
                            value, (action_dis, action_con), (log_prob_dis, log_prob_con) = value_action_logp
                            action_con_pad = (action_con + 1) / 2 * (self.max_green - self.min_green) + self.min_green
                            action_pad = (action_dis, np.array(action_con_pad, dtype=np.int64))
                            self.history[agent] = {'obs': observation_norm, 'act_dis': action_dis,
                                                   'act_con': action_con,
                                                   'val': value, 'logp_act_dis': log_prob_dis,
                                                   'logp_act_con': log_prob_con}
                        actions[agent] = action_pad

                next_states, rewards, terminations, truncations, next_infos = env.step(actions)
                queue_len.append(next_infos['queue'])
                episode_step += 1

                for agent in env_agents:
                    if infos[agent]['agents_to_update']:
                        episode_reward[agent] += rewards[agent]
                        if agent.startswith('cav'):
                            agents[policy_mapping_fn(agent)].buffer.store_con(
                                self.history[agent]['obs'], self.history[agent]['act'], rewards[agent],
                                self.history[agent]['val'], self.history[agent]['logp_act'],
                            )
                        elif agent.startswith('traffic'):
                            agents[policy_mapping_fn(agent)].buffer.store_hybrid(
                                self.history[agent]['obs'], self.history[agent]['act_dis'],
                                self.history[agent]['act_con'], rewards[agent],
                                self.history[agent]['val'],
                                self.history[agent]['logp_act_dis'],
                                self.history[agent]['logp_act_con']
                            )
                # print(reward[0])
                states = next_states
                infos = next_infos
                total_steps += 1
                if truncations['__all__'] or terminations['__all__']:
                    episode += 1
                    [policy.buffer.finish_path(0) for policy in agents.values()]
                    break
            if episode % 2 == 0:
                for agent, policy in agents.items():
                    if policy.buffer.ptr > self.batch_size:
                        if not agent.startswith('traffic'):
                            policy.update(self.batch_size)
                            policy.buffer.clear()
                        elif episode % 10 == 0:
                            policy.update(128)
                            policy.buffer.clear()

            print('***', np.mean(list(env.temp['vehicle_loss'].values())))

            if episode % self.agent_save_freq == 0:
                for agent, policy in agents.items():
                    file_to_save_policy = os.path.join(f"{file_save}/policy/{agent}", f'episode{episode}_{self.seed}')
                    policy.save(file_to_save_policy)
                print('-------------------------------------------------------------------')
                print('model saved')
                print('-------------------------------------------------------------------')
                for agent in agents.keys():
                    path_to_save_npz = os.path.join(f'{file_save}/data/{agent}',
                                                    f'episode{episode}_{self.seed}.npz')
                    state_norm[agent].running_ms.save(path_to_save_npz)
            if self.writer:
                writer.add_scalar('run/episode_len', scalar_value=episode_step, global_step=total_steps)
                writer.add_scalar('run/queue', scalar_value=np.mean(queue_len), global_step=total_steps)
                writer.add_scalar('run/travel_time', scalar_value=np.mean(list(env.temp['vehicle_loss'].values())),
                                  global_step=total_steps)
                for agent in env.possible_agents:
                    writer.add_scalar(f'run/reward_{agent}', scalar_value=episode_reward[agent],
                                      global_step=total_steps)
        env.close()

    def test(self):
        def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
            return agent_id

        current_date = 1110
        file_save = f"data_train/{current_date}"
        agents, state_norm = self.initialize_agents(self.seed, policy_mapping_fn)
        agents['traffic_light'].load('data_train/1018/policy/i_episode990_42_0')
        state_norm['traffic_light'].running_ms.load('data_train/1018/i_episode912_42.npz')
        '''for agent in agents:
            if not agent.startswith('traffic'):
                policy_path = f"{file_save}/policy/{agent}/episode140_42"
                norm_path = f"{file_save}/data/{agent}/episode140_42.npz"
                agents[agent].load(policy_path)
                state_norm[agent].running_ms.load(norm_path)'''

        env = raw_env(
            use_gui=True,
            sumo_config=args.sumocfg,
            sumo_seed=self.seed,
            num_seconds=900,
            begin_time=30,
        )
        episode = 0
        total_steps = 0
        """  TRAINING LOGIC  """
        # collect an episode
        # with torch.no_grad():
        states, infos = env.reset()
        episode_step = 0
        queue_len = []
        episode_reward = {agent: 0.0 for agent in self.env.possible_agents}
        while True:
            # Every update, we will normalize the state_norm(the input of the actor_con and critic) by
            # mean and std retrieve from the last update's buf, in other word observations normalization
            actions = {}
            env_agents = env.possible_agents
            for agent in env_agents:
                if infos[agent]['agents_to_update']:
                    agent_map = policy_mapping_fn(agent)
                    observation = states[agent].reshape(-1)
                    observation_norm = state_norm[agent_map](observation,
                                                             update=bool(infos[agent]['agents_to_update']))
                    # Select action with policy
                    value_action_logp = agents[agent_map].select_action(observation_norm)
                    #  state_value, (action_dis, action_con), (log_prob_dis, log_prob_con)
                    #  state_value, action, log_prob
                    if agent.startswith('cav'):
                        value, action, log_prob = value_action_logp
                        action_pad = action * 3.0
                        self.history[agent] = {'obs': observation_norm, 'act': action, 'val': value,
                                               'logp_act': log_prob}
                    elif agent.startswith('traffic'):
                        value, (action_dis, action_con), (log_prob_dis, log_prob_con) = value_action_logp
                        action_con_pad = (action_con + 1) / 2 * (self.max_green - self.min_green) + self.min_green
                        action_pad = (action_dis, np.array(action_con_pad, dtype=np.int64))
                        self.history[agent] = {'obs': observation_norm, 'act_dis': action_dis,
                                               'act_con': action_con,
                                               'val': value, 'logp_act_dis': log_prob_dis,
                                               'logp_act_con': log_prob_con}
                    actions[agent] = action_pad

            next_states, rewards, terminations, truncations, next_infos = env.step(actions)
            queue_len.append(next_infos['queue'])
            episode_step += 1
            # print(reward[0])
            states = next_states
            infos = next_infos
            total_steps += 1
            if truncations['__all__'] or terminations['__all__']:
                episode += 1
                break
        print('***', np.mean(list(env.temp['vehicle_loss'].values())))
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device.')
    parser.add_argument('--max_episodes', type=int, default=990, help='The max episodes per agent per run.')
    parser.add_argument('--max_steps', type=int, default=int(5e6), help='The max steps in training.')
    parser.add_argument('--buffer_size', type=int, default=20000, help='The maximum size of the PPOBuffer.')
    parser.add_argument('--batch_size', type=int, default=256, help='The sample batch size.')
    parser.add_argument('--rolling_score_window', type=int, default=5,
                        help='Mean of last rolling_score_window.')
    parser.add_argument('--agent_save_freq', type=int, default=5, help='The frequency of the agent saving.')
    parser.add_argument('--agent_update_freq', type=int, default=2, help='The frequency of the agent updating.')
    parser.add_argument('--lr_actor', type=float, default=0.0003, help='The learning rate of actor_con.')  # carefully!
    parser.add_argument('--lr_actor_param', type=float, default=0.001, help='The learning rate of critic.')
    parser.add_argument('--lr_std', type=float, default=0.004, help='The learning rate of log_std.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9966, help='Factor of learning rate decay.')
    parser.add_argument('--mid_dim', type=list, default=[256, 128, 64], help='The middle dimensions of both nets.')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discounted of future rewards.')
    parser.add_argument('--lambda_', type=float, default=0.8,
                        help='Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)')
    parser.add_argument('--epochs_update', type=int, default=10,
                        help='Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)')
    parser.add_argument('--v_iters', type=int, default=1,
                        help='Number of gradient descent steps to take on value function per epoch.')
    parser.add_argument('--target_kl_dis', type=float, default=0.025,
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--target_kl_con', type=float, default=0.05,
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='The clip ratio when calculate surr.')
    parser.add_argument('--max_norm_grad', type=float, default=5.0, help='max norm of the gradients.')
    parser.add_argument('--init_log_std', type=float, default=-1.0,
                        help='The initial log_std of Normal in continuous pattern.')
    parser.add_argument('--coeff_dist_entropy', type=float, default=0.005,
                        help='The coefficient of distribution entropy.')
    parser.add_argument('--random_seed', type=int, default=42, help='The random seed.')
    parser.add_argument('--if_use_active_selection', type=bool, default=False,
                        help='Whether use active selection in the exploration.')
    parser.add_argument('--init_bonus', type=float, default=0.01, help='The initial active selection bonus.')
    parser.add_argument('--sumocfg', type=str, default='net/single-stage2.sumocfg',
                        help='The initial active selection bonus.')
    parser.add_argument('--num_stage', type=int, default=8)
    parser.add_argument('--max_green', type=int, default=60)
    parser.add_argument('--min_green', type=int, default=17)

    args = parser.parse_args()

    # args log
    current_date = datetime.date.today().strftime("%m%d")
    file = f"data_train/{current_date}/"
    if not os.path.exists(file):
        os.makedirs(file)
    argsDict = args.__dict__
    with open(f'data_train/{current_date}/args.txt', 'w') as f:
        f.writelines('------------ start ------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ':' + str(value) + '\n')
        f.writelines('------------ end ------------')

    # training through multiprocess
    trainer = Trainer(args)
    # trainer.train()
    trainer.test()

    # args_tuple = [[31], [32], [33], [34], [35], [36]]
    # pool = Pool(processes=6)
    # for arg in args_tuple:
    #     pool.apply_async(trainer.train, arg)
    # pool.close()
    # pool.join()

    exit(2000)
