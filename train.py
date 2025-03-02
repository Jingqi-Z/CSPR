import argparse
import datetime
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.hppo_noshare import PPO_Hybrid
from agent.ppo2 import PPO_continuous
from cotp_intersection import raw_env
from normalization import Normalization


class Trainer(object):
    def __init__(self, args):
        self.seed = args.random_seed
        self.device = args.device
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
            num_seconds=1860,
            begin_time=60,
        )
        self.history = {}
        self.write = True

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
                agents[agent_policy_dict[agent]] = (
                    PPO_continuous(
                        obs_space.shape[0], act_space.shape[0], 1.0,
                        self.buffer_size, 128,
                        args.max_steps, 1e-4, 3e-4, self.gamma,
                        self.lambda_, self.eps_clip, self.epochs_update, self.coeff_dist_entropy, )
                )
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

    def evaluate(self, state_norm, agents, policy_mapping_fn, total_steps, writer=None):
        env_eva = raw_env(
            use_gui=False,
            sumo_config=args.sumocfg,
            sumo_seed=self.seed,
            num_seconds=1860,
            begin_time=60,
            # additional_sumo_cmd='--tripinfo-output data/cotp_1_10.xml --device.emissions.probability 1.0 ' +
            #                     '--emissions.volumetric-fuel true',
        )
        env_eva.CONNECTION_LABEL = -2
        env_eva.label = 'con' + str(env_eva.CONNECTION_LABEL)
        env_agents = env_eva.possible_agents
        states, infos = env_eva.reset()
        states_norm = {}
        for agent in env_eva.agents:
            agent_map = policy_mapping_fn(agent)
            states_norm[agent_map] = state_norm[agent_map](states[agent].reshape(-1), update=False)
        queue_len = []
        episode_reward = {agent: 0.0 for agent in self.env.possible_agents}
        episode_len = {agent: 0 for agent in self.env.possible_agents}
        done_ = False
        while not done_:
            actions = {}
            self.history = {agent: {} for agent in self.env.possible_agents}
            for agent in env_agents:
                if infos[agent]['agents_to_update']:
                    agent_map = policy_mapping_fn(agent)
                    observation_norm = states_norm[agent]
                    if agent.startswith('cav'):
                        action = agents[agent_map].evaluate(observation_norm)
                        action_pad = action * 3.0

                    elif agent.startswith('traffic'):
                        action_dis, action_con = agents[agent_map].evaluate(observation_norm)
                        action_con_pad = (action_con + 1) / 2 * (self.max_green - self.min_green) + self.min_green
                        action_pad = (action_dis, np.array(action_con_pad, dtype=np.int64))

                    actions[agent] = action_pad
            next_states, rewards, terminations, truncations, next_infos = env_eva.step(actions)
            done_ = terminations['__all__'] or truncations['__all__']
            infos = next_infos
            states_norm_ = {}
            for agent in env_agents:
                if infos[agent]['agents_to_update']:
                    episode_reward[agent] += rewards[agent]
                    episode_len[agent] += 1
                    agent_map = policy_mapping_fn(agent)
                    states_norm_[agent_map] = state_norm[agent_map](next_states[agent].reshape(-1), update=False)
            states_norm = states_norm_
            queue_len.append(next_infos['queue'])

        for key in episode_len:
            if episode_len[key] <= 0:
                episode_len[key] = 1
        veh_loss = np.mean(list(env_eva.temp['vehicle_loss'].values()))
        if writer:
            writer.add_scalar('test/queue', scalar_value=np.mean(queue_len), global_step=total_steps)
            writer.add_scalar('test/delay', scalar_value=veh_loss, global_step=total_steps)
            writer.add_scalar('test/pass_vehicles', scalar_value=len(env_eva.temp['vehicle_loss']),
                              global_step=total_steps)
            for agent in env_eva.possible_agents:
                writer.add_scalar(f'test/reward_{agent}', scalar_value=episode_reward[agent],
                                  global_step=total_steps)
        print('***evaluate***', veh_loss)
        env_eva.close()

    def train(self):
        if self.write:
            writer = SummaryWriter()

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
        # agents['traffic_light'].load('data_train/0109/policy/traffic_light/episode155_42+')
        # state_norm['traffic_light'].running_ms.load('data_train/0109/data/traffic_light/episode155_42+.npz')
        # agents['traffic_light'].agent_old.state_norm.load('data_train/0109/policy/traffic_light/episode155_42+.th')
        env = self.env
        episode = 0
        total_steps = 0
        """  TRAINING LOGIC  """
        while total_steps < self.max_steps:
            # collect an episode
            # with torch.no_grad():
            env_agents = env.possible_agents
            states, infos = env.reset()
            states_norm = {}
            for agent in env.agents:
                agent_map = policy_mapping_fn(agent)
                states_norm[agent_map] = state_norm[agent_map](states[agent].reshape(-1),
                                                               update=bool(infos[agent]['agents_to_update']))
            print(state_norm['traffic_light'].running_ms.mean)
            episode_step = 0
            queue_len = []
            self.history = {agent: {} for agent in self.env.possible_agents}
            episode_len = {agent: 0 for agent in self.env.possible_agents}
            episode_reward = {agent: 0.0 for agent in self.env.possible_agents}
            done = False
            while not done:
                # Every update, we will normalize the state_norm(the input of the actor_con and critic) by
                # mean and std retrieve from the last update's buf, in other word observations normalization
                actions = {}
                for agent in env_agents:
                    if infos[agent]['agents_to_update']:
                        agent_map = policy_mapping_fn(agent)
                        observation_norm = states_norm[agent]
                        if agent.startswith('cav'):
                            action, a_logprob = agents[agent_map].choose_action(observation_norm)
                            action_pad = action * 3.0
                            self.history[agent] = {'obs': observation_norm, 'act': action,
                                                   'logp_act': a_logprob}
                            actions[agent] = action_pad
                        elif agent.startswith('traffic'):
                            value_action_logp = agents[agent_map].select_action(observation_norm)
                            #  state_value, (action_dis, action_con), (log_prob_dis, log_prob_con)
                            value, (action_dis, action_con), (log_prob_dis, log_prob_con) = value_action_logp
                            action_con_pad = (action_con + 1) / 2 * (self.max_green - self.min_green) + self.min_green
                            action_pad = (action_dis, np.array(action_con_pad, dtype=np.int64))
                            self.history[agent] = {'obs': observation_norm, 'act_dis': action_dis,
                                                   'act_con': action_con,
                                                   'val': value, 'logp_act_dis': log_prob_dis,
                                                   'logp_act_con': log_prob_con}
                            actions[agent] = action_pad
                next_states, rewards, terminations, truncations, next_infos = env.step(actions)
                infos = next_infos
                done = terminations['__all__'] or truncations['__all__']

                states_norm_ = {}
                queue_len.append(next_infos['queue'])
                episode_step += 1

                for agent in env_agents:
                    agent_map = policy_mapping_fn(agent)
                    if infos[agent]['agents_to_update']:
                        states_norm_[agent_map] = state_norm[agent_map](next_states[agent].reshape(-1),
                                                                        update=bool(infos[agent]['agents_to_update']))
                        episode_reward[agent] += rewards[agent]
                        episode_len[agent] += 1
                        self.history[agent]['reward'] = rewards[agent]
                        if agent.startswith('cav'):
                            if len(self.history[agent]) >= 4:
                                agents[policy_mapping_fn(agent)].replay_buffer.add(
                                    self.history[agent]['obs'], self.history[agent]['act'],
                                    self.history[agent]['logp_act'], self.history[agent]['reward'],
                                    states_norm_[agent], done, 0
                                )
                        elif agent.startswith('traffic'):
                            if len(self.history[agent]) >= 7:
                                agents[policy_mapping_fn(agent)].buffer.store_hybrid(
                                    self.history[agent]['obs'], self.history[agent]['act_dis'],
                                    self.history[agent]['act_con'], self.history[agent]['reward'],
                                    self.history[agent]['val'],
                                    self.history[agent]['logp_act_dis'],
                                    self.history[agent]['logp_act_con']
                                )
                total_steps += 1
                states_norm = states_norm_
            episode += 1
            for agent, policy in agents.items():
                if not agent.startswith('traffic'):
                    # print(policy.replay_buffer.count)
                    if policy.replay_buffer.count > 512:
                        policy.update(total_steps)
                        policy.replay_buffer.clear()
                        if self.write:
                            writer.add_scalar(f'run/reward_{agent}', scalar_value=episode_reward[agent],
                                              global_step=episode)
                else:
                    policy.buffer.finish_path(0)
                    if policy.buffer.ptr > 256:
                        policy.update(64)
                        policy.buffer.clear()
                        if self.write:
                            writer.add_scalar(f'run/reward_{agent}', scalar_value=episode_reward[agent],
                                              global_step=episode)
            print('***', np.mean(list(env.temp['vehicle_loss'].values())))

            if episode % self.agent_save_freq == 0:
                for agent, policy in agents.items():
                    file_to_save_policy = os.path.join(f"{file_save}/policy/{agent}", f'episode{episode}_{self.seed}')
                    policy.save(file_to_save_policy)
                    if agent.startswith('traffic'):
                        policy.agent_old.state_norm.save(
                            f"{file_save}/policy/{agent}/episode{episode}_{self.seed}.th")
                print('-------------------------------------------------------------------')
                print('model saved')
                print('-------------------------------------------------------------------')
                for agent in agents.keys():
                    path_to_save_npz = os.path.join(f'{file_save}/data/{agent}',
                                                    f'episode{episode}_{self.seed}.npz')
                    state_norm[agent].running_ms.save(path_to_save_npz)
            if episode % 3 == 1:
                if self.write:
                    self.evaluate(state_norm, agents, policy_mapping_fn, total_steps, writer)

            if self.write:
                writer.add_scalar('run/episode_len', scalar_value=episode_step, global_step=episode)
                writer.add_scalar('run/queue', scalar_value=np.mean(queue_len), global_step=episode)
                writer.add_scalar('run/delay', scalar_value=np.mean(list(env.temp['vehicle_loss'].values())),
                                  global_step=episode)
                writer.add_scalar('run/filtered_delay',
                                  scalar_value=np.mean(list(env.temp['filtered_vehicle_delays'].values())),
                                  global_step=episode)
        env.close()

    def test(self):
        def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
            return agent_id

        file_save = f"data_train/0108/"
        agents, state_norm = self.initialize_agents(self.seed, policy_mapping_fn)
        # agents['traffic_light'].load('data_train/1018/policy/i_episode990_42_0')
        # state_norm['traffic_light'].running_ms.load('data_train/1018/i_episode912_42.npz')
        for agent in agents:
            policy_path = f"{file_save}/policy/{agent}/episode90_42"
            norm_path = f"{file_save}/data/{agent}/episode90_42.npz"
            agents[agent].load(policy_path)
            state_norm[agent].running_ms.load(norm_path)
            if agent.startswith('traffic'):
                agents[agent].agent_old.state_norm.load(f"{file_save}/policy/{agent}/episode90_42.th")

        env = raw_env(
            use_gui=True,
            sumo_config=args.sumocfg,
            sumo_seed=self.seed,
            num_seconds=1860,
            begin_time=60,
            additional_sumo_cmd='--tripinfo-output data/nl_1_30.xml --device.emissions.probability 1.0 ' +
                                '--emissions.volumetric-fuel true',
        )

        env_agents = env.possible_agents
        states, infos = env.reset()
        states_norm = {}
        for agent in env_agents:
            agent_map = policy_mapping_fn(agent)
            states_norm[agent_map] = state_norm[agent_map](states[agent].reshape(-1),
                                                           update=bool(infos[agent]['agents_to_update']))
        episode_step = 0
        queue_len = []
        episode_reward = {agent: 0.0 for agent in self.env.possible_agents}
        done = False
        while not done:
            actions = {}
            self.history = {agent: {} for agent in self.env.possible_agents}
            for agent in env_agents:
                if infos[agent]['agents_to_update']:
                    agent_map = policy_mapping_fn(agent)
                    observation_norm = states_norm[agent]
                    if agent.startswith('cav'):
                        action = agents[agent_map].evaluate(observation_norm)
                        action_pad = action * 3.0

                    elif agent.startswith('traffic'):
                        action_dis, action_con = agents[agent_map].evaluate(observation_norm)
                        action_con_pad = (action_con + 1) / 2 * (self.max_green - self.min_green) + self.min_green
                        action_pad = (action_dis, np.array(action_con_pad, dtype=np.int64))

                    actions[agent] = action_pad
            next_states, rewards, terminations, truncations, next_infos = env.step(actions)
            done = terminations['__all__'] or truncations['__all__']
            # print(next_states)
            states_norm_ = {}
            for agent in env_agents:
                agent_map = policy_mapping_fn(agent)
                states_norm_[agent_map] = state_norm[agent_map](next_states[agent].reshape(-1),
                                                                update=bool(infos[agent]['agents_to_update']))
            states_norm = states_norm_
            queue_len.append(next_infos['queue'])
            infos = next_infos
            episode_step += 1
            # print(reward[0])
        print('***', np.mean(list(env.temp['vehicle_loss'].values())))

        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device.')
    parser.add_argument('--max_steps', type=int, default=int(2e6), help='The max steps in training.')
    parser.add_argument('--buffer_size', type=int, default=4000, help='The maximum size of the PPOBuffer.')
    parser.add_argument('--batch_size', type=int, default=128, help='The sample batch size.')
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
    parser.add_argument('--v_iters', type=int, default=10,
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
                        help='The sumoconfig file path.')
    parser.add_argument('--num_stage', type=int, default=8)
    parser.add_argument('--max_green', type=int, default=60)
    parser.add_argument('--min_green', type=int, default=10)

    args = parser.parse_args()

    # args log
    current_date = datetime.date.today().strftime("%m%d")
    # current_date = 1119
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
    trainer.train()
    # trainer.test()

    # args_tuple = [[31], [32], [33], [34], [35], [36]]
    # pool = Pool(processes=6)
    # for arg in args_tuple:
    #     pool.apply_async(trainer.train, arg)
    # pool.close()
    # pool.join()

    exit(2000)
