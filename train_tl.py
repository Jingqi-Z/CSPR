import argparse
import datetime
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.hppo_noshare import PPO_Hybrid
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
            num_seconds=sim_steps,
            begin_time=60,
        )
        self.history = {}
        self.writer = True
        self.set_random_seeds()

    def set_random_seeds(self):
        """
        Sets all possible random seeds to results can be reproduced.
        """
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.cuda.manual_seed(self.seed)

    def train(self):
        writer = SummaryWriter()
        agent_id = 'traffic_light'
        file_save = f"data_train/{current_date}/"
        os.makedirs(f'{file_save}/data/{agent_id}', exist_ok=True)
        os.makedirs(f'{file_save}/policy/{agent_id}', exist_ok=True)

        obs_space, act_space = self.env.observation_space(agent_id), self.env.action_space(agent_id)
        agents = [
            PPO_Hybrid(obs_space.shape[0], act_space[0].n, self.mid_dim, self.lr_actor, self.lr_critic,
                       self.lr_decay_rate, self.buffer_size, self.target_kl_dis, self.target_kl_con,
                       self.gamma, self.lambda_, self.epochs_update, self.eps_clip, self.max_norm_grad,
                       self.coeff_dist_entropy, self.seed, self.device, self.lr_std, self.init_log_std,
                       )
        ]
        state_norm = Normalization(shape=obs_space.shape[0])
        # agents[0].load('data_train/0108/policy/traffic_light/i_episode550_42_0')
        # state_norm.running_ms.load('data_train/0108/data/traffic_light/i_episode550_42_modified.npz')
        # agents[0].agent_old.state_norm.load('data_train/0108/policy/traffic_light/i_episode550_42_0.th')
        env = self.env
        episode = 0
        self.history = {}
        total_steps = 0
        """  TRAINING LOGIC  """
        while total_steps < self.max_steps:
            # collect an episode
            # with torch.no_grad():
            env_agents = env.possible_agents
            states, infos = env.reset()
            observation_norm = state_norm(states[agent_id].reshape(-1),
                                          update=bool(infos[agent_id]['agents_to_update']))
            # print(state_norm.running_ms.mean)
            episode_step = 0
            queue_data = []
            queue_len = []
            tl_len = 0
            # queue_pre = env.get_queue_len()
            episode_reward = 0.0
            done = False
            while not done:
                # Every update, we will normalize the state_norm(the input of the actor_con and critic) by
                # mean and std retrieve from the last update's buf, in other word observations normalization
                actions = {}
                # self.history = {}
                for agent in env_agents:
                    if not infos[agent]['agents_to_update']:
                        continue
                    if agent.startswith('traffic'):
                        # queue_cur = env.get_queue_len()
                        if 'reward' in self.history:
                            agents[0].buffer.store_hybrid(
                                self.history['obs'], self.history['act_dis'],
                                self.history['act_con'], self.history['reward'],
                                self.history['val'],
                                self.history['logp_act_dis'],
                                self.history['logp_act_con']
                            )
                        # queue_pre = env.get_queue_len()
                        tl_len += 1
                        value_action_logp = agents[0].select_action(observation_norm)

                        #  state_value, (action_dis, action_con), (log_prob_dis, log_prob_con)
                        value, (action_dis, action_con), (log_prob_dis, log_prob_con) = value_action_logp
                        action_con_pad = (action_con + 1) / 2 * (self.max_green - self.min_green) + self.min_green
                        # if action_dis == 1:
                        #     action_con_pad = (action_con + 1) / 2 * (self.max_green - 12) + 12
                        action_pad = (action_dis, np.array(action_con_pad, dtype=np.int64))
                        self.history = {'obs': observation_norm, 'act_dis': action_dis,
                                        'act_con': action_con,
                                        'val': value, 'logp_act_dis': log_prob_dis,
                                        'logp_act_con': log_prob_con}
                        actions[agent] = action_pad
                        # print(actions)
                        total_steps += 1
                next_states, rewards, terminations, truncations, next_infos = env.step(actions)
                done = terminations['__all__'] or truncations['__all__']
                infos = next_infos
                if infos[agent_id]['agents_to_update']:
                    self.history['reward'] = rewards[agent_id]
                    # print(env.sim_time(), f"")
                    states_norm_ = state_norm(next_states[agent_id].reshape(-1),
                                              update=bool(infos[agent_id]['agents_to_update']))
                    observation_norm = states_norm_
                    queue_len.append(sum(next_infos['queue']))
                    queue_data.append(next_infos['queue'])
                episode_step += 1
                # for agent in env_agents:
                #     if not infos[agent]['agents_to_update']:
                #         continue
                #     if agent.startswith('traffic'):
                #         episode_reward += rewards[agent]
                #         agents[0].buffer.store_hybrid(
                #             self.history['obs'], self.history['act_dis'],
                #             self.history['act_con'], rewards[agent],
                #             self.history['val'],
                #             self.history['logp_act_dis'],
                #             self.history['logp_act_con']
                #         )
                # print(reward[0])

            episode += 1
            agents[0].buffer.finish_path(0)
            if agents[0].buffer.ptr > 256:
                agents[0].update(64)
                agents[0].buffer.clear()

                if self.writer:
                    writer.add_scalar('run/episode_len', scalar_value=episode_step, global_step=episode)
                    writer.add_scalar('run/queue', scalar_value=np.mean(queue_len), global_step=episode)
                    writer.add_scalar('run/duration', scalar_value=(sim_steps - 60) / tl_len, global_step=episode)
                    writer.add_scalar('run/delay', scalar_value=np.mean(list(env.temp['vehicle_loss'].values())),
                                      global_step=episode)
                    writer.add_scalar(f'run/reward_{agent_id}', scalar_value=episode_reward, global_step=episode)

            print('***', np.mean(list(env.temp['vehicle_loss'].values())))

            if episode % self.agent_save_freq == 0:
                file_to_save_policy = os.path.join(f"{file_save}/policy/{agent_id}", f'episode{episode}_{self.seed}+')
                agents[0].save(file_to_save_policy)
                agents[0].agent_old.state_norm.save(f"{file_save}/policy/{agent_id}/episode{episode}_{self.seed}+.th")
                print('-------------------------------------------------------------------')
                print('model saved')
                print('-------------------------------------------------------------------')
                path_to_save_npz = os.path.join(f'{file_save}/data/{agent_id}',
                                                f'episode{episode}_{self.seed}+.npz')
                state_norm.running_ms.save(path_to_save_npz)
            queue_data = np.array(queue_data)
            print(np.mean(queue_data, axis=0))
        env.close()

    def test(self):
        def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
            return agent_id

        env = raw_env(
            use_gui=True,
            sumo_config=args.sumocfg,
            sumo_seed=self.seed,
            num_seconds=1860,
            begin_time=60,
            additional_sumo_cmd='--tripinfo-output data/hppo_10.xml --device.emissions.probability 1.0 ' +
                                '--emissions.volumetric-fuel true',
        )
        agent_id = 'traffic_light'
        obs_space, act_space = env.observation_space(agent_id), env.action_space(agent_id)
        agents = [
            PPO_Hybrid(obs_space.shape[0], act_space[0].n, self.mid_dim, self.lr_actor, self.lr_critic,
                       self.lr_decay_rate, self.buffer_size, self.target_kl_dis, self.target_kl_con,
                       self.gamma, self.lambda_, self.epochs_update, self.eps_clip, self.max_norm_grad,
                       self.coeff_dist_entropy, self.seed, self.device, self.lr_std, self.init_log_std,
                       )
        ]
        state_norm = Normalization(shape=obs_space.shape[0])
        agents[0].load('data_train/0108/policy/traffic_light/i_episode550_42_0')
        state_norm.running_ms.load('data_train/0108/data/traffic_light/i_episode550_42_modified.npz')
        agents[0].agent_old.state_norm.load('data_train/0108/policy/traffic_light/i_episode550_42_0.th')
        total_steps = 0
        env_agents = env.possible_agents
        states, infos = env.reset()
        observation_norm = state_norm(states[agent_id],
                                      update=bool(infos[agent_id]['agents_to_update']))
        episode_step = 0
        queue_len = []
        episode_reward = 0.0
        done = False
        while not done:
            # Every update, we will normalize the state_norm(the input of the actor_con and critic) by
            # mean and std retrieve from the last update's buf, in other word observations normalization
            actions = {}
            self.history = {}
            for agent in env_agents:
                if infos[agent]['agents_to_update']:
                    if agent.startswith('traffic'):
                        # print(observation_norm)
                        action_dis, action_con = agents[0].evaluate(observation_norm)
                        action_con_pad = (action_con + 1) / 2 * (self.max_green - self.min_green) + self.min_green
                        action_pad = (action_dis, np.array(action_con_pad, dtype=np.int64))
                        actions[agent] = action_pad
                        # print(actions)
            next_states, rewards, terminations, truncations, next_infos = env.step(actions)
            # print(rewards[agent_id])
            done = terminations['__all__'] or truncations['__all__']
            states_norm_ = state_norm(next_states[agent_id].reshape(-1),
                                      update=bool(infos[agent_id]['agents_to_update']))
            observation_norm = states_norm_
            queue_len.append(next_infos['queue'])
            episode_step += 1
            # print(reward[0])
            infos = next_infos
            total_steps += 1
        print('***', np.mean(list(env.temp['vehicle_loss'].values())))

        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device.')
    parser.add_argument('--max_steps', type=int, default=int(4e6), help='The max steps in training.')
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
    sim_steps = 1860
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
