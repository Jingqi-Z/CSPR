import argparse

import numpy as np
import torch

from agent.hppo_noshare import PPO_Hybrid
from tl_env import SumoEnv


def test(args):
    """

    :param worker_idx:
    :return:
    """
    seed = args.seed
    env = SumoEnv(yellow=3,
                  use_gui=True,
                  sumo_config="net/single-stage2.sumocfg",
                  sumo_seed=seed,
                  num_seconds=3600,
                  begin_time=30,
                  observation_pattern="queue",
                  min_green=20,
                  max_green=40,
                  )
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_name = env.metadata['name']
    state_dim = env.observation_space.shape[0]
    action_dis_dim = env.action_space[0].n
    max_action = env.action_space[1].high[0]
    min_action = env.action_space[1].low[0]
    gamma = 0.98
    lam = 0.8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_episode_steps = env.num_seconds  # Maximum number of steps per episode
    agent = PPO_Hybrid(state_dim, action_dis_dim, action_con_dim=1, target_kl_dis=0.08, target_kl_con=0.25,
                       random_seed=seed,  hidden_widths_critic=[64, 64], hidden_widths_actor=[128, 64, 32],
                       epochs_update=10, device=device,
                       )
    checkpoint_path = r'data_train/0930/i_episode295_42'
    agent.load(checkpoint_path)
    obs_norm = np.load('data_train/PPO_hybrid_env_traffic_light_number_1_seed_10.npz')
    # norm_mean = np.zeros(shape=(state_dim,))
    # norm_std = np.ones(shape=(state_dim,))
    norm_mean = obs_norm['mean']
    norm_std = obs_norm['std']
    def pad_action(act_dis, act_con):
        act_para = (act_con.item() + 1.0) / 2 * (max_action - min_action) + min_action
        return tuple((act_dis, np.array(act_para, dtype=int)))


    max_episodes = int(3e6)
    i_episode = 0

    with torch.no_grad():
        state, info = env.reset()
        episode_step = 0
        episode_reward = 0.0
        next_state = state
        agent_to_update = info['agents_to_update']
        queue_len = []
        while True:
            # Every update, we will normalize the state_norm(the input of the actor_con and critic) by
            # mean and std retrieve from the last update's buf, in other word observations normalization
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            # observation = state.reshape(self.num_agent, -1)
            observations_norm = (state - norm_mean) / np.maximum(norm_std, 1e-6)
            # Select action with policy
            value_action_logp = agent.select_action(observations_norm, is_test=True)
            action_dis, action_con = value_action_logp
            action = pad_action(action_dis, action_con)
            next_state, reward, done, truncated, info = env.step(action)
            episode_step += 1
            episode_reward += reward
            # print(reward[0])
            queue_len.append(info['queue'])

            # update observation
            state = next_state
            agents_to_update = info['agents_to_update']
            # for evaluation
            if info['terminated']:
                i_episode += 1
                break

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="PPO")  # Policy name
    parser.add_argument("--env", default="InvertedDoublePendulum-v1")  # OpenAI gym environment name
    # parser.add_argument("--env_name", default="Walker2d-v1")  # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=500, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_epiosides", default=40000, type=float)  # Max time steps to run environment for
    parser.add_argument("--max_timesteps", default=200000, type=float)  # Max time steps to run environment for

    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--epoch-steps", default=128, type=int)  # num of steps to collect for each training iteration
    parser.add_argument("--is-state-norm", default=0, type=int)  # is use state normalization
    parser.add_argument("--seed", default="42", type=int)  # Policy name
    parser.add_argument("--gpu-no", default='-1', type=str)  # Frequency of delayed policy updates
    args = parser.parse_args()
    test(args)
