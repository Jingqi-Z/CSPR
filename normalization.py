import os

import numpy as np


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

    def save(self, file_path):
        np.savez(file_path, mean=self.mean, std=self.std, n=self.n, S=self.S)

    def load(self, file_path):
        data = np.load(file_path)
        self.mean = data['mean']
        self.std = data['std']
        self.n = data['n']
        self.S = data['S']


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-6)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-6)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


if __name__ == '__main__':
    # obs_norm = Normalization(shape=3)
    # obs_norm(np.array([1, 2, 3]))
    # obs_norm(np.array([2, 4, 5]))
    # print(obs_norm.running_ms.mean)
    reward_scaling = RewardScaling(shape=1, gamma=0.98)
    reward_scaling.reset()
    r=-100
    r = reward_scaling(r)
    print(reward_scaling(-200))
    print(reward_scaling(-200))
