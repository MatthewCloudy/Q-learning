import gym
import numpy as np

def q_learning(t_max, gamma, beta):
    env = gym.make("Taxi-v3")
    t = 0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    state = env.reset()
    while t < t_max:
        action = action_choose(state)
        next_observation, reward, terminated, truncated, info = env.step(action)
    return Q

def action_choose(state):
    return state.action_space.sample()