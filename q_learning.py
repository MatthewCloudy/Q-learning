import gym
import numpy as np

def q_learning(t_max, gamma, beta, e_max):
    env = gym.make("Taxi-v3")

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    e = 0
    while e < e_max:
        t = 0
        state = env.reset()
        state = state[0]
        print(state)
        done = False
        while t < t_max:
            action = action_choose(state, Q)
            print(action)
            observation, reward, terminated, truncated, info = env.step(action)
            print("Observation: ")
            print(observation)
            delta = reward + gamma * Q[observation][:] - Q[state][action]
            Q = Q + beta * delta
            t += 1
            state = observation
        e += 1
    return Q

def action_choose(state, Q):
    if np.random.uniform() < 0.1:
        action = np.argmax(Q[state])
    else:
        action = np.argmax(Q[state])
    return int(action)