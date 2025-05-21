import gym
import numpy as np

def q_learning(t_max, gamma, beta, e_max):
    env = gym.make("Taxi-v3")

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    e = 0
    while e < e_max:
        t = 0
        state, _ = env.reset()
        done = False
        while t < t_max and not done:
            print(f"State: {state}")
            action = action_choose(state, Q)
            print(f"Action: {action}")
            next_state, reward, terminated, truncated, info= env.step(action)
            done = terminated or truncated
            row, col, passenger_loc, destination = env.unwrapped.decode(state)
            print(row, col, passenger_loc, destination)
            delta = reward + gamma * np.argmax(Q[next_state]) - Q[state][action]
            Q[state][action] += beta * delta
            t += 1
            state = next_state
        e += 1
    return Q

def action_choose(state, Q):
    if np.random.uniform() < 0.5:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return int(action)