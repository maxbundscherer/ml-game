import gym
import time

# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

env = gym.make('CartPole-v0')

c_default_delay = 0.05

print("Actions", env.action_space)

obs = env.reset()

for _ in range(1000):

    n_step = env.action_space.sample()

    obs, reward, done, info = env.step(n_step)

    # print(reward)

    env.render()

    if done:
        print("WARN", "Reset Env")
        env.reset()

    time.sleep(c_default_delay)

env.close()
