import gym
import time

env = gym.make('CartPole-v0')

c_default_delay = 0.05

print("Actions", env.action_space)

obs = env.reset()

for _ in range(1000):

    n_step = env.action_space.sample()
    env.step(n_step)

    env.render()

    time.sleep(c_default_delay)

env.close()
