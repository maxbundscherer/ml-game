import gym
import time

env = gym.make('Breakout-v0')

num_steps = 1500

obs = env.reset()

# Adapted from https://blog.paperspace.com/getting-started-with-openai-gym/

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs)
    action = env.action_space.sample()

    # apply the action
    obs, reward, done, info = env.step(action)

    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.05)

    # If the epsiode is up, then start another one
    if done:
        env.reset()

env.close()
