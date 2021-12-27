import gym
import time

env = gym.make('Breakout-v0')

c_default_delay = 0.01
c_event_delay = 0.05
c_num_steps = 1500
c_log_interval = 100

obs = env.reset()

# Adapted from https://blog.paperspace.com/getting-started-with-openai-gym/

l_reward = 0.0
l_live = 5

def trigger_event(e_title):
    print("\n[EVENT]\t", e_title)
    time.sleep(c_event_delay)

def print_game_info(g_info, g_reward, step_n):
    print("[GAME]\t", "Reward:", g_reward, "Lives:", g_info["lives"], "Step:", step_n, "/", c_num_steps, "(Debug)")

for step in range(c_num_steps):

    action = env.action_space.sample()
    # action = 3

    obs, reward, done, info = env.step(action)

    env.render()

    l_reward = l_reward + reward

    if info["lives"] != l_live:
        l_live = info["lives"]
        trigger_event("Lost Live (Warn)")
        print_game_info(g_info=info, g_reward=l_reward, step_n=step)

    if info["lives"] == 0:
        l_reward = 0
        trigger_event("Died (Error)")
        print_game_info(g_info=info, g_reward=l_reward, step_n=step)

    if reward != 0:
        trigger_event("Reward +1 (Info)")
        print_game_info(g_info=info, g_reward=l_reward, step_n=step)

    if done:
        trigger_event("Done Game Loop (Info)")
        print_game_info(g_info=info, g_reward=l_reward, step_n=step)
        env.reset()

    time.sleep(c_default_delay)

    if step % c_log_interval == 0:
        print_game_info(g_info=info, g_reward=l_reward, step_n=step)

env.close()
