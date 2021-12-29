import trainer.mlp_trainer_breakout_v0 as mlp_trainer

import gym
import time

env = gym.make('Breakout-v0')

c_default_delay = 0
c_event_delay = 0
c_num_steps = 1500000000
c_log_interval = 100
c_mlp_steps = 10

obs = env.reset()

model, optimizer, criterion, device = mlp_trainer.init_model()

l_reward = 1000.0
l_live = 5

# 0 = no action / 2 = right, 3 = left / 1 = required for init and restart
l_next_step = 1
l_max_game_reward = 0
l_max_single_reward = 0

def trigger_event(e_title):
    print("\n[EVENT]\t", e_title)
    time.sleep(c_event_delay)

def print_game_info(g_info, g_reward, step_n, max_global_reward):
    print("[GAME]\t", "Reward:", g_reward, "Lives:", g_info["lives"], "Step:", step_n, "/", c_num_steps, "Max Reward:", max_global_reward, "(Debug)")

for step in range(c_num_steps):

    obs, reward, done, info = env.step(l_next_step)

    if step % c_mlp_steps == 0:
        l_next_step = mlp_trainer.train_step(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            t_obs=obs,
            t_reward=l_reward,
        )

    env.render()

    if reward != 0:
        trigger_event("Reward +1 (Info)" + str(reward))
        print_game_info(g_info=info, g_reward=l_reward, step_n=step, max_global_reward=l_max_game_reward)
        l_reward = l_reward + 1
        l_max_single_reward = l_max_single_reward + 1

        if l_max_single_reward > l_max_game_reward:
            l_max_game_reward = l_max_single_reward

    if info["lives"] != l_live:
        trigger_event("Lost Live (Warn)")
        print_game_info(g_info=info, g_reward=l_reward, step_n=step, max_global_reward=l_max_game_reward)
        l_live = info["lives"]
        l_reward = l_reward - 1
        for _ in range(3):
            env.step(1)  # restart game (safe way)

    if info["lives"] == 0:
        trigger_event("Died (Error)")
        print_game_info(g_info=info, g_reward=l_reward, step_n=step, max_global_reward=l_max_game_reward)
        # l_reward = l_reward - 1
        l_max_single_reward = 0
        for _ in range(3):
            env.step(1)  # restart game (safe way)

    if l_reward < 0 or l_reward > 1000:
        l_reward = 1000.0

    if done:
        trigger_event("Done Game Loop (Info)")
        print_game_info(g_info=info, g_reward=l_reward, step_n=step, max_global_reward=l_max_game_reward)
        l_max_single_reward = 0
        env.reset()

    if step % c_log_interval == 0:
        print_game_info(g_info=info, g_reward=l_reward, step_n=step, max_global_reward=l_max_game_reward)

    time.sleep(c_default_delay)

env.close()
