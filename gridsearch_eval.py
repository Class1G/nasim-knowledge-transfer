from stable_baselines3 import A2C, PPO, DQN
from env_wrapper import BaseEnvWrapper as wrapper

from sb3_contrib import MaskablePPO
from mask_wrapper import ActionMasking_none_TEST

from tqdm import tqdm
import csv
import json
import os
import statistics


DEVICE =        "cpu"
EVAL_RUNS =     100
STARTING_SEED = 0       

MODEL_DIR =     "trained_agents/gridsearch"
RESULTS_DIR =   "results/gridsearch_eval"



# shared hyperparameters
learning_rates =        [0.3, 0.1, 0.01, 0.001, 0.0001]


# A2C + PPO
n_steps =               [256, 1024, 2048]
ent_coefs =             [0.1, 0.05, 0.001]

# A2C
vf_coefs =              [0.5, 0.1, 0.05]

# PPO
clip_ranges =           [0.5, 0.2, 0.05]

# DQN
learning_starts =       [0, 1000, 10000]
buffer_sizes =          [1000, 5000, 10000]
exploration_fractions = [0.1, 0.3, 0.5]


# -------------------------------

# A2C run
os.makedirs(RESULTS_DIR, exist_ok = True)
a2c_results_path = f"{RESULTS_DIR}/a2c_results.csv"

with open(a2c_results_path, "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["learning_rate", "n_steps", "ent_coef", "vf_coef", "mean_overall", "steps_overall", "goal_rate",
                      "mean_clean", "steps_clean", "reward_sd_clean", "steps_sd_clean", "runs"])

    for l in learning_rates:
        for n in n_steps:
            for e in ent_coefs:
                for vf in vf_coefs:

                    env = wrapper(seed = STARTING_SEED)
                    obs, _ = env.env.reset()

                    model_name = f"a2c_grS_L{str(l)}_N{str(n)}_E{str(e)}_VF{str(vf)}"
                    model_path = f"{MODEL_DIR}/a2c/{model_name}"

                    model = A2C.load(path = model_path, device = DEVICE)

                    print(f"  {model_name}")

                    runs = []
                    for r in tqdm(range(EVAL_RUNS)):
                        done =          False
                        truncated =     False
                        total_reward =  0
                        ep_steps =      0

                        while not done and not truncated:
                            action, _ = model.predict(obs, deterministic = True)
                            obs, reward, done, truncated, _ = env.step(int(action))
                            total_reward += reward
                            ep_steps += 1

                        runs.append([total_reward, ep_steps, done])

                        if r < EVAL_RUNS - 1:
                            obs, _ = env.reset()

                    env.close()

                    rewards = [run[0] for run in runs]
                    steps   = [run[1] for run in runs]
                    dones   = [run[2] for run in runs]

                    mean_overall  = round(sum(rewards) / len(rewards), 3)
                    steps_overall = round(sum(steps) / len(steps), 3)
                    goal_rate     = round(sum(dones) / len(dones), 3)
                    
                    clean_rewards = [r for r, s, d in runs if d]
                    clean_steps   = [s for r, s, d in runs if d]

                    if clean_rewards:
                        mean_clean =    round(statistics.mean(clean_rewards), 3)
                        steps_clean =   round(statistics.mean(clean_steps), 3)
                    else:
                        mean_clean =    float("nan")
                        steps_clean =   float("nan")

                    if len(clean_rewards) >= 2:
                        reward_sd_clean = round(statistics.stdev(clean_rewards), 3)
                        steps_sd_clean  = round(statistics.stdev(clean_steps), 3)
                    else:
                        reward_sd_clean = float("nan")
                        steps_sd_clean  = float("nan")

                    writer.writerow([l, n, e, vf, mean_overall, steps_overall, goal_rate,
                                      mean_clean, steps_clean, reward_sd_clean, steps_sd_clean, json.dumps(runs)])
                    
                    print(l, n, e, vf, " | ", mean_overall, steps_overall, goal_rate, " | ", mean_clean, steps_clean, reward_sd_clean, steps_sd_clean)
                    print()


#DQN runs
os.makedirs(RESULTS_DIR, exist_ok = True)
dqn_results_path = f"{RESULTS_DIR}/dqn_results.csv"

with open(dqn_results_path, "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["learning_rate", "learning_starts", "buffer_size", "exploration_fraction", "mean_overall", "steps_overall", "goal_rate",
                      "mean_clean", "steps_clean", "reward_sd_clean", "steps_sd_clean", "runs"])

    for l in learning_rates:
        for b in buffer_sizes:
            for ls in learning_starts:
                for ef in exploration_fractions:

                    env = wrapper(seed = STARTING_SEED)
                    obs, _ = env.env.reset()

                    model_name = f"dqn_grS_L{str(l)}_B{str(b)}_LS{str(ls)}_EF{ef}"
                    model_path = f"{MODEL_DIR}/dqn/{model_name}"

                    model = DQN.load(path = model_path, device = DEVICE)

                    print(f"  {model_name}")

                    runs = []
                    for r in tqdm(range(EVAL_RUNS)):
                        done =          False
                        truncated =     False
                        total_reward =  0
                        ep_steps =      0

                        while not done and not truncated:
                            action, _ = model.predict(obs, deterministic = True)
                            obs, reward, done, truncated, _ = env.step(int(action))
                            total_reward += reward
                            ep_steps += 1

                        runs.append([total_reward, ep_steps, done])

                        if r < EVAL_RUNS - 1:
                            obs, _ = env.reset()

                    env.close()

                    rewards = [run[0] for run in runs]
                    steps   = [run[1] for run in runs]
                    dones   = [run[2] for run in runs]

                    mean_overall  = round(sum(rewards) / len(rewards), 3)
                    steps_overall = round(sum(steps) / len(steps), 3)
                    goal_rate     = round(sum(dones) / len(dones), 3)
                    
                    clean_rewards = [r for r, s, d in runs if d]
                    clean_steps   = [s for r, s, d in runs if d]

                    if clean_rewards:
                        mean_clean =    round(statistics.mean(clean_rewards), 3)
                        steps_clean =   round(statistics.mean(clean_steps), 3)
                    else:
                        mean_clean =    float("nan")
                        steps_clean =   float("nan")

                    if len(clean_rewards) >= 2:
                        reward_sd_clean = round(statistics.stdev(clean_rewards), 3)
                        steps_sd_clean  = round(statistics.stdev(clean_steps), 3)
                    else:
                        reward_sd_clean = float("nan")
                        steps_sd_clean  = float("nan")

                    writer.writerow([l, ls, b, ef, mean_overall, steps_overall, goal_rate,
                                      mean_clean, steps_clean, reward_sd_clean, steps_sd_clean, json.dumps(runs)])

                    print(l, ls, b, ef, " | ", mean_overall, steps_overall, goal_rate, " | ", mean_clean, steps_clean, reward_sd_clean, steps_sd_clean)
                    print()


# PPO runs
os.makedirs(RESULTS_DIR, exist_ok = True)
ppo_results_path = f"{RESULTS_DIR}/ppo_results.csv"

with open(ppo_results_path, "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["learning_rate", "n_steps", "ent_coef", "clip_range", "mean_overall", "steps_overall", "goal_rate",
                      "mean_clean", "steps_clean", "reward_sd_clean", "steps_sd_clean", "runs"])

    for l in learning_rates:
        for n in n_steps:
            for e in ent_coefs:
                for c in clip_ranges:

                    env = wrapper(seed = STARTING_SEED)
                    obs, _ = env.env.reset()

                    model_name = f"ppo_grS_L{str(l)}_N{str(n)}_E{str(e)}_C{c}"
                    model_path = f"{MODEL_DIR}/ppo/{model_name}"

                    model = PPO.load(path = model_path, device = DEVICE)

                    print(f"  {model_name}")

                    runs = []
                    for r in tqdm(range(EVAL_RUNS)):
                        done =          False
                        truncated =     False
                        total_reward =  0
                        ep_steps =      0

                        while not done and not truncated:
                            action, _ = model.predict(obs, deterministic = True)
                            obs, reward, done, truncated, _ = env.step(int(action))
                            total_reward += reward
                            ep_steps += 1

                        runs.append([total_reward, ep_steps, done])

                        if r < EVAL_RUNS - 1:
                            obs, _ = env.reset()

                    env.close()

                    rewards = [run[0] for run in runs]
                    steps   = [run[1] for run in runs]
                    dones   = [run[2] for run in runs]

                    mean_overall  = round(sum(rewards) / len(rewards), 3)
                    steps_overall = round(sum(steps) / len(steps), 3)
                    goal_rate     = round(sum(dones) / len(dones), 3)

                    clean_rewards = [r for r, s, d in runs if d]
                    clean_steps   = [s for r, s, d in runs if d]

                    if clean_rewards:
                        mean_clean =    round(statistics.mean(clean_rewards), 3)
                        steps_clean =   round(statistics.mean(clean_steps), 3)
                    else:
                        mean_clean =    float("nan")
                        steps_clean =   float("nan")

                    if len(clean_rewards) >= 2:
                        reward_sd_clean = round(statistics.stdev(clean_rewards), 3)
                        steps_sd_clean  = round(statistics.stdev(clean_steps), 3)
                    else:
                        reward_sd_clean = float("nan")
                        steps_sd_clean  = float("nan")

                    writer.writerow([l, n, e, c, mean_overall, steps_overall, goal_rate,
                                      mean_clean, steps_clean, reward_sd_clean, steps_sd_clean, json.dumps(runs)])

                    print(l, n, e, c, " | ", mean_overall, steps_overall, goal_rate, " | ", mean_clean, steps_clean, reward_sd_clean, steps_sd_clean)
                    print()


#Maskable-PPO runs
os.makedirs(RESULTS_DIR, exist_ok = True)
mppo_results_path = f"{RESULTS_DIR}/mppo_results.csv"

with open(mppo_results_path, "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["learning_rate", "n_steps", "ent_coef", "clip_range", "mean_overall", "steps_overall", "goal_rate",
                      "mean_clean", "steps_clean", "reward_sd_clean", "steps_sd_clean", "runs"])

    for l in [0.01, 0.001, 0.0001]:
        for n in n_steps:
            for e in ent_coefs:
                for c in clip_ranges:

                    env = ActionMasking_none_TEST(seed = STARTING_SEED)
                    obs, _ = env.env.reset()

                    model_name = f"mppo_grS_L{str(l)}_N{str(n)}_E{str(e)}_C{c}"
                    model_path = f"{MODEL_DIR}/mppo/{model_name}"

                    model = MaskablePPO.load(path = model_path, device = DEVICE)

                    print(f"  {model_name}")

                    runs = []
                    for r in tqdm(range(EVAL_RUNS)):
                        done =          False
                        truncated =     False
                        total_reward =  0
                        ep_steps =      0

                        while not done and not truncated:
                            action, _ = model.predict(obs, deterministic = True, action_masks = env.action_masks())
                            obs, reward, done, truncated, _ = env.step(int(action))
                            total_reward += reward
                            ep_steps += 1

                        runs.append([total_reward, ep_steps, done])

                        if r < EVAL_RUNS - 1:
                            obs, _ = env.reset()

                    env.close()

                    rewards = [run[0] for run in runs]
                    steps   = [run[1] for run in runs]
                    dones   = [run[2] for run in runs]

                    mean_overall  = round(sum(rewards) / len(rewards), 3)
                    steps_overall = round(sum(steps) / len(steps), 3)
                    goal_rate     = round(sum(dones) / len(dones), 3)

                    clean_rewards = [r for r, s, d in runs if d]
                    clean_steps   = [s for r, s, d in runs if d]

                    if clean_rewards:
                        mean_clean =    round(statistics.mean(clean_rewards), 3)
                        steps_clean =   round(statistics.mean(clean_steps), 3)
                    else:
                        mean_clean =    float("nan")
                        steps_clean =   float("nan")

                    if len(clean_rewards) >= 2:
                        reward_sd_clean = round(statistics.stdev(clean_rewards), 3)
                        steps_sd_clean  = round(statistics.stdev(clean_steps), 3)
                    else:
                        reward_sd_clean = float("nan")
                        steps_sd_clean  = float("nan")

                    writer.writerow([l, n, e, c, mean_overall, steps_overall, goal_rate,
                                      mean_clean, steps_clean, reward_sd_clean, steps_sd_clean, json.dumps(runs)])

                    print(l, n, e, c, " | ", mean_overall, steps_overall, goal_rate, " | ", mean_clean, steps_clean, reward_sd_clean, steps_sd_clean)
                    print()