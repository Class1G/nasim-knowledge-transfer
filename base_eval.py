from sb3_contrib import MaskablePPO
from mask_wrapper import *

from tqdm import tqdm
import csv
import json
import os
import statistics


# unmodified baseline models eval in PART, either trained in FULL or PART

DEVICE =        "cpu"
EVAL_RUNS =     100
STARTING_SEED = 0       

RESULTS_DIR =   "results/"


# hyperparameters

learning_rates =    [0.01, 0.001, 0.0001]
ent_coefs =         [0.1, 0.05, 0.001]
c =                 0.2
n =                 2048



os.makedirs(RESULTS_DIR, exist_ok = True)
rs_results_path = f"{RESULTS_DIR}/base_eval_toPart_results.csv"

with open(rs_results_path, "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["train_env", "learning_rate", "ent_coef", "mean_overall", "steps_overall", "goal_rate",
                      "mean_clean", "steps_clean", "reward_sd_clean", "steps_sd_clean", "runs"])


    for train_env in ["full", "part"]:  
        for l in learning_rates:
            for e in ent_coefs:

                env = ActionMasking_none_PART_TEST(seed = STARTING_SEED)
                obs, _ = env.env.reset()

                if train_env == "full":
                    model_name = f"mppo_grS_L{str(l)}_N{str(n)}_E{str(e)}_C{str(c)}"
                    model_path = f"trained_agents/gridsearch/mppo/{model_name}"
                    
                else:
                    model_name = f"mppo_BASE_PART_L{str(l)}_E{str(e)}"
                    model_path = f"trained_agents/base_part/{model_name}"

                model = MaskablePPO.load(path = model_path, device = DEVICE)

                print(f"  {model_name}")

                runs = []
                for r in tqdm(range(EVAL_RUNS)):
                    done = False
                    truncated = False
                    total_reward = 0
                    ep_steps = 0

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

                mean_overall  = sum(rewards) / len(rewards)
                steps_overall = sum(steps) / len(steps)
                goal_rate     = sum(dones) / len(dones)

                clean_rewards = [r for r, s, d in runs if d]
                clean_steps   = [s for r, s, d in runs if d]

                if clean_rewards:
                    mean_clean = round(statistics.mean(clean_rewards), 3)
                    steps_clean = round(statistics.mean(clean_steps), 3)
                else:
                    mean_clean = float("nan")
                    steps_clean = float("nan")

                if len(clean_rewards) >= 2:
                    reward_sd_clean = round(statistics.stdev(clean_rewards), 3)
                    steps_sd_clean  = round(statistics.stdev(clean_steps), 3)
                else:
                    reward_sd_clean = float("nan")
                    steps_sd_clean  = float("nan")

                writer.writerow([train_env, l, e, mean_overall, steps_overall, goal_rate,
                                    mean_clean, steps_clean, reward_sd_clean, steps_sd_clean, json.dumps(runs)])

                print(train_env, l, e, " | ", mean_overall, steps_overall, goal_rate, " | ", mean_clean, steps_clean, reward_sd_clean, steps_sd_clean)
                print()