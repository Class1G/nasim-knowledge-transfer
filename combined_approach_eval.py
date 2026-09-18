from sb3_contrib import MaskablePPO
from mask_wrapper import *

from tqdm import tqdm
import csv
import json
import os
import statistics


DEVICE =        "cpu"
EVAL_RUNS =     100
STARTING_SEED = 0

MODEL_DIR =     "trained_agents/reward_shaping"
RESULTS_DIR =   "results/combined_approach"



# combined approach - impulse models tested without and with each mask on FULL


# hyperparameters

learning_rates =    [0.01, 0.001, 0.0001]
ent_coefs =         [0.1, 0.05, 0.001]
c =                 0.2
n =                 2048



os.makedirs(RESULTS_DIR, exist_ok = True)

for mask_type, function in [("none",    ActionMasking_none_TEST),
                            ("nodupe",  ActionMasking_noduplicate_TEST),
                            ("poss",    ActionMasking_possible_TEST),
                            ("adv",     ActionMasking_advanced_TEST)]:

    print(f"MASK WRAPPER: {mask_type}")
    print()

    rs_results_path = f"{RESULTS_DIR}/RS_impulse_results_{mask_type}.csv"

    with open(rs_results_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["mask_type", "learning_rate", "ent_coef", "mean_overall", "steps_overall", "goal_rate",
                          "mean_clean", "steps_clean", "reward_sd_clean", "steps_sd_clean", "runs"])


        for l in learning_rates:
            for e in ent_coefs:

                env = function(seed = STARTING_SEED)
                obs, _ = env.env.reset()

                model_name = f"mppo_RS_impulse_L{str(l)}_E{str(e)}"
                model_path = f"{MODEL_DIR}/{model_name}"

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
                    steps_sd_clean = float("nan")

                writer.writerow([mask_type, l, e, mean_overall, steps_overall, goal_rate,
                                    mean_clean, steps_clean, reward_sd_clean, steps_sd_clean, json.dumps(runs)])

                print(mask_type, l, e, " | ", mean_overall, steps_overall, goal_rate, " | ", mean_clean, steps_clean, reward_sd_clean, steps_sd_clean)
                print()
