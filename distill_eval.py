from stable_baselines3 import PPO
from tqdm import tqdm

import os
import csv

from mask_wrapper import ActionMasking_none_TEST


DEVICE =        "cpu"
EVAL_RUNS =     100
STARTING_SEED = 0

RESULTS_DIR =   "results"


learning_rates = [0.01, 0.001, 0.0001]
ent_coefs =      [0.1, 0.05, 0.001]
SPLITS =         [(100, 100), (50, 100), (100, 50), (50, 50), (25, 25), (10, 10)]




os.makedirs(RESULTS_DIR, exist_ok = True)
results_path = f"{RESULTS_DIR}/distill_results.csv"

with open(results_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["actor_split", "critic_split", "learning_rate", "ent_coef", "goal_rate"])

    for a_split, c_split in SPLITS:
        for l in learning_rates:
            for e in ent_coefs:
        
                model_name = f"ppo_PPD_FULL_A{a_split}_C{c_split}_L{l}_E{e}"
        
                model = PPO.load(path = f"trained_agents/distillation/{model_name}", device = DEVICE)

                env = ActionMasking_none_TEST(seed = STARTING_SEED)
                obs, _ = env.env.reset()

                goal_reached = 0
                for r in tqdm(range(EVAL_RUNS)):
                    done = False
                    truncated = False

                    while not done and not truncated:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, _ = env.step(int(action))

                    if done:
                        goal_reached += 1

                    if r < EVAL_RUNS - 1:
                        obs, _ = env.reset()

                env.close()

                goal_rate = goal_reached / EVAL_RUNS
                writer.writerow([a_split, c_split, l, e, goal_rate])

                print(f"{model_name}  |  goal_rate = {goal_rate:.3f}")
