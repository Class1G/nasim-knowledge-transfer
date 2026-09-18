from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from mask_wrapper import *


DEVICE =        "cpu"
TOTALSTEPS =    1000000
N_ENVS =        4

# hyperparameters

learning_rates =    [0.01, 0.001, 0.0001]
ent_coefs =         [0.1, 0.05, 0.001]
c =                 0.2
n =                 2048


for l in learning_rates:
    for e in ent_coefs:
        for func, name in [ (ActionMasking_noduplicate, "nodupe"),
                            (ActionMasking_possible,    "poss"),
                            (ActionMasking_advanced,    "adv")]:
            
            vec_env = DummyVecEnv([lambda i = i: Monitor(func(seed = i, n_env_offset = N_ENVS)) for i in range(N_ENVS)])

            model_name = f"mppo_AM_{name}_L{str(l)}_E{str(e)}"

            print(f"\n{'='*60}")
            print(f"  {model_name}")
            print(f"{'='*60}")

            model = MaskablePPO('MlpPolicy',
                        vec_env,
                        learning_rate =     l,
                        n_steps =           n,
                        ent_coef =          e,
                        clip_range =        c,
                        tensorboard_log =   f"./logs/action_masking/{model_name}", verbose = 0, device = DEVICE)

            model.learn(total_timesteps = TOTALSTEPS, tb_log_name = model_name, progress_bar = True)
            model.save(f"trained_agents/action_masking/{model_name}")
            vec_env.close()