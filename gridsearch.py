from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env_wrapper import SmallEnvWrapper as wrapper

from sb3_contrib import MaskablePPO
from mask_wrapper import ActionMasking_none


DEVICE =        "cpu"
TOTALSTEPS =    1000000
N_ENVS =        4

# shared hyperparameters
learning_rates =    [0.3, 0.1, 0.01, 0.001, 0.0001]


# A2C + PPO
n_steps =           [256, 1024, 2048]
ent_coefs =         [0.1, 0.05, 0.001]

# A2C
vf_coefs =          [0.5, 0.1, 0.05]

# PPO
clip_ranges =       [0.5, 0.2, 0.05]


# DQN
learning_starts =       [0, 1000, 10000]
buffer_sizes =          [1000, 5000, 10000]
exploration_fractions = [0.1, 0.3, 0.5]


#MPPO
mppo_learning_rates =   [0.01, 0.001, 0.0001]


# -------------------------------

# A2C training

for l in learning_rates:
    for n in n_steps:
        for e in ent_coefs:
            for vf in vf_coefs:
                vec_env = DummyVecEnv([lambda i = i: Monitor(wrapper(seed = i, n_env_offset = N_ENVS)) for i in range(N_ENVS)])

                model_name = f"a2c_grS_L{str(l)}_N{str(n)}_E{str(e)}_VF{str(vf)}"

                print(f"\n{'='*60}")
                print(f"  {model_name}")
                print(f"{'='*60}")

                model = A2C('MlpPolicy',
                            vec_env,
                            learning_rate =     l,
                            n_steps =           n,
                            ent_coef =          e,
                            vf_coef =           vf,
                            tensorboard_log =   f"./logs/gridsearch/a2c/{model_name}", verbose = 0, device = DEVICE)

                model.learn(total_timesteps = TOTALSTEPS, tb_log_name = model_name, progress_bar = True)
                model.save(f"trained_agents/gridsearch/a2c/{model_name}")
                vec_env.close()


# PPO

for l in learning_rates:
    for n in n_steps:
        for e in ent_coefs:
            for c in clip_ranges:
                vec_env = DummyVecEnv([lambda i = i: Monitor(wrapper(seed = i, n_env_offset = N_ENVS)) for i in range(N_ENVS)])

                model_name = f"ppo_grS_L{str(l)}_N{str(n)}_E{str(e)}_C{c}"

                print(f"\n{'='*60}")
                print(f"  {model_name}")
                print(f"{'='*60}")

                model = PPO('MlpPolicy',
                            vec_env,
                            learning_rate =     l,
                            n_steps =           n,
                            ent_coef =          e,
                            clip_range =        c,
                            tensorboard_log =   f"./logs/gridsearch/ppo/{model_name}", verbose = 0, device = DEVICE)

                model.learn(total_timesteps = TOTALSTEPS, tb_log_name = model_name, progress_bar = True)
                model.save(f"trained_agents/gridsearch/ppo/{model_name}")
                vec_env.close()


# DQN

for l in learning_rates:
    for b in buffer_sizes:
        for ls in learning_starts:
            for ef in exploration_fractions:
                vec_env = DummyVecEnv([lambda i = i: Monitor(wrapper(seed = i, n_env_offset = N_ENVS)) for i in range(N_ENVS)])

                model_name = f"dqn_grS_L{str(l)}_B{str(b)}_LS{str(ls)}_EF{ef}"

                print(f"\n{'='*60}")
                print(f"  {model_name}")
                print(f"{'='*60}")

                model = DQN('MlpPolicy',
                            vec_env,
                            learning_rate =         l,
                            learning_starts=        ls,
                            buffer_size =           b,
                            exploration_fraction =  ef,
                            tensorboard_log =       f"./logs/gridsearch/dqn/{model_name}", verbose = 0, device = DEVICE)

                model.learn(total_timesteps = TOTALSTEPS, tb_log_name = model_name, progress_bar = True)
                model.save(f"trained_agents/gridsearch/dqn/{model_name}")
                vec_env.close()


#Maskable-PPO

for l in mppo_learning_rates:
    for n in n_steps:
        for e in ent_coefs:
            for c in clip_ranges:
                vec_env = DummyVecEnv([lambda i = i: Monitor(ActionMasking_none(seed = i, n_env_offset = N_ENVS)) for i in range(N_ENVS)])

                model_name = f"mppo_grS_L{str(l)}_N{str(n)}_E{str(e)}_C{c}"

                print(f"\n{'='*60}")
                print(f"  {model_name}")
                print(f"{'='*60}")

                model = MaskablePPO('MlpPolicy',
                            vec_env,
                            learning_rate =     l,
                            n_steps =           n,
                            ent_coef =          e,
                            clip_range =        c,
                            tensorboard_log =   f"./logs/gridsearch/mppo/{model_name}", verbose = 0, device = DEVICE)

                model.learn(total_timesteps = TOTALSTEPS, tb_log_name = model_name, progress_bar = True)
                model.save(f"trained_agents/gridsearch/mppo/{model_name}")
                vec_env.close()