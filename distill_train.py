from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sb3_distill import ProximalPolicyDistillation

from env_wrapper import SmallEnvWrapper


DEVICE =        "cpu"
TOTALSTEPS =    100000

learning_rates =    [0.01, 0.001, 0.0001]
ent_coefs =         [0.1, 0.05, 0.001]
c =                 0.2
n =                 2048

SPLITS = [(100, 100), (50, 100), (100, 50), (50, 50), (25, 25), (10, 10)]   

# (10, 10) is equal to (9, 9) in resulting dimensions for 64 -> 6
# (9, 9) split is closer unrounded to 6 with 5.76 as (10, 10) is

# resulting discrepancy -> actual dimension is [6, 6] but code uses (10, 10), thesis uses (9, 9)


BASE_WIDTH = 64

teacher_model = PPO.load(path = "teacher_models/ppo_grS_L0.001_N1024_E0.05_C0.2", device = DEVICE)    # best ppo model



for a_split, c_split in SPLITS:
    pi = [round(BASE_WIDTH * a_split / 100)] * 2
    vf = [round(BASE_WIDTH * c_split / 100)] * 2
    policy_kwargs = dict(net_arch = dict(pi = pi, vf = vf))

    for l in learning_rates:
        for e in ent_coefs:
        
            model_name = f"ppo_PPD_FULL_A{a_split}_C{c_split}_L{l}_E{e}"

            print(f"\n{'='*60}")
            print(f"  {model_name}")
            print(f"{'='*60}")
            
            env = Monitor(SmallEnvWrapper(seed = 0))

            model = ProximalPolicyDistillation('MlpPolicy',
                        env,
                        learning_rate =     l,
                        n_steps =           n,
                        ent_coef =          e,
                        clip_range =        c,
                        tensorboard_log =   f"./logs/distillation/{model_name}", verbose = 0, device = DEVICE)
            
            model.set_teacher(teacher_model, distill_lambda = 1)

            model.learn(total_timesteps = TOTALSTEPS, tb_log_name = model_name, progress_bar = True)
            model.save(f"trained_agents/distillation/{model_name}")
            env.close()