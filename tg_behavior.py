from collections import Counter

from sb3_contrib import MaskablePPO

from mask_wrapper import ActionMasking_none_TEST


DEVICE =        "cpu"
STARTING_SEED = 0
EVAL_RUNS =     100


MODEL_PATH = "trained_agents/teacher_guidance/mppo_TG_END_L0.001_E0.001"


model = MaskablePPO.load(MODEL_PATH, device = DEVICE)
env = ActionMasking_none_TEST(seed = STARTING_SEED)

action_counts = Counter()

obs, _ = env.env.reset()
for r in range(EVAL_RUNS):
    done = False
    truncated = False

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic = True, action_masks = env.action_masks())
        obs, reward, done, truncated, info = env.step(int(action))

        action_obj = env.env.action_space.get_action(int(action))
        action_counts[type(action_obj).__name__] += 1

    if r < EVAL_RUNS - 1:
        obs, _ = env.reset()

env.close()

for action_type, count in action_counts.items():
    print(f"{action_type}: {count}")

