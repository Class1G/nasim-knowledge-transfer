import hashlib

import scenario_gen
from nasim.envs.environment import NASimEnv


testset = set()

for seed in range(100):
    scenario = scenario_gen.generate_small_scenarios(seed = seed)
    env = NASimEnv(scenario, flat_actions = True, flat_obs = True)
    env.reset()
    full_obs = env.current_state.get_initial_observation(True).numpy_flat()
    testset.add(hashlib.sha256(full_obs.tobytes()).hexdigest())

assert len(testset) == 100


testset_part = set()

for seed in range(100):
    scenario = scenario_gen.generate_part_scenarios(seed = seed)
    env = NASimEnv(scenario, flat_actions = True, flat_obs = True)
    env.reset()
    full_obs = env.current_state.get_initial_observation(True).numpy_flat()
    testset_part.add(hashlib.sha256(full_obs.tobytes()).hexdigest())

assert len(testset_part) == 100


with open("hash_values.py", "w") as f:
    f.write(f"TESTSET = {testset!r}\n")
    f.write(f"TESTSET_PART = {testset_part!r}\n")
