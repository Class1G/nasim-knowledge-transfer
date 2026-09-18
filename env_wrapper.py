import gymnasium as gym
import hashlib

import scenario_gen
from nasim.envs.environment import NASimEnv
from hash_values import TESTSET, TESTSET_PART


# generated scenarios are transformed into useable environments.
# layer where hash check is applied


class BaseEnvWrapper (gym.Wrapper):
    def __init__(self, seed = 0, n_env_offset = 1):
        self._start_seed = seed
        self._seed = seed
        self._offset = n_env_offset
        self._seed, env = self._find_env(self._seed, skip_current = False)
        super().__init__(env)

    def _build_env(self, seed):
        scenario = scenario_gen.generate_small_scenarios(seed = seed)
        return NASimEnv(scenario, flat_actions = True, flat_obs = True)

    def _is_valid(self, env):
        return True

    def _find_env(self, seed, skip_current):
        """ builds a NASimEnv for the given seed, skipping seeds rejected by self._is_valid (incrementing by self._offset) """
        if skip_current:
            seed += self._offset
        while True:
            env = self._build_env(seed)
            if self._is_valid(env):
                return seed, env
            seed += self._offset

    def reset(self, **kwargs):
        """ generates new scenario with offset incremented seed (default = 1) """
        self._seed, self.env = self._find_env(self._seed, skip_current = True)
        return self.env.reset(**kwargs)

    def reset_full(self, **kwargs):
        """ resets seed back to initial seed (default = 0) """
        self._seed, self.env = self._find_env(self._start_seed, skip_current = False)
        return self.env.reset(**kwargs)

class SmallEnvWrapper (BaseEnvWrapper):
    def _is_valid(self, env):
        """ rejects seeds whose initial observation hash is in TESTSET """
        full_obs = env.current_state.get_initial_observation(True).numpy_flat()
        obs_hash = hashlib.sha256(full_obs.tobytes()).hexdigest()
        return obs_hash not in TESTSET


class BaseEnvWrapper_PART (gym.Wrapper):
    def __init__(self, seed = 0, n_env_offset = 1):
        self._start_seed = seed
        self._seed = seed
        self._offset = n_env_offset
        self._seed, env = self._find_env(self._seed, skip_current = False)
        super().__init__(env)

    def _build_env(self, seed):
        scenario = scenario_gen.generate_part_scenarios(seed = seed)
        return NASimEnv(scenario, flat_actions = True, flat_obs = True)

    def _is_valid(self, env):
        return True

    def _find_env(self, seed, skip_current):
        """ builds a NASimEnv for the given seed, skipping seeds rejected by self._is_valid (incrementing by self._offset) """
        if skip_current:
            seed += self._offset
        while True:
            env = self._build_env(seed)
            if self._is_valid(env):
                return seed, env
            seed += self._offset

    def reset(self, **kwargs):
        """ generates new scenario with offset incremented seed (default = 1) """
        self._seed, self.env = self._find_env(self._seed, skip_current = True)
        return self.env.reset(**kwargs)

    def reset_full(self, **kwargs):
        """ resets seed back to initial seed (default = 0) """
        self._seed, self.env = self._find_env(self._start_seed, skip_current = False)
        return self.env.reset(**kwargs)
    
class PartEnvWrapper (BaseEnvWrapper_PART):
    def _is_valid(self, env):
        """ rejects seeds whose initial observation hash is in TESTSET_PART """
        full_obs = env.current_state.get_initial_observation(True).numpy_flat()
        obs_hash = hashlib.sha256(full_obs.tobytes()).hexdigest()
        return obs_hash not in TESTSET_PART