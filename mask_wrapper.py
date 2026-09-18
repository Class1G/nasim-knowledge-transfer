from sb3_contrib import MaskablePPO

from env_wrapper import BaseEnvWrapper, SmallEnvWrapper, PartEnvWrapper, BaseEnvWrapper_PART
from nasim.envs.action import Action
from nasim.envs.utils import AccessLevel
from nasim.envs.host_vector import HostVector
import numpy as np



# --- Reward Shaping Wrappers ---

class RewardShapingWrapper_naive (SmallEnvWrapper):
    
    # delete after manual testing
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flat_actions = self.env.flat_actions
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        a, rew, b, c, d = super().step(*args, **kwargs)
        if rew < 50:
            rew = 0
        return (a, rew, b, c, d)
    
class RewardShapingWrapper_hierarchical (SmallEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # delete after manual testing
        self.flat_actions = self.env.flat_actions
        
        self.os_scanned =       set()
        self.service_scanned =  set()
        self.exploited =        set()
        self.process_scanned =  set()
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        
        action = args[0] if isinstance(args[0], Action) else self.action_space.get_action(args[0])
        
        a, rew, b, c, info = super().step(*args, **kwargs)
        
        if info['success']:
            if action.is_os_scan() and (action.target not in self.os_scanned):
                self.os_scanned.add(action.target)
                rew += 20
            elif action.is_service_scan() and (action.target not in self.service_scanned):
                self.service_scanned.add(action.target)
                rew += 20
            elif action.is_exploit() and (action.target not in self.exploited):
                self.exploited.add(action.target)
                rew += 15
            elif action.is_process_scan() and (action.target not in self.process_scanned):
                self.process_scanned.add(action.target)
                rew += 10


        return (a, rew, b, c, info)

    def _reset_values(self):
        self.os_scanned =       set()
        self.service_scanned =  set()
        self.exploited =        set()
        self.process_scanned =  set()

    def reset(self, **kwargs):
        self._reset_values()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_values() 
        return super().reset_full(**kwargs)

class RewardShapingWrapper_active (SmallEnvWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # delete after manual testing
        self.flat_actions = self.env.flat_actions
        
        self.os_scanned =       set()
        self.service_scanned =  set()
        self.exploited =        set()
        self.process_scanned =  set()
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        
        action = args[0] if isinstance(args[0], Action) else self.action_space.get_action(args[0])
        a, rew, b, c, info = super().step(*args, **kwargs)
        
        # deduction if action is used obviously unnecessary (multiple times or no preconditions met)
        if info['success']:
            
            # if already exploited extra deduction
            if action.is_os_scan():
                if action.target not in self.os_scanned:
                    self.os_scanned.add(action.target)
                elif action.target not in self.exploited:
                    rew -= 20
                else: rew -= 10
                
            # if already exploited extra deduction
            elif action.is_service_scan():
                if action.target not in self.service_scanned:
                    self.service_scanned.add(action.target)
                elif action.target not in self.exploited:
                    rew -= 20
                else: rew -= 10
            
            elif action.is_exploit():
                if action.target not in self.exploited:
                    self.exploited.add(action.target)
                else: rew -= 10
            
            # if not exploited already more deduction    
            elif action.is_process_scan():
                if action.target not in self.process_scanned:
                    self.process_scanned.add(action.target)
                elif action.target not in self.exploited:
                    rew -= 20
                else: rew -= 10
                
        else:
            if action.is_exploit():
                if action.target in self.exploited:
                    rew -= 10

        return (a, rew, b, c, info)

    def _reset_values(self):
        self.os_scanned =       set()
        self.service_scanned =  set()
        self.exploited =        set()
        self.process_scanned =  set()

    def reset(self, **kwargs):
        self._reset_values()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_values() 
        return super().reset_full(**kwargs)
    
class RewardShapingWrapper_impulse (SmallEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # delete after manual testing
        self.flat_actions = self.env.flat_actions
        
        self.exploited =    set()
        self.mask =         [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        
        action = args[0] if isinstance(args[0], Action) else self.action_space.get_action(args[0])
        
        a, rew, b, c, info = super().step(*args, **kwargs)
        
        if info['success'] and action.is_exploit() and (action.target not in self.exploited):
                self.exploited.add(action.target)
                rew += 20

        return (a, rew, b, c, info)

    def _reset_values(self):
        self.exploited = set()

    def reset(self, **kwargs):
        self._reset_values()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_values() 
        return super().reset_full(**kwargs)    


# ---

class RewardShapingWrapper_naive_TEST (BaseEnvWrapper):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flat_actions = self.env.flat_actions
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        a, rew, b, c, d = super().step(*args, **kwargs)
        if rew < 50:
            rew = 0
        return (a, rew, b, c, d)

class RewardShapingWrapper_hierarchical_TEST (BaseEnvWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
       
        self.flat_actions = self.env.flat_actions
        
        self.os_scanned =       set()
        self.service_scanned =  set()
        self.exploited =        set()
        self.process_scanned =  set()
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        
        action = args[0] if isinstance(args[0], Action) else self.action_space.get_action(args[0])
        
        a, rew, b, c, info = super().step(*args, **kwargs)
        
        if info['success']:
            if action.is_os_scan() and (action.target not in self.os_scanned):
                self.os_scanned.add(action.target)
                rew += 20
            elif action.is_service_scan() and (action.target not in self.service_scanned):
                self.service_scanned.add(action.target)
                rew += 20
            elif action.is_exploit() and (action.target not in self.exploited):
                self.exploited.add(action.target)
                rew += 15
            elif action.is_process_scan() and (action.target not in self.process_scanned):
                self.process_scanned.add(action.target)
                rew += 10


        return (a, rew, b, c, info)

    def _reset_values(self):
        self.os_scanned =       set()
        self.service_scanned =  set()
        self.exploited =        set()
        self.process_scanned =  set()

    def reset(self, **kwargs):
        self._reset_values()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_values() 
        return super().reset_full(**kwargs)

class RewardShapingWrapper_active_TEST (BaseEnvWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flat_actions = self.env.flat_actions
        
        self.os_scanned =       set()
        self.service_scanned =  set()
        self.exploited =        set()
        self.process_scanned =  set()
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        
        action = args[0] if isinstance(args[0], Action) else self.action_space.get_action(args[0])
        a, rew, b, c, info = super().step(*args, **kwargs)
        
        # deduction if action is used obviously unnecessary (multiple times or no preconditions met)
        if info['success']:
            
            # if already exploited extra deduction
            if action.is_os_scan():
                if action.target not in self.os_scanned:
                    self.os_scanned.add(action.target)
                elif action.target not in self.exploited:
                    rew -= 20
                else: rew -= 10
                
            # if already exploited extra deduction
            elif action.is_service_scan():
                if action.target not in self.service_scanned:
                    self.service_scanned.add(action.target)
                elif action.target not in self.exploited:
                    rew -= 20
                else: rew -= 10
            
            elif action.is_exploit():
                if action.target not in self.exploited:
                    self.exploited.add(action.target)
                else: rew -= 10
            
            # if not exploited already more deduction    
            elif action.is_process_scan():
                if action.target not in self.process_scanned:
                    self.process_scanned.add(action.target)
                elif action.target not in self.exploited:
                    rew -= 20
                else: rew -= 10
                
        else:
            if action.is_exploit():
                if action.target in self.exploited:
                    rew -= 10

        return (a, rew, b, c, info)

    def _reset_values(self):
        self.os_scanned =       set()
        self.service_scanned =  set()
        self.exploited =        set()
        self.process_scanned =  set()

    def reset(self, **kwargs):
        self._reset_values()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_values() 
        return super().reset_full(**kwargs)



# --- Action Masking Wrappers ---

class ActionMasking_none (SmallEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()

class ActionMasking_possible (SmallEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [0] * self.env.action_space.n
        
    def action_masks(self):
        a_space = self.env.action_space
        for i in range(self.env.action_space.n):
            if self.mask[i] == 1:
                pass
            else:
                action = a_space.get_action(i)
                target = self.env.current_state.get_host(action.target)
                
                if not target.discovered:       # Host not reachable
                    pass
                else:
                    if action.is_os_scan() or action.is_service_scan() or action.is_exploit():
                        self.mask[i] = 1
                    elif target.access == AccessLevel.USER:
                        self.mask[i] = 1
        return self.mask.copy()
        
    def _reset_mask(self):
        self.mask = [0] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)

class ActionMasking_noduplicate (SmallEnvWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [1] * self.env.action_space.n

    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        a, b, c, d, info = super().step(*args, **kwargs)
        if info['success']:
            self.mask[args[0]] = 0
        
        return (a, b, c, d, info)
        
    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)
    
class ActionMasking_advanced (SmallEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_possible = [0] * self.env.action_space.n
        self.mask_duplicate = [1] * self.env.action_space.n
        
    def action_masks(self):
        a_space = self.env.action_space
        for i in range(self.env.action_space.n):
            if self.mask_possible[i] == 1:
                pass
            else:
                action = a_space.get_action(i)
                target = self.env.current_state.get_host(action.target)
                
                if not target.discovered:       # Host not reachable
                    pass
                else:
                    if action.is_os_scan() or action.is_service_scan() or action.is_exploit():
                        self.mask_possible[i] = 1
                    elif target.access == AccessLevel.USER:
                        self.mask_possible[i] = 1
        
        mask_combined = np.array(self.mask_duplicate) * np.array(self.mask_possible)
        return mask_combined.tolist().copy()
        
    def step(self, *args, **kwargs):
        a, b, c, d, info = super().step(*args, **kwargs)
        if info['success']:
            self.mask_duplicate[args[0]] = 0
        
        return (a, b, c, d, info)
        
    def _reset_mask(self):
        self.mask_possible = [0] * self.env.action_space.n
        self.mask_duplicate = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)    
    

# ---

class ActionMasking_none_TEST (BaseEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()
    
class ActionMasking_possible_TEST (BaseEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [0] * self.env.action_space.n
        
    def action_masks(self):
        a_space = self.env.action_space
        for i in range(self.env.action_space.n):
            if self.mask[i] == 1:
                pass
            else:
                action = a_space.get_action(i)
                target = self.env.current_state.get_host(action.target)
                
                if not target.discovered:       # Host not reachable
                    pass
                else:
                    if action.is_os_scan() or action.is_service_scan() or action.is_exploit():
                        self.mask[i] = 1
                    elif target.access == AccessLevel.USER:
                        self.mask[i] = 1
        return self.mask.copy()
        
    def _reset_mask(self):
        self.mask = [0] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)

class ActionMasking_noduplicate_TEST (BaseEnvWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [1] * self.env.action_space.n

    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        a, b, c, d, info = super().step(*args, **kwargs)
        if info['success']:
            self.mask[args[0]] = 0
        
        return (a, b, c, d, info)
        
    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)

class ActionMasking_advanced_TEST (BaseEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_possible = [0] * self.env.action_space.n
        self.mask_duplicate = [1] * self.env.action_space.n
        
    def action_masks(self):
        a_space = self.env.action_space
        for i in range(self.env.action_space.n):
            if self.mask_possible[i] == 1:
                pass
            else:
                action = a_space.get_action(i)
                target = self.env.current_state.get_host(action.target)
                
                if not target.discovered:       # Host not reachable
                    pass
                else:
                    if action.is_os_scan() or action.is_service_scan() or action.is_exploit():
                        self.mask_possible[i] = 1
                    elif target.access == AccessLevel.USER:
                        self.mask_possible[i] = 1
        
        mask_combined = np.array(self.mask_duplicate) * np.array(self.mask_possible)
        return mask_combined.tolist().copy()
        
    def step(self, *args, **kwargs):
        a, b, c, d, info = super().step(*args, **kwargs)
        if info['success']:
            self.mask_duplicate[args[0]] = 0
        
        return (a, b, c, d, info)
        
    def _reset_mask(self):
        self.mask_possible = [0] * self.env.action_space.n
        self.mask_duplicate = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs) 
    
    
# --- Transfer

class ActionMasking_none_PART (PartEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()

class ActionMasking_possible_PART (PartEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [0] * self.env.action_space.n
        
    def action_masks(self):
        a_space = self.env.action_space
        for i in range(self.env.action_space.n):
            if self.mask[i] == 1:
                pass
            else:
                action = a_space.get_action(i)
                target = self.env.current_state.get_host(action.target)
                
                if not target.discovered:       # Host not reachable
                    pass
                else:
                    if action.is_os_scan() or action.is_service_scan() or action.is_exploit():
                        self.mask[i] = 1
                    elif target.access == AccessLevel.USER:
                        self.mask[i] = 1
        return self.mask.copy()
        
    def _reset_mask(self):
        self.mask = [0] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)

class ActionMasking_noduplicate_PART (PartEnvWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [1] * self.env.action_space.n

    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        a, b, c, d, info = super().step(*args, **kwargs)
        if info['success']:
            self.mask[args[0]] = 0
        
        return (a, b, c, d, info)
        
    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)
    
class ActionMasking_advanced_PART (PartEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_possible = [0] * self.env.action_space.n
        self.mask_duplicate = [1] * self.env.action_space.n
        
    def action_masks(self):
        a_space = self.env.action_space
        for i in range(self.env.action_space.n):
            if self.mask_possible[i] == 1:
                pass
            else:
                action = a_space.get_action(i)
                target = self.env.current_state.get_host(action.target)
                
                if not target.discovered:       # Host not reachable
                    pass
                else:
                    if action.is_os_scan() or action.is_service_scan() or action.is_exploit():
                        self.mask_possible[i] = 1
                    elif target.access == AccessLevel.USER:
                        self.mask_possible[i] = 1
        
        mask_combined = np.array(self.mask_duplicate) * np.array(self.mask_possible)
        return mask_combined.tolist().copy()
        
    def step(self, *args, **kwargs):
        a, b, c, d, info = super().step(*args, **kwargs)
        if info['success']:
            self.mask_duplicate[args[0]] = 0
        
        return (a, b, c, d, info)
        
    def _reset_mask(self):
        self.mask_possible = [0] * self.env.action_space.n
        self.mask_duplicate = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)
  
    
# ---

class ActionMasking_none_PART_TEST (BaseEnvWrapper_PART):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [1] * self.env.action_space.n
        
    def action_masks(self):
        return self.mask.copy()

class ActionMasking_possible_PART_TEST (BaseEnvWrapper_PART):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [0] * self.env.action_space.n
        
    def action_masks(self):
        a_space = self.env.action_space
        for i in range(self.env.action_space.n):
            if self.mask[i] == 1:
                pass
            else:
                action = a_space.get_action(i)
                target = self.env.current_state.get_host(action.target)
                
                if not target.discovered:       # Host not reachable
                    pass
                else:
                    if action.is_os_scan() or action.is_service_scan() or action.is_exploit():
                        self.mask[i] = 1
                    elif target.access == AccessLevel.USER:
                        self.mask[i] = 1
        return self.mask.copy()
        
    def _reset_mask(self):
        self.mask = [0] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)

class ActionMasking_noduplicate_PART_TEST (BaseEnvWrapper_PART):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = [1] * self.env.action_space.n

    def action_masks(self):
        return self.mask.copy()
        
    def step(self, *args, **kwargs):
        a, b, c, d, info = super().step(*args, **kwargs)
        if info['success']:
            self.mask[args[0]] = 0
        
        return (a, b, c, d, info)
        
    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)
    
class ActionMasking_advanced_PART_TEST (BaseEnvWrapper_PART):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_possible = [0] * self.env.action_space.n
        self.mask_duplicate = [1] * self.env.action_space.n
        
    def action_masks(self):
        a_space = self.env.action_space
        for i in range(self.env.action_space.n):
            if self.mask_possible[i] == 1:
                pass
            else:
                action = a_space.get_action(i)
                target = self.env.current_state.get_host(action.target)
                
                if not target.discovered:       # Host not reachable
                    pass
                else:
                    if action.is_os_scan() or action.is_service_scan() or action.is_exploit():
                        self.mask_possible[i] = 1
                    elif target.access == AccessLevel.USER:
                        self.mask_possible[i] = 1
        
        mask_combined = np.array(self.mask_duplicate) * np.array(self.mask_possible)
        return mask_combined.tolist().copy()
        
    def step(self, *args, **kwargs):
        a, b, c, d, info = super().step(*args, **kwargs)
        if info['success']:
            self.mask_duplicate[args[0]] = 0
        
        return (a, b, c, d, info)
        
    def _reset_mask(self):
        self.mask_possible = [0] * self.env.action_space.n
        self.mask_duplicate = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        return super().reset(**kwargs)
        
    def reset_full(self, **kwargs):
        self._reset_mask() 
        return super().reset_full(**kwargs)

    
# --- Teacher Guidance Wrappers ---

class TeacherGuidance_start (SmallEnvWrapper):
    
    def __init__(self, start_steps = 5, **kwargs):
        super().__init__(**kwargs)
        self.n = self.env.action_space.n
        self._init_steps = start_steps
        self._remaining_steps = start_steps
        self.mask = [1] * self.env.action_space.n
        
        self.teacher = MaskablePPO.load(path = "teacher_models/mppo_grS_L0.001_N1024_E0.05_C0.2",     # best goal_reached in gridsearch evaluation
                                        device = "cpu")
        
    def action_masks(self):
        if self._remaining_steps <= 0:
            return self.mask.copy()
        else:
            mask = [0] * self.n
            action, _ = self.teacher.predict(self._last_flat_obs, deterministic = True)
            mask[int(action)] = 1
            return mask


    def step(self, *args, **kwargs):
            a, b, c, d, info = super().step(*args, **kwargs)
            self._last_flat_obs = a
            self._remaining_steps -= 1
            return (a, b, c, d, info)

    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        self._remaining_steps = self._init_steps
        obs, info = super().reset(**kwargs)
        self._last_flat_obs = obs
        return obs, info

    def reset_full(self, **kwargs):
        self._reset_mask()
        self._remaining_steps = self._init_steps
        obs, info = super().reset_full(**kwargs)
        self._last_flat_obs = obs
        return obs, info

class TeacherGuidance_end (SmallEnvWrapper):
    
    def __init__(self, after_steps = 10, **kwargs):
        super().__init__(**kwargs)
        self.n = self.env.action_space.n
        self._init_steps = after_steps
        self._remaining_steps = after_steps
        self.mask = [1] * self.env.action_space.n
        
        self.teacher = MaskablePPO.load(path = "teacher_models/mppo_grS_L0.001_N1024_E0.05_C0.2",     # best goal_reached in gridsearch evaluation
                                        device = "cpu")
        
    def action_masks(self):
        if self._remaining_steps > 0:
            return self.mask.copy()
        else:
            mask = [0] * self.n
            action, _ = self.teacher.predict(self._last_flat_obs, deterministic = True)
            mask[int(action)] = 1
            return mask


    def step(self, *args, **kwargs):
            a, b, c, d, info = super().step(*args, **kwargs)
            self._last_flat_obs = a
            self._remaining_steps -= 1
            return (a, b, c, d, info)

    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        self._remaining_steps = self._init_steps
        obs, info = super().reset(**kwargs)
        self._last_flat_obs = obs
        return obs, info

    def reset_full(self, **kwargs):
        self._reset_mask()
        self._remaining_steps = self._init_steps
        obs, info = super().reset_full(**kwargs)
        self._last_flat_obs = obs
        return obs, info

class TeacherGuidance_suggest_exploit (SmallEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = self.env.action_space.n
        self._exploits = []
        self.mask = [1] * self.env.action_space.n
        self.set_exploits()


    def set_exploits(self, **kwargs):
        exploits = []
        for i in range(self.env.action_space.n):
            action = self.env.action_space.get_action(i)
            if action.is_exploit():
                exploits.append((i, action))
        self._exploits = exploits


    def action_masks(self):
        for i, action in self._exploits:
            target_host =       action.target
            target_service =    action.service
            target_os =         action.os

            host_idx = self.env.current_state.get_host_idx(target_host)
            host_vec = HostVector(self._last_obs.tensor[host_idx])

            if host_vec.access >= AccessLevel.USER:
                continue

            service_visible = host_vec.is_running_service(target_service)
            os_visible = target_os is None or host_vec.is_running_os(target_os)
            if not (service_visible and os_visible):
                continue

            mask = [0] * self.n
            mask[i] = 1
            return mask
        return self.mask.copy()

        
    def step(self, *args, **kwargs):
            a, b, c, d, info = super().step(*args, **kwargs)
            self._last_obs = self.env.last_obs
            return (a, b, c, d, info)

    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset(**kwargs)
        self._last_obs = self.env.last_obs
        self._exploited = set()
        self.set_exploits()
        return obs, info

    def reset_full(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset_full(**kwargs)
        self._last_obs = self.env.last_obs
        self._exploited = set()
        self.set_exploits()
        return obs, info


# --- Transfer

class TeacherGuidance_start_PART (PartEnvWrapper):
    
    def __init__(self, start_steps = 5, **kwargs):
        super().__init__(**kwargs)
        self.n = self.env.action_space.n
        self._init_steps = start_steps
        self._remaining_steps = start_steps
        self.mask = [1] * self.env.action_space.n
        
        self.teacher = MaskablePPO.load(path = "teacher_models/mppo_grS_L0.001_N1024_E0.05_C0.2",     # best goal_reached in gridsearch evaluation
                                        device = "cpu")
        
    def action_masks(self):
        if self._remaining_steps <= 0:
            return self.mask.copy()
        else:
            mask = [0] * self.n
            action, _ = self.teacher.predict(self._last_flat_obs, deterministic = True)
            mask[int(action)] = 1
            return mask


    def step(self, *args, **kwargs):
            a, b, c, d, info = super().step(*args, **kwargs)
            self._last_flat_obs = a
            self._remaining_steps -= 1
            return (a, b, c, d, info)

    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        self._remaining_steps = self._init_steps
        obs, info = super().reset(**kwargs)
        self._last_flat_obs = obs
        return obs, info

    def reset_full(self, **kwargs):
        self._reset_mask()
        self._remaining_steps = self._init_steps
        obs, info = super().reset_full(**kwargs)
        self._last_flat_obs = obs
        return obs, info

class TeacherGuidance_suggest_nscan_PART (PartEnvWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = self.env.action_space.n
        self._scanned = set()
        self._compromised = set()
        self.mask = [1] * self.env.action_space.n


    def _find_scan_action(self, target_host):
        for i in range(self.n):
            action = self.env.action_space.get_action(i)
            if action.is_subnet_scan() and action.target == target_host:
                return i
        return None


    def action_masks(self):
        for target_host in self._compromised:
            if target_host in self._scanned:
                continue

            i = self._find_scan_action(target_host)
            if i is None:
                continue

            mask = [0] * self.n
            mask[i] = 1
            return mask
        return self.mask.copy()


    def step(self, *args, **kwargs):
            action = self.env.action_space.get_action(args[0])
            a, b, c, d, info = super().step(*args, **kwargs)
            if info['success']:
                if action.is_subnet_scan():
                    self._scanned.add(action.target)
                elif getattr(action, 'access', None) is not None and action.access >= AccessLevel.USER:
                    self._compromised.add(action.target)
            return (a, b, c, d, info)

    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset(**kwargs)
        self._scanned = set()
        self._compromised = set()
        return obs, info

    def reset_full(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset_full(**kwargs)
        self._scanned = set()
        self._compromised = set()
        return obs, info

class TeacherGuidance_suggest_exploit_PART (PartEnvWrapper):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = self.env.action_space.n
        self._exploits = []
        self.mask = [1] * self.env.action_space.n
        self.set_exploits()


    def set_exploits(self, **kwargs):
        exploits = []
        for i in range(self.env.action_space.n):
            action = self.env.action_space.get_action(i)
            if action.is_exploit():
                exploits.append((i, action))
        self._exploits = exploits


    def action_masks(self):
        for i, action in self._exploits:
            target_host =       action.target
            target_service =    action.service
            target_os =         action.os

            host_idx = self.env.current_state.get_host_idx(target_host)
            host_vec = HostVector(self._last_obs.tensor[host_idx])

            if host_vec.access >= AccessLevel.USER:
                continue

            service_visible = host_vec.is_running_service(target_service)
            os_visible = target_os is None or host_vec.is_running_os(target_os)
            if not (service_visible and os_visible):
                continue

            mask = [0] * self.n
            mask[i] = 1
            return mask
        return self.mask.copy()

        
    def step(self, *args, **kwargs):
            a, b, c, d, info = super().step(*args, **kwargs)
            self._last_obs = self.env.last_obs
            return (a, b, c, d, info)

    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset(**kwargs)
        self._last_obs = self.env.last_obs
        self._exploited = set()
        self.set_exploits()
        return obs, info

    def reset_full(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset_full(**kwargs)
        self._last_obs = self.env.last_obs
        self._exploited = set()
        self.set_exploits()
        return obs, info



# --- Transfer TEST

class TeacherGuidance_suggest_nscan_PART_TEST (BaseEnvWrapper_PART):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = self.env.action_space.n
        self._scanned = set()
        self._compromised = set()
        self.mask = [1] * self.env.action_space.n


    def _find_scan_action(self, target_host):
        for i in range(self.n):
            action = self.env.action_space.get_action(i)
            if action.is_subnet_scan() and action.target == target_host:
                return i
        return None


    def action_masks(self):
        for target_host in self._compromised:
            if target_host in self._scanned:
                continue

            i = self._find_scan_action(target_host)
            if i is None:
                continue

            mask = [0] * self.n
            mask[i] = 1
            return mask
        return self.mask.copy()


    def step(self, *args, **kwargs):
            action = self.env.action_space.get_action(args[0])
            a, b, c, d, info = super().step(*args, **kwargs)
            if info['success']:
                if action.is_subnet_scan():
                    self._scanned.add(action.target)
                elif getattr(action, 'access', None) is not None and action.access >= AccessLevel.USER:
                    self._compromised.add(action.target)
            return (a, b, c, d, info)

    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset(**kwargs)
        self._scanned = set()
        self._compromised = set()
        return obs, info

    def reset_full(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset_full(**kwargs)
        self._scanned = set()
        self._compromised = set()
        return obs, info

class TeacherGuidance_suggest_exploit_PART_TEST (BaseEnvWrapper_PART):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = self.env.action_space.n
        self._exploits = []
        self.mask = [1] * self.env.action_space.n
        self._last_obs = self.env.last_obs
        self.set_exploits()


    def set_exploits(self, **kwargs):
        exploits = []
        for i in range(self.env.action_space.n):
            action = self.env.action_space.get_action(i)
            if action.is_exploit():
                exploits.append((i, action))
        self._exploits = exploits


    def action_masks(self):
        for i, action in self._exploits:
            target_host =       action.target
            target_service =    action.service
            target_os =         action.os

            host_idx = self.env.current_state.get_host_idx(target_host)
            host_vec = HostVector(self._last_obs.tensor[host_idx])

            if host_vec.access >= AccessLevel.USER:
                continue

            service_visible = host_vec.is_running_service(target_service)
            os_visible = target_os is None or host_vec.is_running_os(target_os)
            if not (service_visible and os_visible):
                continue

            mask = [0] * self.n
            mask[i] = 1
            return mask
        return self.mask.copy()

        
    def step(self, *args, **kwargs):
            a, b, c, d, info = super().step(*args, **kwargs)
            self._last_obs = self.env.last_obs
            return (a, b, c, d, info)

    def _reset_mask(self):
        self.mask = [1] * self.env.action_space.n

    def reset(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset(**kwargs)
        self._last_obs = self.env.last_obs
        self._exploited = set()
        self.set_exploits()
        return obs, info

    def reset_full(self, **kwargs):
        self._reset_mask()
        obs, info = super().reset_full(**kwargs)
        self._last_obs = self.env.last_obs
        self._exploited = set()
        self.set_exploits()
        return obs, info
