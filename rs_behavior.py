from sb3_contrib import MaskablePPO

from mask_wrapper import ActionMasking_none_TEST


DEVICE = "cpu"
SEED =   0      


for var, model_path in [("base", "trained_agents/gridsearch/mppo/mppo_grS_L0.001_N2048_E0.001_C0.2"),
                        ("hier", "trained_agents/reward_shaping/mppo_RS_hier_L0.001_E0.001")]:
    model = MaskablePPO.load(model_path, device = DEVICE)

    env = ActionMasking_none_TEST(seed = SEED)
    obs, _ = env.env.reset()
    
    print(f"\n{'='*70}")
    print(f"MODEL: {var}  ({model_path})")
    print(f"{'='*70}")
    

    done = False
    truncated = False
    step = 0
    total_reward = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic = True, action_masks = env.action_masks())
        obs, reward, done, truncated, info = env.step(int(action))

        step += 1
        total_reward += reward

        action_obj = env.env.action_space.get_action(int(action))
        status = "OK  " if info["success"] else "FAIL"
        print(f"  {step:3d}. [{status}] {action_obj}  (reward={reward:+.2f})")

    env.close()
    print(f"  {'-'*70}")
    print(f"  goal reached = {done}, steps = {step}, total_reward = {total_reward:.2f}")

