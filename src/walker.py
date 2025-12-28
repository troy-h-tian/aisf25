import os
import time
import json
import yaml
import imageio
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv




# ---------------------
# CONFIGURATION
# ---------------------
CONFIG_FILE = ""
LOAD_FROM_MODEL = None


# ---------------------
# REWARD WRAPPER
# ---------------------
class RewardShapingWrapper(gym.Wrapper):
    
    def step(self, action):
        
        obs, reward, terminated, truncated, info = self.env.step(action)

        hull_angle = float(obs[0])
        reward -= 0.1 * abs(hull_angle)

        vx = self.env.unwrapped.hull.linearVelocity.x
        vy = self.env.unwrapped.hull.linearVelocity.y
        reward += 0.05 * np.clip(vx, -1.0, 2.0)
        reward -= 0.02 * max(0.0, abs(vy) - 1.0)

        legs = self.env.unwrapped.legs
        dx = abs(legs[1].position[0] - legs[3].position[0])
        reward -= 0.1 * max(0.0, dx - 0.6)

        if not (terminated or truncated):
            reward += 0.01


        return obs, reward, terminated, truncated, info


# ---------------------
# CONFIG LOADING
# ---------------------
def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file '{path}' not found.")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------
# ENV FACTORIES (IMPORTANT: create env INSIDE factory)
# ---------------------
def make_train_env():
    def _init():
        env = gym.make("BipedalWalker-v3")  # no render_mode during training
        env = RewardShapingWrapper(env)
        env = Monitor(env)
        return env
    return _init


def make_eval_env():
    def _init():
        env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        env = RewardShapingWrapper(env)
        env = Monitor(env)
        return env
    return _init


# ---------------------
# TRAINING
# ---------------------
def train_model() -> str:
    cfg = load_config(CONFIG_FILE)

    n_envs = int(cfg.pop("n_envs", 1))   # <-- add this line


    total_timesteps = int(cfg.pop("total_timesteps", 100_000))
    exp_name = str(cfg.pop("experiment_name", "experiment"))

    # VecNormalize flags (pop them so they don't get passed to PPO)
    use_norm_obs = bool(cfg.pop("normalize_obs", False))
    use_norm_rew = bool(cfg.pop("normalize_reward", False))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{exp_name}_{timestamp}"
    tb_dir = f"./ppo_walker_tensorboard/{run_name}/"
    os.makedirs(tb_dir, exist_ok=True)

    new_model_name = f"ppo_bipedal_{run_name}"

    # Build vec env correctly
    vec_env = SubprocVecEnv([make_train_env() for _ in range(n_envs)])

    # Apply or load normalization
    if use_norm_obs or use_norm_rew:
        if LOAD_FROM_MODEL and os.path.exists(f"{LOAD_FROM_MODEL}.pkl"):
            print(f"--- LOADING VecNormalize STATS: {LOAD_FROM_MODEL}.pkl ---")
            vec_env = VecNormalize.load(f"{LOAD_FROM_MODEL}.pkl", vec_env)
            vec_env.training = True
            vec_env.norm_obs = use_norm_obs
            vec_env.norm_reward = use_norm_rew
        else:
            print(f"--- CREATING VecNormalize (Obs={use_norm_obs}, Rew={use_norm_rew}) ---")
            vec_env = VecNormalize(vec_env, norm_obs=use_norm_obs, norm_reward=use_norm_rew, clip_obs=10.0)

    # Load or create model
    if LOAD_FROM_MODEL and os.path.exists(f"{LOAD_FROM_MODEL}.zip"):
        print(f"--- RESUMING FROM: {LOAD_FROM_MODEL}.zip ---")
        model = PPO.load(f"{LOAD_FROM_MODEL}.zip", env=vec_env, tensorboard_log=tb_dir)
    else:
        print(f"--- STARTING FRESH: {run_name} ---")
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=tb_dir, **cfg)

    # Save config used
    config_record = dict(cfg)
    config_record.update({
        "experiment_name": exp_name,
        "total_timesteps": total_timesteps,
        "normalize_obs": use_norm_obs,
        "normalize_reward": use_norm_rew,
        "load_from_model": LOAD_FROM_MODEL,
    })
    with open(os.path.join(tb_dir, "config_used.json"), "w") as f:
        json.dump(config_record, f, indent=2)

    # Train
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)

    # Save model + stats
    model.save(new_model_name)
    print(f"--- SAVED MODEL: {new_model_name}.zip ---")

    if use_norm_obs or use_norm_rew:
        vec_env.save(f"{new_model_name}.pkl")
        print(f"--- SAVED VecNormalize STATS: {new_model_name}.pkl ---")

    vec_env.close()
    return new_model_name



# ---------------------
# GIF RECORDING
# ---------------------
def record_gif(model_name: str, gif_name: str, max_steps: int = 2000):
    print(f"Recording GIF -> {gif_name}")

    vec_env = DummyVecEnv([make_eval_env()])

    stats_path = f"{model_name}.pkl"
    if os.path.exists(stats_path):
        print(f"--- LOADING VecNormalize STATS FOR EVAL: {stats_path} ---")
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False  # show raw env reward in logs; obs still normalized

    model = PPO.load(f"{model_name}.zip", env=vec_env)

    obs = vec_env.reset()
    frames = []

    for _ in range(max_steps):
        frame = vec_env.render()
        frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)

        if bool(dones[0]):
            # capture the final frame after terminal if possible
            frame = vec_env.render()
            frames.append(frame)
            break

    vec_env.close()
    imageio.mimsave(gif_name, frames, fps=30)
    print(f"--- GIF SAVED: {gif_name} ---")


# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    final_model = train_model()
    record_gif(final_model, f"{final_model}.gif")

    print("\nDONE!")
