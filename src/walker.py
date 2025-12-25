import gymnasium as gym
import imageio
import numpy as np
import os
import time
import json
import yaml 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # <--- NEW IMPORT
from stable_baselines3.common.monitor import Monitor

# --- CONFIGURATION ---
CONFIG_FILE = "obs_norm.yaml" # <--- POINTING TO NEW CONFIG
LOAD_FROM_MODEL = None 
# ---------------------

class StabilityRewardWrapper(gym.Wrapper):
    """
    Penalizes the robot for tilting too much.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        hull_angle = obs[0]
        if abs(hull_angle) > 0.4:
            reward -= 0.1
        return obs, reward, terminated, truncated, info

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found!")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train_model():
    # 1. Load Config
    config = load_config(CONFIG_FILE)
    total_timesteps = config.pop("total_timesteps", 100000)
    exp_name = config.pop("experiment_name", "experiment")
    
    # Extract normalization settings (default to False if not in YAML)
    use_norm_obs = config.pop("normalize_obs", False)
    use_norm_rew = config.pop("normalize_reward", False)
    
    # Setup Logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    tensorboard_log_dir = f"./ppo_walker_tensorboard/{run_name}/"
    new_model_name = f"ppo_bipedal_{timestamp}"

    # 2. Setup Environment
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    print("--- Applying Stability Reward Wrapper ---")
    env = StabilityRewardWrapper(env)
    env = Monitor(env)
    
    # Create Vectorized Environment
    vec_env = DummyVecEnv([lambda: env])
    
    # --- NEW: APPLY OBSERVATION NORMALIZATION ---
    if use_norm_obs or use_norm_rew:
        print(f"--- Normalization Enabled (Obs: {use_norm_obs}, Rew: {use_norm_rew}) ---")
        
        # Check if we are resuming (we need to load old stats!)
        if LOAD_FROM_MODEL and os.path.exists(f"{LOAD_FROM_MODEL}.pkl"):
            print(f"--- LOADING NORMALIZATION STATS FROM {LOAD_FROM_MODEL}.pkl ---")
            vec_env = VecNormalize.load(f"{LOAD_FROM_MODEL}.pkl", vec_env)
            # Update settings in case YAML changed (e.g. turning off training)
            vec_env.training = True
            vec_env.norm_obs = use_norm_obs
            vec_env.norm_reward = use_norm_rew
        else:
            # Create fresh normalization
            vec_env = VecNormalize(vec_env, norm_obs=use_norm_obs, norm_reward=use_norm_rew, clip_obs=10.)
    # ---------------------------------------------

    # 3. Load or Create Model
    if LOAD_FROM_MODEL and os.path.exists(f"{LOAD_FROM_MODEL}.zip"):
        print(f"--- RESUMING TRAINING FROM: {LOAD_FROM_MODEL} ---")
        model = PPO.load(f"{LOAD_FROM_MODEL}.zip", env=vec_env, tensorboard_log=tensorboard_log_dir)
    else:
        print(f"--- STARTING FRESH: {run_name} ---")
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log_dir, **config)

    # Save Config Record
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    with open(f"{tensorboard_log_dir}/config_used.json", "w") as f:
        config['total_timesteps'] = total_timesteps
        config['experiment_name'] = exp_name
        config['normalize_obs'] = use_norm_obs
        config['normalize_reward'] = use_norm_rew
        json.dump(config, f, indent=4)

    # 4. Train
    print(f"Training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)
    
    # 5. Save Model AND Normalization Stats
    model.save(new_model_name)
    if use_norm_obs or use_norm_rew:
        vec_env.save(f"{new_model_name}.pkl") # <--- SAVES THE STATS FILE
        print(f"--- SAVED NORMALIZATION STATS: {new_model_name}.pkl ---")
    
    print(f"--- SAVED MODEL: {new_model_name}.zip ---")
    
    env.close()
    return new_model_name

def record_gif(model_name, gif_name, num_steps=1000):
    print(f"Recording execution to {gif_name}...")
    
    # We must match the training environment setup exactly
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    vec_env = DummyVecEnv([lambda: env])
    
    # --- CRITICAL: LOAD NORMALIZATION STATS IF THEY EXIST ---
    stats_path = f"{model_name}.pkl"
    if os.path.exists(stats_path):
        print("--- Detected Normalization Stats (.pkl). Loading... ---")
        # Load the stats from training
        vec_env = VecNormalize.load(stats_path, vec_env)
        # IMPORTANT: Turn off training and reward normalization for testing
        # We want to use the 'frozen' stats, and see the REAL raw score.
        vec_env.training = False
        vec_env.norm_reward = False 
    else:
        print("--- No Normalization Stats found. Using raw environment. ---")
    # --------------------------------------------------------
    
    model = PPO.load(f"{model_name}.zip")
    
    images = []
    obs = vec_env.reset() # VecEnv reset returns just obs
    img = vec_env.render()
    
    for _ in range(num_steps):
        images.append(img)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        img = vec_env.render()
        # VecEnv handles reset automatically, so we don't need manual check here
            
    vec_env.close()
    imageio.mimsave(gif_name, images, fps=30)
    print(f"--- GIF SAVED: {gif_name} ---")

if __name__ == "__main__":
    new_model_name = train_model()
    record_gif(new_model_name, f"{new_model_name}.gif")
    print("\nDONE!")
