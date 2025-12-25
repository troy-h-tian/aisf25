import gymnasium as gym
import imageio
import numpy as np
import os
import time
import json
import yaml  # Requires: pip install pyyaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# --- CONFIGURATION ---
# The script will look for this file to load hyperparameters
CONFIG_FILE = "stability.yaml"

# If you want to continue a previous run, put the name here (e.g., "ppo_bipedal_2025...")
# Set to None or "" to start fresh.
LOAD_FROM_MODEL = "reward_crafted"
# ---------------------

class StabilityRewardWrapper(gym.Wrapper):
    """
    A custom wrapper that penalizes the robot for tilting too much.
    Goal: Encourage the robot to stay upright (Observation [0] is Hull Angle).
    """
    def step(self, action):
        # 1. Take the normal step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Extract Hull Angle (Index 0)
        # 0.0 is perfectly upright. +/- 1.0 is falling over.
        hull_angle = obs[0]
        
        # 3. Apply Stability Penalty
        # If the robot tilts more than 0.4 radians (~23 degrees), penalize it.
        # This forces it to learn "Keep your head up!"
        if abs(hull_angle) > 0.4:
            penalty = -0.1
            reward += penalty
            
        # Optional: Boost survival reward slightly to encourage just existing
        # (Default environment gives small points for not dying, we can boost it)
        # if not terminated:
        #    reward += 0.01

        return obs, reward, terminated, truncated, info

def load_config(config_path):
    """Loads settings from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found! Please create it.")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train_model():
    # 1. Load Configuration
    config = load_config(CONFIG_FILE)
    
    # Extract settings that aren't PPO parameters
    total_timesteps = config.pop("total_timesteps", 100000)
    exp_name = config.pop("experiment_name", "experiment")
    
    # Setup Logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{exp_name}_{timestamp}"
    tensorboard_log_dir = f"./ppo_walker_tensorboard/{run_name}/"
    new_model_name = f"ppo_bipedal_{timestamp}"

    # 2. Setup Environment
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    
    # --- APPLY CUSTOM WRAPPER HERE ---
    print("--- Applying Stability Reward Wrapper ---")
    env = StabilityRewardWrapper(env)
    # ---------------------------------
    
    env = Monitor(env) # Helper for logging episode stats
    vec_env = DummyVecEnv([lambda: env])
    
    # 3. Load or Create Model
    if LOAD_FROM_MODEL and os.path.exists(f"{LOAD_FROM_MODEL}.zip"):
        print(f"--- RESUMING TRAINING FROM: {LOAD_FROM_MODEL} ---")
        model = PPO.load(f"{LOAD_FROM_MODEL}.zip", env=vec_env, tensorboard_log=tensorboard_log_dir)
    else:
        print(f"--- STARTING FRESH: {exp_name} ---")
        # Initialize PPO with arguments unpacked from YAML
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1, 
            tensorboard_log=tensorboard_log_dir,
            **config  # <--- MAGIC: Unpacks learning_rate, n_steps, etc. from YAML
        )

    # Save a copy of the config for records
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    with open(f"{tensorboard_log_dir}/config_used.json", "w") as f:
        # Add back the popped values for the record
        config['total_timesteps'] = total_timesteps
        config['experiment_name'] = exp_name
        json.dump(config, f, indent=4)

    # 4. Train
    print(f"Training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)
    
    # 5. Save Model
    model.save(new_model_name)
    print(f"--- SAVED MODEL: {new_model_name}.zip ---")
    
    env.close()
    return new_model_name

def record_gif(model_path, gif_name, num_steps=1000):
    """Records a GIF of the trained model."""
    print(f"Recording execution to {gif_name}...")
    
    # Note: We do NOT use the wrapper here because we want to see 
    # the 'raw' score performance, but the agent will behave according to its training.
    eval_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    
    model = PPO.load(model_path)
    
    images = []
    obs, info = eval_env.reset()
    img = eval_env.render()
    
    for _ in range(num_steps):
        images.append(img)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        img = eval_env.render()
        if terminated or truncated:
            obs, info = eval_env.reset()
            
    eval_env.close()
    imageio.mimsave(gif_name, images, fps=30)
    print(f"--- GIF SAVED: {gif_name} ---")

if __name__ == "__main__":
    # Train
    new_model_name = train_model()
    
    # Record GIF
    record_gif(new_model_name, f"{new_model_name}.gif")
    
    print("\nDONE! View results via: tensorboard --logdir ./ppo_walker_tensorboard/")
