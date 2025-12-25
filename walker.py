import gymnasium as gym
import imageio
import numpy as np
import os
import time
import json # Import json for saving configs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# --- CONFIGURATION SECTION ---
LOAD_FROM_MODEL = "ppo_bipedal_20251225-120929"
TIMESTEPS = 100_000
# -----------------------------

# --- NEW: Function to save configuration ---
def save_config(model, log_dir, filename):
    """Saves the model's hyperparameters to a JSON file."""
    config = {
        "learning_rate": model.learning_rate,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "ent_coef": model.ent_coef,
        # Add a note if you are using a custom reward wrapper
        "reward_wrapper": "Standard (Change this manually if using custom!)"
    }
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, f"{filename}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"--- SAVED CONFIG: {config_path} ---")
# -------------------------------------------

def train_model(total_timesteps=100_000):
    # Create unique names for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    new_model_name = f"ppo_bipedal_{timestamp}"
    
    # Define where TensorBoard logs will be saved
    # Each run gets its own subfolder, making comparisons easy.
    tensorboard_log_dir = f"./ppo_walker_tensorboard/{run_name}/"

    # 1. Setup Environment
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    # [OPTIONAL] env = CustomRewardWrapper(env) # Add your wrapper here
    env = Monitor(env) # Monitor is still good for basic episode stats
    vec_env = DummyVecEnv([lambda: env])
    
    # 2. Load or Create Model
    model_path_full = f"{LOAD_FROM_MODEL}.zip"
    
    if LOAD_FROM_MODEL and os.path.exists(model_path_full):
        print(f"--- RESUMING TRAINING FROM: {model_path_full} ---")
        # We must pass tensorboard_log when loading to continue logging
        model = PPO.load(model_path_full, env=vec_env, tensorboard_log=tensorboard_log_dir)
    else:
        print("--- STARTING FRESH (New Random Agent) ---")
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1, 
            # --- THE IMPORTANT NEW LINE ---
            # This tells SB3 to log all training metrics to this folder.
            tensorboard_log=tensorboard_log_dir,
            # -----------------------------
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.0 # Entropy coefficient (exploration)
        )

    # --- NEW: Save configs before training starts ---
    save_config(model, tensorboard_log_dir, run_name)

    # 3. Train
    print(f"Training for {total_timesteps} steps. Logs will be in {tensorboard_log_dir}...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)
    
    # 4. Save Model
    model.save(new_model_name)
    print(f"--- SAVED NEW MODEL: {new_model_name}.zip ---")
    
    env.close()
    return new_model_name

# ... (record_gif function and main block remain the same) ...
def record_gif(model_path, gif_name, num_steps=1000):
    print(f"Recording execution to {gif_name}...")
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
    new_model_name = train_model(total_timesteps=TIMESTEPS)
    new_gif_name = f"{new_model_name}.gif"
    record_gif(new_model_name, new_gif_name)
    print("\nDONE!")
