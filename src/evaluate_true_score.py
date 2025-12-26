import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import os

# --- 1. SETTINGS ---
# Put the exact name of the model you want to check here (without .zip)
MODEL_NAME = "ppo_bipedal_strict_20251225-103000" 
STATS_FILE = f"{MODEL_NAME}.pkl"  # This file usually lives next to the .zip

def evaluate():
    print(f"--- EVALUATING TRUE SCORE FOR: {MODEL_NAME} ---")

    # 2. Setup the "Clean" Environment
    # Notice: NO StabilityRewardWrapper here. Just raw BipedalWalker.
    env = gym.make("BipedalWalker-v3", render_mode=None) 
    vec_env = DummyVecEnv([lambda: env])

    # 3. Load the "Glasses" (Normalization Stats)
    # The robot needs these to see the world correctly, but we turn off
    # reward modification so we can see the raw score.
    if os.path.exists(STATS_FILE):
        print(f"Loading normalization stats from {STATS_FILE}...")
        vec_env = VecNormalize.load(STATS_FILE, vec_env)
        vec_env.training = False     # Stop updating stats (freeze mode)
        vec_env.norm_reward = False  # <--- CRITICAL: Show me the REAL points
    else:
        print("WARNING: No .pkl stats file found. If you trained with ObsNorm, the robot is blind!")

    # 4. Load the Brain
    if not os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"Error: Could not find {MODEL_NAME}.zip")
        return

    model = PPO.load(f"{MODEL_NAME}.zip")

    # 5. Run the Test (10 Episodes)
    episodes = 10
    total_score = 0
    
    print("\nStarting 10-Episode Exam...")
    for i in range(1, episodes + 1):
        obs = vec_env.reset()
        done = False
        episode_score = 0
        
        while not done:
            # deterministic=True means "Do your best move" (No random exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_score += reward[0]
            
        print(f"Episode {i}: {episode_score:.2f}")
        total_score += episode_score

    avg_score = total_score / episodes
    print("-" * 30)
    print(f"AVERAGE TRUE SCORE: {avg_score:.2f}")
    
    if avg_score >= 300:
        print("RESULT: 🏆 SOLVED (Official Standards)")
    else:
        print(f"RESULT: Not Solved (Gap: {300 - avg_score:.2f})")

if __name__ == "__main__":
    evaluate()
