import mujoco
import mujoco.viewer
import time
import os
import re
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_env_obstacles import SkydioObstacleEnv

def get_latest_checkpoint(log_dir="./logs_finetune"):
    if not os.path.exists(log_dir):
        return None, None
        
    model_files = []
    for file in os.listdir(log_dir):
        match = re.match(r"skydio_model_(\d+)_steps\.zip", file)
        if match:
            steps = int(match.group(1))
            model_files.append((steps, file))

    if not model_files:
        return None, None

    # Get file with largest step count
    latest_steps, latest_model = max(model_files, key=lambda x: x[0])

    model_path = os.path.join(log_dir, latest_model)
    vec_norm_path = os.path.join(
        log_dir,
        f"vec_normalize_{latest_steps}_steps.pkl"
    )

    if not os.path.exists(vec_norm_path):
        print(f"Warning: Expected VecNormalize at {vec_norm_path} but couldn't find it.")
        return model_path, None

    return model_path, vec_norm_path


def watch_drone():
    model_path, vec_norm_path = get_latest_checkpoint()

    if model_path is None:
        # Fallback to the final saved model if logs directory doesn't have checkpoints yet
        model_path = "skydio_sac_obstacles_final.zip"
        vec_norm_path = "vec_normalize_obstacles_final.pkl"
        
        if not os.path.exists(model_path):
            print("No checkpoints found in ./logs_finetune/ and no final model found either!")
            return

    print(f"Loading model: {model_path}")
    print(f"Loading VecNormalize: {vec_norm_path}")

    def make_env():
        return SkydioObstacleEnv()

    env = DummyVecEnv([make_env])

    if vec_norm_path and os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Running without VecNormalize scaling! (Behavior might be weird)")

    model = SAC.load(model_path)

    with mujoco.viewer.launch_passive(env.envs[0].model, env.envs[0].data) as viewer:
        obs = env.reset()
        episode_reward = 0.0  # Track reward

        for _ in range(100000):

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0] # Reward is an array because of DummyVecEnv
            # # --- ADD THIS TO PRINT POSITION ---
            # # Access the raw MuJoCo data from the first (and only) environment
            # pos = env.envs[0].data.qpos[:3]
            # print(f"X: {pos[0]:5.2f} | Y: {pos[1]:5.2f} | Z: {pos[2]:5.2f} | Reward: {reward[0]:5.2f}")
            # # ----------------------------------
            viewer.sync()
            time.sleep(0.02) # 50 Hz real-time view


            if done:
                print("------------------------------------------------")
                print(f"Episode Ended! Total Reward: {episode_reward:.2f}")
                print("------------------------------------------------")
                
                obs = env.reset()
                episode_reward = 0.0 # Reset tracking for the next episode


if __name__ == "__main__":
    watch_drone()