import mujoco
import mujoco.viewer
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_env_controlling import SkydioNavigateEnv
import os
import re

def get_latest_checkpoint(log_dir="./logs"):
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
        return None, None

    return model_path, vec_norm_path


def watch_drone():
    model_path, vec_norm_path = get_latest_checkpoint()

    if model_path is None:
        print("Model or VecNormalize file not found!")
        return

    print(f"Loading model: {model_path}")
    print(f"Loading VecNormalize: {vec_norm_path}")

    def make_env():
        return SkydioNavigateEnv()

    env = DummyVecEnv([make_env])

    # Load normalization stats
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False

    model = SAC.load(model_path)

    with mujoco.viewer.launch_passive(
        env.envs[0].model,
        env.envs[0].data
    ) as viewer:

        obs = env.reset()

        for _ in range(10000):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            viewer.sync()
            time.sleep(0.02)

            if done:
                obs = env.reset()
                print("Episode Ended. Resetting...")


if __name__ == "__main__":
    watch_drone()