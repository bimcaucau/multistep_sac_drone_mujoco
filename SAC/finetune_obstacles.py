import os
import torch
import numpy as np
import pickle
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from drone_env_obstacles import SkydioObstacleEnv # Import new env

"""
The old checkpoints won't have the same observation space due to the added Lidar inputs. 
This finetuning script performs "network surgery" to copy the learned weights from the 
old model into a new model with the expanded observation space. 
"""

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.training_env.save(path)
            if self.verbose > 0:
                print(f"Saved VecNormalize to {path}")
        return True

def make_env(rank, seed=0):
    def _init():
        return SkydioObstacleEnv()
    set_random_seed(seed)
    return _init

def perform_surgery(old_model_path, new_model, num_old_obs=18, num_new_obs=26):
    """Copies flying capabilities from old model to new model, initializing Lidar to 0"""
    print("Performing Network Surgery to inject Lidar capabilities...")
    old_model = SAC.load(old_model_path)
    
    old_dict = old_model.policy.state_dict()
    new_dict = new_model.policy.state_dict()

    for key in new_dict.keys():
        if key in old_dict and new_dict[key].shape == old_dict[key].shape:
            new_dict[key] = old_dict[key] 
        elif key in old_dict:
            old_w = old_dict[key]
            new_w = new_dict[key]
            
            if len(new_w.shape) == 2:
                if 'actor' in key:
                    new_w[:, :num_old_obs] = old_w
                    new_w[:, num_old_obs:] = 0.0 
                elif 'critic' in key:
                    new_w[:, :num_old_obs] = old_w[:, :num_old_obs] 
                    new_w[:, num_new_obs:] = old_w[:, num_old_obs:] 
                    new_w[:, num_old_obs:num_new_obs] = 0.0 
            new_dict[key] = new_w

    new_model.policy.load_state_dict(new_dict)
    print("Surgery Complete! Drone retains flight capability.")

def perform_vec_surgery(old_vec_path, new_env, num_old_obs=18, num_new_obs=26):
    """Pads VecNormalize so the Lidar inputs don't crash the scaling logic"""
    with open(old_vec_path, "rb") as f:
        old_vec = pickle.load(f)
        
    new_vec = VecNormalize(new_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Copy old mean/var
    new_vec.obs_rms.mean[:num_old_obs] = old_vec.obs_rms.mean
    new_vec.obs_rms.var[:num_old_obs] = old_vec.obs_rms.var
    
    # Initialize Lidar mean/var to safe defaults (assume Lidar avg is ~3 meters)
    new_vec.obs_rms.mean[num_old_obs:] = 3.0
    new_vec.obs_rms.var[num_old_obs:] = 2.0
    new_vec.obs_rms.count = old_vec.obs_rms.count
    
    new_vec.ret_rms = old_vec.ret_rms
    return new_vec

def finetune():
    num_cpu = 12
    log_dir = "./logs_finetune/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecMonitor(env) 
    
    env = perform_vec_surgery("vec_normalize_final.pkl", env)

    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cuda", 
        learning_rate=1e-4, 
        tensorboard_log="./sac_obstacles_tensorboard/", 
        buffer_size=1_000_000,                          
        batch_size=512,                                 
        ent_coef="auto"                                 
    ) 
    
    perform_surgery("skydio_sac_final.zip", model)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // num_cpu, 1), 
        save_path=log_dir, 
        name_prefix="skydio_model"
    )
    
    vec_norm_callback = SaveVecNormalizeCallback(
        save_freq=max(100000 // num_cpu, 1),
        save_path=log_dir
    )

    print(f"Finetuning normal SAC on {num_cpu} cores...")
    
    model.learn(
        total_timesteps=15_000_000, 
        log_interval=10, 
        progress_bar=True, 
        callback=[checkpoint_callback, vec_norm_callback] 
    )
    
    env.save("vec_normalize_obstacles_final.pkl")
    model.save("skydio_sac_obstacles_final")

if __name__ == "__main__":
    finetune()