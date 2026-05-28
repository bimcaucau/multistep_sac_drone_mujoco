import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from drone_env_controlling import SkydioNavigateEnv

# Custom Callback to save VecNormalize statistics alongside the model
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
        env = SkydioNavigateEnv()
        return env
    set_random_seed(seed)
    return _init

def train():
    num_cpu = 12 
    log_dir = "./logs/"
    
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecMonitor(env) 
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=3e-4, 
        tensorboard_log="./sac_drone_tensorboard/",
        buffer_size=1_000_000, 
        batch_size=512,
        ent_coef="auto",
    )

    print(f"Training on {num_cpu} cores...")
    
    # Save Model Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // num_cpu, 1),
        save_path=log_dir,
        name_prefix="skydio_model"
    )
    
    # Save VecNormalize stats 
    vec_norm_callback = SaveVecNormalizeCallback(
        save_freq=max(100000 // num_cpu, 1),
        save_path=log_dir
    )

    model.learn(
        total_timesteps=10_000_000, 
        log_interval=10, 
        progress_bar=True, 
        callback=[checkpoint_callback, vec_norm_callback]
    )
    
    env.save("vec_normalize_final.pkl")
    model.save("skydio_sac_final")

if __name__ == "__main__":
    train()