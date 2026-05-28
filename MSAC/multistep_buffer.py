import numpy as np
import torch
from typing import Optional
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

class MultiStepReplayBuffer(ReplayBuffer):
    """
    Custom Replay Buffer for Multi-Step SAC (MSAC).
    Implements n-step bootstrapping as described in "Multi-step first: A lightweight 
    deep reinforcement learning strategy for robust continuous control".
    """
    def __init__(self, *args, n_steps=5, gamma=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.custom_gamma = gamma
        assert not self.optimize_memory_usage, "MultiStepReplayBuffer does not support optimize_memory_usage=True"

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        batch_size = len(batch_inds)
        n_step_rewards = np.zeros((batch_size, 1), dtype=np.float32)
        
        obs = self.observations[batch_inds, env_indices, :]
        actions = self.actions[batch_inds, env_indices, :]
        n_step_next_obs = self.next_observations[batch_inds, env_indices, :].copy()
        
        n_step_gammas = np.ones((batch_size, 1), dtype=np.float32) * (self.custom_gamma ** self.n_steps)
        active = np.ones(batch_size, dtype=bool)
        
        for k in range(self.n_steps):
            curr_inds = (batch_inds + k) % self.buffer_size
            
            if self.full:
                invalid = (curr_inds == self.pos)
            else:
                invalid = (curr_inds >= self.pos)
            active &= ~invalid
            
            step_rewards = self.rewards[curr_inds, env_indices].copy().reshape(-1, 1)
            if hasattr(self, "_normalize_reward") and env is not None:
                step_rewards = self._normalize_reward(step_rewards, env)
                
            n_step_rewards[active] += (self.custom_gamma ** k) * step_rewards[active]
            
            # --- FIXED EPISODE BOUNDARY LOGIC ---
            step_episode_ends = self.dones[curr_inds, env_indices].copy().reshape(-1, 1)
            
            if self.handle_timeout_termination:
                step_timeouts = self.timeouts[curr_inds, env_indices].copy().reshape(-1, 1)
                # True terminations = crashes/successes. Timeouts = truncated.
                step_true_terms = step_episode_ends * (1.0 - step_timeouts)
            else:
                step_true_terms = step_episode_ends
                
            just_ended = active & (step_episode_ends > 0).squeeze(-1)
            just_terminated = active & (step_true_terms > 0).squeeze(-1)
            just_truncated = just_ended & ~just_terminated
            
            if np.any(just_ended):
                # Episode ended! Grab the exact final observation
                n_step_next_obs[just_ended] = self.next_observations[curr_inds[just_ended], env_indices[just_ended], :]
                
                # If Crash/Success -> Future value is 0
                n_step_gammas[just_terminated] = 0.0 
                # If Timeout -> Bootstrap using the remaining gamma factor
                n_step_gammas[just_truncated] = self.custom_gamma ** (k + 1)
                
                # STOP accumulating for these sequences so we don't leak into the next episode!
                active &= ~just_ended 
                
        if np.any(active):
            last_inds = (batch_inds + self.n_steps - 1) % self.buffer_size
            n_step_next_obs[active] = self.next_observations[last_inds[active], env_indices[active], :]

        if hasattr(self, "_normalize_obs") and env is not None:
            obs = self._normalize_obs(obs, env)
            n_step_next_obs = self._normalize_obs(n_step_next_obs, env)

        pseudo_dones = 1.0 - (n_step_gammas / self.custom_gamma)
        
        data = (obs, actions, n_step_next_obs, pseudo_dones, n_step_rewards)
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))