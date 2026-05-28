import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class SkydioNavigateEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/skydio_x2/world_basic.xml")
        self.data = mujoco.MjData(self.model)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        self.hover_thrust = 3.2495
        
        self.max_steps = 1500 
        self.current_step = 0
        
        # Fixed Waypoints
        self.waypoints = [
            np.array([4.0, 0.0, 2.0]),   
            np.array([-4.0, 4.0, 1.0]),  
            np.array([0.0, -4.0, 3.0]),  
            np.array([0.0, 0.0, 1.5]),   
        ]
        self.waypoint_index = 0
        self.goal_pos = self.waypoints[0]
        self.prev_action = np.zeros(4, dtype=np.float32)

    def _get_obs(self):
        quat = self.data.sensor("body_quat").data.copy()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        
        local_lin_vel = rot.T @ self.data.qvel[:3]
        local_ang_vel = rot.T @ self.data.qvel[3:6]
        local_rel_goal = rot.T @ (self.goal_pos - self.data.qpos[:3])
        
        return np.concatenate([
            quat, local_lin_vel, local_ang_vel, local_rel_goal, self.prev_action, [self.data.qpos[2]]
        ]).astype(np.float32)

    def _update_waypoint_visuals(self):
        for i in range(len(self.waypoints)):
            site_name = f"waypoint{i}"
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id == -1: continue 
            
            if i == self.waypoint_index:
                self.model.site_rgba[site_id] = [0.0, 1.0, 0.0, 0.8] 
            else:
                self.model.site_rgba[site_id] = [1.0, 0.0, 0.0, 0.3] 

    def step(self, action):
        self.current_step += 1
        
        self.action_scale = 3.0 
        thrust = self.hover_thrust + (action * self.action_scale)
        self.data.ctrl[:] = np.clip(thrust, 0, 13.0)

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        pos = self.data.qpos[:3]
        quat = obs[:4]
        
        local_lin_vel = obs[4:7]
        local_ang_vel = obs[7:10] 
        local_rel_goal = obs[10:13]
        dist = np.linalg.norm(local_rel_goal) 

        reward = 0.0
        
        reward += 0.1 
        
        sphere_radius = 1.0
        if dist < sphere_radius:
            reward += (sphere_radius - dist) * 1.0 
        else:
            reward -= (dist - sphere_radius) * 0.05
        
        local_dir_to_goal = local_rel_goal / (dist + 1e-6)
        vel_towards_goal = np.dot(local_lin_vel, local_dir_to_goal)
        reward += 0.3 * vel_towards_goal
        
        up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)
        if up_z < 0.7:
            reward -= (0.7 - up_z) * 5.0 

        reward -= 0.05 * np.linalg.norm(local_ang_vel)

        reward -= 0.05 * np.linalg.norm(action - self.prev_action)

        terminated = False
        
        if dist < 0.25:
            reward += 500.0 
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            self.goal_pos = self.waypoints[self.waypoint_index]
            self._update_waypoint_visuals()
            dist = np.linalg.norm(self.goal_pos - pos) 

        if up_z < 0.2 or pos[2] < 0.15 or pos[2] > 6.0 or abs(pos[0]) > 8 or abs(pos[1]) > 8:
            reward -= 500.0 
            terminated = True
            
        truncated = self.current_step >= self.max_steps
        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "hover")
        if key_id != -1:
            self.data.qpos = self.model.key_qpos[key_id].copy()
            self.data.qvel = self.model.key_qvel[key_id].copy()
            self.data.ctrl = self.model.key_ctrl[key_id].copy()
        else:
            self.data.qpos[:3] = [0, 0, 1.5]
            self.data.qpos[3:7] = [1, 0, 0, 0]
        
        mujoco.mj_forward(self.model, self.data)

        self.waypoint_index = 0
        self.goal_pos = self.waypoints[self.waypoint_index]
        self._update_waypoint_visuals()

        self.current_step = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        return self._get_obs(), {}