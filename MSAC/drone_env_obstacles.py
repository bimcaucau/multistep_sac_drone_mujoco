import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class SkydioObstacleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/skydio_x2/world_with_obstacles.xml")
        self.data = mujoco.MjData(self.model)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        
        self.hover_thrust = 3.2495
        self.max_steps = 2000 
        self.current_step = 0
        
        self.goal_pos = np.array([15.0, 0.0, 1.0])
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        self.lidar_names = ["rf_n", "rf_ne", "rf_e", "rf_se", "rf_s", "rf_sw", "rf_w", "rf_nw"]

    def _get_obs(self):
        quat = self.data.sensor("body_quat").data.copy()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        
        local_lin_vel = rot.T @ self.data.qvel[:3]
        local_ang_vel = rot.T @ self.data.qvel[3:6]
        local_rel_goal = rot.T @ (self.goal_pos - self.data.qpos[:3])
        
        lidar_readings = []
        for name in self.lidar_names:
            val = self.data.sensor(name).data[0]
            if val < 0:
                val = 5.0 
            lidar_readings.append(val)
            
        self.current_lidar = np.array(lidar_readings, dtype=np.float32)
        
        return np.concatenate([
            quat, local_lin_vel, local_ang_vel, local_rel_goal, self.prev_action, [self.data.qpos[2]], self.current_lidar
        ]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        self.action_scale = 3.0 
        thrust = self.hover_thrust + (action * self.action_scale)
        self.data.ctrl[:] = np.clip(thrust, 0, 13.0)
        
        hit_wall = False
        obstacle_names = ["wall_left", "wall_right", "obs1", "obs2"]

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
            if not hit_wall:
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]
                    geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                    geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                    
                    if (geom1_name in obstacle_names) or (geom2_name in obstacle_names):
                        hit_wall = True
                        break
            
            if hit_wall:
                break
        
        obs = self._get_obs()
        pos = self.data.qpos[:3]
        quat = obs[:4]
        
        local_lin_vel = obs[4:7]
        local_ang_vel = obs[7:10] 
        local_rel_goal = obs[10:13]
        dist = np.linalg.norm(local_rel_goal) 

        reward = 0.0
        
        reward -= 0.02

        reward -= 0.02 * dist 

        progress = self.prev_dist - dist
        reward += 10.0 * progress
        self.prev_dist = dist
        
        up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)
        if up_z < 0.7:
            reward -= (0.7 - up_z) * 5.0 

        reward -= 0.05 * np.linalg.norm(local_ang_vel)

        reward -= 0.05 * np.linalg.norm(action - self.prev_action)
            
        min_lidar = np.min(self.current_lidar)
        if min_lidar < 1.2:
            reward -= 0.5 * np.exp(-4.0 * min_lidar)

        terminated = False
        
        if dist < 0.25:
            reward += 1000.0 
            terminated = True

        if up_z < 0.2 or pos[2] < 0.15 or pos[2] > 4.0 or abs(pos[0]) > 20 or abs(pos[1]) > 6:
            reward -= 500.0 
            terminated = True

        if hit_wall:
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
        self.current_step = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_dist = np.linalg.norm(self.goal_pos - self.data.qpos[:3])
        
        return self._get_obs(), {}