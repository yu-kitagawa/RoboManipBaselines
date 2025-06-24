# How to add a new environment

## MuJoCo environments
- [`robo_manip_baselines/envs/mujoco/ur5e/MujocoUR5eCableEnv.py`](../robo_manip_baselines/envs/mujoco/ur5e/MujocoUR5eCableEnv.py)  
  Implement the new environment class. Refer to existing environment files as a template.

- [`robo_manip_baselines/envs/operation/OperationMujocoUR5eCable.py`](../robo_manip_baselines/envs/operation/OperationMujocoUR5eCable.py)  
  Implement the corresponding operation class. Refer to existing environment files as a template.

- [`robo_manip_baselines/envs/assets/mujoco/envs/ur5e/env_ur5e_cable.xml`](../robo_manip_baselines/envs/assets/mujoco/envs/ur5e/env_ur5e_cable.xml)  
  Create a MuJoCo simulation model (XML file). You can base it on other existing environment files.

- [`robo_manip_baselines/envs/mujoco/__init__.py`](../robo_manip_baselines/envs/mujoco/__init__.py)  
  Make the environment importable by adding the following line:
  ```python
  from .ur5e.MujocoUR5eCableEnv import MujocoUR5eCableEnv
  ```

- [`robo_manip_baselines/envs/__init__.py`](../robo_manip_baselines/envs/__init__.py)  
  Register the environment in Gym by adding the following lines:
  ```python
  register(
      id="robo_manip_baselines/MujocoUR5eCableEnv-v0",
      entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eCableEnv",
  )
  ```
