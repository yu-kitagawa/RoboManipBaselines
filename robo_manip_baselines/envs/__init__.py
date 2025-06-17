from gymnasium.envs.registration import register

# Mujoco
## UR5e
register(
    id="robo_manip_baselines/MujocoUR5eCableEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eCableEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eRingEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eRingEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eParticleEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eParticleEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eClothEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eClothEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eInsertEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eInsertEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eDoorEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eDoorEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eCabinetEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eCabinetEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5eToolboxEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eToolboxEnv",
)
register(
    id="robo_manip_baselines/MujocoUR5ePickEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5ePickEnv",
)

## UR5e-Dual
register(
    id="robo_manip_baselines/MujocoUR5eDualCableEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoUR5eDualCableEnv",
)

## xArm7
register(
    id="robo_manip_baselines/MujocoXarm7CableEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoXarm7CableEnv",
)

register(
    id="robo_manip_baselines/MujocoXarm7RingEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoXarm7RingEnv",
)

## ViperX 300S
register(
    id="robo_manip_baselines/MujocoVx300sPickEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoVx300sPickEnv",
)

## ALOHA
register(
    id="robo_manip_baselines/MujocoAlohaCableEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoAlohaCableEnv",
)

## HSR
register(
    id="robo_manip_baselines/MujocoHsrTidyupEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoHsrTidyupEnv",
)

## G1
register(
    id="robo_manip_baselines/MujocoG1BottlesEnv-v0",
    entry_point="robo_manip_baselines.envs.mujoco:MujocoG1BottlesEnv",
)

# Isaac
register(
    id="robo_manip_baselines/IsaacUR5eChainEnv-v0",
    entry_point="robo_manip_baselines.envs.isaac:IsaacUR5eChainEnv",
)
register(
    id="robo_manip_baselines/IsaacUR5eCabinetEnv-v0",
    entry_point="robo_manip_baselines.envs.isaac:IsaacUR5eCabinetEnv",
)

# Real
## UR5e
register(
    id="robo_manip_baselines/RealUR5eDemoEnv-v0",
    entry_point="robo_manip_baselines.envs.real.ur5e:RealUR5eDemoEnv",
)

## xArm7
register(
    id="robo_manip_baselines/RealXarm7DemoEnv-v0",
    entry_point="robo_manip_baselines.envs.real.xarm7:RealXarm7DemoEnv",
)
