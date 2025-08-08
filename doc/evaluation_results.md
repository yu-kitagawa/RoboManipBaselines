# Evaluation results

## Evaluation in MuJoCoUR5e environments
We performed rollouts at six different object positions for each of the five checkpoints trained with different random seeds per task, and computed the success rates.

| <nobr>Policy \\ Task</nobr> | [Cable](./environment_catalog.md#MujocoUR5eCable) | [Ring](./environment_catalog.md#MujocoUR5eRing) | [Particle](./environment_catalog.md#MujocoUR5eParticle) | [Cloth](./environment_catalog.md#MujocoUR5eCloth) | [Door](./environment_catalog.md#MujocoUR5eDoor) | [Toolbox](./environment_catalog.md#MujocoUR5eToolbox) | [CabinetSlide](./environment_catalog.md#MujocoUR5eCabinet) | [CabinetHinge](./environment_catalog.md#MujocoUR5eCabinet) | [Insert](./environment_catalog.md#MujocoUR5eInsert) | Average |
|---|---|---|---|---|---|---|---|---|---|---|
| [ACT](../robo_manip_baselines/policy/act) | 33<br>(&plusmn;42) | 50<br>(&plusmn;26) | 47<br>(&plusmn;30) | 77<br>(&plusmn;22) | 53<br>(&plusmn;43) | 57<br>(&plusmn;15) | 87<br>(&plusmn;14) | 13<br>(&plusmn;18) | 37<br>(&plusmn;18) | **50<br>(&plusmn;22)** |
| [SARNN](../robo_manip_baselines/policy/sarnn) | 80<br>(&plusmn;18) | 80<br>(&plusmn;27) | 83<br>(&plusmn;17) | 100<br>(&plusmn;0) | 63<br>(&plusmn;14) | 47<br>(&plusmn;7) | 93<br>(&plusmn;9) | 60<br>(&plusmn;25) | 37<br>(&plusmn;7) | **71<br>(&plusmn;21)** |
| [Diffusion Policy](../robo_manip_baselines/policy/diffusion_policy) | 27<br>(&plusmn;19) | 7<br>(&plusmn;15) | 60<br>(&plusmn;28) | 93<br>(&plusmn;15) | 60<br>(&plusmn;19) | 77<br>(&plusmn;9) | 67<br>(&plusmn;0) | 87<br>(&plusmn;14) | 33<br>(&plusmn;12) | **57<br>(&plusmn;29)** |
