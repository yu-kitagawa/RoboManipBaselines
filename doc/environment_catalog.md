# Environment catalog

All the following Python commands can be executed in the [teleop](../robo_manip_baselines/teleop/) directory.

## MuJoCo environments
### UR5e
#### MujocoUR5eCableEnv
Task to pass the cable between two poles.
```console
$ python bin/TeleopMujocoUR5eCable.py
```
https://github.com/user-attachments/assets/59736023-a7f7-4aca-a860-176db84579f7

#### MujocoUR5eRingEnv
Task to pick a ring and put it around the pole.
```console
$ python bin/TeleopMujocoUR5eRing.py
```
https://github.com/user-attachments/assets/0eb76bbc-6b9d-43f6-95b1-8600e12a47cf

#### MujocoUR5eParticleEnv
Task to scoop up particles.
```console
$ python bin/TeleopMujocoUR5eParticle.py
```
https://github.com/user-attachments/assets/305300bd-6685-46ab-9704-5a15d901ed7a

#### MujocoUR5eClothEnv
Task to roll up the cloth.
```console
$ python bin/TeleopMujocoUR5eCloth.py
```
https://github.com/user-attachments/assets/88bb9d84-7ca9-4d45-b457-cb9931cfb9a1

### xArm7
#### MujocoXarm7CableEnv
Task to pass the cable between two poles.
```console
$ python bin/TeleopMujocoXarm7Cable.py
```
https://github.com/user-attachments/assets/450d04e8-7154-4c01-849c-d1f9b50be0ae

#### MujocoXarm7RingEnv
Task to pick a ring and put it around the pole.
```console
$ python bin/TeleopMujocoXarm7Ring.py
```
https://github.com/user-attachments/assets/da83db94-7fcb-4690-9b73-41af0c1394a8

### ALOHA
#### MujocoAlohaCableEnv
Task to pass the cable between two poles.
```console
$ python bin/TeleopMujocoAlohaCable.py
```
https://github.com/user-attachments/assets/3bbeeeb8-7034-428c-a17e-f3968a2890b9

## Isaac environments
### UR5e
#### IsaacUR5eChainEnv
Task to pick a chain and hang it on a hook
```console
$ python bin/TeleopIsaacUR5eChain.py
```
https://github.com/user-attachments/assets/803759a5-be70-4530-b9ba-b5d381b1bb7a

#### IsaacUR5eCabinetEnv
Task to open the top lid of the box
```console
$ python bin/TeleopIsaacUR5eCabinet.py
```
https://github.com/user-attachments/assets/b5f0cdc3-f9f6-4b4d-9d03-089b84d44f34

## Real-world environments
### UR5e
#### RealUR5eDemoEnv
Various manipulation tasks with UR5e in the real world
```console
$ python bin/TeleopRealUR5eDemo.py
```
https://github.com/user-attachments/assets/1b894ea0-6b5a-4bf8-abaa-8cd3ec2c4f85

### xArm7
#### RealXarm7DemoEnv
Various manipulation tasks with xArm7 in the real world
```console
$ python bin/TeleopRealXarm7Demo.py
```
https://github.com/user-attachments/assets/ab0c9830-5b33-48e8-9dac-272460a51a39
