# 3d Prognostics

This repository holds a model for a multi-DOF robot with injected degradation for studying the
classification of robot degraded performance. This classification can be used to plan future tasks
in long horizon deep-space missions.

## Setup

Install [robosuite prerequisites](https://robosuite.ai/docs/installation.html)

```
sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
         libosmesa6-dev software-properties-common net-tools unzip vim \
         virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf
```
Then,
```
pip install -r requirements.txt
```

## Habitats Optimized for Missions of Exploration

<img
src="https://homestri.ucdavis.edu/sites/g/files/dgvnsk5651/files/HOME-Project-Logo---Final-transparent_0.png"
width="100" />

HOME is a NASA funded project to answer fundamental questions about deep space habitats.

<https://homestri.ucdavis.edu/about>

## Attribution

- [MuJoCo](https://mujoco.org/)
- [RoboSuite](https://robosuite.ai/)
