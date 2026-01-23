
# Wheel-Legged Robot Reinforcement Learning Library Dreamwaq


## 🧩 Introduction

This repository provides a **Reinforcement Learning (RL) framework** for wheel-legged robots.  
It supports **simulation-based policy training**, **Sim2Sim validation**, and **Sim2Real deployment** on real robots such as **Unitree Go2** and **custom wheel-legged platforms**.

### 🚀 Features
- ✅ Supports vision-proprioception fusion reinforcement learning  
- ✅ End-to-end pipeline for training, validation, and deployment  
- ✅ Extensible to multiple robot platforms (Go2, m20, etc.)  
- ✅ Supports custom reward functions and curriculum learning

---

## 🛠️ 使用指南
### 0. 安装依赖
python环境：3.8
#### Isaacgym 安装

```bash
pip install -e ~/isaacgym/python
```
#### rsl 
```bash
pip install -e ./rsl_rl
```
#### legged_gym
```bash
pip install -e .
```
#### 山猫m20 已经部署实物
![m20](m20_deploy.gif)
## Use

### 1. train 
```bash
cd legged_gym
python legged_gym/scripts/train.py --task=m20 --headless
```

python legged_gym/scripts/train.py --task=go2_handstand --headless

python legged_gym/scripts/train.py --task=go2_legstand --headless

### 2. play policy
```bash
cd legged_gym
python legged_gym/scripts/play.py --task=m20 --num_envs=50
```

python legged_gym/scripts/play.py --task=go2_handstand --num_envs=50

python legged_gym/scripts/play.py --task=go2_legstand --num_envs=50

### problem
如果你发现自己的urdf训练出来轮子不转，而是抬腿，把urdf中的轮子的continuous改为revolute,上下限改一下-99999 99999，即可训练出轮子转的模型

### 为什么可以训练出轮子转的模型

注意到

action_scale = 0.25

vel_scale = 5.0

故轮子的action更能表现出来故更容易训练出轮子转的模型


# 致谢
https://github.com/XinLang2019/Wheel_Legged_Gym