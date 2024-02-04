# Implementation of DQN Network in ABB YuMi Manipulator to do planar pushing under external perturbance. 

Pytorch Implementation of Deep Q Network  (DQN) for the ABB YuMi Robot Arm manipulating a block on a plane using an OpenAI Gym environment. The code is designed to train an agent to perform planar pushing task when dealing with external peturbances where the end goal is to reach a goal destination with defined orientation. This project is done to establish and Reinforcement Learning platform to do other tasks like peg in hole assembly and learning ambidextrous object manipulation policies when encountering semi deformable objects.

---

## Tech Stack

- **Language:** `Python`
- **Libraries:** `Pytorch`, `random`, `numpy`, `matplotlib`, `gym`, `matplotlib`, `pandas`, `itertools`,`collections`

---

## Experimental Setup

<img src="imgs/problem _setup.png" width=600>

---

## Environment Setup

<img src="imgs/env_Setup.png" width=600>


## Pipeline Setup

<img src="imgs/pipeline.png" width=600>

---

## Experience Replay using Priority Buffer for Learning [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)


<img src="imgs/priority_buff.png" width=600>

