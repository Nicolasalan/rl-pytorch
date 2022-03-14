# Multi-Agent

Training of 2 agents that control the rackets to hit a ball over a net, using Multi-Agent Deep Deterministic Policy Gradients (MADDPG) learning.

## Environment
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Goal
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

### Reward
The environment rewards are, if an agent hits the ball over the net, he gets a reward of +0.1. If an agent drops a ball on the ground or hits the ball out of bounds, he receives a reward of -0.01. Thus, the objective of each agent is to keep the ball in play.

## Starting

1. Clone this repository to your local machine using `git clone` .

```
https://github.com/Nicolasalan/Deep-Reinforcement-Learning-pytorch.git
```

2. Download the environment that matches your operating system:
     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
     - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
     - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
     - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. After downloading, extract the **zip** file and place the extracted folder in the root of the **Multi-Agent-rl** repository.

4. Change the unity environment's `file_name` path to match your operating system.
```
env = UnityEnvironment(file_name= "HERE")
```

5. Next, run all cells in the **competition-rl.ipynb** notebook to train the agent.
