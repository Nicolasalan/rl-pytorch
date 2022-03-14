# Collector agent

Training of an intelligent agent that navigates an environment, aiming to collect yellow bananas and avoid blue ones.
## Environment

The environment used is episodic, that is, it contains a beginning and an end. State spaces have 37 dimensions and contain an interaction speed. The agent has 4 discrete actions for interaction, each time slot, it has to select these actions to maximize the reward, corresponding to:

- `0` - forward.
- `1` - moves backwards.
- `2` - turn left.
- `3` - turn right.

To resolve the environment, the agent must obtain an average score of +13 over 100 consecutive episodes.

### Goal

Collect as many yellow bananas while avoiding the blue bananas.

### Reward

When collecting a yellow banana the agent receives a reward of `+1` and when collecting a blue banana he receives a reward of `-1`.


## Starting

1. Clone this repository to your local machine using `git clone` .

```
https://github.com/Nicolasalan/Deep-Reinforcement-Learning-pytorch.git
```

2. Download the environment from one of the links below. You just need to select the environment that matches your operating system:
    - Linux: [ click here ](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32 bit): [ click here ](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [ click here ](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. After downloading, extract the **zip** file and place the extracted folder in the root of the **Agent-dqn** repository.

4. Change the unity environment's `file_name` path to match your operating system.
```
env = UnityEnvironment(file_name= "HERE")
```

5. Next, run all cells in the **navigation-rl.ipynb** notebook to train the agent.

