# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description
```bash
# FrozenLake-v1:
# - 4 x 4 grid environment
# - 16 states
# - 4 actions: Left, Down, Right, Up
# - Agent starts at the initial state
# - Goal state gives reward = 1
# - Holes terminate the episode
# - Other moves give reward = 0
```






## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm
```txt
1. Initialize the FrozenLake environment and Q-table.

2. Set the hyperparameters α, γ, and ε.

3. Generate an episode using the ε-greedy policy.

4. Store the state, action, and reward for each step.

5. Traverse the episode in reverse order.

6. Calculate the return G = γG + reward.

7. Update the Q-value using Q(s,a) = Q(s,a) + α[G - Q(s,a)].

8. Reduce ε gradually after each episode.

9. Repeat the process for the given number of episodes.

10. Extract the optimal policy using the maximum Q-value.

11. Display the learned policy, state-value function, Q-table, and learning curve.
```


## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
num_episodes = 20000
gamma = 0.99
alpha = 0.1
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995
max_steps_per_episode = 100

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((n_states, n_actions))
episode_rewards = []

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
def epsilon_greedy_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------
def generate_episode(epsilon):
    episode = []
    state, info = env.reset()

    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))
        state = next_state

        if terminated or truncated:
            break

    return episode

# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------
epsilon = epsilon_start

for episode_num in range(num_episodes):

    episode = generate_episode(epsilon)

    G = 0
    visited = set()

    for state, action, reward in reversed(episode):

        G = gamma * G + reward

        # Every-visit Monte Carlo
        if (state, action) not in visited:
            Q[state, action] += alpha * (G - Q[state, action])
            visited.add((state, action))

    total_reward = sum([step[2] for step in episode])
    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------
optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

# -------------------------------------------------
# Display Results
# -------------------------------------------------
def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("Name:SOMESHWAR S")
    print("Register Number:212224040322")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print_value_function(state_values)
print_policy(optimal_policy)

print("\nFinal Q-table:")
print(np.round(Q, 3))

success_rate = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:", success_rate)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")
plt.grid(True)
plt.show()

env.close()



```

---

## Output


Final Q-table:


<img width="317" height="376" alt="image" src="https://github.com/user-attachments/assets/736de773-c8a1-45fd-a81e-251a9ca0713c" />



Estimated State-Value Function:

<img width="343" height="177" alt="image" src="https://github.com/user-attachments/assets/ec9e5db5-1f77-43ca-a65c-4aba7994eb2d" />






Learned Policy:


<img width="298" height="125" alt="image" src="https://github.com/user-attachments/assets/031ae52d-9cdb-43c4-bf1e-0dfbd9112bda" />



Average reward over last 1000 episodes: 

<img width="478" height="45" alt="image" src="https://github.com/user-attachments/assets/5253c7cc-de1a-4323-9286-fa67dd4e3b16" />

Monte Carlo Control Learning Curve:

<img width="909" height="608" alt="image" src="https://github.com/user-attachments/assets/cd22a4a0-8fc1-485e-ac7e-3b9f1022c427" />

---

## Result



The Monte Carlo Control algorithm successfully learned an optimal policy for the FrozenLake environment. The Q-table and state-value function were generated, and the learning progress was visualized using a learning curve.

---

## Inference

The results show that Monte Carlo Control can learn a suitable policy through repeated episodes and reward-based Q-value updates. As training progresses, the agent improves its decision-making and learns to reach the goal more effectively.







---

