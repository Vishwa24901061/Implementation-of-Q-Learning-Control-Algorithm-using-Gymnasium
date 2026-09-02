# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Develop an agent that learns to navigate the FrozenLake grid-world using Q-Learning. The agent must find a policy that safely guides it from the start state (S) to the goal state (G) while avoiding holes (H) in the ice, using only the rewards it receives from environment interaction — no explicit model of the environment's transition dynamics is provided to the agent.



## Software Requirements

Python 3.8+

Gymnasium (pip install gymnasium)

NumPy

Matplotlib



## Environment Description

FrozenLake-v1 is a grid-world environment (default 4×4) with 16 discrete states and 4 discrete actions.

S	-Start state
F	-Frozen (safe) tile
H	-Hole (episode ends, reward 0)
G	-Goal (episode ends, reward 1)

Actions: 0 = Left, 1 = Down, 2 = Right, 3 = Up

Reward structure: +1 for reaching G, 0 for every other transition (including falling into H).

When is_slippery=True (the default), the environment is stochastic: the agent's chosen action succeeds with a certain probability, and with the remaining probability the agent slides in a perpendicular direction instead — modeling the effect of ice.


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm

1. Initialize the Q-table with all values set to zero, and set the learning rate, discount factor, exploration rate,
   and number of training episodes.
2. At the start of each episode, reset the environment to obtain the initial state.
3. Select an action using the epsilon-greedy strategy — a random action with probability epsilon,
   or the action with the highest Q-value otherwise.
4. Execute the selected action and observe the reward received and the resulting next state.
5. Update the Q-value for the current state-action pair using the observed reward and the maximum Q-value of the next state.
6. Move to the next state and repeat the action-selection and update steps until the episode ends by reaching the goal
   or falling into a hole, then gradually reduce epsilon.
7. Once training across all episodes is complete, derive the state-value function and the optimal policy from the final Q-table.


## Python Program
```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# Create FrozenLake Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
alpha = 0.1          # learning rate
gamma = 0.99         # discount factor
epsilon = 1.0        # initial exploration rate
epsilon_min = 0.01
epsilon_decay = 0.9995
n_episodes = 15000
max_steps = 100

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((n_states, n_actions))

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
episode_rewards = []

for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update rule
        best_next_action = np.argmax(Q[next_state, :])
        td_target = reward + gamma * Q[next_state, best_next_action] * (not done)
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

# Derive value function and policy from Q-table
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------
def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))

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

    print("\nLearned Policy:")
    print(policy_grid)

# -------------------------------------------------
# Output
# -------------------------------------------------
print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)

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
plt.title("Q-Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
```

## Output
<img width="520" height="693" alt="image" src="https://github.com/user-attachments/assets/af7bcc5b-ed50-4713-931b-2c2f7aa55eeb" />
<img width="864" height="586" alt="image" src="https://github.com/user-attachments/assets/145cc1cf-1350-412a-9cd2-f5ef2ae9591f" />


## Result

```
The Q-Learning algorithm was successfully implemented on the FrozenLake-v1 environment, and the agent was trained over
multiple episodes to learn an optimal action-value function. The final Q-table, state-value function, and optimal policy
were successfully derived and displayed in grid form. The learning curve showed a steady increase in average reward before
stabilizing, confirming that the agent learned to reach the goal while avoiding holes. The average reward over the last 1000
episodes was obtained as the final performance measure of the trained agent.

```


## Inference

```

1. Q-Learning successfully learns an optimal policy through trial-and-error interaction alone, without requiring any prior knowledge
   of the environment's transition dynamics.
2. Since the environment is stochastic (is_slippery=True), the average reward did not reach a perfect score of 1.0,
   reflecting the uncertainty introduced by slippery transitions.

```

