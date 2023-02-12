# CartPole-v1 with Q-learning made by  [Raffaele Pumpo](https://github.com/RaffaelePumpo) 

The Cartpole problem is a classic control problem in the field of reinforcement learning. It involves a pole that is attached to a cart, and the objective is to balance the pole upright on the cart by moving the cart left or right. The system is considered solved when the pole is balanced for a certain amount of time, or a certain number of time steps. 

### Tools used

For the following algorithm, I have used "anaconda3", installed gym with:

*pip install gym* (gym 0.17.3) **Use this version the function used might be different for output and input**

*pip install -U scikit-learn*

*pip install typing*

in the prompt of the anaconda 3.

**Importing required libraries**


```python
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import gym
import time , random , math
from typing import Tuple
```

**Build the environment**


```python
env = gym.make('CartPole-v1')
```

## Q-learning

Q-Learning is a model-free, off-policy reinforcement learning algorithm. It is used to learn the optimal policy for a given Markov Decision Process (MDP) by estimating the optimal action-value function (also known as the Q-function). The Q-function represents the expected reward for taking a specific action in a specific state and following the optimal policy thereafter. The Q-learning algorithm updates the Q-function estimates iteratively based on observed state-action transitions and received rewards. The goal of Q-learning is to find the policy that maximizes the expected cumulative reward. Once the Q-function has converged, the algorithm can be used to determine the optimal policy by selecting the action with the highest Q-value for each state.

**Convert the continuous space in discrete one**


```python
n_bins = (6 ,12)
lower_bounds = [env.observation_space.low[2],-math.radians(50)]
upper_bounds = [env.observation_space.high[2],math.radians(50)]

def discretizer(_,__, angle, pole_velocity) -> Tuple[int, ...]:
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))

```

**Initialise the Q value table with zeros**


```python
Q_table = np.zeros(n_bins + (env.action_space.n,))
```

**Create a policy function using Q-table and  greedly selecting the highest Q value**


```python
def policy ( state : tuple):
    return np.argmax(Q_table[state])
```

**Update the values of the table**


```python
def new_Q_value ( reward : float , state_new : tuple , discount_factor =1) ->float:
    future_optimal_value = np.max(Q_table[state_new])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value
```

**Define the rates** 


```python
def learning_rate ( n : int , min_rate = 0.1) ->float:
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate ( n : int , min_rate = 0.1) ->float:
    return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))
```

### Training 


```python
# Set the number of episodes to run
n_episodes = input("Insert the number of episodes that you want to train the cart, suggest at least 500:")
# Convert in int
n_episodes = int(n_episodes)
# Loop over each episode
for episode in range(n_episodes):

    # Reset the environment and discretize the current state
    current_state, done = discretizer(*env.reset()), False

    # Initialize the episode score to 0
    score = 0

    # Loop until the episode is done
    while not done:

        # Choose an action based on the current policy
        action = policy(current_state)

        # With probability epsilon, choose a random action instead
        if random.random() < exploration_rate(episode):
            action = env.action_space.sample()

        # Take a step in the environment with the chosen action
        obs, reward, done, info = env.step(action)

        # Discretize the new state
        new_state = discretizer(*obs)

        # Update the Q table with the new information
        lr = learning_rate(episode)
        learned_value = new_Q_value(reward, new_state)
        old_value = Q_table[current_state + (action,)]
        Q_table[current_state + (action,)] = old_value * (1 - lr) + lr * learned_value

        # Update the current state and the episode score
        current_state = new_state
        score += reward

        # Render the environment every 5 episodes
        # avoid seeing each episode of training
        if episode % 5 == 0:
            env.render()

    # Print the episode number and score
    print(f'Episode: {episode} Score: {score}')

# Close the environment
env.close()

```

    Insert the number of episodes that you want to train the cart, suggest at least 500:100
    Episode: 0 Score: 18.0
    Episode: 1 Score: 29.0
    Episode: 2 Score: 12.0
    Episode: 3 Score: 16.0
    Episode: 4 Score: 23.0
    Episode: 5 Score: 31.0
    Episode: 6 Score: 38.0
    Episode: 7 Score: 22.0
    Episode: 8 Score: 18.0
    Episode: 9 Score: 59.0
    Episode: 10 Score: 31.0
    Episode: 11 Score: 27.0
    Episode: 12 Score: 18.0
    Episode: 13 Score: 25.0
    Episode: 14 Score: 17.0
    Episode: 15 Score: 23.0
    Episode: 16 Score: 17.0
    Episode: 17 Score: 20.0
    Episode: 18 Score: 20.0
    Episode: 19 Score: 12.0
    Episode: 20 Score: 29.0
    Episode: 21 Score: 22.0
    Episode: 22 Score: 13.0
    Episode: 23 Score: 19.0
    Episode: 24 Score: 21.0
    Episode: 25 Score: 15.0
    Episode: 26 Score: 20.0
    Episode: 27 Score: 21.0
    Episode: 28 Score: 35.0
    Episode: 29 Score: 14.0
    Episode: 30 Score: 13.0
    Episode: 31 Score: 49.0
    Episode: 32 Score: 12.0
    Episode: 33 Score: 12.0
    Episode: 34 Score: 18.0
    Episode: 35 Score: 24.0
    Episode: 36 Score: 19.0
    Episode: 37 Score: 12.0
    Episode: 38 Score: 20.0
    Episode: 39 Score: 18.0
    Episode: 40 Score: 36.0
    Episode: 41 Score: 41.0
    Episode: 42 Score: 43.0
    Episode: 43 Score: 17.0
    Episode: 44 Score: 17.0
    Episode: 45 Score: 76.0
    Episode: 46 Score: 39.0
    Episode: 47 Score: 26.0
    Episode: 48 Score: 18.0
    Episode: 49 Score: 74.0
    Episode: 50 Score: 15.0
    Episode: 51 Score: 69.0
    Episode: 52 Score: 27.0
    Episode: 53 Score: 42.0
    Episode: 54 Score: 31.0
    Episode: 55 Score: 22.0
    Episode: 56 Score: 48.0
    Episode: 57 Score: 65.0
    Episode: 58 Score: 91.0
    Episode: 59 Score: 122.0
    Episode: 60 Score: 154.0
    Episode: 61 Score: 25.0
    Episode: 62 Score: 12.0
    Episode: 63 Score: 21.0
    Episode: 64 Score: 34.0
    Episode: 65 Score: 32.0
    Episode: 66 Score: 20.0
    Episode: 67 Score: 22.0
    Episode: 68 Score: 13.0
    Episode: 69 Score: 34.0
    Episode: 70 Score: 22.0
    Episode: 71 Score: 10.0
    Episode: 72 Score: 40.0
    Episode: 73 Score: 28.0
    Episode: 74 Score: 50.0
    Episode: 75 Score: 53.0
    Episode: 76 Score: 43.0
    Episode: 77 Score: 57.0
    Episode: 78 Score: 20.0
    Episode: 79 Score: 14.0
    Episode: 80 Score: 90.0
    Episode: 81 Score: 25.0
    Episode: 82 Score: 142.0
    Episode: 83 Score: 39.0
    Episode: 84 Score: 145.0
    Episode: 85 Score: 30.0
    Episode: 86 Score: 72.0
    Episode: 87 Score: 22.0
    Episode: 88 Score: 32.0
    Episode: 89 Score: 15.0
    Episode: 90 Score: 33.0
    Episode: 91 Score: 20.0
    Episode: 92 Score: 65.0
    Episode: 93 Score: 58.0
    Episode: 94 Score: 116.0
    Episode: 95 Score: 48.0
    Episode: 96 Score: 75.0
    Episode: 97 Score: 28.0
    Episode: 98 Score: 47.0
    Episode: 99 Score: 90.0
    

## Results

Observing the simulation, we can conclude that the algorithm just implemented It's useful for the CartPole. Initially the pole falls in a very short time, as the following video:


<img src="https://github.com/RaffaelePumpo/CartPole-v1-Q-learning/blob/main/Initial.gif?raw=true" alt="drawing" width="600" height="300"/>

After different episodes, the algorithm allows to move the cart in a such a way that the pole is balanced for a certain time, as shown below:

<img src="https://github.com/RaffaelePumpo/CartPole-v1-Q-learning/blob/main/Initial.gif?raw=true" alt="drawing" width="600" height="300"/>

We can see also that the algorithm is correct, observing the values of the score for each episode, the score is greater when the pole doesn't fall for a longer time and smaller otherwise. The score for the last episode is greater with respect to the initial ones thanks to the algorithm.
