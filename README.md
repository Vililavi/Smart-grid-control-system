Smart grid control using Neuro Evolution/genetic programming

We consider the problem of control of a micro electricity grid that contains residential
electric loads, thermostatically controlled loads TCLs (mainly air conditioning systems), a
battery, an electricity generator as well as a connection to the main electricity grid on which
it can sell or buy electricity. The micro grid manager must balance between the electricity
produced/purchased and the electricity consumed/stored/sold in a cost optimal way. This
problem is formulated as a Markov-decision process where the Agent (The microgrid
manager) interacts with the environment (the microgrid) and tries to maximize the reward.
This is a reinforcement learning problem that can be solved in many ways using DQN,
Policy gradient or proximal policy optimization. However, we want to use a neuro-
evolution method to evolve a neural network that takes optimal actions given a state of the
environment. For this project we provide a simulation of the environment using Openai
gym implemented in Python. Therefore, the focus should be on the neuro evolution method.

![Näyttökuva 2022-10-16 111908](https://user-images.githubusercontent.com/38975896/196025703-9fb22be0-9d4c-4907-837c-3250b8c8884a.png)

The action constitutes of 4 sub actions: TCL action, Price action, Energy deficiency action and Energy excess
action. The possible values of each sub action are positive integers and are as follows:
TCL action: [0:3]
Price action: [0:4]
Energy deficiency action: [0:1]
Energy excess action [0:1]
Each action is a combination of sub-actions, therefore we have 4*5*2*2= 80 possible actions.
Therefore, the NN we want to design has to provide 80 outputs that define the probability distribution of
the actions. (This can be seen as a classification problem: choose one action from 80 possible actions)
The fitness function should be the total reward for one day (24 timesteps)

![Näyttökuva 2022-10-16 111827](https://user-images.githubusercontent.com/38975896/196025681-c7d60c7d-3d28-4be1-9db3-0c3e5b24dbd4.png)

![kuva](https://user-images.githubusercontent.com/38975896/196025715-02cda87e-0dc1-4b5e-99c0-d3e1650e00a2.png)

![kuva](https://user-images.githubusercontent.com/38975896/196025722-b90c223e-b9e3-4691-9dd9-231f6a4dbd84.png)

Literature:

Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2), 99-127.
