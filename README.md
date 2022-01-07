# A basic implementation of Heuristic Search Value Iteration for One-Sided Partially Observable Stochastic Games (Horak, Bosansky, Pechoucek 2017)

The repository also includes implementations of MDP value iteration, Sondik's value iteration for POMDPs, 
HSVI for POMDPs, and Shapley iteration.

**Note**: The implementation is for educational purposes. 
It is terrible inefficient and has not been tested properly. 
I don't recommend using it for any serious use case. 

If you make use of the code, cite the original algorithm:

```bash
@article{horak_bosansky_hsvi, 
title={Heuristic Search Value Iteration for One-Sided Partially Observable Stochastic Games}, 
volume={31}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/10597}, 
abstractNote={ &lt;p&gt; Security problems can be modeled as two-player partially observable stochastic games
 with one-sided partial observability and infinite horizon (one-sided POSGs). 
 We seek for optimal strategies of player 1 that correspond to robust strategies against the worst-case 
 opponent (player 2) that is assumed to have a perfect information about the game. 
 We present a novel algorithm for approximately solving one-sided POSGs based on 
 the heuristic search value iteration (HSVI) for POMDPs. 
 Our results include (1) theoretical properties of one-sided POSGs and their value functions, 
 (2) guarantees showing the convergence of our algorithm to optimal strategies, 
 and (3) practical demonstration of applicability and scalability of our algorithm on 
 three different domains: pursuit-evasion, patrolling, and search games. &lt;/p&gt; }, 
 number={1}, 
 journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
 author={Horák, Karel and Bošanský, Branislav and Pěchouček, Michal}, 
 year={2017}, 
 month={Feb.} }
```

## PWLC value function

<p align="center">
<img src="imgs/tiger_values.png" width="600">
</p>

## Installation

### Install from pypi


### Local installation

```bash
git clone https://github.com/Limmen/os_posg_hsvi_py
cd os_posg_hsvi_py
pip install -e . 
```

## Example

```python
import os_posg_hsvi_py.instances.stopping_intrusion_game_os_posg as stopping_game
import os_posg_hsvi_py.util.util
from os_posg_hsvi_py import os_posg_solvers

Z = stopping_game.observation_tensor()
R = stopping_game.reward_tensor()
T = stopping_game.transition_tensor()
A1, _ = stopping_game.player_1_actions()
A2, _ = stopping_game.player_2_actions()
O, _ = stopping_game.observations()
S, _ = stopping_game.states()
b0 = stopping_game.initial_belief()
os_posg_hsvi_py.util.util.set_seed(1521245)
os_posg_hsvi_py.os_posg_solvers.os_posg_hsvi.hsvi(O=O, Z=Z, R=R, T=T, A1=A1, A2=A2, S=S, gamma=0.9, b0=b0, epsilon=0.01,
                                                  prune_frequency=100, verbose=True, simulation_frequency=1,
                                                  simulate_horizon=100,
                                                  number_of_simulations=50, D=None)
```

## Contributions

Contributions are welcome, please use Github pull requests and issues.

## Author & Maintainer

Kim Hammar <kimham@kth.se>

## Copyright and license

[LICENSE](LICENSE.md)

Creative Commons

(C) 2022, Kim Hammar