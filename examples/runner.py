"""
Solves different simple discrete control/game problems
"""

import instances.tiger_problem_pomdp as tiger_problem
import instances.stopping_intrusion_game_os_posg as stopping_game
import os_posg_hsvi.instances.lottery_game_sg as lottery_game_sg
import instances.machine_replacement_mdp as machine_replacement_mdp
import util.util
from os_posg_hsvi import pomdp_solvers, os_posg_solvers
import os_posg_hsvi.sg_solvers.shapley_iteration_fully_observed
import mdp_solvers.vi


def sg_lottery_game():
    R = lottery_game_sg.reward_tensor()
    T = lottery_game_sg.transition_tensor()
    A1, _ = lottery_game_sg.player_1_actions()
    A2, _ = lottery_game_sg.player_2_actions()
    S, _ = lottery_game_sg.states()
    util.util.set_seed(1521245)
    os_posg_hsvi.sg_solvers.shapley_iteration_fully_observed.shapley_iteration(S=S, A1=A1, A2=A2, R=R, T=T, gamma=1,
                                                                               max_iterations=1000, delta_threshold=0.05, log=True)


def pomdp_tiger():
    Z = tiger_problem.observation_matrix()
    R = tiger_problem.reward_matrix()
    T = tiger_problem.transition_tensor()
    A, _ = tiger_problem.actions()
    O, _ = tiger_problem.observations()
    S, _ = tiger_problem.states()
    b0 = tiger_problem.initial_belief()
    util.util.set_seed(1521245)
    pomdp_solvers.hsvi.hsvi(O=O, Z=Z, R=R, T=T, A=A, S=S, gamma=0.9, b0=b0, epsilon=0.01,
                            lp=False, prune_frequency=100, verbose=False, simulation_frequency=10, simulate_horizon=100,
                            number_of_simulations=50)


def mdp_machine_replacement():
    S, _ = machine_replacement_mdp.states()
    A, _ = machine_replacement_mdp.actions()
    T = machine_replacement_mdp.transition_tensor()
    R = machine_replacement_mdp.reward_matrix()
    V, policy = \
        mdp_solvers.vi.value_iteration(T=T, num_states=len(S), num_actions=len(A), R=R, theta=0.0001,
                                       discount_factor=0.99)


def os_posg_stopping_game():
    Z = stopping_game.observation_tensor()
    R = stopping_game.reward_tensor()
    T = stopping_game.transition_tensor()
    A1, _ = stopping_game.player_1_actions()
    A2, _ = stopping_game.player_2_actions()
    O, _ = stopping_game.observations()
    S, _ = stopping_game.states()
    b0 = stopping_game.initial_belief()
    util.util.set_seed(1521245)
    os_posg_solvers.os_posg_hsvi.hsvi_os_posg(O=O, Z=Z, R=R, T=T, A1=A1, A2=A2, S=S, gamma=0.9, b0=b0, epsilon=0.01,
                 prune_frequency=100, verbose=True, simulation_frequency=1, simulate_horizon=100,
                 number_of_simulations=50, D=None)


if __name__ == '__main__':
    # pomdp_tiger()
    # os_posg_stopping_game()
    # sg_lottery_game()
    mdp_machine_replacement()