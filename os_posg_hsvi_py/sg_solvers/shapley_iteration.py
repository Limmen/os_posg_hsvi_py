from typing import  Tuple
import numpy as np
import pulp


def auxillary_game(V: np.ndarray, gamma :float, S: np.ndarray, s: int,
                   A1: np.ndarray, A2: np.ndarray, R: np.ndarray,
                   T: np.ndarray) -> np.ndarray:
    """
    Creates an auxillary matrix game based on the value function V

    :param V: the value function
    :param gamma: the discount factor
    :param S: the set of states
    :param s: the state s
    :param A1: the set of actions of player 1
    :param A2: the set of actions of player 2
    :param R: the reward tensor
    :param T: the transition tensor
    :return: the matrix auxillary game
    """
    A = np.zeros((len(A1), len(A2)))
    for a1 in A1:
        for a2 in A2:
            immediate_reward = R[a1][a2][s]
            expected_future_reward = 0
            for s_prime in S:
                expected_future_reward += T[a1][a2][s][s_prime]*V[s_prime]
            expected_future_reward = expected_future_reward*gamma
            A[a1][a2]=immediate_reward + expected_future_reward
    return A


def compute_matrix_game_value(A: np.ndarray, A1: np.ndarray, A2: np.ndarray, maximizer: bool = True):
    """

    :param A: the matrix game
    :param A1: the set of actions of player 1
    :param A2: the set of acitons of player 2
    :param maximizer: a boolean flag indicating whether the maximin or minimax strategy should be computed
    :return: (val(A), maximin/minimax)
    """
    if maximizer:
        problem = pulp.LpProblem("AuxillaryGame", pulp.LpMaximize)
        Ai = A1
    else:
        problem = pulp.LpProblem("AuxillaryGame", pulp.LpMinimize)
        Ai = A2

    # Decision variables, strategy-weights
    s = []
    for ai in Ai:
        si = pulp.LpVariable("s_" + str(ai), lowBound=0, upBound=1, cat=pulp.LpContinuous)
        s.append(si)

    # Auxillary decision variable, value of the game v
    v = pulp.LpVariable("v", lowBound=None, upBound=None, cat=pulp.LpContinuous)

    # The objective function
    problem += v, "Value of the game"

    # The constraints
    if maximizer:
        for j in range(A.shape[1]):
            sum = 0
            for i in range(A.shape[0]):
                sum += s[i] * A[i][j]
            problem += sum >= v, "SecurityValueConstraint_" + str(j)
    else:
        for i in range(A.shape[0]):
            sum = 0
            for j in range(A.shape[1]):
                sum += s[j] * A[i][j]
            problem += sum <= v, "SecurityValueConstraint_" + str(i)

    strategy_weights_sum = 0
    for si in s:
        strategy_weights_sum += si
    problem += strategy_weights_sum == 1, "probabilities sum"

    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    # Obtain solution
    optimal_strategy = list(map(lambda x: x.varValue, s))
    value = v.varValue
    return value, optimal_strategy


def si(S: np.ndarray, A1: np.ndarray, A2: np.ndarray, R: np.ndarray, T: np.ndarray,
       gamma : float = 1, max_iterations : int = 500,
       delta_threshold : float = 0.1, log = False) -> Tuple[np.ndarray, np.ndarray,
                                                                           np.ndarray, np.ndarray]:
    """
    Shapley Iteration (L. Shapley 1953)

    :param S: the set of states of the SG
    :param A1: the set of actions of player 1 in the SG
    :param A2: the set of actions of player 2 in the SG
    :param R: the reward tensor in the SG
    :param T: the transition tensor in the SG
    :param gamma: the discount factor
    :param max_iterations: the maximum number of iterations
    :param delta_threshold: the stopping threshold
    :param log: a boolean flag whether to use verbose logging or not
    :return: the value function, the set of maximin strategies for all stage games,
    the set of minimax strategies for all stage games, and the stage games themselves
    """
    num_states = len(S)

    V = np.zeros(num_states)

    for i in range(max_iterations):
        delta = 0.0
        auxillary_games = []
        for s in S:
            A = auxillary_game(V=V, gamma=gamma, S=S, s=s, A1=A1, A2=A2, R=R, T=T)
            auxillary_games.append(A)

        for s in S:
            value, _ = compute_matrix_game_value(A=auxillary_games[s],A1=A1, A2=A2, maximizer=True)
            delta += abs(V[s] - value)
            V[s] = value

        if log:
            print(f"i:{i}, delta: {delta}, V: {V}")

        if delta <= delta_threshold:
            break

    maximin_strategies = []
    minimax_strategies = []
    auxillary_games = []
    for s in S:
        A = auxillary_game(V=V, gamma=gamma, S=S, s=s, A1=A1, A2=A2, R=R, T=T)
        v1, maximin_strategy = compute_matrix_game_value(A=A,A1=A1, A2=A2, maximizer=True)
        v2, minimax_strategy = compute_matrix_game_value(A=A, A1=A1, A2=A2, maximizer=False)
        maximin_strategies.append(maximin_strategy)
        minimax_strategies.append(minimax_strategy)
        auxillary_games.append(A)

    return V, np.array(maximin_strategies), np.array(minimax_strategies), np.array(auxillary_games)


if __name__ == '__main__':
    pass
    # V, maximin_strategies, minimax_strategies, auxillary_games = shapley_iteration(
    #     gamma=1, iterations=100, delta_threshold = 0.0, log=True)
    # for s in range(len(V)):
    #     print(f"state s: {s}, value:{V[s]}")
    #     print(f"Game: \n {auxillary_games[s]}")
    #     print(f"P1 strategy: {maximin_strategies[s]}, P2 strategy: {minimax_strategies[s]}")