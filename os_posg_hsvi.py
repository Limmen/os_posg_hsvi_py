from typing import List, Tuple
import numpy as np
import pulp
from instances import stopping_intrusion_game_os_posg as stopping_game
from sg_solvers import shapley_iteration_fully_observed as shapley
from mdp_solvers import vi as vi
import random
import itertools


def hsvi_os_posg(O: np.ndarray, Z: np.ndarray, R: np.ndarray, T: np.ndarray, A1: np.ndarray,
                 A2: np.ndarray, S: np.ndarray, gamma: float, b0: np.ndarray,
                 epsilon: float, prune_frequency: int = 10, verbose=False,
                 simulation_frequency: int = 10, simulate_horizon: int = 10, number_of_simulations: int = 10,
                 D: float = None):
    """
    Heuristic Search Value Iteration for zero-sum OS-POSGs (Horak, Bosansky, Pechoucek, 2017)

    The value function represents the utility player 1 can achieve in each possible initial belief of the game.

    :param O: set of observations of the OS-POSG
    :param Z: observation tensor of the OS-POSG
    :param R: reward tensor of the OS-POSG
    :param T: transition tensor of the OS-POSG
    :param A1: action set of P1 in the OS-POSG
    :param A2: action set of P2 in the OS-POSG
    :param S: state set of the OS-POSG
    :param gamma: discount factor
    :param b0: initial belief point
    :param epsilon: accuracy parameter
    :param prune_frequency: how often to prune the upper and lower bounds
    :param verbose: verbose flag for logging
    :param simulation_frequency: how frequently to simulate the OS-POSG to compure rewards of current policy
    :param simulate_horizon: length of simulations to compute rewards
    :param number_of_simulations: number of simulations to estimate reward
    :param D: neighborhood parameter
    :return: None
    """
    lower_bound = initialize_lower_bound(S=S, A1=A1, A2=A2, gamma=gamma, b0=b0)
    upper_bound = initialize_upper_bound(T=T, R=R, A1=A1, A2=A2, S=S, gamma=gamma)

    if verbose:
        print(f"init LB:{lower_bound},\n init UB:{upper_bound}")

    delta = compute_delta(S=S, A1=A1, A2=A2, gamma=gamma)

    if D is None:
        D = sample_D(gamma=gamma, epsilon=epsilon, delta=delta)

    excess_val, w = excess(lower_bound=lower_bound, upper_bound=upper_bound, b=b0, S=S, epsilon=epsilon,
                           gamma=gamma, t=0, delta=delta, D=D)

    iteration = 0
    cumulative_r = 0

    while excess_val > 0:

        lower_bound, upper_bound = explore(
            b=b0, epsilon=epsilon, t=0, lower_bound=lower_bound, upper_bound=upper_bound,
            gamma=gamma, S=S, O=O, R=R, T=T, A1=A1, A2=A2, Z=Z, delta=delta, D=D)

        excess_val, w = excess(lower_bound=lower_bound, upper_bound=upper_bound, b=b0, S=S, epsilon=epsilon,
                               gamma=gamma, t=0, delta=delta, D=D)

        if iteration % simulation_frequency == 0:
            print("TODO simulation")

        if iteration > 1 and iteration % prune_frequency == 0:
            size_before_lower_bound = len(lower_bound)
            size_before_upper_bound = len(upper_bound)
            lower_bound = prune_lower_bound(lower_bound=lower_bound, S=S)
            upper_bound = prune_upper_bound(upper_bound=upper_bound, delta=delta)
            if verbose:
                print(f"Pruning, LB before:{size_before_lower_bound},after:{len(lower_bound)}, "
                      f"UB before: {size_before_upper_bound},after:{len(upper_bound)}")

        initial_belief_V_star_upper = upper_bound_value(upper_bound=upper_bound, b=b0, delta=delta)
        initial_belief_V_star_lower = lower_bound_value(lower_bound=lower_bound, b=b0, S=S)
        iteration += 1

        print(f"iteration: {iteration}, excess: {excess_val}, w: {w}, epsilon: {epsilon}, R: {cumulative_r}, "
              f"UB size:{len(upper_bound)}, LB size:{len(lower_bound)}")
        if verbose:
            print(f"Upper V*[b0]: {initial_belief_V_star_upper}, "
                  f"Lower V*[b0]:{initial_belief_V_star_lower}")

    with open('aleph_t.npy', 'wb') as f:
        np.save(f, lower_bound)


def compute_delta(S: np.ndarray, A1: np.ndarray, A2: np.ndarray, gamma: float) -> float:
    """
    The optimal value function V* of a OS-POSG is delta-Lipschitz continuous.
    To prove convergence of HSVI, we require that V_UB and V_LB are delta-Lipschitz continuous as well.
    This function computes the delta value according to (Horák, Bosansky, Kovařík, Kiekintveld, 2020)
    Specifically, delta = (U-L)/2 where L and U are teh lower respectively upper bounds on the game values.

    :param S: the set of states of the OS-POSG
    :param A1: the set of actions of player 1
    :param A2: the set of actions of player 2
    :param gamma: the discount factor
    :return: the delta value
    """
    temp = []
    for s in S:
        for a1 in A1:
            for a2 in A2:
                temp.append(R[a1][a2][s] / (1 - gamma))
    L = min(temp)
    U = max(temp)
    delta = (U - L) / 2
    return delta


def sample_D(gamma: float, epsilon: float, delta: float) -> float:
    """
    Samples the neighborhood parameter in the legal range.

    To ensure that the sequence rho is monotonically increasing and unbounded, we need to select the parameter D
    from the interval (0, (1-gamma)*epsilon/2delta)

    :param gamma: the discount factor
    :param epsilon: the epsilon accuracy parameter
    :param delta: the Lipschitz-continuity parameter
    :return: the neighborhood parameter D
    """
    min_val = 1.e-10
    max_val = ((1 - gamma) * epsilon) / (2 * delta)
    return (max_val - min_val) / 2


def obtain_equilibrium_strategy_profiles_in_stage_game(
        lower_bound: List, upper_bound: List, b: np.ndarray, delta: float,
        S: np.ndarray, A1: np.ndarray, A2: np.ndarray, gamma: float) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes equilibrium strategy profiles in the stage game constructed from the lower and upper bound value functions

    :param lower_bound: the lower bound  of V
    :param upper_bound:  the upper bound of V
    :param b: the belief point
    :param delta: the Lipschitz-continuity parameter
    :param S: the set of states
    :param A1: the set of acitons of player 1
    :param A2: the set of actions of player 2
    :param gamma: the discount factor
    :return: Equilibrium strategy profiles in the upper and lower bound stage games
    """

    upper_bound_stage_game = np.zeros((len(A1), len(S) * len(A2)))
    lower_bound_stage_game = np.zeros((len(A1), len(S) * len(A2)))

    p2_policies = []
    policy_combinations = np.array(list(itertools.product(S, A2)))
    for p in policy_combinations:
        policy = np.zeros((len(S), len(A2)))
        policy[0][p[0]] = 1
        policy[1][p[1]] = 1
        p2_policies.append(policy)

    for a1 in A1:
        for i, pi_2 in enumerate(p2_policies):
            immediate_reward = 0
            expected_future_reward_upper_bound = 0
            expected_future_reward_lower_bound = 0
            payoff_upper_bound = 0
            payoff_lower_bound = 0
            for s in S:
                a2 = int(np.argmax(pi_2[s]))
                immediate_reward += b[s] * R[a1][a2][s]
                for o in O:
                    new_belief = next_belief(o=o, a1=a1, b=b, S=S, Z=Z, T=T, pi_2=pi_2)
                    upper_bound_new_belief_value = upper_bound_value(upper_bound=upper_bound, b=new_belief, delta=delta)
                    lower_bound_new_belief_value = lower_bound_value(lower_bound=lower_bound, b=b, S=S)
                    prob = p_o_given_b_a1_a2(o=o, b=b, a1=a1, a2=a2, S=S, Z=Z)
                    expected_future_reward_upper_bound += prob * upper_bound_new_belief_value
                    expected_future_reward_lower_bound += prob * lower_bound_new_belief_value

            expected_future_reward_upper_bound = gamma * expected_future_reward_upper_bound
            expected_future_reward_lower_bound = gamma * expected_future_reward_lower_bound
            payoff_upper_bound += immediate_reward + expected_future_reward_upper_bound
            payoff_lower_bound += immediate_reward + expected_future_reward_lower_bound
            upper_bound_stage_game[a1][i] = payoff_upper_bound
            lower_bound_stage_game[a1][i] = payoff_upper_bound

    upper_bound_equlibrium_strategies = compute_equilibrium_strategies_in_matrix_game(
        A=upper_bound_stage_game, A1=A1, A2=np.array(list(range(len(S) * len(A2)))))
    lower_bound_equlibrium_strategies = compute_equilibrium_strategies_in_matrix_game(
        A=upper_bound_stage_game, A1=A1, A2=np.array(list(range(len(S) * len(A2)))))

    pi_1_upper_bound, temp = upper_bound_equlibrium_strategies
    pi_2_upper_bound = combine_weights_and_pure_strategies_into_mixed_strategy(weights=temp, strategies=p2_policies)
    pi_1_lower_bound, temp = lower_bound_equlibrium_strategies
    pi_2_lower_bound = combine_weights_and_pure_strategies_into_mixed_strategy(weights=temp, strategies=p2_policies)

    return pi_1_upper_bound, pi_2_upper_bound, pi_1_lower_bound, pi_2_lower_bound


def combine_weights_and_pure_strategies_into_mixed_strategy(weights: np.ndarray, strategies: np.ndarray):
    """
    Uses a set of mixture weights and strategies to compute a mixed strategy

    :param weights: the mixture weights
    :param strategies: the set of strategies
    :return:  the mixed strategy
    """
    mixed_strategy = np.zeros(strategies[0].shape)
    for i in range(len(weights)):
        mixed_strategy = mixed_strategy + strategies[i] * weights[i]
    return mixed_strategy


def compute_equilibrium_strategies_in_matrix_game(A: np.ndarray, A1: np.ndarray, A2: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes equilibrium strategies in a matrix game

    :param A: the matrix game
    :param A1: the action set of player 1 (the maximizer)
    :param A2: the action set of player 2 (the minimizer)
    :return: the equilibrium strategy profile
    """
    v1, maximin_strategy = compute_matrix_game_value(A=A, A1=A1, A2=A2, maximizer=True)
    v2, minimax_strategy = compute_matrix_game_value(A=A, A1=A1, A2=A2, maximizer=False)
    return maximin_strategy, minimax_strategy


def compute_matrix_game_value(A: np.ndarray, A1: np.ndarray, A2: np.ndarray, maximizer: bool = True) \
        -> Tuple[any, np.ndarray]:
    """
    Uses LP to compute the value of a a matrix game, also computes the maximin or minimax strategy

    :param A: the matrix game
    :param A1: the action set of player 1
    :param A2: the action set of player 2
    :param maximizer: boolean flag whether to compute the maximin strategy or minimax strategy
    :return: (val(A), maximin or minimax strategy)
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

    # Extract solution
    optimal_strategy = np.array(list(map(lambda x: x.varValue, s)))
    value = v.varValue

    return value, optimal_strategy


def explore(b: np.ndarray, epsilon: float, t: int, lower_bound: List, upper_bound: List,
            gamma: float, S: np.ndarray, O: np.ndarray, Z: np.ndarray, R: np.ndarray,
            T: np.ndarray, A1: np.ndarray, A2: np.ndarray, delta: float, D: float) \
        -> Tuple[List, List]:
    """
    Explores the OS-POSG tree

    :param b: current belief
    :param epsilon: accuracy parameter
    :param t: the current depth of the exploration
    :param lower_bound: the lower bound on the value function
    :param upper_bound: the upper bound on the value function
    :param gamma: discount factor
    :param S: set of states in the OS-POSG
    :param O: set of observations in the OS-POSG
    :param Z: observation tensor in the OS-POSG
    :param R: reward tensor in the OS-POSG
    :param T: transition tensor in the OS-POSG
    :param A1: set of actions of P1 in the OS-POSG
    :param A2: set of actions of P2 in the OS-POSG
    :param delta: the delta parameter in Lipschitz-continuity
    :return: new lower and upper bounds
    """
    pi_1_upper_bound, pi_2_upper_bound, pi_1_lower_bound, pi_2_lower_bound = \
        obtain_equilibrium_strategy_profiles_in_stage_game(lower_bound=lower_bound, upper_bound=upper_bound, b=b,
                                                           delta=delta, S=S, A1=A1, A2=A2, gamma=gamma)

    a_star, o_star, weighted_excess, excess_val, w, new_belief = choose_a_o_for_exploration(
        A1=A1, O=O, t=t + 1, b=b, pi_1_upper_bound=pi_1_upper_bound,
        pi_2_lower_bound=pi_2_lower_bound, lower_bound=lower_bound,
        upper_bound=upper_bound, gamma=gamma, epsilon=epsilon, delta=delta, D=D)

    print(f"weighted_excess: {weighted_excess}, w:{w}, excess_val:{excess_val}")

    if weighted_excess > 0:
        lower_bound, upper_bound = explore(b=new_belief, epsilon=epsilon, t=t + 1, lower_bound=lower_bound,
                                           upper_bound=upper_bound, gamma=gamma, S=S, O=O, Z=Z, R=R, T=T,
                                           A1=A1, A2=A2, delta=delta, D=D)

        lower_bound, upper_bound = \
            local_updates(lower_bound=lower_bound, upper_bound=upper_bound, b=b, A1=A1, A2=A2, S=S, Z=Z, O=O, R=R,
                          T=T, gamma=gamma, delta=delta)

    return lower_bound, upper_bound


def choose_a_o_for_exploration(A1: np.ndarray, O: np.ndarray, t: int, b: np.ndarray, pi_1_upper_bound: List,
                               pi_2_lower_bound: List,
                               lower_bound: List, upper_bound: List, gamma: float, epsilon: float,
                               delta: float, D: float) -> Tuple[int, int, float, float, float, np.ndarray]:
    """
    Selects the action a* and observation * for exploration according to the heuristic:

    (a1*,o*) = argmax_(a1,o)[P[a1,o]*excess(b_1(a_1,o))]

    (Horák, Bosansky, Kovařík, Kiekintveld, 2020)

    :param A1: the action set of player 1
    :param O: the set of observations in the OS-POSG
    :param t: the time-step t
    :param b: the belief point
    :param pi_1_upper_bound: equilibrium strategy of player 1 in the stage game constructed from the upper bound V*
    :param pi_2_lower_bound: equilibrium strategy of player 2 in the stage game constructed from the lower bound V*
    :param lower_bound: the lower bound
    :param upper_bound: the upper bound
    :param gamma: the discount factor
    :param epsilon: the epsilon accuracy parameter
    :param delta: the Lipschitz-continuity parameter
    :param D: the neighboorhood parameter
    :return: a*,o*,weighted_excess(a*,o*), excess(a*,o*), width(a*,o*), b_prime(a*,o*)
    """
    weighted_excess_values = []
    widths = []
    excess_values = []
    a_o_list = []
    new_beliefs = []
    for a1 in A1:
        for o in O:
            weighted_excess_val, excess_val, w, new_belief = weighted_excess_gap(lower_bound=lower_bound,
                                                                                 upper_bound=upper_bound, a1=a1, o=o,
                                                                                 b=b, t=t,
                                                                                 gamma=gamma,
                                                                                 pi_1_upper_bound=pi_1_upper_bound,
                                                                                 pi_2_lower_bound=pi_2_lower_bound,
                                                                                 epsilon=epsilon, delta=delta, D=D)
            weighted_excess_values.append(weighted_excess_val)
            widths.append(w)
            excess_values.append(excess_val)
            a_o_list.append([a1, o])
            new_beliefs.append(new_belief)

    max_index = int(np.argmax(weighted_excess_values))
    max_a_o = a_o_list[max_index]
    a_star = max_a_o[0]
    o_star = max_a_o[1]

    return a_star, o_star, weighted_excess_values[max_index], excess_values[max_index], \
           widths[max_index], new_beliefs[max_index]


def weighted_excess_gap(lower_bound: List, upper_bound: List, a1: int, o: int, b: np.ndarray, t: int,
                        gamma: float, pi_1_upper_bound, pi_2_lower_bound, epsilon: float, delta: float, D: float) \
        -> Tuple[float, float, float, np.ndarray]:
    """
    Computes the weighted excess gap

    :param lower_bound: the lower bound
    :param upper_bound: the upper bound
    :param a1: the action of player 1
    :param o: the observation
    :param b: the belief point
    :param t: the time-step
    :param gamma: the discount factor
    :param pi_1_upper_bound a maximin strategy of player 1 in the stage game based on upper bound V*
    :param pi_2_lower_bound: a minimax strategy of player 1 in the stage game based on lower bound V*
    :param epsilon: the epsilon accuracy parameter
    :param delta: the  Lipschitz-continuity parameter
    :param D: the neighborhood parameter
    :return: (weighted excess, excess, width, b_prime)
    """
    new_belief = next_belief(o=o, a1=a1, b=b, S=S, Z=Z, T=T, pi_2=pi_2_lower_bound)
    excess_val, w = excess(lower_bound=lower_bound, upper_bound=upper_bound, b=new_belief, S=S, epsilon=epsilon,
                           gamma=gamma, t=t, delta=delta, D=D)
    weight = p_o_given_b_pi_1_pi_2(o=o, b=b, pi_1=pi_1_upper_bound, pi_2=pi_2_lower_bound, S=S, Z=Z, A1=A1, A2=A2)
    return weight * excess_val, excess_val, w, new_belief


def initialize_lower_bound(S: np.ndarray, A1: np.ndarray, A2: np.ndarray, gamma: float, b0: np.ndarray) -> List:
    """
    Initializes the lower bound by computing the state-values of the POMDP induced by player 1 playing a uniform
    strategy

    :param S: the set of states
    :param A1: the set of actions of player 1
    :param A2: the set of actions of player 2
    :param gamma: the discount factor
    :param b0: the initial belief point
    :return: the lower bound (singleton set with an alpha vector)
    """
    uniform_strategy = np.zeros(len(A1))
    uniform_strategy.fill(1 / len(A1))
    alpha_vector, _ = value_of_p1_strategy_static(S=S, A1=A1, A2=A2, gamma=gamma, P1_strategy=uniform_strategy, b0=b0)
    lower_bound = []
    lower_bound.append(alpha_vector)
    return lower_bound


def value_of_p1_strategy_static(S: np.ndarray, A1: np.ndarray, A2: np.ndarray, gamma: float,
                                P1_strategy: np.ndarray, b0: np.ndarray) \
        -> Tuple[np.ndarray, float]:
    """
    Computes the value of PI's strategy P1_strategy, assuming that P1's strategy is static and independent of
    observations/actions/beliefs in the game. For example a uniform strategy.

    The value function is computed by solving the Bellman equation of the induced MDP for P2.

    :param S: the set of states of the OS-POSG
    :param A1: the set of actions of P1 in the OS-POSG
    :param A2: the set of actions of P2 in the OS-POSG
    :param gamma: the discount factor
    :param P1_strategy: the static strategy of P1 to evaluate
    :param b0: the initial belief
    :return: the value vector and the value given the initial belief b0.
    """
    R_mdp = stopping_game.mdp_reward_matrix_p2(P1_strategy=P1_strategy, A1=A1)
    T_mdp = stopping_game.mdp_transition_tensor_p2(P1_strategy=P1_strategy, A1=A1)
    V, _ = vi.value_iteration(T=T_mdp, num_states=len(S), num_actions=len(A2), R=R_mdp,
                              theta=0.0001, discount_factor=gamma)
    V = np.array(V)
    b0 = np.array(b0)
    return V, b0.dot(V)


def valcomp(pi_1: np.ndarray, alpha_bar: np.ndarray, s: int, A1: np.ndarray, A2: np.ndarray,
            O: np.ndarray, S: np.ndarray, Z: np.ndarray,
            T: np.ndarray, R: np.ndarray, gamma: float, substituted_alpha: bool = False) -> float:
    """
    Computes the value of a compositional strategy of player 1 in a given state s.
    The compositional strategy consists of the one-stage strategy pi_1 (a probability distribution over A1)
    and the expected value when taking action a according to pi_1 and then observing the next observation o and then
    following a behavioral strategy C^(a,o) \in (Sigma_1)^(O x A). It is assumed that the value of following strategy
    C^(a,o) is represented by alpha^(a,o) \in alpha_bar. Further, to compute the value, since the game is zero-sum,
    we assume that P2 plays a best response against the compositional strategy.

    :param pi_1: the one-stage strategy of P1 to evaluate
    :param alpha_bar: the set of alpha vectors representing the value of the behavioral strategies in
                      subsequent subgames
    :param s: the state to evaluate the value of
    :param A1: the action set of player 1 in the OS-POSG
    :param A2: the action set of player 2 in the OS-POSG
    :param O: the set of observations in the OS-POSG
    :param S: the set of states in the OS-POSG
    :param Z: the observation tensor in the OS-POSG
    :param T: the transition tensor in the OS-POSG
    :param R: the reward tensor in the OS-POSG
    :param substituted_alpha: if True, treat the alpha vectors as alpha^(a,o)(s) = pi_1[a]alpha^(a,o)(s)
    :return: the value of the compositional strategy of P1
    """
    values = []
    for a2 in A2:
        for a1 in A1:
            immediate_reward = R[a1][a2][s]
            expected_future_reward = 0
            for o in O:
                for s_prime in S:
                    expected_future_reward += Z[a1][a2][s_prime][o] * T[a1][a2][s][s_prime] * alpha_bar[a1][o][s_prime]
            expected_future_reward = gamma * expected_future_reward
            immediate_reward = pi_1[a1] * immediate_reward
            if not substituted_alpha:
                expected_future_reward = expected_future_reward * pi_1[a1]
            total_value = immediate_reward + expected_future_reward
            values.append(total_value)
    val = min(values)
    return val


def initialize_upper_bound(T: np.ndarray, R: np.ndarray, A1: np.ndarray, A2: np.ndarray,
                           S: np.ndarray, gamma: float) -> List:
    """
    Initializes the upper bound by computing the values of the fully observed version of the OS-POSG using
    Shapley iteration.

    :param T: the transition tensor of the OS-POSG
    :param R: the reward tensor of the OS-POSG
    :param A1: the action set of player 1 in the the OS-POSG
    :param A2: the action set of player 2 in the the OS-POSG
    :param S: the set of states of the OS-POSG
    :param gamma: the discount factor
    :return: the initial upper bound
    """
    V, maximin_strategies, minimax_strategies, auxillary_games = shapley.shapley_iteration(
        S=S, A1=A1, A2=A2, T=T, R=R, gamma=gamma, max_iterations=1000, delta_threshold=0.001, log=False)
    point_set = []
    for s in S:
        b = generate_corner_belief(s=s, S=S)
        point_set.append([b, V[s]])
    return point_set


def delta_lipschitz_envelope_of_upper_bound_value(upper_bound: List, b: np.ndarray, delta: float) -> float:
    """
    This function computes the delta-Lipschitz envelop of the upper bound value at a given belief point b.


    :param upper_bound: the upper bound
    :param b: the belief point
    :param delta: the delta parameter for Lipschitz-continuity
    :return: the belief value
    """
    problem = pulp.LpProblem("Delta-Lipschitz-Envelope", pulp.LpMinimize)

    # ----    Decision variables   ------

    # Convex hull coefficients
    lamb = []
    for i in range(len(upper_bound)):
        lamb_i = pulp.LpVariable("lambda_" + str(i), lowBound=0, upBound=1, cat=pulp.LpContinuous)
        lamb.append(lamb_i)

    # b_prime weights
    b_prime = []
    for i in range(len(b)):
        b_prime_i = pulp.LpVariable("b_prime_" + str(i), lowBound=0, upBound=1, cat=pulp.LpContinuous)
        b_prime.append(b_prime_i)

    # Delta variables
    state_deltas = []
    for i in range(len(b)):
        state_deltas_i = pulp.LpVariable("state_Delta_" + str(i), lowBound=0, upBound=1, cat=pulp.LpContinuous)
        state_deltas.append(state_deltas_i)

    # --- The objective ---

    # The objective function
    objective = 0
    for i, point in enumerate(upper_bound):
        objective += lamb[i] * point[1]
    for s in range(len(b)):
        objective += delta * state_deltas[s]
    problem += objective, "Lipschitz-Delta envelop"

    # --- The constraints ---

    # Belief probability constraint
    for j in range(len(S)):
        belief_sum = 0
        for i, point in enumerate(upper_bound):
            belief_sum += lamb[i] * point[0][j]
        problem += belief_sum == b_prime[j], "BeliefVectorConstraint_" + str(j)

    # Delta s constraints
    for s in range(len(b)):
        problem += state_deltas[s] >= (b_prime[s] - b[s]), "Delta_s_constraint_1_" + str(s)
        problem += state_deltas[s] >= (b[s] - b_prime[s]), "Delta_s_constraint_2_" + str(s)

    # Convex Hull constraint
    lambda_weights_sum = 0
    for i in range(len(lamb)):
        lambda_weights_sum += lamb[i]
    problem += lambda_weights_sum == 1, "ConvexHullWeightsSumConstraint"

    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract solution
    projected_lamb_coefficients = []
    belief_value = 0
    for i in range(len(upper_bound)):
        projected_lamb_coefficients.append(lamb[i].varValue)
        belief_value += projected_lamb_coefficients[i] * upper_bound[i][1]

    state_deltas_coefficients = []
    state_deltas_sum = 0
    for s in range(len(b)):
        state_deltas_coefficients.append(state_deltas[s].varValue)
        state_deltas_sum += state_deltas_coefficients[s]
    state_deltas_sum = delta * state_deltas_sum

    belief_value += state_deltas_sum

    return belief_value


def maxcomp_shapley_bellman_operator(Gamma: np.ndarray, A1: np.ndarray, S: np.ndarray, O: np.ndarray, A2: np.ndarray,
                                     gamma: float, b: np.ndarray, R, T, Z) -> Tuple[np.ndarray, np.ndarray]:
    """
    A dear child with many names: Maxcomp/Shapley/Bellman operator that computes [HV](b) where V is represented by
    the pointwise maximum over the convex hull of a set of alpha vectors Gamma.
    By representing the set of vectors as the convex hull, the solution to the operator can be found by an LP.
    I.e the whole purpose of using the convex hull of Gamma is to be able to compute the operator through LP.

    The operator is represented as the solution to a linear program given in
    (Karel Horák PhD thesis, 2019).

    That is, the operator performs a backup to update the lower bound in HSVI or to perform iterations in exact
    value iteration for OS-POSGs.

    :param Gamma: the set of alpha vectors representing the value function
    :param A1: the action space of player 1
    :param S: the set of states in the OS-POSG
    :param O: the set of observations in the OS-POSG
    :param A2: the action space of player 2 in the OS-POSG
    :param gamma: the discount factor
    :param b: the belief point
    :param R: the reward tensor of the OS-POSG
    :param T: the transition tensor of the OS-POSG
    :param Z: the observation tensor of the OS-POSG
    :return: the "optimal" stage strategy of Player 1 and the set of alpha-vectors representing the  values of the
             behavioral strategy following a1,o for any combination of a1,o.
    """
    problem = pulp.LpProblem("ShapleyOperator", pulp.LpMaximize)

    # ----    Decision variables   ------

    # Strategy weights of the stage strategy pi_1 of player 1
    pi_1 = []
    for a1 in A1:
        pi_1_1 = pulp.LpVariable("pi_1_" + str(a1), lowBound=0, upBound=1, cat=pulp.LpContinuous)
        pi_1.append(pi_1_1)

    # State-value function
    V = []
    for s in S:
        V_1 = pulp.LpVariable("V_" + str(s), lowBound=None, upBound=None, cat=pulp.LpContinuous)
        V.append(V_1)

    # Convex hull coefficients of Conv(Gamma_k)
    lamb = []
    for a1 in A1:
        a_o_lambs = []
        for o in O:
            o_lambs = []
            for i in range(len(Gamma)):
                lamb_i = pulp.LpVariable("lambda_" + str(i) + "_" + str(o) + "_" + str(a1), lowBound=0,
                                         upBound=1, cat=pulp.LpContinuous)
                o_lambs.append(lamb_i)
            a_o_lambs.append(o_lambs)
        lamb.append(a_o_lambs)

    # Alpha bar: set of alpha^(a,o) vectors representing the value of the behavioral strategy C^(a,o) in the subgame
    # after taking action a and observing o in belief point b
    alpha_bar = []
    for a1 in A1:
        a_o_alphas = []
        for o in O:
            o_alphas = []
            for s in S:
                alpha_bar_i = pulp.LpVariable("alpha_bar_" + str(s) + "_" + str(o) + "_" + str(a1),
                                              lowBound=None, upBound=None, cat=pulp.LpContinuous)
                o_alphas.append(alpha_bar_i)
            a_o_alphas.append(o_alphas)
        alpha_bar.append(a_o_alphas)

    # The objective function
    objective = 0
    for s in S:
        objective = b[s] * V[s]
    problem += objective, "Expected state-value"

    # --- The constraints ---

    # State-Value function constraints
    for s in S:
        for a2 in A2:
            immediate_reward = 0
            for a1 in A1:
                immediate_reward += pi_1[a1] * R[a1][a2][s]

            expected_future_reward = 0
            for a1 in A1:
                for o in O:
                    for s_prime in S:
                        expected_future_reward += T[a1][a2][s][s_prime] * \
                                                  Z[a1][a2][s_prime][o] * alpha_bar[a1][o][s_prime]

            expected_future_reward = gamma * expected_future_reward
            sum = immediate_reward + expected_future_reward
            problem += sum >= V[s], "SecurityValueConstraint_" + str(s) + "_a2" + str(a2)

    # Alpha-bar constraints
    for a1 in A1:
        for o in O:
            for s_prime in S:
                weighted_alpha_sum = 0
                for i in range(len(Gamma)):
                    weighted_alpha_sum += lamb[a1][o][i] * Gamma[i][s_prime]
                problem += weighted_alpha_sum == alpha_bar[a1][o][s_prime], "AlphaBarConstraint_" \
                           + str(s_prime) + "_" + str(a1) + "_" + str(o)

    # Lambda constraints
    for a1 in A1:
        for o in O:
            lambda_sum = 0
            for i in range(len(Gamma)):
                lambda_sum += lamb[a1][o][i]

            problem += lambda_sum == pi_1[a1], "Lambconstraint_" + str(a1) + "_" + str(o)

    # Strategy constraints
    strategy_weights_sum = 0
    for i in range(len(pi_1)):
        strategy_weights_sum += pi_1[i]
    problem += strategy_weights_sum == 1, "probabilities sum"

    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    value = []
    for s in S:
        value.append(V[s].varValue)

    pi_1_val = []
    for a1 in A1:
        pi_1_val.append(pi_1[a1].varValue)

    lamba_1_val = []
    for a1 in A1:
        lamba_1_val_a1 = []
        for o in O:
            lamba_1_val_o = []
            for k in range(len(Gamma)):
                lamba_1_val_o.append(lamb[a1][o][k].varValue)
            lamba_1_val_a1.append(lamba_1_val_o)
        lamba_1_val.append(lamba_1_val_a1)

    alpha_bar_val = []
    for a1 in A1:
        alpha_1_val_a1 = []
        for o in O:
            alpha_1_val_o = []
            for s in S:
                alpha_1_val_o.append(alpha_bar[a1][o][s].varValue)
            alpha_1_val_a1.append(alpha_1_val_o)
        alpha_bar_val.append(alpha_1_val_a1)

    return np.array(pi_1_val), np.array(alpha_bar_val)


def generate_corner_belief(s: int, S: np.ndarray):
    """
    Generate the corner of the simplex that corresponds to being in some state with probability 1

    :param s: the state
    :param S: the set of States
    :return: the corner belief corresponding to state s
    """
    b = np.zeros(len(S))
    b[s] = 1
    return b


def local_updates(lower_bound: List, upper_bound: List, b: np.ndarray, A1: np.ndarray,
                  A2: np.ndarray, S: np.ndarray,
                  O: np.ndarray, R: np.ndarray, T: np.ndarray, gamma: float, Z: np.ndarray, delta: float)\
        -> Tuple[List, List]:
    """
    Perform local updates to the upper and  lower bounds for the given belief in the heuristic-search-exploration

    :param lower_bound: the lower bound on V
    :param upper_bound: the upper bound on V
    :param b: the current belief point
    :param A1: the set of actions of player 1 in the OS-POSG
    :param A2: the set of actions of player 2 in the OS-POSG
    :param S: the set of states in the OS-POSG
    :param O: the set of observations in the OS-POSG
    :param R: the reward tensor in the OS-POSG
    :param T: the transition tensor in the OS-POSG
    :param gamma: the discount factor
    :param Z: the set of observations in the OS-POSG
    :param delta: the Lipschitz-Delta parameter
    :return: The updated lower and upper bounds
    """
    new_upper_bound = local_upper_bound_update(upper_bound=upper_bound, b=b, A1=A1, A2=A2,
                                               S=S, O=O, R=R, T=T, gamma=gamma,
                                               Z=Z, delta=delta)
    new_lower_bound = local_lower_bound_update(lower_bound=lower_bound, b=b, Z=Z, A1=A1, A2=A2, O=O, S=S,
                                               T=T, R=R, gamma=gamma)
    return new_lower_bound, new_upper_bound


def local_upper_bound_update(upper_bound: List, b: np.ndarray, A1: np.ndarray, A2: np.ndarray,
                             S: np.ndarray, O: np.ndarray, R: np.ndarray, T: np.ndarray,
                             gamma: float, Z: np.ndarray, delta: float) -> np.ndarray:
    """
    Performs a local update to the upper bound during the heuristic-search exploration

    :param upper_bound: the upper bound to update
    :param b: the current belief point
    :param A1: the set of actions of player 1 in the OS-POSG
    :param S: the set of states in the OS-POSG
    :param O: the set of observations in the OS-POSG
    :param R: the reward tensor in the OS-POSG
    :param T: the transition tensor in the OS-POSG
    :param gamma: the discount factor in the OS-POSG
    :param Z: the set of observations in the OS-POSG
    :param delta: the Lipschitz-Delta parameter
    :return: the updated upper bound
    """
    new_val = upper_bound_backup(upper_bound=upper_bound, b=b, A1=A1, A2=A2, S=S,
                                 Z=Z, O=O, R=R, T=T, gamma=gamma, delta=delta)
    upper_bound.append([b, new_val])
    return upper_bound


def local_lower_bound_update(lower_bound: List, b: np.ndarray, A1: np.ndarray,
                             A2: np.ndarray, O: np.ndarray, Z: np.ndarray, S: np.ndarray,
                             T: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """
    Performs a local update to the lower bound given a belief point in the heuristic search.

    The lower bound update preserves the lower bound property and the Delta-Lipschitz continuity property.

    :param lower_bound: the current lower bound
    :param b: the current belief point
    :param A1: the set of actions of player 1 in the OS-POSG
    :param A2: the set of actions of player 2 in the OS-POSG
    :param O: the set of observations in the OS-POSG
    :param Z: the observation tensor in the OS-POSG
    :param S: the set of states in the OS-POSG
    :param T: the transition tensor in the OS-POSG
    :param R: the reward tensor in the OS-POSG
    :param gamma: the discount factor in the OS-POSG
    :return: the updated lower bound
    """
    alpha_vec = lower_bound_backup(lower_bound=lower_bound, b=b, A1=A1, Z=Z, O=O, S=S, T=T, R=R, gamma=gamma)
    if not check_duplicate(lower_bound, alpha_vec):
        lower_bound.append(alpha_vec)
    return lower_bound


def lower_bound_backup(lower_bound: List, b: np.ndarray, A1: np.ndarray, O: np.ndarray,
                       Z: np.ndarray, S: np.ndarray,
                       T: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """
    Generates a new alpha-vector for the lower bound

    :param lower_bound: the current lower bound
    :param b: the current belief point
    :param A1: the set of actions of player 1 in the OS-POSG
    :param A2: the set of actions of player 2 in the OS-POSG
    :param O: the set of observations in the OS-POSG
    :param Z: the observation tensor in the OS-POSG
    :param S: the set of states in the OS-POSG
    :param T: the transition tensor in the OS-POSG
    :param R: the reward tensor in the OS-POSG
    :param gamma: the discount factor
    :return: the new alpha vector
    """

    # Shapley operator to obtain optimal value composition
    pi_1_LB, alpha_bar_LB = maxcomp_shapley_bellman_operator(Gamma=lower_bound, A1=A1, S=S, O=O, A2=A2,
                                                             gamma=gamma, b=b, R=R, T=T, Z=Z)
    alpha_vec = []
    for s in S:
        s_val = valcomp(pi_1=pi_1_LB, alpha_bar=alpha_bar_LB, s=s, A1=A1, A2=A2, O=O, S=S, Z=Z, T=T, R=R,
                        gamma=gamma, substituted_alpha=True)
        alpha_vec.append(s_val)

    return alpha_vec


def upper_bound_backup(upper_bound: List, b: np.ndarray, A1: np.ndarray, A2: np.ndarray,
                       S: np.ndarray, O: np.ndarray, Z: np.ndarray, R: np.ndarray, T: np.ndarray,
                       gamma: float, delta: float) -> Tuple[np.ndarray, float]:
    """
    Adds a point to the upper bound

    :param upper_bound: the current upper bound
    :param b: the current belief point
    :param A1: the set of actions of player 1 in the OS-POSG
    :param A2: the set of actions of player 1 in the OS-POSG
    :param S: the set of states in the OS-POSG
    :param O: the set of observations in the OS-POSG
    :param Z: the observation tensor in the OS-POSG
    :param R: the reward tensor in the OS-POSG
    :param T: the transition tensor in the OS-POSG
    :param gamma: the discount factor in the OS-POSG
    :param lp: a boolean flag whether to use LP to compute the upper bound belief
    :param delta: the Lipschitz-delta parameter
    :return: the new point
    """
    problem = pulp.LpProblem("Local Upper Bound Backup", pulp.LpMinimize)

    # ----    Decision variables   -----

    # Policy weights of player 2
    pi_2 = []
    for s in S:
        pi_2_s = []
        for a2 in A2:
            pi_2_i = pulp.LpVariable("pi_2_" + str(s) + "_" + str(a2), lowBound=0, upBound=1, cat=pulp.LpContinuous)
            pi_2_s.append(pi_2_i)
        pi_2.append(pi_2_s)

    # V
    V = pulp.LpVariable("V", lowBound=None, upBound=None, cat=pulp.LpContinuous)

    # tau hat
    tau_hat = []
    for a1 in A1:
        a_o_tau_hats = []
        for o in O:
            o_tau_hats = []
            for s_prime in S:
                tau_hat_i = pulp.LpVariable("tau_hat_" + str(a1) + "_" + str(o) + "_" + str(s_prime),
                                            lowBound=0, upBound=1, cat=pulp.LpContinuous)
                o_tau_hats.append(tau_hat_i)
            a_o_tau_hats.append(o_tau_hats)
        tau_hat.append(a_o_tau_hats)

    # b_prime weights
    b_prime = []
    for a1 in A1:
        o_a1_b_primes = []
        for o in O:
            o_b_primes = []
            for s_prime in S:
                b_prime_i = pulp.LpVariable("b_prime_" + str(s_prime) + "_" + str(o) + "_" + str(a1),
                                            lowBound=0, upBound=1, cat=pulp.LpContinuous)
                o_b_primes.append(b_prime_i)
            o_a1_b_primes.append(o_b_primes)
        b_prime.append(o_a1_b_primes)

    # V-hat
    V_hat = []
    for a1 in A1:
        a_o_V_hats = []
        for o in O:
            V_hat_a_o_i = pulp.LpVariable("V_hat_" + str(a1) + "_" + str(o),
                                          lowBound=0, upBound=1, cat=pulp.LpContinuous)
            a_o_V_hats.append(V_hat_a_o_i)
        V_hat.append(a_o_V_hats)

    # Convex hull coefficients of the upper_bound_point_set
    lamb = []
    for a1 in A1:
        a_o_lambs = []
        for o in O:
            o_lambs = []
            for i in range(len(upper_bound)):
                lamb_i = pulp.LpVariable("lambda_" + str(i) + "_" + str(o) + "_" + str(a1), lowBound=0,
                                         upBound=1, cat=pulp.LpContinuous)
                o_lambs.append(lamb_i)
            a_o_lambs.append(o_lambs)
        lamb.append(a_o_lambs)

    # Delta variables
    state_action_observation_deltas = []
    for a1 in A1:
        a_o_state_deltas = []
        for o in O:
            o_state_deltas = []
            for s_prime in S:
                state_deltas_i = pulp.LpVariable("state_Delta_" + str(a1) + "_" + str(o) + "_" + str(s_prime),
                                                 lowBound=0, upBound=1, cat=pulp.LpContinuous)
                o_state_deltas.append(state_deltas_i)
            a_o_state_deltas.append(o_state_deltas)
        state_action_observation_deltas.append(a_o_state_deltas)

    # --- The objective ---

    # The objective function
    problem += V, "Upper bound local update objective"

    # --- The constraints ---

    # Value constraints
    for a1 in A1:
        sum = 0
        weighted_immediate_rew_sum = 0
        for a2 in A2:
            for s in S:
                weighted_immediate_rew_sum += b[s] * pi_2[s][a2] * R[a1][a2][s]
        future_val_sum = 0
        for o in O:
            future_val_sum += V_hat[a1][o]
        future_val_sum = gamma * future_val_sum
        sum += future_val_sum
        problem += V >= sum, "V_constraint_" + str(a1)

    # Tau-hat constraints
    for a1 in A1:
        for o in O:
            for s_prime in S:
                sum = 0
                for s in S:
                    for a2 in A2:
                        sum += T[a1][a2][s][s_prime] * pi_2[s][a2]
                problem += sum == tau_hat[a1][o][s_prime], "belief_constraint_" + str(a1) + "_" + str(o) \
                           + "_" + str(s_prime)

    # Pi_2 constraints
    for s in S:
        sum = 0
        for a2 in A2:
            sum += pi_2[s][a2]

        problem += sum == b[s], "pi_2_constraint_" + str(s)

    # V hat constraints
    for a1 in A1:
        for o in O:
            sum = 0
            for i, point in enumerate(upper_bound):
                sum += lamb[a1][o][i] * point[1]
            deltas_sum = 0
            for s_prime in S:
                deltas_sum += state_action_observation_deltas[a1][o][s_prime]
            deltas_sum = delta * deltas_sum
            sum += deltas_sum
            problem += sum == V_hat[a1][o], "V_hat_constraint_" + str(a1) + "_" + str(o)

    # Belief_prime constraints
    for a1 in A1:
        for o in O:
            for s_prime in S:
                sum = 0
                for i, point in enumerate(upper_bound):
                    sum += lamb[a1][o][i] * point[0][s_prime]
                problem += sum == b_prime[a1][o][s_prime], "b_prime constraint_" + str(a1) + "_" + str(o) \
                           + "_" + str(s_prime)

    # Deltas constraints
    for a1 in A1:
        for o in O:
            for s_prime in S:
                problem += state_action_observation_deltas[a1][o][s_prime] \
                           >= (b_prime[a1][o][s_prime] - tau_hat[a1][o][s_prime]), "Delta_contraints_1_" + \
                           str(a1) + "_" + str(o) + "_" + str(s_prime)
                problem += state_action_observation_deltas[a1][o][s_prime] >= (tau_hat[a1][o][s_prime] -
                                                                               b_prime[a1][o][s_prime]), \
                           "Delta_contraints_2_" + \
                           str(a1) + "_" + str(o) + "_" + str(s_prime)

    # Lambda constraints
    for a1 in A1:
        for o in O:
            lambdas_sum = 0
            for i, point in enumerate(upper_bound):
                lambdas_sum += lamb[a1][o][i]

            tau_hat_sum = 0
            for s_prime in S:
                tau_hat_sum += tau_hat[a1][o][s_prime]

            problem += lambdas_sum == tau_hat_sum, "lambda_constraint_" + str(a1) + "_" + str(o)

    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    # Obtain solution
    belief_value_var = V.varValue
    return belief_value_var


def upper_bound_value(upper_bound: List, b: np.ndarray, delta: float) -> float:
    """
    Computes the upper bound value of a given belief point

    :param upper_bound: the upper bound
    :param b: the belief point
    :param delta: the delta-parameter for Lipschitz-continuity
    :param lp: boolean flag that decides whether to use LP to compute the upper bound value or not
    :return: the upper bound value
    """
    return delta_lipschitz_envelope_of_upper_bound_value(upper_bound=upper_bound, b=b, delta=delta)


def lower_bound_value(lower_bound: List, b: np.ndarray, S: np.ndarray) -> float:
    """
    Computes the lower bound value of a given belief point

    :param lower_bound: the lower bound
    :param b: the belief point
    :param S: the set of states
    :return: the lower bound value
    """
    alpha_vals = []
    for alpha_vec in lower_bound:
        sum = 0
        for s in S:
            sum += b[s] * alpha_vec[s]
        alpha_vals.append(sum)
    return max(alpha_vals)


def next_belief(o: int, a1: int, b: np.ndarray, S: np.ndarray, Z: np.ndarray, T: np.ndarray, pi_2: np.ndarray) -> np.ndarray:
    """
    Computes the next belief using a Bayesian filter

    :param o: the latest observation
    :param a1: the latest action of player 1
    :param b: the current belief
    :param S: the set of states
    :param Z: the observation tensor
    :param T: the transition tensor
    :param pi_2: the policy of player 2
    :return: the new belief
    """
    b_prime = np.zeros(len(S))
    for s_prime in S:
        b_prime[s_prime] = bayes_filter(s_prime=s_prime, o=o, a1=a1, b=b, S=S, Z=Z, T=T, pi_2=pi_2, A2=A2)

    assert round(sum(b_prime), 5) == 1
    return b_prime


def bayes_filter(s_prime: int, o: int, a1: int, b: np.ndarray, S: np.ndarray, Z: np.ndarray,
                 T: np.ndarray, pi_2: np.ndarray, A2: np.ndarray) -> float:
    """
    A Bayesian filter to compute the belief of player 1
    of being in s_prime when observing o after taking action a in belief b given that the opponent follows
    strategy pi_2

    :param s_prime: the state to compute the belief of
    :param o: the observation
    :param a1: the action of player 1
    :param b: the current belief point
    :param S: the set of states
    :param Z: the observation tensor
    :param T: the transition tensor
    :param pi_2: the policy of player 2
    :param A2: the action set of player 2
    :return: b_prime(s_prime)
    """
    norm = 0
    for s in S:
        for a2 in A2:
            for s_prime_1 in S:
                prob_1 = Z[a1][a2][s_prime_1][o]
                norm += b[s] * prob_1 * T[a1][a2][s][s_prime_1] * pi_2[s][a2]
    temp = 0

    for s in S:
        for a2 in A2:
            temp += Z[a1][a2][s_prime][o] * T[a1][a2][s][s_prime] * b[s] * pi_2[s][a2]

    if norm != 0:
        b_prime_s_prime = temp / norm
    else:
        b_prime_s_prime = temp

    assert b_prime_s_prime <= 1
    return b_prime_s_prime


def p_o_given_b_a1_a2(o: int, b: np.ndarray, a1: int, a2: int, S: np.ndarray, Z: np.ndarray) -> float:
    """
    Computes P[o|a,b]

    :param o: the observation
    :param b: the belief point
    :param a1: the action of player 1
    :param a2: the action of player 2
    :param S: the set of states
    :param Z: the observation tensor
    :return: the probability of observing o when taking action a in belief point b
    """
    prob = 0
    for s in S:
        for s_prime in S:
            prob += b[s] * T[a1][a2][s][s_prime] * Z[a1][a2][s_prime][o]
    assert prob <= 1
    return prob


def p_o_given_b_pi_1_pi_2(o: int, b: np.ndarray, pi_1: np.ndarray, pi_2: np.ndarray, S: np.ndarray,
                          Z: np.ndarray, A1: np.ndarray, A2: np.ndarray) -> float:
    """
    Computes P[o|a,b]

    :param o: the observation
    :param b: the belief point
    :param pi_1: the policy of player 1
    :param pi_2: the policy of player 2
    :param S: the set of states
    :param Z: the observation tensor
    :return: the probability of observing o when taking action a in belief point b
    """
    prob = 0
    for a1 in A1:
        for a2 in A2:
            for s in S:
                for s_prime in S:
                    prob += b[s] * pi_1[a1] * pi_2[s][a2] * T[a1][a2][s][s_prime] * Z[a1][a2][s_prime][o]
    assert prob < 1
    return prob


def excess(lower_bound: List, upper_bound: List, b: np.ndarray, S: np.ndarray,
           epsilon: float, gamma: float, t: int, delta: float, D: float) -> Tuple[float, float]:
    """
    Computes the excess gap and width (Horak, Bosansky, Pechoucek, 2017)

    :param lower_bound: the lower bound
    :param upper_bound: the upper bound
    :param D: the neighborhood parameter
    :param delta: the Lipschitz-continuity parameter
    :param b: the current belief point
    :param S: the set of states
    :param epsilon: the epsilon accuracy parameter
    :param gamma: the discount factor
    :param t: the current exploration depth
    :return: the excess gap and gap width
    """
    w = width(lower_bound=lower_bound, upper_bound=upper_bound, b=b, S=S, delta=delta)
    return (w - rho(t=t, epsilon=epsilon, gamma=gamma, delta=delta, D=D)), w


def rho(t: int, epsilon: float, gamma: float, delta: float, D: float) -> float:
    """
    During the exploration, the HSVI algorithms tries to keep the gap between V_UB and V_LB to be at most
    rho(t), which is monotonically increasing and unbounded

    rho(0) = epsilon
    rho(t+1) = (rho(t) -2*delta*D)/gamma

    :param t: the time-step of the exploration
    :param epsilon: the epsilon parameter
    :param gamma: the discount factor
    :param delta: the Lipshitz-continuity parameter
    :param D: the neighborhood parameter
    :return: rho(t)
    """
    if t == 0:
        return epsilon
    else:
        return (rho(t=t - 1, epsilon=epsilon, gamma=gamma, delta=delta, D=D) - 2 * delta * D) / gamma


def width(lower_bound: List, upper_bound: List, b: np.ndarray, S: np.ndarray, delta: float) -> float:
    """
    Computes the bounds width (Trey Smith and Reid Simmons, 2004)

    :param lower_bound: the current lower bound
    :param upper_bound: the current upper bound
    :param b: the current belief point
    :param S: the set of states
    :param delta: the delta parameter for Lipschitz-continuity
    :return: the width of the bounds
    """
    ub = upper_bound_value(upper_bound=upper_bound, b=b, delta=delta)
    lb = lower_bound_value(lower_bound=lower_bound, b=b, S=S)
    return ub - lb


def check_duplicate(alpha_set: np.ndarray, alpha: np.ndarray) -> bool:
    """
    Check whether alpha vector av is already in set a

    :param alpha_set: the set of alpha vectors
    :param alpha: the vector to check
    :return: true or false
    """
    for alpha_i in alpha_set:
        if np.allclose(alpha_i, alpha):
            return True
    return False


def prune_lower_bound(lower_bound: List, S: np.ndarray) -> np.ndarray:
    """
    Lark's filtering algorithm to prune the lower bound, (Cassandra, Littman, Zhang, 1997)

    :param lower_bound: the current lower bound
    :param S: the set of states
    :return: the pruned lower bound
    """
    # dirty set
    F = set()
    for i in range(len(lower_bound)):
        F.add(tuple(lower_bound[i]))

    # clean set
    Q = []

    for s in S:
        max_alpha_val_s = -np.inf
        max_alpha_vec_s = None
        for alpha_vec in F:
            if alpha_vec[s] > max_alpha_val_s:
                max_alpha_val_s = alpha_vec[s]
                max_alpha_vec_s = alpha_vec
        if max_alpha_vec_s is not None and len(F) > 0:
            # Q.update({max_alpha_vec_s})
            Q.append(np.array(list(max_alpha_vec_s)))
            F.remove(max_alpha_vec_s)
    while F:
        alpha_vec = F.pop()  # just to get a reference
        F.add(alpha_vec)
        x = check_dominance_lp(alpha_vec, np.array(Q))
        if x is None:
            F.remove(alpha_vec)
        else:
            max_alpha_val = -np.inf
            max_alpha_vec = None
            for phi in F:
                phi_vec = np.array(list(phi))
                if phi_vec.dot(alpha_vec) > max_alpha_val:
                    max_alpha_val = phi_vec.dot(alpha_vec)
                    max_alpha_vec = phi_vec
            Q.append(max_alpha_vec)
            F.remove(tuple(list(max_alpha_vec)))
    return Q


def check_dominance_lp(alpha_vec: np.ndarray, Q: np.ndarray):
    """
    Uses LP to check whether a given alpha vector is dominated or not (Cassandra, Littman, Zhang, 1997)

    :param alpha_vec: the alpha vector to check
    :param Q: the set of vectors to check dominance against
    :return: None if dominated, otherwise return the vector
    """

    problem = pulp.LpProblem("AlphaDominance", pulp.LpMaximize)

    # --- Decision Variables ----

    # x
    x_vars = []
    for i in range(len(alpha_vec)):
        x_var_i = pulp.LpVariable("x_" + str(i), lowBound=0, upBound=1, cat=pulp.LpContinuous)
        x_vars.append(x_var_i)

    # delta
    delta = pulp.LpVariable("delta", lowBound=None, upBound=None, cat=pulp.LpContinuous)

    # --- Objective Function ----
    problem += delta, "maximize delta"

    # --- The constraints ---

    # x sum to 1
    x_sum = 0
    for i in range(len(x_vars)):
        x_sum += x_vars[i]
    problem += x_sum == 1, "XSumWeights"

    # alpha constraints
    for i, alpha_vec_prime in enumerate(Q):
        x_dot_alpha_sum = 0
        x_dot_alpha_prime_sum = 0
        for j in range(len(alpha_vec)):
            x_dot_alpha_sum += x_vars[j] * alpha_vec[j]
            x_dot_alpha_prime_sum += x_vars[j] * alpha_vec_prime[j]
        problem += x_dot_alpha_sum >= delta + x_dot_alpha_prime_sum, "alpha_constraint _" + str(i)

    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    delta = delta.varValue
    if delta > 0:
        return alpha_vec
    else:
        return None


def prune_upper_bound(upper_bound: List, delta: float) -> List:
    """
    Prunes the points in the upper bound

    :param upper_bound: the current upper bound
    :param delta: the delta parameter for lipschitz-continuity
    :return: the pruned upper bound
    """
    pruned_upper_bound_point_set = []

    for point in upper_bound:
        true_val = upper_bound_value(upper_bound=upper_bound, b=point[0], delta=delta)
        if not (point[1] > true_val):
            pruned_upper_bound_point_set.append(point)

    return pruned_upper_bound_point_set


def set_seed(seed: float) -> None:
    """
    Deterministic seed config

    :param seed: random seed for the PRNG
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    Z = stopping_game.observation_tensor()
    R = stopping_game.reward_tensor()
    T = stopping_game.transition_tensor()
    A1, _ = stopping_game.player_1_actions()
    A2, _ = stopping_game.player_2_actions()
    O, _ = stopping_game.observations()
    S, _ = stopping_game.states()
    b0 = stopping_game.initial_belief()
    set_seed(1521245)
    hsvi_os_posg(O=O, Z=Z, R=R, T=T, A1=A1, A2=A2, S=S, gamma=0.9, b0=b0, epsilon=0.01,
                 prune_frequency=100, verbose=True, simulation_frequency=1, simulate_horizon=100,
                 number_of_simulations=50, D=None)
