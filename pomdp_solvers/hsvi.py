from typing import List, Tuple
import numpy as np
import pulp
import pomdp.tiger_problem as tiger_problem
import pomdp.vi as vi
import random
import math


def hsvi(O: List, Z: List, R: List, T: List, A: List, S: List, gamma: float, b0: List,
         epsilon: float, opt_alphas: List = None, lp : bool = False, prune_frequency : int = 10, verbose=False,
         simulation_frequency: int = 10, simulate_horizon : int = 10, number_of_simulations: int = 10):
    """
    Heuristic Search Value Iteration for POMDPs (Trey Smith and Reid Simmons, 2004)

    :param O: set of observations of the POMDP
    :param Z: observation tensor of the POMDP
    :param R: reward tensor of the POMDP
    :param T: transition tensor of the POMDP
    :param A: action set of the POMDP
    :param S: state set of the POMDP
    :param gamma: discount factor
    :param b0: initial belief point
    :param epsilon: accuracy parameter
    :param opt_alphas: optimal set of alphas to compare with
    :param lp: whether to use LP to compute upper bound values or SawTooth approximation
    :param prune_frequency: how often to prune the upper and lower bounds
    :param verbose: verbose flag for logging
    :param simulation_frequency: how frequently to simulate the POMDP to compure rewards of current policy
    :param simulate_horizon: length of simulations to compute rewards
    :param number_of_simulations: number of simulations to estimate reward
    :return: None
    """
    lower_bound = initialize_lower_bound(R=R,S=S,A=A,gamma=gamma)
    upper_bound = initialize_upper_bound(T=T,A=A,S=S,gamma=gamma)
    print(f"init LB:{lower_bound},\n init UB:{upper_bound}")

    w = width(lower_bound=lower_bound, upper_bound=upper_bound, b=b0, S=S, lp=lp)
    iteration = 0
    cumulative_r = 0

    while w > epsilon:
        lower_bound, upper_bound = explore(
            b=b0, epsilon=epsilon, t=0, lower_bound=lower_bound, upper_bound=upper_bound,
            gamma=gamma, S=S, O=O, R=R, T=T, A=A,Z=Z, lp=lp)
        w = width(lower_bound=lower_bound, upper_bound=upper_bound, b=b0, S=S, lp=lp)

        if iteration % simulation_frequency == 0:
            r = 0
            for i in range(number_of_simulations):
                r += simulate(horizon=simulate_horizon, b0=b0, lower_bound=lower_bound, Z=Z,
                                        R=R, gamma=gamma, T=T, A=A, O=O)
            cumulative_r = r/number_of_simulations

        if iteration > 1 and iteration % prune_frequency == 0:
            size_before_lower_bound=len(lower_bound)
            size_before_upper_bound = len(upper_bound[0])
            lower_bound = prune_lower_bound(lower_bound=lower_bound, S=S)
            upper_bound = prune_upper_bound(upper_bound=upper_bound, S=S, lp=lp)
            if verbose:
                print(f"Pruning, LB before:{size_before_lower_bound},after:{len(lower_bound)}, "
                      f"UB before: {size_before_upper_bound},after:{len(upper_bound[0])}")

        initial_belief_V_star_upper = upper_bound_value(upper_bound=upper_bound, b=b0, S=S, lp=lp)
        initial_belief_V_star_lower = lower_bound_value(lower_bound=lower_bound, b=b0, S=S)
        optimal_V_star = None
        if opt_alphas is not None:
            optimal_V_star = lower_bound_value(lower_bound=opt_alphas, b=b0, S=S)
        iteration += 1

        print(f"iteration: {iteration}, width: {w}, epsilon: {epsilon}, R: {cumulative_r}, "
              f"UB size:{len(upper_bound[0])}, LB size:{len(lower_bound)}")
        if verbose:
            print(f"Upper V*[b0]: {initial_belief_V_star_upper}, "
                  f"Lower V*[b0]:{initial_belief_V_star_lower}, V*[b0]:{optimal_V_star}")

    with open('../aleph_t.npy', 'wb') as f:
        np.save(f, np.asarray(list(lower_bound)))


def explore(b : List, epsilon : float, t : int, lower_bound : List, upper_bound: Tuple[List, List],
            gamma: float, S: List, O: List, Z: List, R: List, T: List, A: List, lp: bool) \
        -> Tuple[List, Tuple[List, List]]:
    """
    Explores the POMDP tree

    :param b: current belief
    :param epsilon: accuracy parameter
    :param t: the current depth of the exploration
    :param lower_bound: the lower bound on the value function
    :param upper_bound: the upper bound on the value function
    :param gamma: discount factor
    :param S: set of states
    :param O: set of observations
    :param Z: observation tensor
    :param R: reward tensor
    :param T: transition tensor
    :param A: set of actions
    :param lp: whether to use LP to compute upper bound values
    :return: new lower and upper bounds
    """
    w = width(lower_bound=lower_bound, upper_bound=upper_bound, b=b, S=S, lp=lp)
    if (gamma > 0 and w <= epsilon*math.pow(gamma, -t)) or (w<=epsilon):
        return lower_bound, upper_bound

    # Determine a*
    a_Q_vals = []
    for a in A:
        upper_Q = q(b=b, a=a, lower_bound=lower_bound, upper_bound=upper_bound, S=S, O=O, Z=Z,
                    R=R, gamma=gamma, T=T, upper=True, lp=lp)
        a_Q_vals.append(upper_Q)
    a_star = np.argmax(np.array(a_Q_vals))

    # Determine o*
    o_vals = []
    for o in O:
        new_belief = next_belief(o=o, a=a_star, b=b, S=S, Z=Z, T=T)
        o_val = p_o_given_b_a(o=o, b=b, a=a_star, S=S, Z=Z) * \
                excess(lower_bound=lower_bound,upper_bound=upper_bound,b=new_belief,S=S,epsilon=epsilon,
                       gamma=gamma,t=(t+1), lp=lp)
        o_vals.append(o_val)
    o_star = np.argmax(np.array(o_vals))

    b_prime = next_belief(o=o_star, a=a_star, b=b, S=S, Z=Z, T=T)
    lower_bound, upper_bound = explore(b=b_prime,epsilon=epsilon,t=t+1,lower_bound=lower_bound,
            upper_bound=upper_bound,gamma=gamma,S=S,O=O,R=R,T=T,A=A, Z=Z, lp=lp)

    lower_bound, upper_bound = \
        local_updates(lower_bound=lower_bound, upper_bound=upper_bound, b=b, A=A, S=S, Z=Z, O=O, R=R,
                      T=T, gamma=gamma, lp=lp)

    return lower_bound, upper_bound


def initialize_lower_bound(R: List, S: List, A: List, gamma: float) -> List:
    """
    Initializes the lower bound

    :param R: reward tensor
    :param S: set of states
    :param A: set of actions
    :param gamma: discount factor
    :return: the initialized lower bound
    """
    vals_1 = []
    for a in A:
        vals_2 = []
        for s in S:
            vals_2.append(R[a][s]/(1-gamma))
        vals_1.append(min(vals_2))
    R_underbar = max(vals_1)
    alpha_vector = np.zeros(len(S))
    alpha_vector.fill(R_underbar)
    lower_bound = []
    lower_bound.append(alpha_vector)
    return lower_bound


def initialize_upper_bound(T: List, A: List, S: List, gamma: float) -> Tuple[List, List]:
    """
    Initializes the upper bound

    :param T: the transition tensor
    :param A: the set of actions
    :param S: the set of states
    :param gamma: the discount factor
    :return: the initialized upper bound
    """
    V, pi = vi.value_iteration(T=T, num_states=len(S), num_actions=len(A), R=R, theta=0.0001, discount_factor=gamma)
    print("Initial state values: V:{}".format(V))
    point_set = []
    for s in S:
        b = generate_corner_belief(s=s, S=S)
        point_set.append([b, V[s]])
    return (point_set, point_set.copy())


def generate_corner_belief(s: int, S: List):
    """
    Generate the corner of the simplex that corresponds to being in some state with probability 1

    :param s: the state
    :param S: the set of States
    :return: the corner belief corresponding to state s
    """
    b = np.zeros(len(S))
    b[s] = 1
    return b


def local_updates(lower_bound: List, upper_bound: Tuple[List, List], b: List, A: List, S: List,
                  O: List, R: List, T: List, gamma: float, Z: List, lp: bool) -> Tuple[List, Tuple[List, List]]:
    """
    Perform local updates to the upper and  lower bounds for the given belief in the heuristic-search-exploration

    :param lower_bound: the lower bound on V
    :param upper_bound: the upper bound on V
    :param b: the current belief point
    :param A: the set of actions
    :param S: the set of states
    :param O: the set of observations
    :param R: the reward tensor
    :param T: the transition tensor
    :param gamma: the discount factor
    :param Z: the set of observations
    :param lp: a boolean flag whether to use LP to compute upper bound beliefs
    :return: The updated lower and upper bounds
    """
    new_upper_bound = local_upper_bound_update(upper_bound=upper_bound, b=b, A=A, S=S, O=O, R=R, T=T, gamma=gamma,
                                               Z=Z, lp=lp)
    new_lower_bound = local_lower_bound_update(lower_bound=lower_bound, b=b, Z=Z, A=A, O=O, S=S, T=T, R=R, gamma=gamma)
    return new_lower_bound, new_upper_bound


def local_upper_bound_update(upper_bound: Tuple[List, List], b: List, A: List, S: List, O: List, R: List, T: List,
                             gamma: float, Z: List, lp: bool) -> Tuple[List, List]:
    """
    Performs a local update to the upper bound during the heuristic-search exploration

    :param upper_bound: the upper bound to update
    :param b: the current belief point
    :param A: the set of actions
    :param S: the set of states
    :param O: the set of observations
    :param R: the reward tensor
    :param T: the transition tensor
    :param gamma: the discount factor
    :param Z: the set of observations
    :param lp: whether or not to use LP to compute upper bound beliefs
    :return: the updated upper bound
    """
    b, new_val = upper_bound_backup(upper_bound=upper_bound, b=b, A=A, S=S, Z=Z, O=O, R=R, T=T, gamma=gamma, lp=lp)
    upper_bound_point_set, corner_points = upper_bound
    upper_bound_point_set.append([b,new_val])
    new_corner_points = update_corner_points(corner_points=corner_points, new_point=(b, new_val))
    upper_bound = (upper_bound_point_set, new_corner_points)
    return upper_bound


def update_corner_points(corner_points: List, new_point: Tuple[List, float]) -> List:
    """
    (Maybe) update the corner points of the upper bound

    :param corner_points: the current set of corner points
    :param new_point: the new point to add to the upper bound
    :return: the new set of corner points
    """
    new_corner_points = []
    for cp in corner_points:
        corner_match = True
        for i in range(len(cp[0])):
            if cp[0][i] != new_point[0][i]:
                corner_match = False
        if corner_match:
            new_corner_points.append((cp[0], new_point[1]))
        else:
            new_corner_points.append(cp)
    return new_corner_points


def local_lower_bound_update(lower_bound: List, b: List, A: List, O: List, Z: List, S: List,
                             T: List, R: List, gamma: float) -> List:
    """
    Performs a local update to the lower bound given a belief point in the heuristic search

    :param lower_bound: the current lower bound
    :param b: the current belief point
    :param A: the set of actions
    :param O: the set of observations
    :param Z: the observation tensor
    :param S: the set of states
    :param T: the transition tensor
    :param R: the reward tensor
    :param gamma: the discount factor
    :return: the updated lower bound
    """
    beta = lower_bound_backup(lower_bound=lower_bound, b=b, A=A, Z=Z, O=O, S=S, T=T, R=R, gamma=gamma)
    if not check_duplicate(lower_bound, beta):
        lower_bound.append(beta)
    return lower_bound


def lower_bound_backup(lower_bound: List, b: List, A: List, O: List, Z: List, S: List,
                       T: List, R: List, gamma: float) -> List:
    """
    Generates a new alpha-vector for the lower bound

    :param lower_bound: the current lower bound
    :param b: the current belief point
    :param A: the set of actions
    :param O: the set of observations
    :param Z: the observation tensor
    :param S: the set of states
    :param T: the transition tensor
    :param R: the reward tensor
    :param gamma: the discount factor
    :return: the new alpha vector
    """
    max_beta_a_o_alpha_vecs = []
    for a in A:
        max_beta_a_vecs = []
        for o in O:
            new_belief = np.array(next_belief(o=o, a=a, b=b, S=S, Z=Z, T=T))
            max_alpha_vec = lower_bound[0]
            max_alpha_val = float("-inf")
            for alpha_vec in lower_bound:
                alpha_val = new_belief.dot(alpha_vec)
                if alpha_val > max_alpha_val:
                    max_alpha_val = alpha_val
                    max_alpha_vec=alpha_vec
            max_beta_a_vecs.append(max_alpha_vec)
        max_beta_a_o_alpha_vecs.append(max_beta_a_vecs)
    beta_a_vecs = []
    for a in A:
        beta_a_vec = []
        for s in S:
            beta_a_s_val = 0
            beta_a_s_val += R[a][s]
            expected_future_vals = 0
            for o in O:
                for s_prime in S:
                    expected_future_vals += max_beta_a_o_alpha_vecs[a][o][s_prime]*Z[a][s_prime][o]*T[a][s][s_prime]
            beta_a_s_val += gamma*expected_future_vals
            beta_a_vec.append(beta_a_s_val)
        beta_a_vecs.append(beta_a_vec)

    beta = beta_a_vecs[0]
    max_val = float("-inf")
    for beta_a_vec in beta_a_vecs:
        val = np.array(beta_a_vec).dot(np.array(b))
        if val > max_val:
            max_val = val
            beta = beta_a_vec

    return beta


def upper_bound_backup(upper_bound: Tuple[List, List], b: List, A: List, S: List, O: List, Z: List, R: List, T: List,
                       gamma: float, lp: bool) -> Tuple[List, float]:
    """
    Adds a point to the upper bound

    :param upper_bound: the current upper bound
    :param b: the current belief point
    :param A: the set of actions
    :param S: the set of states
    :param O: the set of observations
    :param Z: the observation tensor
    :param R: the reward tensor
    :param T: the transition tensor
    :param gamma: the discount factor
    :param lp: a boolean flag whether to use LP to compute the upper bound belief
    :return: the new point
    """
    q_vals = []
    for a in A:
        v = 0
        for s in S:
            immediate_r = b[s] * R[a][s]
            expected_future_rew = 0
            for o in O:
                new_belief = next_belief(o=o, a=a, b=b, S=S, Z=Z, T=T)
                for s_prime in S:
                    expected_future_rew += \
                        b[s]*T[a][s][s_prime]*Z[a][s_prime][o]*upper_bound_value(
                            upper_bound=upper_bound, b=new_belief, S=S, lp=lp)
            v += immediate_r + gamma*expected_future_rew
        q_vals.append(v)
    new_val = max(q_vals)
    return b, new_val


def lp_convex_hull_projection_lp(upper_bound : Tuple[List, List], b: List, S: List) -> float:
    """
    Reference: (Hauskreht 2000)

    Computes the upper bound belief by performing a projection onto the convex hull of the upper bound, it is computed
    by solving an LP

    :param upper_bound: the upper bound
    :param b: the belief point to compute the value for
    :param S: the set of states
    :return: the upper bound value of the belief point
    """
    upper_bound_point_set, corner_points = upper_bound

    problem = pulp.LpProblem("ConvexHullProjection", pulp.LpMinimize)

    # Convex hull coefficients
    lamb = []
    for i in range(len(upper_bound_point_set)):
        lamb_i = pulp.LpVariable("lambda_" + str(i), lowBound=0, upBound=1, cat=pulp.LpContinuous)
        lamb.append(lamb_i)

    # The objective function
    objective = 0
    for i, point in enumerate(upper_bound_point_set):
        objective += lamb[i]*point[1]
    problem += objective, "Convex hull projection"

    # --- The constraints ---

    # Belief probability constraint
    for j in range(len(S)):
        belief_sum = 0
        for i, point in enumerate(upper_bound_point_set):
            belief_sum += lamb[i] * point[0][j]
        problem += belief_sum == b[j], "BeliefVectorConstraint_" + str(j)

    # Convex Hull constraint
    lambda_weights_sum = 0
    for i in range(len(lamb)):
        lambda_weights_sum += lamb[i]
    problem += lambda_weights_sum == 1, "ConvexHullWeightsSumConstraint"

    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    projected_lamb_coefficients = []
    belief_value = 0
    for i in range(len(upper_bound_point_set)):
        projected_lamb_coefficients.append(lamb[i].varValue)
        belief_value += projected_lamb_coefficients[i]*upper_bound_point_set[i][1]

    return belief_value


def approximate_projection_sawtooth(upper_bound: Tuple[List, List], b :List) -> float:
    """
    Reference: (Hauskreht 2000)

    Performs an approximate projection of the belief onto the convex hull of the upepr bound to compute the upper bound
    value of the belief

    :param upper_bound: the upper bound
    :param b: the belief point
    :return: the value of the belief point
    """
    upper_bound_point_set, corner_points = upper_bound
    alpha_corner = np.array(corner_points)[:,1]
    # corner_points_belief_value = alpha_corner.dot(np.array(b))
    # non_zero_belief_states = list(filter(lambda x: b[x]>0, range(len(b))))
    # zero_belief_states = list(filter(lambda x: b[x]==0, range(len(b))))

    # min_val = corner_points_belief_value
    interior_belief_values = []
    for point in upper_bound_point_set:
        interior_belief_values.append(interior_point_belief_val(interior_point=point, b=b, alpha_corner=alpha_corner,
                                                                S=S))
    return min(interior_belief_values)


def interior_point_belief_val(interior_point: Tuple[List, float], b: List, alpha_corner: List, S: List) -> float:
    """
    Computes the value induced on the belief point b projected onto the convex hull by a given interior belief point

    :param interior_point: the interior point
    :param b: the belief point
    :param alpha_corner: the alpha vector corresponding to the corners of the belief simplex
    :param S: the set of states
    :return: the value of the belief point induced by the interior point
    """
    min_ratios = []
    for s in S:
        if interior_point[0][s] > 0:
            min_ratio = b[s]/interior_point[0][s]
            min_ratios.append(min_ratio)
        else:
            min_ratios.append(float("inf"))
    min_ratio = min(min_ratios)

    if min_ratio > 1:
        min_ratio = 1

    interior_alpha_corner_dot = alpha_corner.dot(interior_point[0])

    return interior_alpha_corner_dot + min_ratio*(interior_point[1]-interior_alpha_corner_dot)


def upper_bound_value(upper_bound: Tuple[List, List], b: List, S: List, lp : bool = False) -> float:
    """
    Computes the upper bound value of a given belief point

    :param upper_bound: the upper bound
    :param b: the belief point
    :param S: the set of states
    :param lp: boolean flag that decides whether to use LP to compute the upper bound value or not
    :return: the upper bound value
    """

    if lp:
        return lp_convex_hull_projection_lp(upper_bound=upper_bound, b=b, S=S)
    else:
        return approximate_projection_sawtooth(upper_bound=upper_bound, b=b)


def lower_bound_value(lower_bound: List, b: List, S: List) -> float:
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
            sum += b[s]*alpha_vec[s]
        alpha_vals.append(sum)
    return max(alpha_vals)


def next_belief(o: int, a: int, b: List, S: List, Z: List, T: List) -> List:
    """
    Computes the next belief using a Bayesian filter

    :param o: the latest observation
    :param a: the latest action
    :param b: the current belief
    :param S: the set of states
    :param Z: the observation tensor
    :param T: the transition tensor
    :return: the new belief
    """
    b_prime = np.zeros(len(S))
    for s_prime in S:
        b_prime[s_prime] = bayes_filter(s_prime=s_prime, o=o, a=a, b=b, S=S, Z=Z, T=T)
    assert round(sum(b_prime), 5) == 1
    return b_prime


def bayes_filter(s_prime: int, o: int, a: int, b: List, S: List, Z: List, T: List) -> float:
    """
    A Bayesian filter to compute the belief of being in s_prime when observing o after taking action a in belief b

    :param s_prime: the state to compute the belief of
    :param o: the observation
    :param a: the action
    :param b: the current belief point
    :param S: the set of states
    :param Z: the observation tensor
    :param T: the transition tensor
    :return: b_prime(s_prime)
    """
    norm = 0
    for s in S:
        for s_prime_1 in S:
            prob_1 = Z[a][s_prime_1][o]
            norm += b[s]*prob_1*T[a][s][s_prime_1]
    obs_prob = Z[a][s_prime][o]
    temp = 0
    for s in S:
        temp += b[s]*T[a][s][s_prime]
    b_prime = (obs_prob*temp)/norm
    assert b_prime <=1
    return b_prime


def p_o_given_b_a(o: int, b: List, a: int, S: List, Z: List) -> float:
    """
    Computes P[o|a,b]

    :param o: the observation
    :param b: the belief point
    :param a: the action
    :param S: the set of states
    :param Z: the observation tensor
    :return: the probability of observing o when taking action a in belief point b
    """
    prob = 0
    for s in S:
        for s_prime in S:
            prob += b[s] * T[a][s][s_prime] * Z[a][s_prime][o]
    assert prob < 1
    return prob


def excess(lower_bound: List, upper_bound: Tuple[List, List], b: List, S: List, epsilon: float, gamma : float, t: int,
           lp: bool) -> float:
    """
    Computes the excess uncertainty  (Trey Smith and Reid Simmons, 2004)

    :param lower_bound: the lower bound
    :param upper_bound: the upper bound
    :param b: the current belief point
    :param S: the set of states
    :param epsilon: the epsilon accuracy parameter
    :param gamma: the discount factor
    :param t: the current exploration depth
    :param lp: whether to use LP or not to compute upper bound belief values
    :return: the excess uncertainty
    """
    w = width(lower_bound=lower_bound, upper_bound=upper_bound, b=b, S=S, lp=lp)
    if gamma == 0:
        return w
    else:
        return w - epsilon*math.pow(gamma, -(t))


def width(lower_bound: List, upper_bound: Tuple[List,List], b: List, S: List, lp: bool) -> float:
    """
    Computes the bounds width (Trey Smith and Reid Simmons, 2004)

    :param lower_bound: the current lower bound
    :param upper_bound: the current upper bound
    :param b: the current belief point
    :param S: the set of states
    :param lp: boolean flag that decides whether to use LP to compute upper bound belief values
    :return: the width of the bounds
    """
    ub = upper_bound_value(upper_bound=upper_bound, b=b, S=S, lp=lp)
    lb = lower_bound_value(lower_bound=lower_bound, b=b, S=S)
    return ub-lb


def q_hat_interval(b: List, a: int, S: List, O: List, Z: List, R: List,
                   T: List, gamma: float, lower_bound: List, upper_bound: Tuple[List, List], lp: float) -> float:
    """
    Computes the interval (Trey Smith and Reid Simmons, 2004)

    :param b: the current belief point
    :param a: the action
    :param S: the set of states
    :param O: the set of observations
    :param Z: the observation tensor
    :param R: the reward tensor
    :param T: the transition tensor
    :param gamma: the discount factor
    :param lower_bound: the lower bound
    :param upper_bound: the upper bound
    :param lp: boolean flag that decides whether to use LP to compute upper bound belief values
    :return: the interval
    """
    upper_Q = q(b=b, a=a, lower_bound=lower_bound, upper_bound=upper_bound, S=S, O=O, Z=Z,
                R=R, gamma=gamma, T=T, upper=True, lp=lp)
    lower_Q = q(b=b, a=a, lower_bound=lower_bound, upper_bound=upper_bound, S=S, O=O, Z=Z,
                R=R, gamma=gamma, T=T, upper=False, lp=lp)
    return [lower_Q, upper_Q]


def q(b: List, a: int, lower_bound: List, upper_bound: Tuple[List, List], S: List, O: List,
      Z: List, R: List, gamma: float, T: List, upper : bool = True, lp : bool = False) -> float:
    """
    Applies the Bellman equation to compute Q values

    :param b: the belief point
    :param a: the action
    :param lower_bound: the lower bound
    :param upper_bound: the upper bound
    :param S: the set of states
    :param O: the set of observations
    :param Z: the observation tensor
    :param R: the reward tensor
    :param gamma: the discount factor
    :param T: the transition tensor
    :param upper: boolean flag that decides whether to use the upper bound or lower bound on V to compute the Q-value
    :param lp: boolean flag that decides whether to use LP to compute upper bound belief values
    :return: the Q-value
    """
    Q_val = 0
    for s in S:
        immediate_r = R[a][s]*b[s]
        expected_future_rew = 0
        for o in O:
            new_belief = next_belief(o=o, a=a, b=b, S=S, Z=Z, T=T)
            for s_prime in S:
                if upper:
                    future_val = upper_bound_value(upper_bound=upper_bound, b=new_belief, S=S, lp=lp)
                else:
                    future_val = lower_bound_value(lower_bound=lower_bound, b=new_belief, S=S)
                expected_future_rew += \
                    b[s] * T[a][s][s_prime] * Z[a][s_prime][o] * future_val
        Q_val += immediate_r + expected_future_rew* gamma
    return Q_val


def check_duplicate(alpha_set: List, alpha: List) -> bool:
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


def prune_lower_bound(lower_bound: List, S: List) -> List:
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
        alpha_vec = F.pop() # just to get a reference
        F.add(alpha_vec)
        x = check_dominance_lp(alpha_vec, Q)
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


def check_dominance_lp(alpha_vec: List, Q: List):
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
            x_dot_alpha_sum += x_vars[j]*alpha_vec[j]
            x_dot_alpha_prime_sum += x_vars[j] * alpha_vec_prime[j]
        problem += x_dot_alpha_sum >= delta + x_dot_alpha_prime_sum, "alpha_constraint _" + str(i)

    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    delta = delta.varValue
    if delta > 0:
        return alpha_vec
    else:
        return None


def prune_upper_bound(upper_bound: Tuple[List, List], S: List, lp: bool) -> Tuple[List, List]:
    """
    Prunes the points in the upper bound

    :param upper_bound: the current upper bound
    :param S: the set of states
    :param lp: boolean flag that decides whether to use LP to compute upper bound belief values
    :return: the pruned upper bound
    """

    upper_bound_point_set, corner_points = upper_bound
    pruned_upper_bound_point_set = []

    for point in upper_bound_point_set:
        true_val = upper_bound_value(upper_bound=upper_bound, S=S, b=point[0], lp=lp)
        if not (point[1] > true_val):
            pruned_upper_bound_point_set.append(point)

    return pruned_upper_bound_point_set, corner_points


def simulate(horizon: int, b0: List, lower_bound: List, Z: List, R: List, gamma: float, T: List, A: List,
             O: List) -> float:
    """
    Simulates the POMDP to estimate the reward of the greedy policy with respect to the value function represented
    by the lower bound

    :param horizon: the horizon for the simulation
    :param b0: the initial belief
    :param lower_bound: the lower bound which represents the value function
    :param Z: the observation tensor
    :param R: the reward tensor
    :param gamma: the discount factor
    :param T: the transition operator
    :param A: the action set
    :param O: the observation set
    :return: the cumulative discounted reward
    """
    t = 0
    b = b0
    cumulative_r = 0
    while t < horizon:
        q_values = list(map(lambda a: q(b=b,a=a,lower_bound=lower_bound, upper_bound=([], []),S=S,O=O,Z=Z,R=R,
                                    gamma=gamma,T=T,upper=False, lp=False), A))
        a = int(np.argmax(np.array(q_values)))
        r = 0
        for s in S:
            r += R[a][s]*b[s]
        cumulative_r += math.pow(gamma, t) * r

        observation_probabilities = []
        for o in O:
            p = 0
            for s in S:
                for s_prime in S:
                    p += b[s]*T[a][s][s_prime]*Z[a][s_prime][o]
            observation_probabilities.append(p)
        o = np.random.choice(np.arange(0, len(O)), p=observation_probabilities)
        b = next_belief(o=o, a=a, b=b, S=S, Z=Z, T=T)
        t += 1

    return cumulative_r


def set_seed(seed: float) -> None:
    """
    Deterministic seed config

    :param seed: random seed for the PRNG
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    Z = tiger_problem.observation_matrix()
    R = tiger_problem.reward_matrix()
    T = tiger_problem.transition_tensor()
    A, _ = tiger_problem.actions()
    O, _ = tiger_problem.observations()
    S, _ = tiger_problem.states()
    b0 = tiger_problem.initial_belief()
    set_seed(1521245)
    optimal_alphas = tiger_problem.optimal_alphas_2()
    hsvi(O=O,Z=Z,R=R,T=T,A=A,S=S,gamma=0.9, b0=b0, epsilon=0.01, opt_alphas=optimal_alphas,
         lp=False,prune_frequency=100, verbose=False, simulation_frequency=1, simulate_horizon=100,
         number_of_simulations=50)
