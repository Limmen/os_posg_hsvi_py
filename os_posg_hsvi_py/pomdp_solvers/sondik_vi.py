import numpy as np
from scipy.optimize import linprog
from itertools import product


def compute_all_conditional_plans_conditioned_on_a_t(n_alpha_vectors_t_plus_one, n_obs):
    """
    Compute the number of conditional plans conditioned on an action a. It produces all possible combinations of
    (observation -> conditional_plan)

    :param n_alpha_vectors_t_plus_one: Number of alpha-vectors (number of conditional plans) for t+1
    :param n_obs: Number of observations
    :return: list of lists, where each list contains n_obs elements, and each element is in [0, n_alpha_vectors-1].

    The number of conditional plans will be be n_alpha_vectors^n_obs elements.
    The plan is of the form: (o^(1)_i, o^(2)_j, ..., o^(n_alpha_vectors_t_plus_one)_k)
    where o^(1)_i means that if observation o_i is observed, conditional plan 1 should be followed,
    o^(2)_j means that if observation o_j is observed, conditional plan 2 should be followed,
    o^(n_alpha_vectors_t_plus_one)_k means that if observation o_k is observed, conditional plan
    n_alpha_vectors_t_plus_one should be followed.
    """
    x = list(range(n_alpha_vectors_t_plus_one))
    return [p for p in product(x, repeat=n_obs)]


def vi(P, Z, R, T, gamma, n_states, n_actions, n_obs, b0, use_pruning = True):
    """

    :param P: The transition probability matrix
    :param Z: The observation probability matrix
    :param R: The immediate rewards matrix
    :param T: The planning horizon
    :param gamma: The discount factor
    :param n_states: The number of states
    :param n_actions: The number of actions
    :param n_obs: The number of observations
    :param b0: The initial belief
    :return:
    """
    alepth_t_plus_1 = set()
    zero_alpha_vec = (-1, tuple(np.zeros(n_states))) # an alpha vector is associated with an action and a set of values
    alepth_t_plus_1.add(zero_alpha_vec)
    first = True
    num_alpha_vectors = []
    num_alpha_vectors.append(len(alepth_t_plus_1))

    # Backward induction
    for t in range(T):
        print('[Value Iteration] planning horizon {}, |aleph|:{} ...'.format(t, len(alepth_t_plus_1)))

        # New set of alpha vectors which will be constructed from the previous (backwards) set aleph_t+1.
        aleph_t = set()

        # Weight the alpha vectors in aleph_t by the transition probabilities alpha(s)*Z(s'|s,o)*P(s'|s,a) forall a,o,s,s'
        # alpha'(s) = alpha(s)*Z(s'|s,o)*P(s'|s,a) forall a,o,s,s'
        alpha_new = np.zeros(shape=(len(alepth_t_plus_1), n_actions, n_obs, n_states))
        n_alpha_vectors = 0
        for old_alpha_vec in alepth_t_plus_1:
            for a in range(n_actions):
                for o in range(n_obs):
                    for s in range(n_states):
                        for s_prime in range(n_states):
                            # Half of Sondik's one-pass DP backup, alpha'_(a,o)(s)=alpha(s')*Z(s'|s,o)*P(s'|s,a) forall a,o,s,s'
                            # note that alpha(s) is a representation of $V(s)$
                            alpha_new[n_alpha_vectors][a][o][s] += np.array(old_alpha_vec[1][s_prime]) * Z[a][s_prime][o] * P[a][s][s_prime]
            n_alpha_vectors +=1

        # Compute the new alpha vectors by adding the discounted immediate rewards and the expected alpha vectors at time t+1
        # There are in total |Gamma^(k+1)|=|A|*|Gamma^k|^(|Z|) number of conditional plans, which means that there
        # is |Gamma^(k+1)|=|A|*|Gamma^k|^(|Z|) number of alpha vectors
        for a in range(n_actions):

            # |Gamma^k|^(|Z|) number of conditional plans conditioned on 'a'
            conditional_plans_conditioned_on_a = compute_all_conditional_plans_conditioned_on_a_t(n_alpha_vectors, n_obs)

            # Each conditional plan is of the form (o^(1)_i, o^(2)_j, ..., o^(n_alpha_vectors_t_plus_one)_k)
            # where o^(p)_i means that if observation o_i is observed, conditional plan p should be followed
            for conditional_plan_conditioned_on_a in conditional_plans_conditioned_on_a:
                for o in range(n_obs):
                    conditional_plan_to_follow_when_observing_o = conditional_plan_conditioned_on_a[o]
                    temp = np.zeros(n_states)
                    for s in range(n_states):
                        # Second half of Sondik's one-pass DP backup,
                        # alpha_(a,o,beta)'(s) = gamma*(R(a,s) alpha_beta(s)*Z(s'|s,o)*P(s'|s,a) forall a,o,s,s')
                        temp[s] = gamma * (R[a][s] + alpha_new[conditional_plan_to_follow_when_observing_o][a][o][s])
                    aleph_t.add((a, tuple(temp)))

        alepth_t_plus_1.update(aleph_t)
        num_alpha_vectors.append(len(alepth_t_plus_1))

        if first:
            # remove the dummy alpha vector
            alepth_t_plus_1.remove(zero_alpha_vec)
            first = False

        if use_pruning:
            alepth_t_plus_1 = prune(n_states, alepth_t_plus_1) # remove dominated alpha vectors

    # The optimal value function is implicitly represented by aleph^0. Note that aleph^0 is a much larger set of
    # elements than the set of states. To compute the optimal value function V^*(b0) given an initial belief b0,
    # compute
    #V^*(b) = max_alpha b0*alpha for all alpha in aleph^0
    max_v = -np.inf
    best_belief_weighted_alpha = None
    b0 = [0.5,0.5]
    for alpha in aleph_t:
        v = np.dot(np.array(alpha[1]), b0)

        if v > max_v:
            max_v = v
            best_belief_weighted_alpha = alpha

    value_fun = best_belief_weighted_alpha
    return value_fun, aleph_t, num_alpha_vectors


def prune(n_states, aleph):
    """
    Remove dominated alpha-vectors using Lark's filtering algorithm
    :param n_states
    :return:
    """
    # parameters for linear program
    delta = 0.0000000001
    # equality constraints on the belief states
    A_eq = np.array([np.append(np.ones(n_states), [0.])])
    b_eq = np.array([1.])

    # dirty set
    F = aleph.copy()

    # clean set
    Q = set()

    for i in range(n_states):
        max_i = -np.inf
        best = None
        for av in F:
            # av[1] = np.array(av[1])
            if av[1][i] > max_i:
                max_i = av[1][i]
                best = av
        if best is not None and len(F) > 0:
            Q.update({best})
            F.remove(best)
    while F:
        av_i = F.pop()  # get a reference to av_i
        F.add(av_i)  # don't want to remove it yet from F
        dominated = False
        for av_j in Q:
            c = np.append(np.zeros(n_states), [1.])
            A_ub = np.array([np.append(-(np.array(av_i[1]) - np.array(av_j[1])), [-1.])])
            b_ub = np.array([-delta])

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
            if res.x[n_states] > 0.0:
                # this one is dominated
                dominated = True
                F.remove(av_i)
                break

        if not dominated:
            max_k = -np.inf
            best = None
            for av_k in F:
                b = res.x[0:2]
                v = np.dot(av_k.v, b)
                if v > max_k:
                    max_k = v
                    best = av_k
            F.remove(best)
            if not check_duplicate(Q, best):
                Q.update({best})
    return Q


def check_duplicate(a, av):
    """
    Check whether alpha vector av is already in set a

    :param a:
    :param av:
    :return:
    """
    for av_i in a:
        if np.allclose(av_i[1], av.v):
            return True
        if av_i[1][0] == av[1][0] and av_i[1][1] > av[1][1]:
            return True
        if av_i[1][1] == av[1][1] and av_i[1][0] > av[1][0]:
            return True
