import numpy as np

R_ST=20.0
R_SLA = 5.0
R_COST=-5.0
R_INT=-10.0
p=0.01
L=1.0


def discretized_belief_space():
    """
    There are only two states s1, s2, so the belief can be represented by P(s1), where P(s2) = 1-P(s1)

    :return: the discretized belief points P(s1)
    """
    return np.array([0.0, 0.25, 0.5, 0.75, 1]), {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4}

def terminal_reward():
    return 0

def states():
    return [0,1,2], {0: "NO_INTRUSION", 1: "INTRUSION", 2:"TERMINAL"}


def defender_actions():
    return [0, 1], {0: "CONTINUE", 1: "STOP"}


def attacker_actions():
    return [0, 1], {0: "CONTINUE", 1: "STOP"}


def observations():
    return [0, 1, 2, 3, 4], {0: "NO ALERT", 1: "ONE ALERT", 2: "TWO ALERTS", 3: "THREE ALERTS", 4: "TERMINAL"}


def reward_function(s, a1, a2):
    # Terminal state
    if s == 2:
        return 0

    # No intrusion state
    if s == 0:
        # Continue and Wait
        if a1==0 and a2 == 0:
            return R_SLA
        # Continue and Attack
        if a1 == 0 and a2 == 1:
            return R_SLA + p*R_ST/L
        # Stop and Wait
        if a1 == 1 and a2 == 0:
            return R_COST/L
        # Stop and Attack
        if a1 == 1 and a2 == 1:
            return R_COST / L + R_ST/L

    # Intrusion state
    if s == 1:
        # Continue and Continue
        if a1 == 0 and a2 == 0:
            return R_SLA + R_INT
        # Continue and Stop
        if a1 == 0 and a2 == 1:
            return R_SLA
        # Stop and Continue
        if a1 == 1 and a2 == 0:
            return R_COST / L + R_ST/L
        # Stop and Stop
        if a1 == 1 and a2 == 1:
            return R_COST / L

    raise ValueError("Invalid input, s:{}, a1:{}, a2:{}".format(s,a1,a2))


def transition_operator(s, a1, a2, s_prime, l):

    # Terminal state
    if s == 2:
        return 1 if s_prime==2 else 0

    # No intrusion state
    if s == 0:
        # Continue and Wait
        if a1==0 and a2 == 0:
            if s_prime == 0:
                # Stays in no intrusion state
                return 1-p
            elif s_prime == 2:
                # Game ends
                return p
            else:
                return 0

        # Stop and Wait
        if a1 == 1 and a2 == 0:

            if l > 1:
                if s_prime == 0:
                    # Stays in no intrusion state
                    return 1 - p
                if s_prime == 2:
                    # Game ends
                    return p
                else:
                    return 0
            else:
                if s_prime == 2:
                    # Game ends
                    return 1
                else:
                    return 0

        # Continue and Attack
        if a1 == 0 and a2 == 1:

            if s_prime == 1:
                # Intrusion starts
                return 1 - p
            elif s_prime == 2:
                # Game ends
                return p
            else:
                return 0

        # Stop and Attack
        if a1 == 1 and a2 == 1:

            if l > 1:
                if s_prime == 1:
                    # Intrusion starts
                    return 1 - p
                if s_prime == 2:
                    # Game ends
                    return p
                else:
                    return 0
            else:
                if s_prime == 2:
                    # Game ends
                    return 1
                else:
                    return 0


    # Intrusion state
    if s == 1:
        # Continue and Continue
        if a1==0 and a2 == 0:
            if s_prime == 1:
                # Stays in intrusion state
                return 1-p
            elif s_prime == 2:
                # Game ends
                return p
            else:
                return 0

        # Stop and Continue
        if a1 == 1 and a2 == 0:

            if l > 1:
                if s_prime == 1:
                    # Stays in  intrusion state
                    return 1 - p
                if s_prime == 2:
                    # Game ends
                    return p
                else:
                    return 0
            else:
                if s_prime == 2:
                    # Game ends
                    return 1
                else:
                    return 0

        # Continue and Stop
        if a1 == 0 and a2 == 1:
            if s_prime == 2:
                # Game ends
                return 1
            else:
                return 0

        # Stop and Stop
        if a1 == 1 and a2 == 1:
            if s_prime == 2:
                # Game ends
                return 1
            else:
                return 0

    raise ValueError("s:{}, a1:{}, a2:{}, s_prime:{} not recognized".format(s, a1, a2, s_prime))


def belief_reward_function(b, a1, a2):
    S, _ = states()
    rew = 0
    for s in S:
        rew += b[s]*reward_function(s=s, a1=a1, a2=a2)
    return rew


def observation_function(s_prime, a1, a2, o):

    # Terminal state
    if s_prime == 2:
        if o == 4:
            return 1
        else:
            return 0

    # No intrusion state
    if s_prime == 0:
        if o == 0:
            # 0 alerts
            return 1/3
        elif o == 1:
            # 1 alerts
            return 1 / 3
        elif o == 2:
            # 2 alerts
            return 1 / 3
        else:
            return 0

    # Intrusion state
    if s_prime == 1:
        if o == 0:
            # 0 alerts
            return 1 / 4
        elif o == 1:
            # 1 alerts
            return 1 / 4
        elif o == 2:
            # 2 alerts
            return 1 / 4
        elif o == 3:
            # 3 alerts
            return 1 / 4
        else:
            return 0

    raise ValueError("s_prime:{}, a1:{}, a2:{}, o:{} not recognized".format(s_prime, a1, a2, o))


def bayes_filter(s_prime, o, a1, pi2, A2, b, l, normalize=True):
    norm = 0
    S, _ = states()
    for s in S:
        for a2 in A2:
            for s_prime_1 in S:
                prob_1 = observation_function(s_prime=s_prime_1, a1=a1, a2=a2, o=o)*pi2[s][a2]
                # print("prob:{}, s:{},a:{},o:{}".format(prob_1, s, a, o))
                norm += b[s]*prob_1*transition_operator(s=s, a1=a1, a2=a2, s_prime=s_prime_1,l=l)
    temp = 0
    for s in S:
        for a2 in A2:
            obs_prob = observation_function(s_prime=s_prime,a1=a1,a2=a2,o=o)*pi2[s][a2]
            temp += b[s]*transition_operator(s=s,a1=a1,a2=a2,s_prime=s_prime,l=l)*obs_prob
    if normalize:
        b_prime = temp/norm
    else:
        b_prime = temp
    # if b_prime >1:
    #     print(b_prime)
    # assert b_prime <=1
    return b_prime


def initial_state():
    return 0


