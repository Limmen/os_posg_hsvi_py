"""
Defines a simple stopping game that is OS-POSG
"""

from typing import List, Tuple
import numpy as np

# Constants
R_ST=20.0
R_SLA = 5.0
R_COST=-5.0
R_INT=-10.0
p=0.01


def states() -> Tuple[np.ndarray, dict]:
    """
    :return: the set of states and a lookup dict
    """
    return np.array([0,1]), {0: "NO_INTRUSION", 1: "INTRUSION", 2:"TERMINAL"}


def player_1_actions() -> Tuple[np.ndarray, dict]:
    """
    :return: the set of actions of player 1 and a lookup dict
    """
    return np.array([0, 1]), {0: "CONTINUE", 1: "STOP"}


def player_2_actions() -> Tuple[np.ndarray, dict]:
    """
    :return: the set of actions of player 2 and a lookup dict
    """
    return np.array([0, 1]), {0: "CONTINUE", 1: "STOP"}


def observations() -> Tuple[np.ndarray, dict]:
    """
    :return: the set of observations and a lookup dict
    """
    return np.array([0, 1, 2]), {0: "NO ALERT", 1: "ONE ALERT", 2: "TERMINAL"}


def observation_tensor() -> np.ndarray:
    """
    :return:  a |A1|x|A2|x|S|x|O| tensor
    """
    O = np.array(
        [
            [
                [
                    [0.48, 0.48, 0.01],
                    [0.01, 0.01, 0.98]
                ],
                [
                    [0.01, 0.01, 0.98],
                    [0.01, 0.01, 0.98]
                ]
            ],
            [
                [
                    [0.01, 0.01, 0.98],
                    [0.01, 0.01, 0.98]
                ],
                [
                    [0.01, 0.01, 0.98],
                    [0.01, 0.01, 0.98]
                ]
            ]
        ]
    )

    return O


def reward_tensor() -> np.ndarray:
    """
    :return: return a |A1|x|A2|x|S| tensor
    """
    R = np.array(
        [
            [
                [R_SLA + R_INT, 0],
                [R_SLA, 0]
            ],
            [
                [R_COST + R_ST, 0],
                [R_COST, 0]
            ]
        ]
    )
    return R


def mdp_reward_matrix_p2(P1_strategy: np.ndarray, A1: List) -> np.ndarray:
    """
    :return: return a |A2|x|S| matrix
    """
    R = reward_tensor()
    R_mdp = np.zeros(R[0].shape)
    for a1 in A1:
        R_i = R[a1]
        R_i = R_i*-P1_strategy[a1]
        R_mdp = R_mdp + R_i
    return R_mdp


def transition_tensor() -> np.ndarray:
    """
    :return: a |A1|x|A2||S|^2 tensor
    """
    return np.array(
        [
            [
                [
                    [1-p, p],
                    [0, 1]
                ],
                [
                    [0, 1],
                    [0, 1]
                ]
            ],
            [
                [
                    [0, 1],
                    [0, 1]
                ],
                [
                    [0, 1],
                    [0, 1]
                ]
            ]
        ]
    )


def mdp_transition_tensor_p2(P1_strategy: np.ndarray, A1: List) -> np.ndarray:
    """
    :return: a |A2||S|^2 tensor
    """
    T = transition_tensor()
    T_mdp = np.zeros(T[0].shape)
    for a1 in A1:
        T_i = T[a1]
        T_i = T_i * P1_strategy[a1]
        T_mdp = T_mdp + T_i

    return T_mdp


def initial_belief() -> np.ndarray:
    """
    :return: the initial belief point
    """
    return np.array([1,0])


