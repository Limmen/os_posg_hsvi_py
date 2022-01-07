"""
The tiger problem from (Kaelbling, Littman, Cassandra 1998)
"""

from typing import Tuple
import numpy as np


def states() -> Tuple[np.ndarray, dict]:
    """
    :return: set of states and lookup dict
    """
    return np.array([0,1]), {0: "LEFT", 1: "RIGHT"}


def actions() -> Tuple[np.ndarray, dict]:
    """
    :return: set of actions and lookup dict
    """
    return np.array([0, 1, 2]), {0: "LEFT", 1: "RIGHT", 2: "LISTEN"}


def observations() -> Tuple[np.ndarray, dict]:
    """
    :return: set of observations and lookup dict
    """
    return np.array([0, 1]), {0: "TL", 1: "TR"}


def transition_tensor() -> np.ndarray:
    """
    :return: a |A|x|S|^2 tensor
    """
    P = np.array([
        [
            [0.5,0.5],
            [0.5, 0.5]
        ],
        [
            [0.5, 0.5],
            [0.5, 0.5]
        ],
        [
            [1, 0],
            [0, 1]
        ]
    ])
    return P


def observation_matrix() -> np.ndarray:
    """
    :return: a |A|x|S|x|O| matrix
    """
    O = np.array([
        [
            [0.5,0.5],
            [0.5, 0.5]
        ],
        [
            [0.5, 0.5],
            [0.5, 0.5]
        ],
        [
            [0.85, 0.15],
            [0.15, 0.85]
        ]
    ])
    return O


def reward_matrix() -> np.ndarray:
    """
    :return: return a |A|x|S| matrix
    """
    R = np.array([
        [-100, 10],
        [10, -10],
        [-1, -1]
    ])
    return R


def initial_belief() -> np.ndarray:
    """
    :return: initial belief vector
    """
    return np.array([0.5, 0.5])

