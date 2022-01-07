"""
The machine replacement problem from (Bertsekas, 2005)
"""

from typing import Tuple
import numpy as np

# Constants
theta = 0.1


def states() -> Tuple[np.ndarray, dict]:
    """
    :return: set of states and lookup dict
    """
    return np.array([0,1]), {0: "functional", 1: "broken"}


def actions() -> Tuple[np.ndarray, dict]:
    """
    :return: set of actions and lookup dict
    """
    return np.array([0, 1]), {0: "Continue", 1: "Replace"}


def transition_tensor() -> np.ndarray:
    """
    :return: a |A|x|S|^2 tensor
    """
    P = np.array([
        [
            [1-theta,theta],
            [0, 1]
        ],
        [
            [1, 0],
            [1, 0]
        ]
    ])
    return P


def reward_matrix() -> np.ndarray:
    """
    :return: return a |A|x|S| matrix
    """
    R = np.array([
        [1, 0],
        [0, -2]
    ])
    return R


