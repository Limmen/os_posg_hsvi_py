"""
Defines a simple lottery game that is an SG
"""

from typing import Tuple
import numpy as np


def states() -> Tuple[np.ndarray, dict]:
    """
    :return: the set of states and a lookup dict
    """
    return np.array([0,1,2]), {0: "0", 1: "1", 2:"TERMINAL"}


def player_1_actions() -> Tuple[np.ndarray, dict]:
    """
    :return: the set of actions of player 1 and a lookup dict
    """
    return np.array([0, 1]), {0: "0", 1: "1"}


def player_2_actions() -> Tuple[np.ndarray, dict]:
    """
    :return: the set of actions of player 2 and a lookup dict
    """
    return np.array([0, 1]), {0: "0", 1: "1"}



def reward_tensor() -> np.ndarray:
    """
    :return: return a |A1|x|A2|x|S| tensor
    """
    R = np.array(
        [
            [
                [2, -4, 0],
                [2, 0, 0],
            ],
            [
                [0, -2, 0],
                [4, -4, 0]
            ]
        ]
    )
    return R


def transition_tensor() -> np.ndarray:
    """
    :return: a |A1|x|A2||S|^2 tensor
    """
    return np.array(
        [
            [
                [
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1]
                ],
                [
                    [0, 0.5, 0.5],
                    [0, 0, 1],
                    [0, 0, 1]
                ]
            ],
            [
                [
                    [0, 0, 1],
                    [0.5, 0, 0.5],
                    [0, 0, 1]
                ],
                [
                    [0.5, 0, 0.5],
                    [0.5, 0, 0.5],
                    [0, 0, 1]
                ]
            ]
        ]
    )


def initial_belief() -> np.ndarray:
    """
    :return: the initial belief point
    """
    return np.array([1,0])


