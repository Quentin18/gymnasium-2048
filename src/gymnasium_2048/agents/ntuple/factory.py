from typing import Sequence

import numpy as np


def get_all_straight_tuples(state: np.ndarray) -> Sequence[Sequence[int]]:
    """
    Gets all horizontal and vertical straight tuples.

    :param state: The board state.
    :return: The horizontal and vertical straight tuples.
    """
    tuples = []

    # horizontal
    for row in range(state.shape[0]):
        tuples.append(tuple(state[row, :]))

    # vertical
    for col in range(state.shape[1]):
        tuples.append(tuple(state[:, col]))

    return tuples


def get_all_rectangles_tuples(state: np.ndarray) -> Sequence[Sequence[int]]:
    """
    Gets all 2x2 square tuples.

    :param state: The board state.
    :return: The 2x2 square tuples.
    """
    tuples = []

    # square
    for row in range(state.shape[0] - 1):
        for col in range(state.shape[1] - 1):
            tuples.append(
                (
                    state[row, col],
                    state[row, col + 1],
                    state[row + 1, col + 1],
                    state[row + 1, col],
                )
            )

    return tuples
