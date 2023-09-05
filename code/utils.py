import numpy as np


def random_argmax_choice(arr):
    max_value = np.max(arr)

    # Find the indices where the maximum value occurs
    max_indices = np.where(arr == max_value)[0]

    # Randomly choose one index from the maximum indices
    return np.random.choice(max_indices)



