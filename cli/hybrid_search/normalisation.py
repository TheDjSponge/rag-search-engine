import numpy as np


def normalize(number_list: list[float]) -> list[float]:
    min_val = min(number_list)
    max_val = max(number_list)

    if min_val == max_val:
        return [1.0] * len(number_list)

    arr = np.array(number_list)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return list(normalized_arr.tolist())
