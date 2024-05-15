import numpy as np

def calculate_reorientation_matrix(floor_equation):
    if floor_equation[3] != 0:
            scale_factor = 1 / c
            translation = np.array([0, 0, -d / c])
            transformation_matrix = np.array([[scale_factor, 0, 0, 0],
                                               [0, scale_factor, 0, 0],
                                               [0, 0, scale_factor, 0],
                                               [0, 0, 0, 1]])
    return transformation_matrix