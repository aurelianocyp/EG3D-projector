import json
from collections import OrderedDict
import glob
import shutil
import os
import numpy as np

def compute_rotation(angles):
    x, y, z = angles
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    rot_y = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    rot_z = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    return np.matmul(rot_z, np.matmul(rot_y, rot_x))

c = np.load('./eg3d/projector_test_data/06015.npy',allow_pickle=True)
print(c)
# print(compute_rotation([-0.01638367, -0.41674608,  0.19435087]))
# print([1.034335449621267, 0.013151852036503497, 2.493988212972303])
#
# trans = np.array([-0.01638367, -0.41674608,  0.19435087])
# print(trans/np.linalg.norm(trans)*2.7)


# b = np.array([
# [0.912443220615387, -0.03427037224173546, -0.4077657461166382, 1.034335449621267],
#    [ -0.03810766339302063, -0.9992728233337402, -0.0012890419457107782,
# 0.013151852036503497],
#     [-0.40742504596710205, 0.01671517640352249, -0.9130856394767761, 2.493988212972303], [0.0, 0.0, 0.0, 1.0]
# ])

# [0.912443220615387, -0.03427037224173546, -0.4077657461166382, 1.034335449621267, -0.03810766339302063, -0.9992728233337402, -0.0012890419457107782,
# 0.013151852036503497, -0.40742504596710205, 0.01671517640352249, -0.9130856394767761, 2.493988212972303, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]