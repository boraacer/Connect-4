import numpy as np


horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]\
    
from scipy.signal import convolve2d

def winning_move(board, player):
    for kernel in detection_kernels:
        if (convolve2d(board == player, kernel, mode="valid") == 4).any():
            return True
    return False

board = np.array([[0,0,0,0,0,0,0],
          [1,0,0,0,0,0,0],
          [1, 0, 1, 0, 2, 0, 0],
          [1, 0, 1, 0, 2, 0 ,1],
          [1, 1 ,1, 1 ,2, 0 ,1],
          [1, 1 ,1 ,1 ,2 ,0 ,1]])

print(winning_move(board, 2))