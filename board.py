import numpy as np

import random

from scipy.signal import convolve2d

class Board:
    def __init__(self):
        self.board_height = 6
        self.board_width = 7
        self.board = np.zeros((self.board_height, self.board_width), dtype=np.uint8)
        # Player 1: Red | Player 2: Yellow
        self.player_values = {'p1': 1, 'p2': 2, 'empty': 0}
        self.complete = False
        self.reward = {'win': 1, 'loss': 0, 'draw': 0.25}
        
    def setBoard(self, board):
        self.board = board
    
    def possible_moves(self):
        available_columns = []
        for i in range(self.board_width):
            if self.board[0][i] == 0:
                available_columns.append(i)
        return available_columns
    
    def check_if_game_finished(self):
        # check if the connect 4 game on the board is finished
        horizontal_kernel = np.array([[ 1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        
        for kernel in detection_kernels:
            if (convolve2d(self.board == 1, kernel, mode="valid") == 4).any():
                return True
            if (convolve2d(self.board == 2, kernel, mode="valid") == 4).any():
                return True
        return False
    
    def board_full(self):
        for i in range(self.board_width):
            if self.board[0][i] == 0:
                return False
        return True
    
    # takes the value of the column and the value of the player
    def make_move(self, a, p):
        # place the piece in the column by iterating through the rows
        for i in range(self.board_height-1, -1, -1):
            if self.board[i][a] == 0:
                self.board[i][a] = self.player_values[p]
                break
        
        reward = self.check_if_game_finished()
        state = self.board.copy()
        return (state, reward)
    
    def reset(self):
        self.__init__()


if __name__ == '__main__':
    b = Board()
    print(b.board)
    print(b.possible_moves())
