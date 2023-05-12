import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import copy

def get_win_and_invalid_move_counts(samples):
    wins = 0
    invalid_moves = 0
    for i in range(len(samples)):
        if samples[i][1] > 0:
            wins += 1
        elif samples[i][1] == -100:
            invalid_moves += 1
    return wins, invalid_moves

def get_won_games(samples):
    samples.sort(key=lambda x: x[1], reverse=True)
    wins, _invalid_moves = get_win_and_invalid_move_counts(samples)
    return list(map(lambda x: x[0], samples[:wins]))

def get_inverted_board(board):
    inverted_board = copy(board)
    nrows, ncols = inverted_board.shape
    for i in range(nrows):
        for j in range(ncols):
            if inverted_board[i][j] == 0:
                continue
            elif inverted_board[i][j] == 1:
                inverted_board[i][j] = 2
            else:
                inverted_board[i][j] = 1
    return inverted_board

def get_mirrored_board(board):
    mirrored_board = copy(board)
    nrows, ncols = mirrored_board.shape
    for i in range(nrows):
        mirrored_board[i] = mirrored_board[i][::-1]
    return mirrored_board

def get_action_between_boards(board1, board2, agent):
    nrows, ncols = board2.shape
    for i in range(nrows):
        for j in range(ncols):
            if board1[i][j] != board2[i][j] and board2[i][j] == agent:
                return i,j

def transform_losses_to_wins(samples):
    for (i,(game,score)) in enumerate(samples):
        if score < 0 and score != -100:
            for (j,(board,move)) in enumerate(game):
                inverted_board = get_inverted_board(board)
                inverted_move = move
                if j != 0:
                    changed_pos = get_action_between_boards(game[j-1][0],inverted_board,1)
                    inverted_board[changed_pos[0]][changed_pos[1]] = 0 #Undo this move
                    inverted_move = changed_pos[1]
                game[j] = (inverted_board,inverted_move)
            samples[i] = (game,-score)
            
def mirror_games(samples):
    mirrored_samples = []
    for (i,(game,score)) in enumerate(samples):
        mirrored_game = []
        for (j,(board,move)) in enumerate(game):
            mirrored_board = get_mirrored_board(board)
            mirrored_move = 6-move
            mirrored_game.append((mirrored_board,mirrored_move))
        mirrored_samples.append((mirrored_game,score))
    samples += mirrored_samples
    
# def get_elite_games(samples, elite_frac=0.2):
#     num_samples = len(samples)
#     num_elite = int(num_samples*elite_frac)
#     samples.sort(key=lambda x: x[1], reverse=True)
#     for i in range(1,num_elite):
#         if samples[i][1] <= 0:
#             num_elite = i-1 #We only want to train on games that we won so we discard all elite states that result in a loss
#             break
#     return list(map(lambda x: x[0], samples[:num_elite]))

def board_to_matrix(board):
    return np.array(board).reshape(6,7)

device = 'mps'

class DQN_Connect4(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 128, kernel_size=4)
        
        fc_input_dim = 128 * 6 * 7
        self.fc1 = nn.Linear(fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 7)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        # flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output(x))
        
        return x

class Agent:
    def __init__(self):
        self.action_vector = [*range(7)]
        self.batch_size = 64
        self.lr = 1e-4
        self.model = DQN_Connect4().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def predict(self, board):
        board_reshaped = torch.from_numpy(np.array(board).reshape(6, 7, 1)).to(device)
        predictions = self.model(board_reshaped)
        
        for i in range(7):
            if board[0][i] != 0:
                predictions[0][i] = 0
        
        acc = np.sum(predictions)
        for i in range(7):
            predictions[0][i] /= acc
        
        return predictions
    
    def sample_states(self, sample_state, num_samples):
        samples = []
        for i in range(num_samples):
            sample_state.reset()
            sample_state_moves = []
            reward = None
            done = False
            while done == False:
                board = board_to_matrix(sample_state.obs['board'])
                predictions = self.predict(board)
                action = np.random.choice(self.action_list,p=predictions) # Choose next action based on predicted probablities
                sample_state_moves.append((board,action))
                _, reward, done, _ = sample_state.step(action)
            samples.append((sample_state_moves,reward))
        return samples
    
    def train(self, elite_games):
        training_boards, training_moves = [], []
        for elite_game in elite_games:
            for (elite_board,elite_move) in elite_game:
                training_boards.append(elite_board)
                training_move_one_hot = torch.from_numpy(np.array([0 if i != elite_move else 1 for i in range(self.num_actions)],dtype=np.float32)).to(device)
                training_moves.append(training_move_one_hot)
        # self.model.fit(np.array([training_board.reshape(6,7,1) for training_board in training_boards]),np.array(training_moves))
        
        board_values = torch.tensor([torch.from_numpy(training_board.reshape(6,7,1)) for training_board in training_boards])
        