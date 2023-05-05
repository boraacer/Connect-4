import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from board import Board
from DQN import DQN_1
from buffer import ReplayBuffer

import random

from itertools import count

# hyperparameters for models
BATCH_SIZE = 256
GAMMA = 0.999

WIN_RATE_SAMPLES = 100

# target network lag
TARGET_UPDATE = 10

# values for epsilon decay in epsilon-greedy approach
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000

device = 'mps'

# also bora i stole this function because i had no idea how to do this myself
def optimize_step(policy_model, target_model, mem, optim):
    if len(mem) < BATCH_SIZE:
        return

    transitions = mem.sample(BATCH_SIZE)

    state_batch, action_batch, reward_batch, next_state_batch = zip(*[(np.expand_dims(m[0], axis=0), [m[1]], m[2], np.expand_dims(m[3], axis=0)) for m in transitions])
    
    # wrap tensors
    state_batch = torch.tensor(state_batch, dtype=torch.float, device=device)
    action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)

    state_action_values = policy_model(state_batch).gather(1, action_batch)
    
    non_final_mask = torch.tensor(tuple(map(lambda s_: s_[0] is not None, next_state_batch)), device=device)
    non_final_next_state = torch.cat([torch.tensor(s_, dtype=torch.float, device=device).unsqueeze(0) for s_ in next_state_batch])
    
    # truth from target_net, initialize with zeros since terminal state value = 0
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # tensor.detach() creates a tensor that shares storage with tensor that does not require grad
    next_state_values[non_final_mask] = target_model(non_final_next_state).max(1)[0].detach()
    # compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss which is robust to outliers
    loss = F.huber_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # torch.tensor.unsqueeze returns a copy

    optim.zero_grad()
    loss.backward()
    optim.step()


# select best possible action using an epsilon-greedy approach
def select_action(policy_model, state, possible_actions, steps=None, training=True):
    state = torch.Tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unqueeze(dim=0)
    eps = random.random()
    
    if training:
        # epsilon decay
        threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps / EPS_DECAY)
    else:
        threshold = 0
    
    if eps > threshold:
        with torch.no_grad():
            # select action with highest Q value
            action_vector = policy_model(state)[0, :]
            state_action_vals = [action_vector[action] for action in possible_actions]
            greedy_action = possible_actions[np.argmax(state_action_vals)]
            return greedy_action
    else:
        # if epsilon is greater than threshold then select random action
        return random.choice(possible_actions)

# play 100 games to sample win rate of agent 1
def generate_win_rate(p1_policy, p2, env, num_games=WIN_RATE_SAMPLES):
    win_moves_lst = []
    wins = 0
    for i in range(num_games):
        env.reset()
        win_moves_taken = 0
        
        while not env.complete:
            state = env.board.copy()
            available_actions = env.possible_moves()
            action_1 = select_action(p1_policy, state, available_actions, training=False)
            
            state1, reward1 = env.make_move(action_1, 'p1')
            win_moves_taken += 1
            
            if reward1 == 1:
                win_moves_lst.append(win_moves_taken)
                wins += 1
                
            available_actions = env.possible_moves()
            action_2 = p2.act(state, available_actions)
            state, reward = env.make_move(action_2, 'p2')
            
    return wins / num_games, np.mean(win_moves_lst)    

# initialize board and parameters
b = Board()

n_actions = b.board_width

height, width = (b.board_height, b.board_width)

# neural network for player 1
policy_net_1 = DQN_1(n_actions).to(device)
target_net_1 = DQN_1(n_actions).to(device)

target_net_1.load_state_dict(policy_net_1.state_dict())

target_net_1.eval()

optimizer1 = torch.optim.Adam(policy_net_1.parameters(), lr=0.0001)

memory1 = ReplayBuffer()

# avoid reset
steps_done_1 = 0
training_history_1 = []

agent_2 = 2


# number of episodes to train, this is really small for testing purposes
num_episodes = 1000

# training loop
for i in range(num_episodes):
    b.reset()
    state_p1 = b.board.copy()
    state_p2 = b.board.copy()

    # record each 20 epochs
    if i % 20 == 1:
        winrate, moves = generate_win_rate(policy_net_1, policy_net_2, b)
        training_history_1.append([i + 1, winrate, moves])
        tmp_hist = np.array(training_history_1)
        
        if i % 100 == 1:
            print('Episode: ', i + 1, ' Winrate: ', winrate, ' Moves: ', moves)
    
    for t in count():
        available_actions = b.possible_moves()
        action_p1 = select_action(state_p1, available_actions, steps_done)
        steps_done_1 += 1
        state_p1_, reward_p1 = b.make_move(action_p1, 'p1')
        
        
        
        optimize_step(policy_net_1, target_net_1, memory1, optimizer1)
