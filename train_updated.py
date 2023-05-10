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

from copy import deepcopy

device = 'mps'

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000

BATCH_SIZE = 256
GAMMA = 0.999

policy_network = DQN_1(7).to(device)
target_network = DQN_1(7).to(device)

target_network.load_state_dict(policy_network.state_dict())

target_network.eval()

model_optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

model_memory = ReplayBuffer()

b = Board()

def select_action(state, possible_actions, steps=None, training=True):
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
    eps = random.random()
    
    if training:
        # epsilon decay
        threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps / EPS_DECAY)
    else:
        threshold = 0
    
    if eps > threshold:
        with torch.no_grad():
            # select action with highest Q value
            action_vector = policy_network(state)[0, :]
            state_action_vals = [action_vector[action] for action in possible_actions]
            greedy_action = possible_actions[np.argmax(torch.Tensor(state_action_vals).cpu())]
            return greedy_action
    else:
        # if epsilon is greater than threshold then select random action
        return random.choice(possible_actions)


def optimize_model():
    if len(model_memory) < BATCH_SIZE:
        return
    transitions = model_memory.sample(BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch = zip(*[(np.expand_dims(m[0], axis=0), [m[1]], m[2], np.expand_dims(m[3], axis=0)) for m in transitions])
    # tensor wrapper
    state_batch = torch.tensor(np.array(state_batch), dtype=torch.float, device=device)
    action_batch = torch.tensor(np.array(action_batch), dtype=torch.long, device=device)
    reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float, device=device)
    
    # for assigning terminal state value = 0 later
    non_final_mask = torch.tensor(tuple(map(lambda s_: s_[0] is not None, next_state_batch)), device=device)
    non_final_next_state = torch.cat([torch.tensor(s_, dtype=torch.float, device=device).unsqueeze(0) for s_ in next_state_batch if s_[0] is not None])
    
    # prediction from policy_net
    state_action_values = policy_network(state_batch).gather(1, action_batch)
    
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    next_state_values[non_final_mask] = target_network(non_final_next_state).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.huber_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()

def random_selection(possible_actions):
    return random.choice(possible_actions)

def sample_win_rate():
    win_moves_taken_list = []
    win = []
    for _ in range(100):
        b.reset()
        win_moves_taken = 0

        while not b.complete:
            state = b.board.copy()
            available_actions = b.possible_moves()
            action = select_action(state, available_actions, training=False)
            state, reward = b.make_move(action, 'p1')
            win_moves_taken += 1

            if reward == 1:
                win_moves_taken_list.append(win_moves_taken)
                win.append(1)
                break

            available_actions = b.possible_moves()
            action = random_selection(available_actions)
            state, reward = b.make_move(action, 'p2')

    return np.sum(win)/100, np.mean(win_moves_taken_list)

steps_done = 0
training_history = []

from itertools import count

num_episodes = 1000

TARGET_UPDATE = 20

for i in range(num_episodes): 
    b.reset()
    state_p1 = b.board.copy()

    # record every 20 epochs
    if i % 20 == 19:
        win_rate, moves_taken = sample_win_rate()
        training_history.append([i + 1, win_rate, moves_taken])
        th = np.array(training_history)
        # print training message every 200 epochs
        if i % 200 == 199:
            print('Episode {}: | win_rate: {} | moves_taken: {}'.format(i, th[-1, 1], th[-1, 2]))

    for t in count():
        available_actions = b.possible_moves()
        action_p1 = select_action(state_p1, available_actions, steps_done)
        steps_done += 1
        state_p1_, reward_p1 = b.make_move(action_p1, 'p1')
        
        if b.complete:
            if reward_p1 == 1:
                # reward p1 for p1's win
                model_memory.add([state_p1, action_p1, 1, None])
            else:
                # state action value tuple for a draw
                model_memory.add([state_p1, action_p1, 0.5, None])
            break
        
        available_actions = b.possible_moves()
        action_p2 = random_selection(available_actions)
        state_p2_, reward_p2 = b.make_move(action_p2, 'p2')
        
        if b.complete:
            if reward_p2 == 1:
                # punish p1 for (random agent) p2's win 
                model_memory.add([state_p1, action_p1, -1, None])
            else:
                # state action value tuple for a draw
                model_memory.add([state_p1, action_p1, 0.5, None])
            break
        
        # punish for taking too long to win
        model_memory.add([state_p1, action_p1, -0.05, state_p2_])
        state_p1 = state_p2_
        
        # Perform one step of the optimization
        optimize_model()
        
    if i % TARGET_UPDATE == TARGET_UPDATE - 1:
        policy_network.load_state_dict(policy_network.state_dict())

print('Complete')

torch.save(policy_network, 'policy_network_DQN1_new.pth')