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

# hyperparameters for models
BATCH_SIZE = 32
GAMMA = 0.999

WIN_RATE_SAMPLES = 100

# target network lag
TARGET_UPDATE = 10

# values for epsilon decay in epsilon-greedy approach
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000

device = 'mps'

def random_agent(moves):
    return random.choice(moves)


# this is a function to perform one iteration of optimization on the policy model using the optimizer passed into the function and based on the results of the memory
def optimize_step(policy_model, target_model, mem, optim):
    if len(mem) < BATCH_SIZE:
        return

    transitions = mem.sample(BATCH_SIZE)

    state_batch, action_batch, reward_batch, next_state_batch = zip(*[(np.expand_dims(m[0], axis=0), [m[1]], m[2], np.expand_dims(m[3], axis=0)) for m in transitions])
    
    # wrap tensors
    state_batch = torch.tensor(np.array(state_batch), dtype=torch.float, device=device)
    
    tmp = []
    for i in action_batch:
        if i == [None]:
            tmp.append([0])
        else: tmp.append(i)
    action_batch = torch.tensor(tmp, dtype=torch.long, device=device)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)

    state_action_values = policy_model(state_batch).gather(1, action_batch)
    
    non_final_mask = torch.tensor(tuple(map(lambda s_: s_[0] is not None, next_state_batch)), device=device)
    non_final_next_state = torch.cat([torch.tensor(s_.astype(np.float32), dtype=torch.float, device=device).unsqueeze(0) for s_ in next_state_batch if s_.shape != (1,)])
    
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


# select the best possible action the model can take using an epsilon-greedy approach
def select_action(policy_model, state, possible_actions, steps=None, training=True):
    # convert the state passed in to a torch.tensor so the model can use it as input
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
    # take a random epsilon
    eps = random.random()
    
    if training:
        # epsilon decay
        threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps / EPS_DECAY)
    else:
        threshold = 0
    
    # as epsilon decays, the probability of taking a random action decreases meaning the model converges to a policy faster
    if eps > threshold:
        with torch.no_grad():
            # select action with highest Q value
            action_vector = policy_model(state)[0, :]
            state_action_vals = [action_vector[action] for action in possible_actions]
            greedy_action = possible_actions[np.argmax(torch.Tensor(state_action_vals).cpu())]
            return greedy_action
    else:
        # if epsilon is greater than threshold then select a random action to play
        return random.choice(possible_actions)

# play 100 games against a random agent to sample win rate of agent 1
def generate_win_rate(p1_policy, env, num_games=WIN_RATE_SAMPLES):
    win_moves_lst = []
    wins = 0
    # loop to play 100 games
    for i in range(num_games):
        env.reset()
        # start the moves taken at 0
        win_moves_taken = 0
        
        # perform the game loop until the game is over
        while not env.complete:
            # copy the board state
            state = env.board.copy()
            
            # get the possible actions, select an action based on the model, and make the move
            available_actions = env.possible_moves()
            action_1 = select_action(p1_policy, state, available_actions, training=False)
            _state1, reward1 = env.make_move(action_1, 'p1')
            # add to number of moves that it took the agent to win
            win_moves_taken += 1
            
            # if there is a win, add to the wins
            if reward1 == 1:
                win_moves_lst.append(win_moves_taken)
                wins += 1
               
            # get the possible actions and make the random agent choose a move 
            available_actions = env.possible_moves()
            action_2 = random_agent(available_actions)
            _state2, _reward2 = env.make_move(action_2, 'p2')
    
    # return win rate and the average number of moves that the agent took to win
    return wins / num_games, np.mean(win_moves_lst)    

# initialize board and parameters
b = Board()

n_actions = b.board_width

height, width = (b.board_height, b.board_width)

# neural networks for the dqn agent
policy_net_1 = DQN_1(n_actions).to(device)
target_net_1 = DQN_1(n_actions).to(device)

target_net_1.load_state_dict(policy_net_1.state_dict())

# put the target network in evaluation mode
target_net_1.eval()

# optimizer for the neural network
optimizer1 = torch.optim.Adam(policy_net_1.parameters(), lr=0.001)

# memory for the dqn agent to remember state transitions
memory1 = ReplayBuffer()

# training history of the model(steps done, winrate)
steps_done_1 = 0
training_history_1 = []


# number of episodes to train the model
num_episodes = 2000

# training loop
for i in range(num_episodes):
    b.reset()
    state_p1 = b.board.copy()

    # record the results each 20 epochs(put them in the training history)
    if i % 20 == 1:
        winrate, moves = generate_win_rate(policy_net_1, b)
        training_history_1.append([i + 1, winrate, moves])
        tmp_hist = np.array(training_history_1)
        
        # every 100 epochs print out the results so far to the console
        if i % 100 == 1:
            print('Episode: ', i + 1, ' Winrate: ', winrate, ' Moves: ', moves)
            
    previous_action_p2 = None
    
    # game loop
    for t in count():
        # get possible moves for the neural network player, select the action, and make the move
        available_actions = b.possible_moves()
        action_p1 = select_action(policy_net_1, state_p1, available_actions, steps_done_1)
        steps_done_1 += 1
        state_p1_, reward_p1_ = b.make_move(action_p1, 'p1')
        b.check_if_game_finished()
        
        # if the game is finished, add the transition vector to the memory, which is structured as (state, action, reward, next state)
        if b.complete:
            if reward_p1_ == 1:
                memory1.add([state_p1, action_p1, 1, None])
            else:
                memory1.add([state_p1, action_p1, 0.5, None])
            break
        
        # if the game is finished then break the loop
        if b.complete: break
        
        # get possible moves for the random agent, select the action using the random agent, and make the move
        state_p2 = b.board.copy()

        available_actions = b.possible_moves()
        action_p2 = random_agent(available_actions)
        state_p2_, reward_p2_= b.make_move(action_p2, 'p2')
        b.check_if_game_finished()

        # if the game is finished, add the transition vector to the memory        
        if b.complete:
            if reward_p2_ == 1:
                reward_p1_2 = -1
                memory1.add([state_p1, action_p1, -1, None])
            else:
                memory1.add([state_p1, action_p1, 0.5, None])
            break
        
        if b.complete: break
        
        # punish model for taking too long to win by adding a transition vector to the memory with a slight negative reward every turn the model goes without winning
        # this is to prevent the model from trying to attain a score which gets stuck very close to zero
        memory1.add([state_p1, action_p1, -0.01, state_p1_])
        
        # perform the model optimization step based on the memory
        optimize_step(policy_net_1, target_net_1, memory1, optimizer1)
    
    # every TARGET_UPDATE epochs, update the target network
    if i % TARGET_UPDATE == TARGET_UPDATE - 1:
        target_net_1.load_state_dict(policy_net_1.state_dict())

# training is finished!
print('complete')

# save the model
model_filename_1 = 'DQN_1_player1_mps_randomagent.pth'
torch.save(policy_net_1, model_filename_1)