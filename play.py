import numpy as np
import pygame
import torch
from board import Board
from DQN import DQN_1
from copy import deepcopy

pygame.init()

def get_action(policy_model, state, possible_actions):
    with torch.no_grad():
        # select action with highest Q value
        action_vector = policy_model(state)[0, :]
        state_action_vals = [action_vector[action] for action in possible_actions]
        greedy_action = possible_actions[np.argmax(torch.Tensor(state_action_vals).cpu())]
        return greedy_action

RESOLUTION = (1260, 900)

# Set up the drawing window
screen = pygame.display.set_mode([RESOLUTION[0], RESOLUTION[1]])

# Run until the user asks to quit
CirclePos = 0
playerOne = True

model_state_dict = torch.load('DQN_1_player2')
model = DQN_1(7)
model.load_state_dict(model_state_dict)
model.eval()

def AICHOICE(board):
    b = Board()
    b.board = board
    action = get_action(model, b.board, b.possible_moves())
    return action

from scipy.signal import convolve2d

horizontal_kernel = np.array([[ 1.0, 1.0, 1.0, 1.0]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]\
    
def winning_move(board, player):
    for kernel in detection_kernels:
        if (convolve2d(board == player, kernel, mode="valid") == 4).any():
            return True
    return False


board = np.zeros((6, 7), dtype=np.float32)

def placeDisc(pos, player):
    global board
    for i in range(5, -1, -1):
        if board[i][pos] == 0:
            board[i][pos] = player
            break



running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if playerOne == True: 
                    CirclePos -= 1
            if event.key == pygame.K_RIGHT:
                if playerOne == True: 
                    CirclePos += 1
            if event.key == pygame.K_RETURN:
                if playerOne == True: 
                    placeDisc(CirclePos, 1)
                    playerOne = False
                    print(board)
                
    screen.fill(0)
    if playerOne == True:        
        CircleCords = (int((RESOLUTION[0]/2))-15-(128*3)+(128*CirclePos+1), int((RESOLUTION[1]/2)-375))
        pygame.draw.circle(screen, (255, 0, 0), CircleCords, 50)
        
    else:
        print("Is Red Winning?: " + str(winning_move(board, 1)))
        board = placeDisc(AICHOICE(board), 2)
        playerOne = True
        print("Is AI Winning?: " + str(winning_move(board, 2)))
        

        

    # Draw a solid blue circle in the center
    pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(int((100/RESOLUTION[0])+125), int((100/RESOLUTION[1])*1000), int(RESOLUTION[0]-175)-100, int(RESOLUTION[1]-200)), border_radius=50)
    for x in range (0, 7):
        for y in range (0, 6):
            if board[y][x] == 1:
                pygame.draw.circle(screen, (255, 0, 0), (int((RESOLUTION[0]/2)-400+(x*128)), int((RESOLUTION[1]/2)-260+(y*110))), 50)
            elif board[y][x] == 2:
                pygame.draw.circle(screen, (0, 255, 0), (int((RESOLUTION[0]/2)-400+(x*128)), int((RESOLUTION[1]/2)-260+(y*110))), 50)
            else:
                pygame.draw.circle(screen, (0, 0, 0), (int((RESOLUTION[0]/2)-400+(x*128)), int((RESOLUTION[1]/2)-260+(y*110))), 50)

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()