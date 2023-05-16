import numpy as np
import pygame
import torch
from board import Board
from DQN import DQN_1
from copy import deepcopy
import random

# Initializing Pygame
pygame.init()

# Setting System Font and Resolution
font = pygame.font.SysFont("Arial", 120)
RESOLUTION = (1260, 900)

# Setting up the board
board = np.zeros((6, 7), dtype=np.uint8)


# Link to the AI DQN Class
def select_action(policy_model, state, possible_actions, steps=None, training=True):
    state = torch.tensor(state, dtype=torch.float, device='cpu').unsqueeze(dim=0).unsqueeze(dim=0)
    eps = random.random()
    
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
        # if epsilon is greater than threshold then select random action
        return random.choice(possible_actions)



# Set up the drawing window
screen = pygame.display.set_mode([RESOLUTION[0], RESOLUTION[1]])

# Run until the user asks to quit
CirclePos = 0
playerOne = True

# Loading in the model from file
model = torch.load('DQN_1_player1_mps_randomagent.pth', map_location=torch.device('cpu') )
model.eval()

def AICHOICE(board):
    b = Board()
    b.board = board
    action = select_action(model, b.board.copy(), b.possible_moves())
    print(f"AI ACTION: {action}")
    return action


# Check if a player has won with
def winning_move(board, player):
    from scipy.signal import convolve2d
    horizontal_kernel = np.array([[ 1.0, 1.0, 1.0, 1.0]])
    vertical_kernel = np.transpose(horizontal_kernel)
    diag1_kernel = np.eye(4, dtype=np.uint8)
    diag2_kernel = np.fliplr(diag1_kernel)
    detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
    for kernel in detection_kernels:
        if (convolve2d(board == player, kernel, mode="valid") == 4).any():
            return True
    return False



def placeDisc(pos, player):
    global board
    for i in range(5, -1, -1):
        if board[i][pos] == 0:
            board[i][pos] = player
            break


# As the adversary place the first disc randomly
placeDisc(random.randint(0,6), 2)

# Main Game Loop
running = True
while running:
    
    # User input
    for event in pygame.event.get():
        
        # Quitting the game with the X button
        if event.type == pygame.QUIT:
            running = False

        # Key Presses
        if event.type == pygame.KEYDOWN:
            
            # Called if the user presses the left arrow key
            if event.key == pygame.K_LEFT:
                if playerOne == True: 
                    if CirclePos > 0:
                        CirclePos -= 1
            # Called if the user presses the right arrow key
            if event.key == pygame.K_RIGHT:
                if playerOne == True: 
                    if CirclePos < 6:
                        CirclePos += 1
            # Called if the user presses the enter key
            if event.key == pygame.K_RETURN:
                if playerOne == True: 
                    # Called if the user places a disc in a column that is not full
                    if board[0][CirclePos] == 0:
                        placeDisc(CirclePos, 1)
                        playerOne = False
                        print(board)
                    else:
                        # Called if the user tries to place a disc in a full column
                        print("Bad Move!")
                        txtsurf = font.render("Bad Move!!!", True, (255,255,255), (0))
                        screen.blit(txtsurf, (int((RESOLUTION[0]/2))-260, int((RESOLUTION[1]/2)-100)))
                        pygame.display.flip()
                        pygame.time.delay(1000)
        
    # Fill the background with black           
    screen.fill(0)
    if playerOne == True:        
        CircleCords = (int((RESOLUTION[0]/2))-15-(128*3)+(128*CirclePos+1), int((RESOLUTION[1]/2)-375))
        pygame.draw.circle(screen, (255, 0, 0), CircleCords, 50)
        
    else:
        # AI Turn
        placeDisc(AICHOICE(board), 2)
        playerOne = True
        print("Is AI Winning?: " + str(winning_move(board, 2)))
        

        

    # Draw a solid blue circle in the center
    pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(int((100/RESOLUTION[0])+125), int((100/RESOLUTION[1])*1000), int(RESOLUTION[0]-175)-100, int(RESOLUTION[1]-200)), border_radius=50)
    # Draw the circles on the board
    for x in range (0, 7):
        for y in range (0, 6):
            if board[y][x] == 1:
                pygame.draw.circle(screen, (255, 0, 0), (int((RESOLUTION[0]/2)-400+(x*128)), int((RESOLUTION[1]/2)-260+(y*110))), 50)
            elif board[y][x] == 2:
                pygame.draw.circle(screen, (0, 0, 255), (int((RESOLUTION[0]/2)-400+(x*128)), int((RESOLUTION[1]/2)-260+(y*110))), 50)
            else:
                pygame.draw.circle(screen, (0, 0, 0), (int((RESOLUTION[0]/2)-400+(x*128)), int((RESOLUTION[1]/2)-260+(y*110))), 50)
    # Check if Human player has won
    if winning_move(board, 1) == True:
            print("Red Wins!")
            txtsurf = font.render("RED WINS!", True, (255, 0, 0), (0))
            screen.blit(txtsurf, (int((RESOLUTION[0]/2))-280, int((RESOLUTION[1]/2)-100)))
            break
    # Check if AI has won
    elif winning_move(board, 2) == True:
            print("Blue Wins!")
            txtsurf = font.render("BLUE WINS!", True, (0, 0, 255), (0))
            screen.blit(txtsurf, (int((RESOLUTION[0]/2))-260, int((RESOLUTION[1]/2)-100)))
            break
    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()