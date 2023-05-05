# Simple pygame program

# Import and initialize the pygame library
import numpy as np
import pygame

pygame.init()

RESOLUTION = (1260, 900)

# Set up the drawing window
screen = pygame.display.set_mode([RESOLUTION[0], RESOLUTION[1]])

# Run until the user asks to quit
CirclePos = 0
playerOne = True

def AICHOICE(board):
    return board

from scipy.signal import convolve2d

horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]\
    
def winning_move(board, player):
    for kernel in detection_kernels:
        if (convolve2d(board == player, kernel, mode="valid") == 4).any():
            return True
    return False


board = np.zeros((6, 7), dtype=np.uint8)

def placeDisc(pos, player):
    global board
    for i in range(5, -1, -1):
        if board[i][pos] == 0:
            board[i][pos] = player
            break



running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill(0)
    if playerOne == True:
        if pygame.key.get_pressed()[pygame.K_LEFT]:
            if CirclePos > 0:
                CirclePos -= 1
                pygame.time.wait(100)
        if pygame.key.get_pressed()[pygame.K_RIGHT]:
            if CirclePos < 6:
                CirclePos += 1
                pygame.time.wait(100)
        if pygame.key.get_pressed()[pygame.K_RETURN]:
            placeDisc(CirclePos, 1)
            playerOne = False
            pygame.time.wait(150)
            print(board)
                        
        CircleCords = (int((RESOLUTION[0]/2))-15-(128*3)+(128*CirclePos+1), int((RESOLUTION[1]/2)-375))
        pygame.draw.circle(screen, (255, 0, 0), CircleCords, 50)
        
    else:
        print("Is Red Winning?: " + str(winning_move(board, 1)))
        board = AICHOICE(board)
        playerOne = True
        pygame.time.wait(500)
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