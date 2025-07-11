import pygame
import numpy as np
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from chain_reaction_env import ChainReactionEnv  # Import the class from earlier

# --- Config ---
BOARD_SIZE = 6
CELL_SIZE = 80
MARGIN = 5
WINDOW_SIZE = BOARD_SIZE * (CELL_SIZE + MARGIN) + MARGIN
FPS = 30

# --- Colors ---
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
RED = (220, 50, 50)
BLUE = (50, 100, 220)
BLACK = (0, 0, 0)

# --- Init ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Chain Reaction: Human (Red) vs AI (Blue)")
font = pygame.font.SysFont(None, 36)
clock = pygame.time.Clock()

# --- Load AI ---
model = MaskablePPO.load("ppo_chainreaction")
env = ChainReactionEnv(size=BOARD_SIZE)
obs, _ = env.reset()

def draw_board(env):
    screen.fill(GRAY)
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            value = env.board[x, y]
            owner = env.owner[x, y]
            color = WHITE
            if owner == 1:
                color = RED
            elif owner == 2:
                color = BLUE
            rect = pygame.Rect(
                MARGIN + y * (CELL_SIZE + MARGIN),
                MARGIN + x * (CELL_SIZE + MARGIN),
                CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, color, rect)
            if value > 0:
                text = font.render(str(value), True, BLACK)
                screen.blit(text, (rect.x + CELL_SIZE // 3, rect.y + CELL_SIZE // 4))
    pygame.display.flip()

def get_cell_from_mouse(pos):
    mx, my = pos
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            rx = MARGIN + y * (CELL_SIZE + MARGIN)
            ry = MARGIN + x * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(rx, ry, CELL_SIZE, CELL_SIZE)
            if rect.collidepoint(mx, my):
                return x, y
    return None

# --- Game Loop ---
running = True
done = False
obs, _ = env.reset()

while running:
    clock.tick(FPS)
    draw_board(env)

    if env.done:
        print("Game Over. Press ESC to exit.")
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and env.current_player == 1 and not env.done:
            cell = get_cell_from_mouse(pygame.mouse.get_pos())
            if cell:
                x, y = cell
                action = x * BOARD_SIZE + y
                obs, reward, done, _, _ = env.step(action)
                draw_board(env)

                # Now AI plays
                if not done and env.current_player == 2:
                    ai_action, _ = model.predict(obs)
                    obs, reward, done, _, _ = env.step(ai_action)
                    draw_board(env)

pygame.quit()
