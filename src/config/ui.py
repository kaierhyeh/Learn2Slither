"""UI configuration constants"""

# Pygame keycode mapping for manual control
try:
    import pygame
    KEY_TO_DIR = {
        pygame.K_UP: "UP",
        pygame.K_w: "UP",       # 順便支援 WASD
        pygame.K_LEFT: "LEFT",
        pygame.K_a: "LEFT",
        pygame.K_DOWN: "DOWN",
        pygame.K_s: "DOWN",
        pygame.K_RIGHT: "RIGHT",
        pygame.K_d: "RIGHT",
    }
except ImportError:
    KEY_TO_DIR = {}

# Window dimensions
CELL_SIZE = 48
HEADER_HEIGHT = 60

# Colors (R, G, B)
COLOR_BG = (30, 30, 34)
COLOR_GRID = (50, 50, 55)
COLOR_SNAKE_HEAD = (0, 160, 255)
COLOR_SNAKE_BODY = (30, 120, 220)
COLOR_GREEN_APPLE = (40, 200, 80)
COLOR_RED_APPLE = (230, 45, 45)
COLOR_TEXT = (240, 240, 240)
