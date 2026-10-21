"""Game configuration and constants"""

# Board configuration
BOARD_SIZE = 10

# Directions / Actions (Single Source of Truth)
DIRECTIONS = {
    "UP": (-1, 0),
    "LEFT": (0, -1),
    "DOWN": (1, 0),
    "RIGHT": (0, 1),
}

# Object / Vision characters (Subject Specification)
CHAR_WALL = "W"
CHAR_HEAD = "H"
CHAR_BODY = "S"
CHAR_GREEN = "G"
CHAR_RED = "R"
CHAR_EMPTY = "0"

# Initial environment parameters
INITIAL_SNAKE_LENGTH = 3
INITIAL_GREEN_APPLES = 2
INITIAL_RED_APPLES = 1
