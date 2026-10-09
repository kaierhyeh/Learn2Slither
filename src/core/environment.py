"""Game environment engine"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from config.game import (
    BOARD_SIZE,
    CHAR_BODY,
    CHAR_EMPTY,
    CHAR_GREEN,
    CHAR_HEAD,
    CHAR_RED,
    CHAR_WALL,
    DIRECTIONS,
    INITIAL_GREEN_APPLES,
    INITIAL_RED_APPLES,
    INITIAL_SNAKE_LENGTH,
)
from core.snake import Snake


class Environment:
    """10x10 Environment, rules, and vision."""

    def __init__(self, board_size: int = BOARD_SIZE):
        self.board_size = board_size
        self.snake_model = Snake(
            board_size=board_size,
            initial_length=INITIAL_SNAKE_LENGTH,
        )
        self.green_apples: List[Tuple[int, int]] = []
        self.red_apple: Optional[Tuple[int, int]] = None
        self.duration: int = 0
        self.max_length: int = INITIAL_SNAKE_LENGTH
        self.game_over: bool = False
        self.game_over_reason: Optional[str] = None
        self.reset()

    @property
    def snake(self) -> List[Tuple[int, int]]:
        """Return snake coordinates as a list with head at index 0."""
        return self.snake_model.body_list

    def reset(self) -> Tuple[Any, ...]:
        """Reset environment with contiguous 3-cell snake and fresh apples."""
        initial_snake = self.snake_model.reset(length=INITIAL_SNAKE_LENGTH)

        # Place 2 green apples and 1 red apple on empty cells
        occupied = set(initial_snake)
        self.green_apples = []
        for _ in range(INITIAL_GREEN_APPLES):
            g_pos = self._spawn_apple(occupied)
            self.green_apples.append(g_pos)
            occupied.add(g_pos)

        self.red_apple = (
            self._spawn_apple(occupied) if INITIAL_RED_APPLES > 0 else None
        )
        self.duration = 0
        self.max_length = len(self.snake_model)
        self.game_over = False
        self.game_over_reason = None
        return self.get_state()

    def _spawn_apple(self, occupied: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Find a random empty board cell not in occupied set."""
        empty_cells = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if (r, c) not in occupied
        ]
        if not empty_cells:
            return (0, 0)
        return random.choice(empty_cells)

    def get_vision_cross_string(self) -> str:
        """Generate terminal cross vision matching 42 Subject specification."""
        if len(self.snake_model) == 0:
            return ""

        head_r, head_c = self.snake_model.head
        body_set = set(self.snake_model.body_segments)
        green_set = set(self.green_apples)

        def cell_char(r: int, c: int) -> str:
            pos = (r, c)
            if pos == (head_r, head_c):
                return CHAR_HEAD
            if pos in body_set:
                return CHAR_BODY
            if pos in green_set:
                return CHAR_GREEN
            if pos == self.red_apple:
                return CHAR_RED
            return CHAR_EMPTY

        # UP ray from Wall (row -1) to cell above head (head_r - 1)
        up_chars = [CHAR_WALL]
        for r in range(head_r):
            up_chars.append(cell_char(r, head_c))

        # DOWN ray from cell below head (head_r + 1) to Wall (row board_size)
        down_chars = []
        for r in range(head_r + 1, self.board_size):
            down_chars.append(cell_char(r, head_c))
        down_chars.append(CHAR_WALL)

        # LEFT ray from Wall (col -1) to cell left of head (head_c - 1)
        left_chars = [CHAR_WALL]
        for c in range(head_c):
            left_chars.append(cell_char(head_r, c))

        # RIGHT ray from cell right of head to Wall
        right_chars = []
        for c in range(head_c + 1, self.board_size):
            right_chars.append(cell_char(head_r, c))
        right_chars.append(CHAR_WALL)

        indent = " " * len(left_chars)
        lines = []

        # UP ray characters
        for ch in up_chars:
            lines.append(f"{indent}{ch}")

        # Horizontal line: LEFT + H + RIGHT
        horiz_line = "".join(left_chars) + CHAR_HEAD + "".join(right_chars)
        lines.append(horiz_line)

        # DOWN ray characters
        for ch in down_chars:
            lines.append(f"{indent}{ch}")

        return "\n".join(lines)

    def get_state(self) -> Tuple[Any, ...]:
        """Extract compact state feature tuple from 4 visible rays."""
        if len(self.snake_model) == 0:
            return ()

        head_r, head_c = self.snake_model.head
        body_set = set(self.snake_model.body_segments)
        green_set = set(self.green_apples)

        features = []
        for action, (dr, dc) in DIRECTIONS.items():

            # 1. Immediate danger at distance 1
            adj_r, adj_c = head_r + dr, head_c + dc
            danger = 0
            if not (0 <= adj_r < self.board_size and 0 <= adj_c < self.board_size):
                danger = 1
            elif (adj_r, adj_c) in body_set:
                danger = 1

            # 2. Ray scan to locate first visible object and apple presence
            first_item = CHAR_WALL
            first_dist = 0
            saw_green = 0
            saw_red = 0
            step = 1

            while True:
                curr_r, curr_c = head_r + dr * step, head_c + dc * step
                if not (0 <= curr_r < self.board_size and 0 <= curr_c < self.board_size):
                    if first_dist == 0:
                        first_item = CHAR_WALL
                        first_dist = step
                    break

                pos = (curr_r, curr_c)
                if pos in body_set:
                    if first_dist == 0:
                        first_item = CHAR_BODY
                        first_dist = step
                elif pos in green_set:
                    saw_green = 1
                    if first_dist == 0:
                        first_item = CHAR_GREEN
                        first_dist = step
                elif pos == self.red_apple:
                    saw_red = 1
                    if first_dist == 0:
                        first_item = CHAR_RED
                        first_dist = step

                step += 1

            dist_bucket = min(first_dist, 3)
            features.extend([
                danger,
                first_item,
                dist_bucket,
                saw_green,
                saw_red,
            ])

        return tuple(features)

    def step(self, action: str) \
            -> Tuple[Tuple[Any, ...], float, bool, Dict[str, Any]]:
        """Advance the environment by one step given action."""
        if self.game_over:
            return self.get_state(), 0.0, True, {}

        self.duration += 1
        new_head = self.snake_model.compute_next_head(action)

        # 1. Wall collision
        if self.snake_model.is_wall_collision(new_head):
            self.game_over = True
            self.game_over_reason = "wall"
            return self.get_state(), -50.0, True, {"reason": "wall"}

        # 2. Self collision
        if self.snake_model.is_self_collision(new_head):
            self.game_over = True
            self.game_over_reason = "self"
            return self.get_state(), -50.0, True, {"reason": "self"}

        reward = -0.1

        # 3. Eating green apple
        if new_head in self.green_apples:
            self.snake_model.grow(new_head)
            self.green_apples.remove(new_head)
            occupied = set(self.snake_model.body_list) | set(self.green_apples)
            if self.red_apple:
                occupied.add(self.red_apple)
            self.green_apples.append(self._spawn_apple(occupied))
            reward = 10.0
        # 4. Eating red apple
        elif new_head == self.red_apple:
            self.snake_model.shrink(new_head)
            if len(self.snake_model) == 0:
                self.game_over = True
                self.game_over_reason = "zero_length"
                return self.get_state(), -50.0, True, {"reason": "zero_length"}

            occupied = set(self.snake_model.body_list) | set(self.green_apples)
            self.red_apple = self._spawn_apple(occupied)
            reward = -10.0
        # 5. Normal movement
        else:
            self.snake_model.move(new_head)

        self.max_length = max(self.max_length, len(self.snake_model))
        return self.get_state(), reward, False, {}
