"""Snake model"""

from collections import deque
import random
from typing import Deque, List, Optional, Tuple
from config.game import DIRECTIONS, BOARD_SIZE, INITIAL_SNAKE_LENGTH


class Snake:
    """Represents the snake, its movement, growth, and collision mechanics."""

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        initial_length: int = INITIAL_SNAKE_LENGTH,
        initial_positions: Optional[List[Tuple[int, int]]] = None,
    ):
        self.board_size = board_size
        self.body: Deque[Tuple[int, int]] = deque()

        if initial_positions is not None:
            self.body = deque(initial_positions)
        else:
            self.reset(initial_length)

    def reset(self, length: int = INITIAL_SNAKE_LENGTH) \
            -> List[Tuple[int, int]]:
        """Initialize/reset snake with randomly placed contiguous segments."""
        segments = self._generate_contiguous_snake(length)
        self.body = deque(segments)
        return list(self.body)

    def _generate_contiguous_snake(self, length: int) -> List[Tuple[int, int]]:
        """Generate contiguous segments on the board."""
        dirs = list(DIRECTIONS.values())

        for _ in range(1000):
            head_r = random.randint(1, self.board_size - 2)
            head_c = random.randint(1, self.board_size - 2)
            snake = [(head_r, head_c)]
            valid = True

            for _ in range(length - 1):
                neighbors = []
                last_r, last_c = snake[-1]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = last_r + dr, last_c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        if (nr, nc) not in snake:
                            neighbors.append((nr, nc))
                if not neighbors:
                    valid = False
                    break
                snake.append(neighbors[0])

            if valid and len(snake) == length:
                return snake

        # Fallback default contiguous segments in board center
        mid = self.board_size // 2
        return [(mid, mid), (mid + 1, mid), (mid + 2, mid)]

    @property
    def head(self) -> Tuple[int, int]:
        """Return the current head position."""
        return self.body[0]

    @property
    def body_list(self) -> List[Tuple[int, int]]:
        """Return snake segments as a standard list."""
        return list(self.body)

    @property
    def body_segments(self) -> List[Tuple[int, int]]:
        """Return body segments excluding the head."""
        return list(self.body)[1:]

    def __len__(self) -> int:
        """Return the current length of the snake."""
        return len(self.body)

    def compute_next_head(self, action: str) -> Tuple[int, int]:
        """Compute the next head position given an action."""
        dr, dc = DIRECTIONS[action]
        head_r, head_c = self.head
        return (head_r + dr, head_c + dc)

    def move(self, new_head: Tuple[int, int]):
        """Advance snake forward by one cell without growing."""
        self.body.appendleft(new_head)
        self.body.pop()

    def grow(self, new_head: Tuple[int, int]):
        """Advance snake forward and grow length by 1 (tail is not popped)."""
        self.body.appendleft(new_head)

    def shrink(self, new_head: Tuple[int, int]):
        """Advance snake forward and shrink length by 1."""
        self.body.appendleft(new_head)
        self.body.pop()  # Normal advance
        if self.body:
            self.body.pop()  # Shrink by 1

    def is_wall_collision(self, position: Tuple[int, int]) -> bool:
        """Check if target position collides with board boundaries."""
        r, c = position
        return not (0 <= r < self.board_size and 0 <= c < self.board_size)

    def is_self_collision(self, position: Tuple[int, int]) -> bool:
        """Check if target position collides with snake's own body."""
        # A normal forward step frees the current tail unless the snake grows
        body_list = list(self.body)
        return position in body_list[:-1]
