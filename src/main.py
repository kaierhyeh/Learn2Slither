"""Learn2Slither - Snake Reinforcement Learning.

Main entry point handling CLI arguments, execution modes, and orchestration
between Environment, Interpreter, Agent, and GUI.
"""

import argparse
import json
import os
import random
import time

# Action definitions
UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3
ACTION_NAMES = ["UP", "LEFT", "DOWN", "RIGHT"]
ACTION_DELTAS = {
    UP: (-1, 0),
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
}

# Vision Characters (42 Subject Specification)
CHAR_WALL = "W"
CHAR_HEAD = "H"
CHAR_BODY = "S"
CHAR_GREEN = "G"
CHAR_RED = "R"
CHAR_EMPTY = "0"


class SnakeEnvironment:
    """10x10 Snake Environment following 42 Subject specifications."""

    def __init__(self, board_size=10):
        self.board_size = board_size
        self.snake = []
        self.green_apples = []
        self.red_apple = None
        self.duration = 0
        self.max_length = 3
        self.game_over = False

    def reset(self):
        """Reset environment with contiguous 3-cell snake and apples."""
        self.duration = 0
        self.game_over = False
        self.snake = self._generate_contiguous_snake(length=3)
        self.max_length = len(self.snake)

        # Place 2 green apples and 1 red apple
        occupied = set(self.snake)
        self.green_apples = []
        for _ in range(2):
            g_pos = self._random_empty_cell(occupied)
            self.green_apples.append(g_pos)
            occupied.add(g_pos)

        self.red_apple = self._random_empty_cell(occupied)
        return self.get_state()

    def _generate_contiguous_snake(self, length=3):
        """Generate a random contiguous snake inside board boundaries."""
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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

        # Fallback default contiguous snake in center
        mid = self.board_size // 2
        return [(mid, mid), (mid + 1, mid), (mid + 2, mid)]

    def _random_empty_cell(self, occupied):
        """Find a random empty cell not in occupied set."""
        empty_cells = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if (r, c) not in occupied
        ]
        if not empty_cells:
            return (0, 0)
        return random.choice(empty_cells)

    def get_vision_cross_string(self):
        """Generate terminal cross vision matching subject specification."""
        if not self.snake:
            return ""

        head_r, head_c = self.snake[0]
        body_set = set(self.snake[1:])
        green_set = set(self.green_apples)

        def cell_char(r, c):
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

        # RIGHT ray from cell right of head (head_c + 1) to Wall (col board_size)
        right_chars = []
        for c in range(head_c + 1, self.board_size):
            right_chars.append(cell_char(head_r, c))
        right_chars.append(CHAR_WALL)

        indent = " " * len(left_chars)
        lines = []

        # Print UP ray characters
        for ch in up_chars:
            lines.append(f"{indent}{ch}")

        # Print horizontal line: LEFT + H + RIGHT
        horiz_line = "".join(left_chars) + CHAR_HEAD + "".join(right_chars)
        lines.append(horiz_line)

        # Print DOWN ray characters
        for ch in down_chars:
            lines.append(f"{indent}{ch}")

        return "\n".join(lines)

    def get_state(self):
        """Extract compact state feature tuple strictly from 4 visible rays.

        Provides ONLY the information visible to the snake to strictly avoid
        any -42 penalty.
        """
        if not self.snake:
            return ()

        head_r, head_c = self.snake[0]
        body_set = set(self.snake[1:])
        green_set = set(self.green_apples)

        features = []
        for action in (UP, LEFT, DOWN, RIGHT):
            dr, dc = ACTION_DELTAS[action]

            # 1. Immediate danger (distance 1)
            adj_r, adj_c = head_r + dr, head_c + dc
            danger = 0
            if not (0 <= adj_r < self.board_size and 0 <= adj_c < self.board_size):
                danger = 1
            elif (adj_r, adj_c) in body_set:
                danger = 1

            # 2. Ray scan to find first visible item and green/red apple presence
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

            # Discretize first obstacle distance: 1 (adjacent), 2 (near), 3 (far)
            dist_bucket = min(first_dist, 3)
            features.extend([danger, first_item, dist_bucket, saw_green, saw_red])

        return tuple(features)

    def step(self, action):
        """Execute action, update state, and return (next_state, reward, done)."""
        if self.game_over:
            return self.get_state(), 0.0, True, {}

        self.duration += 1
        dr, dc = ACTION_DELTAS[action]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        # 1. Wall collision
        if not (0 <= new_head[0] < self.board_size and 0 <= new_head[1] < self.board_size):
            self.game_over = True
            return self.get_state(), -50.0, True, {"reason": "wall"}

        # 2. Self collision (hitting any body segment except potentially tail being removed)
        # In strict snake rules, hitting any current body part is game over
        if new_head in self.snake[:-1]:
            self.game_over = True
            return self.get_state(), -50.0, True, {"reason": "self"}

        reward = -0.1

        # 3. Eating green apple
        if new_head in self.green_apples:
            self.snake.insert(0, new_head)
            self.green_apples.remove(new_head)
            occupied = set(self.snake) | set(self.green_apples)
            if self.red_apple:
                occupied.add(self.red_apple)
            self.green_apples.append(self._random_empty_cell(occupied))
            reward = 10.0
        # 4. Eating red apple
        elif new_head == self.red_apple:
            self.snake.insert(0, new_head)
            self.snake.pop()  # Normal head movement
            if self.snake:
                self.snake.pop()  # Length decrease by 1

            if len(self.snake) == 0:
                self.game_over = True
                return self.get_state(), -50.0, True, {"reason": "zero_length"}

            occupied = set(self.snake) | set(self.green_apples)
            self.red_apple = self._random_empty_cell(occupied)
            reward = -10.0
        # 5. Normal movement
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()

        self.max_length = max(self.max_length, len(self.snake))
        return self.get_state(), reward, False, {}


class QTableAgent:
    """Q-learning agent for Snake decision making and persistence."""

    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.2, dontlearn=False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 0.0 if dontlearn else epsilon
        self.dontlearn = dontlearn
        self.q_table = {}

    def get_q_values(self, state):
        """Get or initialize Q-values for state."""
        key = str(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0, 0.0, 0.0, 0.0]
        return self.q_table[key]

    def choose_action(self, state):
        """Select action via epsilon-greedy strategy."""
        if not self.dontlearn and random.random() < self.epsilon:
            return random.choice([UP, LEFT, DOWN, RIGHT])

        q_vals = self.get_q_values(state)
        max_q = max(q_vals)
        # Break ties randomly to avoid directional bias
        best_actions = [idx for idx, val in enumerate(q_vals) if val == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Update Q-table entry using Bellman equation."""
        if self.dontlearn:
            return

        key = str(state)
        curr_q = self.get_q_values(state)[action]
        if done:
            target = reward
        else:
            next_q_vals = self.get_q_values(next_state)
            target = reward + self.gamma * max(next_q_vals)

        self.q_table[key][action] = curr_q + self.alpha * (target - curr_q)

    def save(self, filepath):
        """Save Q-table to text file in subject specified format."""
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.q_table, file, indent=2)
        print(f"Save learning state in {filepath}")

    def load(self, filepath):
        """Load Q-table from text file in subject specified format."""
        if not os.path.exists(filepath):
            print(f"Warning: Model file {filepath} not found.")
            return False
        with open(filepath, "r", encoding="utf-8") as file:
            self.q_table = json.load(file)
        print(f"Load trained model from {filepath}")
        return True


def run_training_loop(env, agent, sessions, save_path=None):
    """Ultra-fast training loop (visual off) without GUI or per-step terminal IO."""
    overall_max_len = 0
    overall_max_dur = 0

    for _ in range(sessions):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        overall_max_len = max(overall_max_len, env.max_length)
        overall_max_dur = max(overall_max_dur, env.duration)
        print(f"Game over, max length = {env.max_length}, max duration = {env.duration}")

    if save_path:
        agent.save(save_path)

    return overall_max_len, overall_max_dur


def run_visual_loop(env, agent, sessions, save_path=None, step_by_step=False):
    """Visual demonstration loop with Pygame GUI and per-step terminal vision."""
    pygame_available = False
    screen = None
    clock = None

    # Colors
    color_bg = (30, 30, 34)
    color_grid = (50, 50, 55)
    color_snake_head = (0, 160, 255)
    color_snake_body = (30, 120, 220)
    color_green_apple = (40, 200, 80)
    color_red_apple = (230, 45, 45)

    cell_size = 48
    header_height = 60
    board_px = env.board_size * cell_size
    win_w = board_px
    win_h = board_px + header_height

    try:
        import pygame
        pygame.init()
        pygame.display.set_caption("Learn2Slither - Snake")
        screen = pygame.display.set_mode((win_w, win_h))
        clock = pygame.time.Clock()
        pygame_available = True
        font = pygame.font.SysFont("monospace", 18, bold=True)
    except Exception as exc:
        print(f"Note: Running in terminal visualization mode ({exc})")
        pygame_available = False

    fps = 10
    paused = False

    for sess_idx in range(sessions):
        state = env.reset()
        done = False

        while not done:
            # Handle Pygame UI events
            if pygame_available:
                import pygame
                step_triggered = not step_by_step

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            pygame.quit()
                            return
                        if event.key == pygame.K_SPACE:
                            if step_by_step:
                                step_triggered = True
                            else:
                                paused = not paused
                        if event.key in (pygame.K_UP, pygame.K_EQUALS):
                            fps = min(60, fps + 5)
                        if event.key in (pygame.K_DOWN, pygame.K_MINUS):
                            fps = max(1, fps - 5)

                if step_by_step and not step_triggered:
                    clock.tick(30)
                    continue

                if paused and not step_by_step:
                    clock.tick(15)
                    continue

            # 1. Print cross vision to terminal before asking agent
            vision_str = env.get_vision_cross_string()
            print(vision_str)

            # 2. Agent decides action
            action = agent.choose_action(state)

            # 3. Print action taken to terminal
            print(f"\n{ACTION_NAMES[action]}\n")

            # 4. In terminal-only step-by-step mode, wait for Enter
            if step_by_step and not pygame_available:
                try:
                    input("[Press Enter for next step...]")
                except (EOFError, KeyboardInterrupt):
                    return

            # 5. Advance environment
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

            # 6. Render Pygame GUI
            if pygame_available:
                import pygame
                screen.fill(color_bg)

                # Header info
                info_text = (
                    f"Sess: {sess_idx + 1}/{sessions}  "
                    f"Len: {len(env.snake)} (Max: {env.max_length})  "
                    f"Dur: {env.duration}  Speed: {fps} FPS"
                )
                txt_surf = font.render(info_text, True, (240, 240, 240))
                screen.blit(txt_surf, (12, 18))

                # Grid offset
                ox, oy = 0, header_height

                # Draw grid lines
                for r in range(env.board_size + 1):
                    pygame.draw.line(
                        screen, color_grid,
                        (ox, oy + r * cell_size),
                        (ox + board_px, oy + r * cell_size), 1
                    )
                for c in range(env.board_size + 1):
                    pygame.draw.line(
                        screen, color_grid,
                        (ox + c * cell_size, oy),
                        (ox + c * cell_size, oy + board_px), 1
                    )

                # Draw Green apples
                for gr, gc in env.green_apples:
                    rect = pygame.Rect(
                        ox + gc * cell_size + 4, oy + gr * cell_size + 4,
                        cell_size - 8, cell_size - 8
                    )
                    pygame.draw.rect(screen, color_green_apple, rect, border_radius=6)

                # Draw Red apple
                if env.red_apple:
                    rr, rc = env.red_apple
                    rect = pygame.Rect(
                        ox + rc * cell_size + 4, oy + rr * cell_size + 4,
                        cell_size - 8, cell_size - 8
                    )
                    pygame.draw.rect(screen, color_red_apple, rect, border_radius=6)

                # Draw Snake body & head
                for idx, (sr, sc) in enumerate(env.snake):
                    rect = pygame.Rect(
                        ox + sc * cell_size + 3, oy + sr * cell_size + 3,
                        cell_size - 6, cell_size - 6
                    )
                    color = color_snake_head if idx == 0 else color_snake_body
                    pygame.draw.rect(screen, color, rect, border_radius=5)

                pygame.display.flip()
                if not step_by_step:
                    clock.tick(fps)
            else:
                if not step_by_step:
                    time.sleep(0.1)

        print(f"Game over, max length = {env.max_length}, max duration = {env.duration}")

    if pygame_available:
        import pygame
        pygame.quit()

    if save_path:
        agent.save(save_path)


def parse_args():
    """Parse CLI arguments supporting both single-dash and double-dash flags."""
    parser = argparse.ArgumentParser(
        description="Learn2Slither - Snake Reinforcement Learning with Q-learning"
    )
    parser.add_argument(
        "-sessions", "--sessions",
        type=int,
        default=1,
        help="Number of training / evaluation sessions"
    )
    parser.add_argument(
        "-save", "--save",
        type=str,
        default=None,
        help="Path to save trained Q-learning model"
    )
    parser.add_argument(
        "-load", "--load",
        type=str,
        default=None,
        help="Path to load pre-trained Q-learning model"
    )
    parser.add_argument(
        "-visual", "--visual",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable ('on') or disable ('off') visual GUI and terminal vision"
    )
    parser.add_argument(
        "-dontlearn", "--dontlearn",
        action="store_true",
        default=False,
        help="Exploitation without learning: Q-values are not updated"
    )
    parser.add_argument(
        "-step-by-step", "--step-by-step",
        dest="step_by_step",
        action="store_true",
        default=False,
        help="Wait for user input before executing each step"
    )
    parser.add_argument(
        "-board-size", "--board-size",
        dest="board_size",
        type=int,
        default=10,
        help="Board size dimension (default: 10)"
    )

    return parser.parse_args()


def main():
    """Main launcher configuring environment, agent, and selected loop."""
    args = parse_args()

    # Initialize Environment
    env = SnakeEnvironment(board_size=args.board_size)

    # Initialize Q-learning Agent
    agent = QTableAgent(dontlearn=args.dontlearn)

    # Load model if requested
    if args.load:
        agent.load(args.load)

    # Dispatch to appropriate execution loop
    if args.visual == "off":
        run_training_loop(
            env=env,
            agent=agent,
            sessions=args.sessions,
            save_path=args.save
        )
    else:
        run_visual_loop(
            env=env,
            agent=agent,
            sessions=args.sessions,
            save_path=args.save,
            step_by_step=args.step_by_step
        )


if __name__ == "__main__":
    main()
