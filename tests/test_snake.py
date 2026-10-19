"""Unit tests for Learn2Slither environment, snake mechanics, and Q-agent."""

import os
import sys
import tempfile
import unittest

# Ensure src/ is importable when running unittest from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ai.q_table import QTableAgent  # noqa: E402
from config.ai import (  # noqa: E402
    REWARD_COLLISION,
    REWARD_GREEN_APPLE,
    REWARD_RED_APPLE,
    REWARD_STEP,
)
from config.game import (  # noqa: E402
    CHAR_EMPTY,
    CHAR_GREEN,
    CHAR_HEAD,
    CHAR_RED,
    CHAR_WALL,
)
from core.environment import Environment  # noqa: E402
from core.snake import Snake  # noqa: E402


class TestSnakeMechanics(unittest.TestCase):
    """Test core Snake initialization, movement, growth, and collisions."""

    def test_contiguous_initialization(self):
        """Snake must initialize with 3 contiguous segments inside bounds."""
        for _ in range(20):
            snake = Snake(board_size=10, initial_length=3)
            self.assertEqual(len(snake), 3)
            self.assertEqual(len(set(snake.body_list)), 3)
            for r, c in snake.body_list:
                self.assertTrue(0 <= r < 10 and 0 <= c < 10)
            for i in range(len(snake) - 1):
                r1, c1 = snake.body_list[i]
                r2, c2 = snake.body_list[i + 1]
                manhattan = abs(r1 - r2) + abs(c1 - c2)
                self.assertEqual(manhattan, 1)

    def test_move_grow_shrink(self):
        """Verify normal move, growth (+1), and shrinking (-1)."""
        snake = Snake(
            board_size=10,
            initial_positions=[(5, 5), (5, 6), (5, 7)],
        )
        # Normal move UP to (4, 5)
        next_head = snake.compute_next_head("UP")
        self.assertEqual(next_head, (4, 5))
        snake.move(next_head)
        self.assertEqual(snake.body_list, [(4, 5), (5, 5), (5, 6)])

        # Grow UP to (3, 5)
        snake.grow((3, 5))
        self.assertEqual(len(snake), 4)
        self.assertEqual(snake.body_list, [(3, 5), (4, 5), (5, 5), (5, 6)])

        # Shrink LEFT to (3, 4) -> advances head and removes 2 tail segments
        snake.shrink((3, 4))
        self.assertEqual(len(snake), 3)
        self.assertEqual(snake.body_list, [(3, 4), (3, 5), (4, 5)])

    def test_wall_and_self_collisions(self):
        """Verify wall boundary checks and self-collision detection."""
        snake = Snake(
            board_size=10,
            initial_positions=[(0, 5), (1, 5), (1, 6), (0, 6)],
        )
        self.assertTrue(snake.is_wall_collision((-1, 5)))
        self.assertTrue(snake.is_wall_collision((10, 5)))
        self.assertTrue(snake.is_wall_collision((0, -1)))
        self.assertTrue(snake.is_wall_collision((0, 10)))
        self.assertFalse(snake.is_wall_collision((0, 4)))

        # Neck (1, 5) is a self-collision; tail (0, 6) vacates on normal move
        self.assertTrue(snake.is_self_collision((1, 5)))
        self.assertFalse(snake.is_self_collision((0, 6)))


class TestEnvironmentRules(unittest.TestCase):
    """Test 42 Subject environment rules, rewards, and vision."""

    def test_reset_spawns_non_overlapping_items(self):
        """Board must have 3-cell snake, 2 green apples, and 1 red apple."""
        env = Environment(board_size=10)
        self.assertEqual(len(env.snake), 3)
        self.assertEqual(len(env.green_apples), 2)
        self.assertIsNotNone(env.red_apple)

        all_positions = (
            set(env.snake) | set(env.green_apples) | {env.red_apple}
        )
        self.assertEqual(len(all_positions), 6)

    def test_eat_green_apple_grows_and_respawns(self):
        """Eating green apple increases length by 1 and gives +10 reward."""
        env = Environment(board_size=10)
        env.snake_model = Snake(
            board_size=10,
            initial_positions=[(5, 5), (5, 6), (5, 7)],
        )
        env.green_apples = [(4, 5), (0, 0)]
        env.red_apple = (9, 9)

        _, reward, done, _ = env.step("UP")
        self.assertFalse(done)
        self.assertEqual(reward, REWARD_GREEN_APPLE)
        self.assertEqual(len(env.snake), 4)
        self.assertEqual(env.max_length, 4)
        self.assertEqual(len(env.green_apples), 2)

    def test_eat_red_apple_shrinks_and_zero_length_ends_game(self):
        """Eating red apple decreases length by 1; length 0 ends game."""
        env = Environment(board_size=10)
        env.snake_model = Snake(
            board_size=10,
            initial_positions=[(5, 5), (5, 6)],
        )
        env.green_apples = [(0, 0), (0, 1)]
        env.red_apple = (4, 5)

        # First red apple drops length from 2 to 1
        _, reward, done, _ = env.step("UP")
        self.assertFalse(done)
        self.assertEqual(reward, REWARD_RED_APPLE)
        self.assertEqual(len(env.snake), 1)

        # Second red apple drops length from 1 to 0 -> Game Over
        env.red_apple = (3, 5)
        _, reward, done, info = env.step("UP")
        self.assertTrue(done)
        self.assertEqual(reward, REWARD_COLLISION)
        self.assertEqual(info.get("reason"), "zero_length")

    def test_wall_and_self_collision_game_over(self):
        """Hitting wall or own body immediately triggers game over (-50)."""
        env = Environment(board_size=10)
        env.snake_model = Snake(
            board_size=10,
            initial_positions=[(0, 5), (1, 5), (2, 5)],
        )
        _, reward, done, info = env.step("UP")
        self.assertTrue(done)
        self.assertEqual(reward, REWARD_COLLISION)
        self.assertEqual(info.get("reason"), "wall")

        env.reset()
        env.snake_model = Snake(
            board_size=10,
            initial_positions=[(5, 5), (5, 6), (6, 6)],
        )
        _, reward, done, info = env.step("RIGHT")
        self.assertTrue(done)
        self.assertEqual(reward, REWARD_COLLISION)
        self.assertEqual(info.get("reason"), "self")

    def test_vision_cross_and_state_tuple(self):
        """Verify 4-directional ray vision and 20-element state extraction."""
        env = Environment(board_size=10)
        env.snake_model = Snake(
            board_size=10,
            initial_positions=[(5, 5), (6, 5), (7, 5)],
        )
        env.green_apples = [(2, 5), (0, 0)]
        env.red_apple = (5, 6)

        cross = env.get_vision_cross_string()
        self.assertIn(CHAR_HEAD, cross)
        self.assertIn(CHAR_WALL, cross)
        self.assertIn(CHAR_GREEN, cross)

        state = env.get_state()
        self.assertEqual(len(state), 20)
        # UP sees Green apple at (2, 5): (0, 'G', 0, 1, 0)
        self.assertEqual(state[0:5], (0, CHAR_GREEN, 0, 1, 0))
        # LEFT has wall at dist 6 (>4 -> bin 2): (0, '0', 2, 0, 0)
        self.assertEqual(state[5:10], (0, CHAR_EMPTY, 2, 0, 0))
        # DOWN has body at dist 1 (bin 0): (1, 'W', 0, 0, 0)
        self.assertEqual(state[10:15], (1, CHAR_WALL, 0, 0, 0))
        # RIGHT has red apple at dist 1 (bin 0): (0, 'R', 0, 0, 1)
        self.assertEqual(state[15:20], (0, CHAR_RED, 0, 0, 1))


class TestQTableAgentAndModels(unittest.TestCase):
    """Test Q-learning Bellman updates, -dontlearn, and model persistence."""

    def test_bellman_update_and_dontlearn(self):
        """Verify Bellman equation update and dontlearn freeze."""
        agent = QTableAgent(alpha=0.5, gamma=0.5, epsilon=0.0, dontlearn=False)
        s1 = (0, "G", 1, 1, 0)
        s2 = (0, "0", 2, 0, 0)

        agent.update(s1, "UP", REWARD_GREEN_APPLE, s2, done=False)
        self.assertAlmostEqual(agent.get_q_values(s1)["UP"], 5.0)

        # Enable dontlearn and verify Q-values are frozen
        agent.dontlearn = True
        agent.update(s1, "UP", REWARD_COLLISION, s2, done=True)
        self.assertAlmostEqual(agent.get_q_values(s1)["UP"], 5.0)

    def test_save_and_load_persistence(self):
        """Verify saving and loading Q-table preserves exact values."""
        agent = QTableAgent()
        state = (1, "W", 1, 0, 0)
        agent.update(state, "UP", REWARD_COLLISION, state, done=True)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            agent.save(tmp_path)
            loaded_agent = QTableAgent(dontlearn=True)
            self.assertTrue(loaded_agent.load(tmp_path))
            self.assertAlmostEqual(
                loaded_agent.get_q_values(state)["UP"],
                agent.get_q_values(state)["UP"],
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_pretrained_models_exist_and_1000sess_reaches_goal(self):
        """Verify 1, 10, 100, 1000 session models exist and 1000sess >= 10."""
        models_dir = os.path.join(PROJECT_ROOT, "models")
        model_files = (
            "1sess.txt",
            "10sess.txt",
            "100sess.txt",
            "1000sess.txt",
        )
        for fname in model_files:
            fpath = os.path.join(models_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing {fpath}")

        agent = QTableAgent(dontlearn=True)
        agent.load(os.path.join(models_dir, "1000sess.txt"))
        env = Environment(board_size=10)

        max_achieved = 0
        for _ in range(10):
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < 2000:
                action = agent.choose_action(state)
                state, _, done, _ = env.step(action)
                steps += 1
            max_achieved = max(max_achieved, env.max_length)

        self.assertGreaterEqual(max_achieved, 10)
        self.assertGreater(REWARD_STEP, REWARD_RED_APPLE)


if __name__ == "__main__":
    unittest.main()
