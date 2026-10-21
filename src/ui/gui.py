"""Pygame Graphical User Interface"""

from typing import Optional, Tuple

from config.ui import (
    CELL_SIZE,
    COLOR_BG,
    COLOR_GREEN_APPLE,
    COLOR_GRID,
    COLOR_RED_APPLE,
    COLOR_SNAKE_BODY,
    COLOR_SNAKE_HEAD,
    COLOR_TEXT,
    COLOR_VISION,
    HEADER_HEIGHT,
    KEY_TO_DIR,
)


class UI:
    """User interface managing Pygame rendering, HUD, and keyboard inputs."""

    def __init__(
        self,
        board_size: int = 10,
        cell_size: int = CELL_SIZE,
        header_height: int = HEADER_HEIGHT,
    ):
        self.board_size = board_size
        self.cell_size = cell_size
        self.header_height = header_height
        self.board_px = board_size * cell_size
        self.win_w = self.board_px
        self.win_h = self.board_px + header_height

        self.available = False
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        self._init_pygame()

    def _init_pygame(self) -> None:
        """Attempt to initialize pygame display and font subsystem."""
        try:
            import pygame

            pygame.init()
            pygame.display.set_caption("Learn2Slither")
            self.screen = pygame.display.set_mode(
                (self.win_w, self.win_h),
                pygame.RESIZABLE,
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 18, bold=True)
            self.small_font = pygame.font.SysFont("monospace", 13)
            self.available = True
        except Exception as exc:
            print(f"Note: Running in terminal visualization mode ({exc})")
            self.available = False

    def handle_events(
        self,
        step_by_step: bool,
        paused: bool,
        fps: int,
    ) -> Tuple[bool, bool, bool, Optional[str], int]:
        """Poll pygame events and return state changes.

        Returns:
            (should_quit, step_triggered, paused, manual_action, fps)
        """
        if not self.available:
            return False, True, paused, None, fps

        import pygame

        should_quit = False
        step_triggered = not step_by_step
        manual_action = None

        active_key_to_dir = KEY_TO_DIR if KEY_TO_DIR else {
            pygame.K_UP: "UP",
            pygame.K_w: "UP",
            pygame.K_LEFT: "LEFT",
            pygame.K_a: "LEFT",
            pygame.K_DOWN: "DOWN",
            pygame.K_s: "DOWN",
            pygame.K_RIGHT: "RIGHT",
            pygame.K_d: "RIGHT",
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True, False, paused, None, fps

            if event.type == pygame.VIDEORESIZE:
                self.win_w = max(200, event.w)
                self.win_h = max(200 + self.header_height, event.h)
                self.screen = pygame.display.set_mode(
                    (self.win_w, self.win_h),
                    pygame.RESIZABLE,
                )

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return True, False, paused, None, fps

                if event.key == pygame.K_SPACE:
                    if step_by_step:
                        step_triggered = True
                    else:
                        paused = not paused
                elif event.key == pygame.K_r:
                    manual_action = "UNDO"
                elif event.key in active_key_to_dir:
                    manual_action = active_key_to_dir[event.key]
                    if step_by_step:
                        step_triggered = True
                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                    pygame.K_KP_PLUS,
                    pygame.K_RIGHTBRACKET,
                ):
                    fps = min(60, fps + 5)
                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                    pygame.K_LEFTBRACKET,
                ):
                    fps = max(1, fps - 5)

        return should_quit, step_triggered, paused, manual_action, fps

    def render(
        self,
        env,
        sess_idx: int,
        sessions: int,
        fps: int,
        step_by_step: bool,
        paused: bool,
    ) -> None:
        """Render board, apples, snake, and HUD to screen."""
        if not self.available or self.screen is None:
            return

        import pygame

        self.screen.fill(COLOR_BG)

        win_w, win_h = self.screen.get_size()
        avail_w = max(self.board_size * 8, win_w)
        avail_h = max(self.board_size * 8, win_h - self.header_height)
        cell_size = max(
            8,
            min(avail_w // self.board_size, avail_h // self.board_size),
        )
        board_px = cell_size * self.board_size
        ox = max(0, (win_w - board_px) // 2)
        oy = self.header_height + max(
            0, (win_h - self.header_height - board_px) // 2
        )

        # 1. Header info
        mode_str = "STEP" if step_by_step else ("PAUSE" if paused else "AUTO")
        info_text = (
            f"Sess: {sess_idx + 1}/{sessions}  "
            f"Len: {len(env.snake)} (Max: {env.max_length})  "
            f"Dur: {env.duration}  {fps}FPS [{mode_str}]"
        )
        txt_surf = self.font.render(info_text, True, COLOR_TEXT)
        self.screen.blit(txt_surf, (12, 10))

        hint_text = (
            "[SPACE]: Step/Pause | [R]: Undo | [ARROWS/WASD]: Move | [+/-]"
        )
        hint_surf = self.small_font.render(hint_text, True, (160, 160, 170))
        self.screen.blit(hint_surf, (12, 34))

        # 2. Highlight 4-directional vision rays from snake head in gray
        if env.snake:
            hr, hc = env.snake[0]
            for c in range(self.board_size):
                rect = pygame.Rect(
                    ox + c * cell_size,
                    oy + hr * cell_size,
                    cell_size,
                    cell_size,
                )
                pygame.draw.rect(self.screen, COLOR_VISION, rect)
            for r in range(self.board_size):
                rect = pygame.Rect(
                    ox + hc * cell_size,
                    oy + r * cell_size,
                    cell_size,
                    cell_size,
                )
                pygame.draw.rect(self.screen, COLOR_VISION, rect)

        # 3. Grid lines
        for r in range(self.board_size + 1):
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (ox, oy + r * cell_size),
                (ox + board_px, oy + r * cell_size),
                1,
            )
        for c in range(self.board_size + 1):
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (ox + c * cell_size, oy),
                (ox + c * cell_size, oy + board_px),
                1,
            )

        apple_pad = max(2, cell_size // 12)
        snake_pad = max(1, cell_size // 16)
        apple_radius = max(2, cell_size // 8)
        snake_radius = max(2, cell_size // 10)

        # 3. Green apples
        for gr, gc in env.green_apples:
            rect = pygame.Rect(
                ox + gc * cell_size + apple_pad,
                oy + gr * cell_size + apple_pad,
                cell_size - apple_pad * 2,
                cell_size - apple_pad * 2,
            )
            pygame.draw.rect(
                self.screen,
                COLOR_GREEN_APPLE,
                rect,
                border_radius=apple_radius,
            )

        # 4. Red apple
        if env.red_apple:
            rr, rc = env.red_apple
            rect = pygame.Rect(
                ox + rc * cell_size + apple_pad,
                oy + rr * cell_size + apple_pad,
                cell_size - apple_pad * 2,
                cell_size - apple_pad * 2,
            )
            pygame.draw.rect(
                self.screen,
                COLOR_RED_APPLE,
                rect,
                border_radius=apple_radius,
            )

        # 5. Snake body & head
        for idx, (sr, sc) in enumerate(env.snake):
            rect = pygame.Rect(
                ox + sc * cell_size + snake_pad,
                oy + sr * cell_size + snake_pad,
                cell_size - snake_pad * 2,
                cell_size - snake_pad * 2,
            )
            color = COLOR_SNAKE_HEAD if idx == 0 else COLOR_SNAKE_BODY
            pygame.draw.rect(
                self.screen,
                color,
                rect,
                border_radius=snake_radius,
            )

        pygame.display.flip()

        if not step_by_step and not paused and self.clock:
            self.clock.tick(fps)

    def tick(self, fps: int) -> None:
        """Advance pygame clock."""
        if self.available and self.clock:
            self.clock.tick(fps)

    def quit(self) -> None:
        """Safely terminate pygame instance."""
        if self.available:
            import pygame

            pygame.quit()
            self.available = False
