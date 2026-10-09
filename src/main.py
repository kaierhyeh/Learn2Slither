"""
Learn2Slither - AI Reinforcement Learning.

Main entry point handling CLI arguments and orchestrating Environment,
Agent, and UI.
"""

import argparse
import time

from ai import QTableAgent
from core import Environment
from ui import UI


def run_training_loop(
    env: Environment,
    agent: QTableAgent,
    sessions: int,
    save_path=None,
):
    """Training loop w/o UI or per-step terminal IO."""
    overall_max_len = 0
    overall_max_dur = 0

    for _ in range(sessions):
        state = env.reset()
        done = False

        # Agent-Environment Loop
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        overall_max_len = max(overall_max_len, env.max_length)
        overall_max_dur = max(overall_max_dur, env.duration)
        print(
            f"Game over, max length = {env.max_length}, "
            f"max duration = {env.duration}"
        )

    if save_path:
        agent.save(save_path)

    return overall_max_len, overall_max_dur


def run_visual_loop(
    env: Environment,
    agent: QTableAgent,
    sessions: int,
    save_path=None,
    step_by_step: bool = False,
):
    """Visual demonstration loop with UI and per-step terminal vision."""
    ui = UI(board_size=env.board_size)
    fps = 10
    paused = False

    for sess_idx in range(sessions):
        state = env.reset()
        done = False
        manual_action = None

        while not done:
            # 1. Handle UI events if UI is available
            if ui.available:
                should_quit, step_triggered, paused, key_act, fps = (
                    ui.handle_events(step_by_step, paused, fps)
                )
                if should_quit:
                    ui.quit()
                    return
                if key_act:
                    manual_action = key_act

                if step_by_step and not step_triggered:
                    ui.tick(30)
                    continue

                if paused and not step_by_step:
                    ui.tick(15)
                    continue

            # 2. Print cross vision to terminal before choosing action
            print(env.get_vision_cross_string())

            # 3. Decide action (manual input or agent)
            if manual_action is not None:
                action = manual_action
                manual_action = None
            else:
                action = agent.choose_action(state)

            # 4. Print action taken to terminal
            print(f"\n{action}\n")

            # 5. Advance environment
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

            # 6. Render UI frame
            if ui.available:
                ui.render(
                    env=env,
                    sess_idx=sess_idx,
                    sessions=sessions,
                    fps=fps,
                    step_by_step=step_by_step,
                    paused=paused,
                )
            elif not step_by_step:
                time.sleep(0.1)

        print(
            f"Game over, max length = {env.max_length}, "
            f"max duration = {env.duration}."
        )

    ui.quit()

    if save_path:
        agent.save(save_path)


def parse_args():
    """Parse CLI arguments supporting "-" and "--"."""
    parser = argparse.ArgumentParser(
        description="Learn2Slither - AI Reinforcement Learning w/ Q-learning"
    )
    parser.add_argument(
        "-sessions", "--sessions",
        type=int,
        default=1,
        help="Number of training / evaluation sessions",
    )
    parser.add_argument(
        "-save", "--save",
        type=str,
        default=None,
        help="Path to save trained Q-learning model",
    )
    parser.add_argument(
        "-load", "--load",
        type=str,
        default=None,
        help="Path to load pre-trained Q-learning model",
    )
    parser.add_argument(
        "-visual", "--visual",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable ('on') or disable ('off') visual UI and terminal vision",
    )
    parser.add_argument(
        "-dontlearn", "--dontlearn",
        action="store_true",
        default=False,
        help="Exploitation without learning: Q-values are not updated",
    )
    parser.add_argument(
        "-step-by-step", "--step-by-step",
        dest="step_by_step",
        action="store_true",
        default=False,
        help="Wait for user input before executing each step",
    )
    parser.add_argument(
        "-board-size", "--board-size",
        dest="board_size",
        type=int,
        default=10,
        help="Board size dimension (default: 10)",
    )

    return parser.parse_args()


def main():
    """Main launcher configuring environment, agent, and execution loop."""
    args = parse_args()

    env = Environment(board_size=args.board_size)
    agent = QTableAgent(dontlearn=args.dontlearn)

    if args.load:
        agent.load(args.load)

    if args.visual == "off":
        run_training_loop(
            env=env,
            agent=agent,
            sessions=args.sessions,
            save_path=args.save,
        )
    else:
        run_visual_loop(
            env=env,
            agent=agent,
            sessions=args.sessions,
            save_path=args.save,
            step_by_step=args.step_by_step,
        )


if __name__ == "__main__":
    main()
