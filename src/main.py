"""
Learn2Slither - AI Reinforcement Learning.

Main entry point handling CLI arguments and orchestrating Environment,
Agent, and UI.
"""

import argparse
import sys
import time

from ai import QTableAgent
from core import Environment
from plot import generate_eval
from ui import UI


def run_training_loop(
    env: Environment,
    agent: QTableAgent,
    sessions: int,
    save_path=None,
    model_path=None,
):
    """Training loop w/o UI or per-step terminal IO."""
    overall_max_len = 0
    overall_max_dur = 0
    session_lengths = []
    session_durations = []
    session_reasons = []

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
        session_lengths.append(env.max_length)
        session_durations.append(env.duration)
        session_reasons.append(env.game_over_reason or "unknown")
        print(
            f"Game over, max length = {env.max_length}, "
            f"max duration = {env.duration}."
        )

    if save_path:
        agent.save(save_path)

    if agent.dontlearn and sessions > 1:
        model_src = model_path or agent.loaded_model_path
        generate_eval(
            lengths=session_lengths,
            durations=session_durations,
            reasons=session_reasons,
            model_path=model_src,
        )

    return overall_max_len, overall_max_dur


def run_visual_loop(
    env: Environment,
    agent: QTableAgent,
    sessions: int,
    save_path=None,
    step_by_step: bool = False,
    model_path=None,
):
    """Visual demonstration loop with UI and per-step terminal vision."""
    ui = UI(board_size=env.board_size)
    fps = 10
    paused = False
    session_lengths = []
    session_durations = []
    session_reasons = []

    for sess_idx in range(sessions):
        state = env.reset()
        done = False
        manual_action = None
        history = []

        if ui.available:
            ui.render(
                env=env,
                sess_idx=sess_idx,
                sessions=sessions,
                fps=fps,
                step_by_step=step_by_step,
                paused=paused,
            )

        while not done:
            # 1. Handle UI events if UI is available
            if ui.available:
                should_quit, step_triggered, paused, key_act, fps = (
                    ui.handle_events(step_by_step, paused, fps)
                )
                if should_quit:
                    ui.quit()
                    return

                if key_act == "UNDO":
                    if history:
                        snap, q_key, q_act, old_q = history.pop()
                        state = env.restore_snapshot(snap)
                        if not agent.dontlearn and q_key in agent.q_table:
                            agent.q_table[q_key][q_act] = old_q
                        if not step_by_step:
                            paused = True
                        print(f"\n[UNDO] Reverted to step {env.duration}.\n")
                        print(env.get_vision())
                    continue

                if key_act:
                    manual_action = key_act

                if step_by_step and not step_triggered:
                    ui.render(
                        env=env,
                        sess_idx=sess_idx,
                        sessions=sessions,
                        fps=fps,
                        step_by_step=step_by_step,
                        paused=paused,
                    )
                    ui.tick(30)
                    continue

                if paused and not step_by_step:
                    ui.render(
                        env=env,
                        sess_idx=sess_idx,
                        sessions=sessions,
                        fps=fps,
                        step_by_step=step_by_step,
                        paused=paused,
                    )
                    ui.tick(15)
                    continue

            # 2. Print cross vision to terminal before choosing action
            print(env.get_vision())

            # 3. Decide action (manual input or agent)
            if manual_action is not None:
                action = manual_action
                manual_action = None
            else:
                action = agent.choose_action(state)

            # 4. Print action taken to terminal
            print(f"\n{action}\n")

            # 5. Save snapshot for undo before advancing environment
            old_q = agent.get_q_values(state)[action]
            history.append((env.save_snapshot(), str(state), action, old_q))

            # 6. Advance environment
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

            # 7. Render UI frame
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

            # In step-by-step mode, allow pressing [R] to undo on Game Over
            if done and step_by_step and ui.available:
                print(
                    f"Game over, max length = {env.max_length}, "
                    f"max duration = {env.duration}. "
                    "(Press [R] to undo or [SPACE] to continue)"
                )
                while done:
                    should_quit, step_triggered, paused, key_act, fps = (
                        ui.handle_events(step_by_step, paused, fps)
                    )
                    if should_quit:
                        ui.quit()
                        return
                    if key_act == "UNDO" and history:
                        snap, q_key, q_act, old_q = history.pop()
                        state = env.restore_snapshot(snap)
                        if not agent.dontlearn and q_key in agent.q_table:
                            agent.q_table[q_key][q_act] = old_q
                        done = False
                        print(f"\n[UNDO] Reverted to step {env.duration}\n")
                        print(env.get_vision())
                        break
                    if step_triggered:
                        break
                    ui.render(
                        env=env,
                        sess_idx=sess_idx,
                        sessions=sessions,
                        fps=fps,
                        step_by_step=step_by_step,
                        paused=paused,
                    )
                    ui.tick(30)
                if not done:
                    continue

        session_lengths.append(env.max_length)
        session_durations.append(env.duration)
        session_reasons.append(env.game_over_reason or "unknown")
        print(
            f"Game over, max length = {env.max_length}, "
            f"max duration = {env.duration}."
        )

    ui.quit()

    if save_path:
        agent.save(save_path)

    if agent.dontlearn and len(session_lengths) > 1:
        model_src = model_path or agent.loaded_model_path
        generate_eval(
            lengths=session_lengths,
            durations=session_durations,
            reasons=session_reasons,
            model_path=model_src,
        )


def parse_args():
    """Parse CLI arguments supporting "-" and "--"."""
    parser = argparse.ArgumentParser(
        description="Learn2Slither - AI Reinforcement Learning w/ Q-learning"
    )
    parser.add_argument(
        "-n", "--sessions",
        type=int,
        default=1,
        help="Number of training/evaluation sessions",
    )
    parser.add_argument(
        "-s", "--save",
        type=str,
        default=None,
        help="Path to save trained Q-learning model",
    )
    parser.add_argument(
        "-l", "--load",
        type=str,
        default=None,
        help="Path to load pre-trained Q-learning model",
    )
    parser.add_argument(
        "-v", "--visual",
        action="store_true",
        default=False,
        help="Enable visual UI and terminal vision (default: off)",
    )
    parser.add_argument(
        "-x", "--dontlearn",
        action="store_true",
        default=False,
        help="Exploitation without learning: Q-values are not updated",
    )
    parser.add_argument(
        "-step", "--step-by-step",
        dest="step_by_step",
        action="store_true",
        default=False,
        help="Wait for user input before executing each step",
    )
    parser.add_argument(
        "-b", "--board-size",
        dest="board_size",
        type=int,
        default=10,
        help="Board size dimension (default: 10)",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def main():
    """Main launcher configuring environment, agent, and execution loop."""
    args = parse_args()

    env = Environment(board_size=args.board_size)
    agent = QTableAgent(dontlearn=args.dontlearn)

    if args.load:        # -load <path to model>
        agent.load(args.load)

    if not args.visual:  # -visual
        run_training_loop(
            env=env,
            agent=agent,
            sessions=args.sessions,
            save_path=args.save,
            model_path=args.load,
        )
    else:
        run_visual_loop(
            env=env,
            agent=agent,
            sessions=args.sessions,
            save_path=args.save,
            step_by_step=args.step_by_step,
            model_path=args.load,
        )


if __name__ == "__main__":
    main()
