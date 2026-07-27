#!/usr/bin/env python3
"""
Replay a recorded LeRobot dataset episode directly on the Trossen AI Stationary kit.

No policy server, no EMA smoothing: the recorded actions are sent to the follower
arms exactly as stored. The max_relative_target clamp inside
ManipulatorRobot.send_action() stays active for safety.

Why no EMA: the EMA filter in record_gr00t_trossen.py runs *before* the action is
sent, so recorded actions are already post-EMA. Filtering again on replay would
double-smooth and would not reproduce the original motion.

Motion sequence, matching the other modes:
    connect        -> arms move to home pose (trossen_arm_driver line 133)
    reset pause    -> spoken warning, time to clear the workspace
    approach       -> cosine-eased move from home to the episode's first action
    replay         -> recorded actions at the dataset frame rate
    disconnect     -> arms move to home, then to sleep pose (all zeros)

Diagnostics:
  - dry run prints action ranges, step sizes, the clamp-relevant offset, and a
    joint-limit check, all without connecting the robot
  - clamp hits: how often send_action() had to clip a recorded action. On data
    recorded before the send_action-return fix, a high count means the stored
    actions overstate the motion the robot actually made

Usage:
    # Inspect the episode, robot untouched
    python replay.py --repo_id ANRedlich/my_dataset \
        --dataset_root demo_data/my_dataset --episode 80 --dry_run

    # Approach only, no replay steps
    python replay.py ... --episode 80 --max_steps 0

    # First cautious hardware run
    python replay.py ... --episode 80 --max_steps 10

    # Full episode
    python replay.py ... --episode 80

Abort: escape, left arrow, or right arrow. The e-stop is the real backstop.
"""

import argparse
import logging
import time

import numpy as np
import torch

from robots.configs import TrossenAIStationaryRobotConfig
from robots.utils import (
    make_robot_from_config,
    TROSSEN_AI_STATIONARY_JOINT_MIN,
    TROSSEN_AI_STATIONARY_JOINT_MAX,
)
from utils import init_keyboard_listener, say_tts, busy_wait

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HOME_POSE = [0, 0.261799, 0.261799, 0, 0, 0, 0.044]


def aborted(events):
    """Escape sets stop_recording; left and right arrows set exit_early."""
    return events["exit_early"] or events["stop_recording"]


def load_episode(repo_id, dataset_root, episode):
    """Return (actions, states, fps) for one episode, without decoding any video."""
    dataset = LeRobotDataset(repo_id, root=dataset_root)

    fps = getattr(dataset, "fps", None)
    if fps is None:
        fps = dataset.meta.fps

    num_episodes = getattr(dataset, "num_episodes", None)
    if num_episodes is None:
        num_episodes = dataset.meta.total_episodes
    if not (0 <= episode < num_episodes):
        raise ValueError(f"Episode {episode} out of range (dataset has {num_episodes})")

    from_idx = int(dataset.episode_data_index["from"][episode].item())
    to_idx = int(dataset.episode_data_index["to"][episode].item())

    # select_columns avoids touching the video keys, so nothing gets decoded
    cols = dataset.hf_dataset.select_columns(["action", "observation.state"])
    chunk = cols[from_idx:to_idx]

    actions = torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in chunk["action"]])
    states = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in chunk["observation.state"]])

    logger.info(f"Episode {episode}: frames {from_idx}..{to_idx} "
                f"({len(actions)} steps, {len(actions)/fps:.1f} s at {fps} fps)")
    return actions, states, fps


def describe(actions, states, fps, max_relative_target):
    """Print what the episode contains. Returns False if actions violate joint limits."""
    steps = torch.diff(actions, dim=0).abs()
    per_step_max = steps.max(dim=1).values

    logger.info(f"Action range per joint (min..max):")
    for j in range(actions.shape[1]):
        logger.info(f"  joint {j:2d}: {actions[:, j].min():+.4f} .. {actions[:, j].max():+.4f} rad")

    logger.info(f"Largest single-step action change: {per_step_max.max():.4f} rad "
                f"at step {int(per_step_max.argmax())}")
    logger.info(f"Median single-step action change:  {per_step_max.median():.4f} rad")

    if max_relative_target is not None:
        # NOTE: this is action-to-action change, not the action-vs-measured-position
        # difference the clamp actually tests. A useful proxy, not the real thing.
        over = int((per_step_max > max_relative_target).sum())
        pct = 100.0 * over / max(len(per_step_max), 1)
        logger.info(f"Steps whose action change exceeds max_relative_target="
                    f"{max_relative_target}: {over}/{len(per_step_max)} ({pct:.1f}%)")

    # Clamp-relevant offset: how far each action was from the position measured just
    # before it was sent. state[t-1] is the closest recorded proxy for that read, and
    # it runs high because the arm kept moving during the intervening control period.
    #
    # CAUTION: a large value here does NOT by itself mean the recorded actions were
    # unclamped. Intervention teleop sets max_relative_target=None, so teleop frames
    # legitimately store raw leader positions the follower had not reached yet.
    pre = (actions[1:] - states[:-1]).abs().max(dim=1).values
    logger.info(f"Max |action[t] - state[t-1]|: {pre.max():.4f} rad")

    if max_relative_target is not None:
        over = int((pre > max_relative_target).sum())
        logger.info(f"Steps exceeding the clamp by this measure: "
                    f"{over}/{len(pre)} ({100.0 * over / len(pre):.1f}%)")

        # Long contiguous runs suggest fast task phases or teleop segments.
        # Many short scattered runs suggest the clamp saturating during rollout.
        over_mask = (pre > max_relative_target).numpy()
        runs, start = [], None
        for i, v in enumerate(over_mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(over_mask) - 1))
        if runs:
            lengths = [e - s + 1 for s, e in runs]
            logger.info(f"Exceedances span {len(runs)} contiguous runs "
                        f"(median length {int(np.median(lengths))} steps); 5 longest:")
            for s, e in sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:5]:
                logger.info(f"  steps {s}-{e} ({e - s + 1} long)")

    gap = (actions[:-1] - states[1:]).abs().max()
    logger.info(f"Max |action[t] - state[t+1]| across episode: {gap:.4f} rad "
                f"(how well the follower tracked during recording)")

    # Joint-limit check: catches a corrupt or mis-scaled action array before
    # anything connects. Runs during --dry_run, so it costs nothing.
    lo, hi = TROSSEN_AI_STATIONARY_JOINT_MIN, TROSSEN_AI_STATIONARY_JOINT_MAX
    if actions.shape[1] != len(lo):
        logger.warning(f"Action width {actions.shape[1]} != {len(lo)}; "
                       f"skipping joint-limit check")
        return True

    tol = 0.01  # rad, absorbs encoder noise at the limits (0.0005 is typical)
    bad = [j for j in range(len(lo))
           if bool((actions[:, j] < lo[j] - tol).any() or (actions[:, j] > hi[j] + tol).any())]
    if bad:
        logger.error(f"Actions exceed joint limits on joints {bad} - DO NOT REPLAY")
        for j in bad:
            logger.error(f"  joint {j}: {actions[:, j].min():+.4f}..{actions[:, j].max():+.4f} "
                         f"(limit {lo[j]:+.4f}..{hi[j]:+.4f})")
        return False

    logger.info("All actions within Trossen joint limits")
    return True


def read_present(robot):
    """Cheap joint-position read: follower arms only, no cameras."""
    return np.concatenate([
        robot.follower_arms[name].read("Present_Position")
        for name in robot.follower_arms
    ])


def approach(robot, target, duration, fps, events):
    """Ease the arms from the home pose to the episode's first action."""
    start = read_present(robot)
    target = target.numpy().astype(np.float32)
    delta = np.abs(target - start).max()
    logger.info(f"Approaching first action: max joint move {delta:.4f} rad "
                f"over {duration:.1f} s")

    if delta < 1e-4:
        logger.info("Already at the first action, skipping approach")
        return True

    say_tts("moving to episode start position")
    dt = 1.0 / fps
    n = max(int(duration * fps), 1)
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        # cosine ease so the move starts and stops gently
        s = 0.5 - 0.5 * np.cos(np.pi * i / n)
        goal = start + s * (target - start)
        robot.send_action(torch.from_numpy(goal).float())

        if aborted(events):
            logger.warning("Aborted during approach")
            say_tts("aborted")
            return False
        busy_wait(dt - (time.perf_counter() - t0))

    time.sleep(0.5)
    return True


def replay(robot, actions, states, fps, events, diag_every, max_duration_s):
    """Send each recorded action in turn, at the dataset's frame rate."""
    dt = 1.0 / fps
    n_clamped = 0
    max_clamp = 0.0
    max_track = 0.0
    overruns = 0

    if len(actions) == 0:
        logger.info("No replay steps requested (--max_steps 0), approach only")
        return

    logger.info(f"Replaying {len(actions)} steps at {fps} fps "
                f"(time limit {max_duration_s:.1f} s)")
    say_tts("replay starting")
    time.sleep(1.0)

    t_start = time.perf_counter()

    for i, a in enumerate(actions):
        t0 = time.perf_counter()

        sent = robot.send_action(a)

        # send_action returns the post-clamp goal, so this costs nothing extra
        if sent is not None:
            d = float((sent - a).abs().max())
            if d > 1e-6:
                n_clamped += 1
                max_clamp = max(max_clamp, d)

        if diag_every and i % diag_every == 0:
            present = read_present(robot)
            err = float(np.abs(present - states[i].numpy()).max())
            max_track = max(max_track, err)
            logger.info(f"step {i:5d}/{len(actions)}  tracking err {err:.4f} rad")

        if aborted(events):
            logger.warning(f"Aborted by keypress at step {i}")
            say_tts("aborted")
            break

        elapsed = time.perf_counter() - t_start
        if elapsed > max_duration_s:
            logger.warning(f"Time limit {max_duration_s:.1f} s reached at step {i} "
                           f"of {len(actions)}")
            say_tts("time limit reached")
            break

        remaining = dt - (time.perf_counter() - t0)
        if remaining < 0:
            overruns += 1
        busy_wait(remaining)

    logger.info("--- replay summary ---")
    logger.info(f"ran {time.perf_counter() - t_start:.1f} s")
    logger.info(f"clamp hits:       {n_clamped}/{len(actions)} steps, "
                f"largest clip {max_clamp:.4f} rad")
    if diag_every:
        logger.info(f"max tracking err: {max_track:.4f} rad "
                    f"(sampled every {diag_every} steps)")
    else:
        logger.info("tracking check disabled (--diag_every 0)")
    logger.info(f"loop overruns:    {overruns}/{len(actions)} steps exceeded "
                f"the {dt*1000:.1f} ms budget")


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded episode on the Trossen robot")
    parser.add_argument("--repo_id", required=True, help="Dataset repo id")
    parser.add_argument("--dataset_root", default=None, help="Local dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--max_relative_target", type=float, default=0.1,
                        help="Per-step clamp against measured position. Match the "
                             "value the episode was recorded with.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override replay rate (default: the dataset's own fps)")
    parser.add_argument("--approach_s", type=float, default=5.0,
                        help="Seconds to ease from the home pose to the first recorded action")
    parser.add_argument("--reset_time_s", type=float, default=5.0,
                        help="Spoken pause after connecting, to clear the workspace")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Replay only the first N steps (default: whole episode). "
                             "0 runs the approach and stops. Use a small value for "
                             "a first cautious run.")
    parser.add_argument("--max_duration_s", type=float, default=None,
                        help="Hard stop after this many seconds "
                             "(default: replayed length + 25%% + 5 s)")
    parser.add_argument("--diag_every", type=int, default=30,
                        help="Sample tracking error every N steps (0 disables)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Analyse the episode and exit without connecting the robot")
    parser.add_argument("--force", action="store_true",
                        help="Replay even if the joint-limit check fails")
    args = parser.parse_args()

    actions, states, ds_fps = load_episode(args.repo_id, args.dataset_root, args.episode)
    fps = args.fps if args.fps else ds_fps
    if args.fps and args.fps != ds_fps:
        logger.warning(f"Replaying at {fps} fps but the episode was recorded at {ds_fps} fps")

    limits_ok = describe(actions, states, fps, args.max_relative_target)
    if not limits_ok and not args.force:
        logger.error("Refusing to replay. Pass --force to override.")
        return

    if args.dry_run:
        logger.info("Dry run: robot not connected, nothing moved.")
        return

    # Truncate after describe(), so the diagnostics always cover the full episode
    # and the replay summary counts stay honest for the steps actually run.
    replay_actions, replay_states = actions, states
    if args.max_steps is not None:
        replay_actions = actions[:args.max_steps]
        replay_states = states[:args.max_steps]
        logger.warning(f"TRUNCATED: replaying first {len(replay_actions)} "
                       f"of {len(actions)} steps")

    duration = max(len(replay_actions) / fps, 1.0)
    max_duration_s = args.max_duration_s if args.max_duration_s else duration * 1.25 + 5.0

    robot_config = TrossenAIStationaryRobotConfig(
        args.max_relative_target,
        home_pose=HOME_POSE,
    )
    robot = make_robot_from_config(robot_config)
    robot.leader_arms = {}          # replay drives followers only

    listener, events = init_keyboard_listener()
    if listener is None:
        logger.warning("Headless mode: keyboard abort is NOT available. Use the e-stop.")

    logger.info("Connecting. The arms will move to the home pose.")
    robot.connect(hold=True)

    say_tts("reset environment")
    logger.info(f"Clear the workspace. Approach starts in {args.reset_time_s:.0f} s. "
                f"Escape or an arrow key aborts.")
    time.sleep(args.reset_time_s)

    # Discard any stale key presses from before the run started
    events["exit_early"] = False
    events["stop_recording"] = False

    try:
        if approach(robot, replay_actions[0] if len(replay_actions) else actions[0],
                    args.approach_s, fps, events):
            replay(robot, replay_actions, replay_states, fps, events,
                   args.diag_every, max_duration_s)
            say_tts("replay finished")
            time.sleep(2)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
    finally:
        if listener is not None:
            listener.stop()
        logger.info("Disconnecting: arms will move to home, then to the sleep pose.")
        robot.disconnect()


if __name__ == "__main__":
    main()
