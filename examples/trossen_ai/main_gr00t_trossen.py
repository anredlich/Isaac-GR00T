#!/usr/bin/env python3
"""
Trossen Arm <-> GR00T Policy Server Bridge (Bimanual Version)

Bridge between a bimanual trossen-ai stationary kit and the GR00T policy server.
Handles:
1. Collecting observations from the arm (joint positions, images)
2. Sending observations to the policy server via ZMQ (PolicyClient)
3. Receiving action predictions (16-step chunks for GR00T N1.7)
4. Executing actions on the arm

Adapted from the openpi version of main.py for trossen-ai, modified for
GR00T's network protocol and observation/action format.

Usage:
    python main_gr00t_trossen.py --mode autonomous --task_prompt "transfer the cube"

    Test mode (no movement):
    python main_gr00t_trossen.py --mode test --task_prompt "transfer the cube"
"""

import argparse
from collections import defaultdict
import logging
import time
import torch

import cv2
from robots.configs import TrossenAIStationaryRobotConfig
from robots.utils import make_robot_from_config
import numpy as np

# GR00T-specific client (instead of openpi's websocket client)
from gr00t.policy.server_client import PolicyClient

from scipy.interpolate import PchipInterpolator
from utils import init_keyboard_listener, say_tts
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Separate diagnostics logger - silenced by default, enabled via --diagnostics flag
diag = logging.getLogger("diagnostics")
diag.setLevel(logging.WARNING)  # WARNING by default = INFO calls suppressed
diag.propagate = True  # Use root handler for output

class TrossenGR00TBridge:
    """Bridge between a Trossen AI Stationary Kit and GR00T policy server."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 5555,        # GR00T default port (vs openpi 8000)
        control_frequency: int = 30,
        test_mode: str = "autonomous",
        max_steps: int = 1000,
        action_chunk_size: int = 16,           # GR00T uses H=16 (vs pi0's H=50)
        open_loop_horizon: int = 8,            # Execute 8 of 16 actions before replanning
        max_relative_target: float = 0.05,
        adjust_for_sim_to_real: bool = False,
        use_rtc: bool = False,
        rtc_overlap_steps: int = 4,
        rtc_frozen_steps: int = 2,
        rtc_ramp_rate: float = 15.0,
        action_smooth_alpha: float = 1.0,
    ):        
        """
        ...
        action_smooth_alpha: EMA filter coefficient for commanded actions.
            1.0 = disable smoothing (raw policy output, default)
            0.15-0.3 = recommended for GR00T N1.7 on small training sets to reduce jerk
            < 0.1 = very heavy smoothing, may cause sluggishness
            0.0 = NEVER USE (would freeze robot output)
        """

        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

        self.adjust_for_sim_to_real = adjust_for_sim_to_real
        self.display = True

        logger.info(f"Connecting to GR00T policy server at {policy_server_host}:{policy_server_port}")
        # GR00T uses ZMQ-based PolicyClient (vs openpi's WebSocket)
        self.policy_client = PolicyClient(
            host=policy_server_host,
            port=policy_server_port,
        )

        # Get modality config from server to understand what observations to send
        self.modality_config = self.policy_client.get_modality_config()
        logger.info(f"Server modality config:")
        logger.info(f"  Video keys: {self.modality_config['video'].modality_keys}")
        logger.info(f"  State keys: {self.modality_config['state'].modality_keys}")
        logger.info(f"  Language keys: {self.modality_config['language'].modality_keys}")
        logger.info(f"  Action keys: {self.modality_config['action'].modality_keys}")
        logger.info(f"  Action horizon: {len(self.modality_config['action'].delta_indices)}")

        # Initialize trossen robot (same as openpi version)
        robot_config = TrossenAIStationaryRobotConfig(
            max_relative_target,
            home_pose=[0, 0.261799, 0.261799, 0, 0, 0, 0.044]
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.leader_arms = {}
        self.robot.connect()

        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self.action_chunk_size = action_chunk_size
        self.open_loop_horizon = open_loop_horizon  # GR00T-specific: execute fewer than full chunk
        self.episode_step = 0
        # RTC config
        self.use_rtc = use_rtc
        self.rtc_overlap_steps = rtc_overlap_steps
        self.rtc_frozen_steps = rtc_frozen_steps
        self.rtc_ramp_rate = rtc_ramp_rate
        self.prev_normalized_chunk = None        
        self.is_running = False

        self._last_action = None
        self.action_smooth_alpha = action_smooth_alpha

        # Validate alpha range
        if not (0.0 < self.action_smooth_alpha <= 1.0):
            raise ValueError(
                f"action_smooth_alpha must be in (0.0, 1.0]. Got {self.action_smooth_alpha}. "
                f"Use 1.0 to disable smoothing, ~0.2 for moderate smoothing."
            )

        # Action dimension
        self.action_dim = len(self.robot.features['action']['names'])  # 14 for trossen bimanual

        # State key splits (matches your training modality config)
        # 14-dim action/state vector layout:
        #   indices 0-5   : left_arm
        #   index   6     : left_gripper
        #   indices 7-12  : right_arm
        #   index   13    : right_gripper
        self.state_splits = {
            "left_arm": (0, 6),
            "left_gripper": (6, 7),
            "right_arm": (7, 13),
            "right_gripper": (13, 14),
        }

    def execute_action(self, action: np.ndarray):
        """Execute action on the arm."""

        # Low-pass filter on commanded actions
        if self._last_action is None:
            smoothed = action.copy()
        else:
            alpha = self.action_smooth_alpha
            smoothed = alpha * action + (1.0 - alpha) * self._last_action
        self._last_action = smoothed.copy()

        #full_action = action.copy()
        full_action = smoothed.copy()
        full_action = torch.from_numpy(full_action).float()

        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return
        if self.test_mode == "autonomous":
            self.robot.send_action(full_action)
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")

    def build_gr00t_observation(self, joint_positions: np.ndarray,
                                 observation_dict: dict, task_prompt: str) -> dict:
        """
        Build GR00T-format observation dict.

        Different from openpi: GR00T expects nested dict with batch (B=1) and time (T=1) dims,
        and per-key state splits matching the trained model's modality config.
        """
        # Build video dict — GR00T expects (B=1, T=1, H, W, C) HWC format at 480x640
        video_dict = {}
        for cam_key in self.modality_config["video"].modality_keys:
            # Map "cam_high" -> "observation.images.cam_high" to match robot's output keys
            full_cam_key = f"observation.images.{cam_key}"
            if full_cam_key in observation_dict:
                image_hwc = observation_dict[full_cam_key].numpy()
            else:
                # Fallback if robot uses short names
                image_hwc = observation_dict[cam_key].numpy()

            # GR00T expects HWC (H, W, 3) — same as recording format, no transpose needed
            # Add batch and time dims: (1, 1, H, W, C)
            video_dict[cam_key] = image_hwc[None, None, ...]

        # Build state dict — split 14-dim joint vector into 4 keys
        state_dict = {}
        for key, (start, end) in self.state_splits.items():
            # (B=1, T=1, D) shape
            state_dict[key] = joint_positions[start:end][None, None, ...].astype(np.float32)

        # Language input
        language_key = self.modality_config["language"].modality_keys[0]  # 'annotation.human.task_description'
        language_dict = {language_key: [[task_prompt]]}

        return {
            "video": video_dict,
            "state": state_dict,
            "language": language_dict,
        }

    def parse_gr00t_action(self, response: tuple) -> np.ndarray:
        """
        Parse GR00T action response into a (chunk_size, action_dim) numpy array.

        Different from openpi: GR00T returns (action_dict, info) where action_dict
        has per-key arrays that need concatenation back into 14-dim vector.
        """
        action_dict, info = response

        # Each action_dict[key] has shape (B=1, chunk_size, dim_per_key)
        # We need to: unbatch and concatenate keys to get (chunk_size, 14)
        action_keys = self.modality_config["action"].modality_keys
        action_chunks = []
        for key in action_keys:
            chunk = np.atleast_2d(action_dict[key][0])  # (chunk_size, dim_per_key)
            action_chunks.append(chunk)

        # Concatenate along action dim
        full_chunk = np.concatenate(action_chunks, axis=-1)  # (chunk_size, 14)
        return full_chunk

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):
        """Smoothly move arm to start position to avoid jumps. (Same as openpi version)"""
        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith(".pos")]
        current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0, duration])
        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + timepoints[-1]

        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            positions = interpolator_position(current_time)
            self.execute_action(positions)

    def run_episode(self, task_prompt: str = "transfer the cube"):
        """Run a single episode of policy execution."""
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.prev_normalized_chunk = None  # reset RTC state per episode
        self._last_action = None  # reset action filter per episode
        self.is_running = True

        listener, events = init_keyboard_listener()

        camera_features = list(self.robot.camera_features.keys())
        for cam in camera_features:
            cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam, 640, 480)

        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        say_tts('starting episode')
        time.sleep(2)

        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            # Display cameras live (optional)
            if self.display:
                observation_dict = self.robot.capture_observation()
                for cam in camera_features:
                    image_hwc = observation_dict[cam].numpy()
                    cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

            # Request new action chunk after consuming the previous one
            if self.current_action_chunk is None or self.action_chunk_idx >= self.open_loop_horizon:
                # Save previous chunk for RTC freeze diagnostic
                prev_chunk_for_diag = None
                if self.use_rtc and self.current_action_chunk is not None:
                    prev_chunk_for_diag = self.current_action_chunk.copy()

                observation_dict = self.robot.capture_observation()

                # Extract joint positions (14-dim)
                joint_positions = observation_dict['observation.state'].numpy()

                # Build GR00T-format observation
                gr00t_obs = self.build_gr00t_observation(
                    joint_positions, observation_dict, task_prompt
                )

                # Build RTC options if enabled and we have a previous chunk
                rtc_opts = None
                if self.use_rtc and self.prev_normalized_chunk is not None:
                    rtc_opts = {
                        "previous_action_chunk": self.prev_normalized_chunk,
                        "action_horizon": self.action_chunk_size,
                        "rtc_overlap_steps": self.rtc_overlap_steps,
                        "rtc_frozen_steps": self.rtc_frozen_steps,
                        "rtc_ramp_rate": self.rtc_ramp_rate,
                    }
                    diag.info(f"RTC active: prev_chunk shape={self.prev_normalized_chunk.shape}")
                elif self.use_rtc:
                    diag.info("RTC enabled but no previous chunk yet (first inference)")
                else:
                    diag.info("RTC disabled")
                
                # Send to server and get action chunk
                server_start = time.time()
                response = self.policy_client.get_action(gr00t_obs, options=rtc_opts)
                server_time_ms = (time.time() - server_start) * 1000
                logger.info(f"Server inference: {server_time_ms:.1f} ms")

                # Parse response into (chunk_size, 14) array AND save normalized chunk for RTC
                self.current_action_chunk = self.parse_gr00t_action(response)
                if self.use_rtc:
                    _, info = response
                    if isinstance(info, dict) and "normalized_action" in info:
                        self.prev_normalized_chunk = info["normalized_action"]
                # DIAGNOSTIC: RTC freeze check and first-action delta (only when diagnostics enabled)
                if diag.isEnabledFor(logging.INFO):
                    if self.use_rtc and prev_chunk_for_diag is not None:
                        diff_0 = np.abs(self.current_action_chunk[0] - prev_chunk_for_diag[12]).max()
                        diff_1 = np.abs(self.current_action_chunk[1] - prev_chunk_for_diag[13]).max()
                        diff_random = np.abs(self.current_action_chunk[8] - prev_chunk_for_diag[8]).max()
                        diag.info(f"RTC freeze: new[0] vs prev[12] = {diff_0:.4f}, new[1] vs prev[13] = {diff_1:.4f}, new[8] vs prev[8] = {diff_random:.4f}")
                    first_action = self.current_action_chunk[0]
                    initial_delta = np.abs(first_action - joint_positions)
                    diag.info(f"First action delta from current state: max={initial_delta.max():.4f} rad on joint {initial_delta.argmax()}")
                self.action_chunk_idx = 0

            # Select current action from chunk
            a_t = self.current_action_chunk[self.action_chunk_idx]

            # Optional sim-to-real adjustment (same as openpi version)
            if self.adjust_for_sim_to_real:
                a_t = a_t.copy()
                a_t[7] = 1.05 * (a_t[7] + 0.01)
                a_t[8] = a_t[8] - 0.025
                a_t[9] = a_t[9] + 0.025

            # Execute the action
            self.execute_action(a_t)

            self.action_chunk_idx += 1
            self.episode_step += 1

            # Maintain control frequency
            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)

            # Exit conditions
            if events["exit_early"]:
                events["exit_early"] = False
                break

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def autonomous_mode(self, task_prompt: str = "transfer the cube"):
        """Run in autonomous mode where the arm executes policy predictions."""
        logger.info("Starting autonomous mode")
        self.run_episode(task_prompt=task_prompt)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trossen AI Stationary Kit <-> GR00T Policy Server Bridge"
    )
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=5555,
                        help="Policy server port (GR00T default: 5555)")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "test"],
        default="autonomous",
        help="Operation mode: autonomous (execute) or test (no movement)",
    )
    parser.add_argument("--task_prompt", default="transfer the cube",
                        help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--action_chunk_size", type=int, default=16,
                        help="Action chunk size (GR00T default: 16)")
    parser.add_argument("--open_loop_horizon", type=int, default=8,
                        help="Execute N actions before replanning (default 8 of 16)")
    parser.add_argument("--max_relative_target", type=float, default=0.01,
                        help="Max delta action for robot safety")
    parser.add_argument("--adjust_for_sim_to_real", type=bool, default=False,
                        help="True for sim to real adjustment")
    parser.add_argument("--use_rtc", action="store_true",
                        help="Enable Real-Time Chunking for smoother actions")
    parser.add_argument("--rtc_overlap_steps", type=int, default=4,
                        help="Number of RTC overlap steps (default 4)")
    parser.add_argument("--rtc_frozen_steps", type=int, default=2,
                        help="Number of RTC frozen steps (default 2)")
    parser.add_argument("--rtc_ramp_rate", type=float, default=15.0,
                        help="RTC exponential ramp rate (default 15.0)")   
    parser.add_argument("--action_smooth_alpha", type=float, default=1.0,
                    help="EMA filter coefficient. 1.0=disable smoothing (default), "
                         "lower=more smoothing. Typical 0.15-0.3. "
                         "Do NOT use 0.0 (would freeze robot).")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Enable diagnostic logging (RTC freeze checks, action deltas, etc.)")
    args = parser.parse_args()

    # Enable diagnostics logging if requested
    if args.diagnostics:
        diag.setLevel(logging.INFO)
        logger.info("Diagnostics logging enabled")

    bridge = TrossenGR00TBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        action_chunk_size=args.action_chunk_size,
        open_loop_horizon=args.open_loop_horizon,
        max_relative_target=args.max_relative_target,
        adjust_for_sim_to_real=args.adjust_for_sim_to_real,
        use_rtc=args.use_rtc,
        rtc_overlap_steps=args.rtc_overlap_steps,
        rtc_frozen_steps=args.rtc_frozen_steps,
        rtc_ramp_rate=args.rtc_ramp_rate,
        action_smooth_alpha=args.action_smooth_alpha,
    )

    bridge.autonomous_mode(task_prompt=args.task_prompt)
    bridge.cleanup()
