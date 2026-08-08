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

# Additional imports for recording and intervention (from openpi record.py)
import trossen_arm as trossen
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

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
        record_mode: str = "rollout",
        max_teleop_time_s: float = 0.0,
        tpad_s: float = 2.0,
        clip_teleop: bool = False,        
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

        # Initialize trossen robot
        robot_config = TrossenAIStationaryRobotConfig(
            max_relative_target,
            home_pose=[0, 0.261799, 0.261799, 0, 0, 0, 0.044]
        )
        self.robot = make_robot_from_config(robot_config)

        # Capture dataset features BEFORE clearing leader_arms
        self.dataset_features = self.robot.features.copy()
        for key, ft in self.dataset_features.items():
            if 'images' in key:
                self.dataset_features[key] = {'dtype': 'video', **ft}

        self.record_mode = record_mode
        self.max_teleop_time_s = max_teleop_time_s
        self.tpad_s = tpad_s
        self.clip_teleop = clip_teleop        
        if record_mode == 'rollout':
            self.robot.leader_arms = {}

        self.robot.connect(hold=True)

        # Save CLI clamp value so we can restore after teleop/intervention phases
        self._cli_max_relative_target = max_relative_target

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
        """Execute action on the arm and return the actually-executed action (post-EMA)."""

        # Low-pass filter on commanded actions
        if self._last_action is None:
            smoothed = action.copy()
        else:
            alpha = self.action_smooth_alpha
            smoothed = alpha * action + (1.0 - alpha) * self._last_action
        self._last_action = smoothed.copy()

        full_action = torch.from_numpy(smoothed.copy()).float()

        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return full_action
        if self.test_mode == "autonomous":
            return self.robot.send_action(full_action)
            #return full_action
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")
            return full_action

    def _log_chunk_diagnostics(self, prev_chunk_for_diag, joint_positions):
        """Log RTC freeze/ramp/free-region checks, boundary jerk, and first-action delta.

        Time alignment between old and new chunks:
            new_chunk[i] corresponds in time to prev_chunk[OLH + i]
        because the old chunk was executed for OLH steps before replanning.
        """
        if self.use_rtc and prev_chunk_for_diag is not None:
            olh = self.open_loop_horizon
            n_frozen = self.rtc_frozen_steps
            n_overlap = self.rtc_overlap_steps
            prev_len = len(prev_chunk_for_diag)
            new_len = len(self.current_action_chunk)

            # --- Frozen region: should be nearly identical (RTC server enforces) ---
            frozen_end_prev = min(olh + n_frozen, prev_len)
            frozen_count = max(0, frozen_end_prev - olh)
            if frozen_count > 0:
                frozen_new = self.current_action_chunk[:frozen_count]
                frozen_prev = prev_chunk_for_diag[olh:olh + frozen_count]
                frozen_max = np.abs(frozen_new - frozen_prev).max()
                diag.info(
                    f"RTC frozen[0:{frozen_count}] vs prev[{olh}:{olh + frozen_count}]: "
                    f"max diff = {frozen_max:.4f} (expect ~0)"
                )

            # --- Ramp region: blended, should diverge gradually ---
            ramp_end_prev = min(olh + n_overlap, prev_len)
            ramp_start = n_frozen
            ramp_end_new = min(n_overlap, new_len)
            if ramp_end_new > ramp_start and ramp_end_prev > olh + ramp_start:
                ramp_count = min(ramp_end_new - ramp_start,
                                 ramp_end_prev - (olh + ramp_start))
                ramp_new = self.current_action_chunk[ramp_start:ramp_start + ramp_count]
                ramp_prev = prev_chunk_for_diag[olh + ramp_start:olh + ramp_start + ramp_count]
                ramp_max = np.abs(ramp_new - ramp_prev).max()
                diag.info(
                    f"RTC ramp[{ramp_start}:{ramp_start + ramp_count}] vs "
                    f"prev[{olh + ramp_start}:{olh + ramp_start + ramp_count}]: "
                    f"max diff = {ramp_max:.4f} (expect small, growing)"
                )

            # --- Free region: outside overlap, no RTC constraint ---
            free_idx_new = n_overlap + max(0, (new_len - n_overlap) // 2)
            free_idx_prev = olh + free_idx_new
            if free_idx_new < new_len and free_idx_prev < prev_len:
                free_diff = np.abs(
                    self.current_action_chunk[free_idx_new]
                    - prev_chunk_for_diag[free_idx_prev]
                ).max()
                diag.info(
                    f"RTC free new[{free_idx_new}] vs prev[{free_idx_prev}]: "
                    f"max diff = {free_diff:.4f} (expect larger - policy free here)"
                )

            # --- Boundary continuity: actual physical discontinuity at handoff ---
            if olh - 1 < prev_len:
                boundary = np.abs(
                    self.current_action_chunk[0] - prev_chunk_for_diag[olh - 1]
                )
                diag.info(
                    f"Boundary jerk: |new[0] - prev[{olh - 1}]| "
                    f"max={boundary.max():.4f} rad on joint {boundary.argmax()}"
                )

        first_action = self.current_action_chunk[0]
        initial_delta = np.abs(first_action - joint_positions)
        diag.info(
            f"First action delta from current state: "
            f"max={initial_delta.max():.4f} rad on joint {initial_delta.argmax()}"
        )

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

    def run_episode_rollout(self, task_prompt: str = "transfer the cube",
                            dataset=None, events=None):
        """Run a single episode of policy execution, optionally recording to a dataset."""        
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.prev_normalized_chunk = None  # reset RTC state per episode
        self._last_action = None  # reset action filter per episode
        self.is_running = True

        # events comes from autonomous_mode's shared listener; fallback if not passed
        if events is None:
            events = {"exit_early": False, "rerecord_episode": False,
                      "stop_recording": False,
                      "switch_to_teleop": False, "switch_to_rollout": False}
            
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
                # # DIAGNOSTIC: RTC freeze check and first-action delta (only when diagnostics enabled)
                # if diag.isEnabledFor(logging.INFO):
                #     if self.use_rtc and prev_chunk_for_diag is not None:
                #         diff_0 = np.abs(self.current_action_chunk[0] - prev_chunk_for_diag[12]).max()
                #         diff_1 = np.abs(self.current_action_chunk[1] - prev_chunk_for_diag[13]).max()
                #         diff_random = np.abs(self.current_action_chunk[8] - prev_chunk_for_diag[8]).max()
                #         diag.info(f"RTC freeze: new[0] vs prev[12] = {diff_0:.4f}, new[1] vs prev[13] = {diff_1:.4f}, new[8] vs prev[8] = {diff_random:.4f}")
                #     first_action = self.current_action_chunk[0]
                #     initial_delta = np.abs(first_action - joint_positions)
                #     diag.info(f"First action delta from current state: max={initial_delta.max():.4f} rad on joint {initial_delta.argmax()}")
                # DIAGNOSTIC: RTC freeze/ramp/free-region checks and boundary continuity
                if diag.isEnabledFor(logging.INFO):
                    self._log_chunk_diagnostics(prev_chunk_for_diag, joint_positions)
                self.action_chunk_idx = 0

            # Select current action from chunk
            a_t = self.current_action_chunk[self.action_chunk_idx]

            # Optional sim-to-real adjustment (same as openpi version)
            if self.adjust_for_sim_to_real:
                a_t = a_t.copy()
                a_t[7] = 1.05 * (a_t[7] + 0.01)
                a_t[8] = a_t[8] - 0.025
                a_t[9] = a_t[9] + 0.025

            # Execute the action and get the actually-executed action for recording
            actual_action = self.execute_action(a_t)

            # Record frame if recording
            if dataset is not None:
                obs = self.robot.capture_observation()
                frame = {
                    **obs,
                    "action": actual_action,
                    "task": task_prompt,
                }
                dataset.add_frame(frame)

            self.action_chunk_idx += 1
            self.episode_step += 1

            # Maintain control frequency
            dt_s = time.perf_counter() - start_loop_time
            if self.episode_step % 100 == 0:
                        logger.info(f"loop dt: {dt_s*1000:.1f} ms (budget {self.dt*1000:.1f} ms)")            
            busy_wait_time = self.dt - dt_s
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)

            # Exit conditions
            if events["exit_early"]:
                events["exit_early"] = False
                break

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def run_episode_teleoperate(self, task_prompt: str = "transfer the cube",
                                dataset=None, events=None):
        """Run a single episode of pure teleoperation, optionally recording to a dataset."""
        self.episode_step = 0
        self.is_running = True

        if events is None:
            events = {"exit_early": False, "rerecord_episode": False,
                      "stop_recording": False,
                      "switch_to_teleop": False, "switch_to_rollout": False}

        camera_features = list(self.robot.camera_features.keys())
        for cam in camera_features:
            cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam, 640, 480)

        logger.info(f"Starting teleop episode with prompt: '{task_prompt}'")
        say_tts('starting teleop episode')
        time.sleep(2)

        # Disable clamping for teleop — human motion is inherently smooth
        self._set_max_relative_target(None)
        try:

            while self.is_running and self.episode_step < self.max_steps:
                start_loop_time = time.perf_counter()

                # Display cameras live
                if self.display:
                    observation_dict = self.robot.capture_observation()
                    for cam in camera_features:
                        image_hwc = observation_dict[cam].numpy()
                        cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

                # One teleop step: reads leaders, sends to followers, returns obs + action
                observation, action = self.robot.teleop_step(record_data=True)

                if dataset is not None:
                    frame = {
                        **observation,
                        "action": action["action"],
                        "task": task_prompt,
                    }
                    dataset.add_frame(frame)

                self.episode_step += 1

                # Maintain control frequency
                dt_s = time.perf_counter() - start_loop_time
                if self.episode_step % 100 == 0:
                        logger.info(f"loop dt: {dt_s*1000:.1f} ms (budget {self.dt*1000:.1f} ms)")
                busy_wait_time = self.dt - dt_s
                if busy_wait_time > 0:
                    time.sleep(busy_wait_time)

                if events["exit_early"]:
                    events["exit_early"] = False
                    break
        finally:
            self.restore_max_relative_target()

        self.is_running = False
        logger.info(f"Teleop episode completed after {self.episode_step} steps")

    def run_episode_intervention(self, task_prompt: str = "transfer the cube",
                                 dataset=None, events=None):
        """Run a policy episode with keyboard-triggered switch to teleop intervention."""
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.prev_normalized_chunk = None
        self._last_action = None
        self.is_running = True
        # clip_teleop: track first-teleop [start, end) frame indices (offline trim)
        self._clip_teleop_start = None
        self._clip_teleop_end = None

        if events is None:
            events = {"exit_early": False, "rerecord_episode": False,
                      "stop_recording": False,
                      "switch_to_teleop": False, "switch_to_rollout": False}

        camera_features = list(self.robot.camera_features.keys())
        for cam in camera_features:
            cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam, 640, 480)

        logger.info(f"Starting intervention episode with prompt: '{task_prompt}'")
        say_tts('starting intervention episode')
        time.sleep(2)

        n_interventions = 0
        # Discard any stale arrow presses from before the episode started
        events["switch_to_teleop"] = False
        events["switch_to_rollout"] = False

        try:
            # Intervention cycle: rollout -> align -> teleop -> paused -> rollout ...
            logger.info(f"Rollout entry: max_relative_target={self.robot.config.max_relative_target}, "
                            f"EMA fresh={self._last_action is None}")
            while True:
                # --- Phase 1: Policy rollout until intervention triggered ---
                while self.is_running and self.episode_step < self.max_steps:
                    start_loop_time = time.perf_counter()

                    if self.display:
                        observation_dict = self.robot.capture_observation()
                        for cam in camera_features:
                            image_hwc = observation_dict[cam].numpy()
                            cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                            cv2.waitKey(1)

                    # Request new action chunk when consumed
                    if self.current_action_chunk is None or self.action_chunk_idx >= self.open_loop_horizon:
                        observation_dict = self.robot.capture_observation()
                        joint_positions = observation_dict['observation.state'].numpy()

                        gr00t_obs = self.build_gr00t_observation(
                            joint_positions, observation_dict, task_prompt
                        )

                        rtc_opts = None
                        if self.use_rtc and self.prev_normalized_chunk is not None:
                            rtc_opts = {
                                "previous_action_chunk": self.prev_normalized_chunk,
                                "action_horizon": self.action_chunk_size,
                                "rtc_overlap_steps": self.rtc_overlap_steps,
                                "rtc_frozen_steps": self.rtc_frozen_steps,
                                "rtc_ramp_rate": self.rtc_ramp_rate,
                            }

                        response = self.policy_client.get_action(gr00t_obs, options=rtc_opts)
                        self.current_action_chunk = self.parse_gr00t_action(response)
                        if self.use_rtc:
                            _, info = response
                            if isinstance(info, dict) and "normalized_action" in info:
                                self.prev_normalized_chunk = info["normalized_action"]

                        self.action_chunk_idx = 0
                        #diagnostics (2 lines plus 2 loggers)
                        first_delta = self.current_action_chunk[0] - joint_positions
                        chunk_deltas = np.diff(self.current_action_chunk, axis=0)
                        logger.info(f"New chunk: |first_action - state| max={np.abs(first_delta).max():.4f} rad, "
                                    f"per-joint={np.round(first_delta, 3)}")
                        logger.info(f"New chunk: largest within-chunk step={np.abs(chunk_deltas).max():.4f} rad")

                    a_t = self.current_action_chunk[self.action_chunk_idx]

                    if self.adjust_for_sim_to_real:
                        a_t = a_t.copy()
                        a_t[7] = 1.05 * (a_t[7] + 0.01)
                        a_t[8] = a_t[8] - 0.025
                        a_t[9] = a_t[9] + 0.025

                    actual_action = self.execute_action(a_t)

                    if dataset is not None:
                        obs = self.robot.capture_observation()
                        frame = {
                            **obs,
                            "action": actual_action,
                            "task": task_prompt,
                        }
                        dataset.add_frame(frame)

                    self.action_chunk_idx += 1
                    self.episode_step += 1

                    dt_s = time.perf_counter() - start_loop_time
                    if self.episode_step % 100 == 0:
                        logger.info(f"loop dt: {dt_s*1000:.1f} ms (budget {self.dt*1000:.1f} ms)")
                    busy_wait_time = self.dt - dt_s
                    if busy_wait_time > 0:
                        time.sleep(busy_wait_time)

                    if events["exit_early"]:
                        events["exit_early"] = False
                        break

                    # Protocol: up arrow is inconsistent during rollout -- ignore it
                    if events["switch_to_rollout"]:
                        events["switch_to_rollout"] = False

                    if events["switch_to_teleop"]:
                        break

                # No intervention -- exit_early or max_steps; episode over
                if not events["switch_to_teleop"]:
                    break

                # --- Phase 2: Transition from policy to teleop ---
                events["switch_to_teleop"] = False
                n_interventions += 1
                logger.info("Intervention triggered — freezing follower, aligning leader")
                say_tts("intervention: hold the leader arms")

                # Disable clamping — Phase 2 alignment + Phase 3 teleop
                self._set_max_relative_target(None)

                # Move leader arms to match current follower positions
                for name in self.robot.follower_arms:
                    follower_pos = self.robot.follower_arms[name].read("Present_Position")
                    self.robot.leader_arms[name].driver.set_all_modes(trossen.Mode.position)
                    self.robot.leader_arms[name].driver.set_all_positions(follower_pos, 5.0, False)
                time.sleep(2)

                logger.info("Leaders aligned — grip the leaders, then press down arrow again")
                say_tts("grip leaders and press down arrow")

                # Wait for down arrow; ignore up arrow; honor exit_early
                while not events["switch_to_teleop"] and not events["exit_early"]:
                    if events["switch_to_rollout"]:
                        events["switch_to_rollout"] = False
                    time.sleep(0.1)
                if events["exit_early"]:
                    events["exit_early"] = False
                    break
                events["switch_to_teleop"] = False

                logger.info("Teleop active")
                say_tts("teleop active")
                teleop_start_t = time.perf_counter()
                pause_requested = False

                # Release leader torque LAST — motion becomes possible immediately
                # before the first capture, minimizing the position jump.
                for name in self.robot.leader_arms:
                    self.robot.leader_arms[name].write("Torque_Enable", 0)

                # --- Phase 3: Teleop until episode ends ---
                first_teleop = True
                while self.is_running and self.episode_step < self.max_steps:
                    start_loop_time = time.perf_counter()

                    # Skip the display capture on the FIRST teleop frame so the
                    # first recorded frame lands as soon as possible after torque
                    # release (minimizes the rollout->teleop position jump).
                    if self.display and not first_teleop:
                        observation_dict = self.robot.capture_observation()
                        for cam in camera_features:
                            image_hwc = observation_dict[cam].numpy()
                            cv2.imshow(cam, cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))
                            cv2.waitKey(1)

                    observation, action = self.robot.teleop_step(record_data=True)
                    first_teleop = False

                    if dataset is not None:
                        frame = {
                            **observation,
                            "action": action["action"],
                            "task": task_prompt,
                        }
                        dataset.add_frame(frame)

                    # clip_teleop: mark frame index where first teleop begins
                    if self.clip_teleop and self._clip_teleop_start is None:
                        self._clip_teleop_start = self.episode_step

                    self.episode_step += 1

                    dt_s = time.perf_counter() - start_loop_time
                    if self.episode_step % 100 == 0:
                        logger.info(f"loop dt: {dt_s*1000:.1f} ms (budget {self.dt*1000:.1f} ms)")                    
                    busy_wait_time = self.dt - dt_s
                    if busy_wait_time > 0:
                        time.sleep(busy_wait_time)

                    if events["exit_early"]:
                        events["exit_early"] = False
                        break

                    # Protocol: down arrow is inconsistent during teleop -- ignore it
                    if events["switch_to_teleop"]:
                        events["switch_to_teleop"] = False

                    if events["switch_to_rollout"]:
                        events["switch_to_rollout"] = False
                        logger.info("Up arrow -- pausing teleop")
                        pause_requested = True
                        break

                    if (self.max_teleop_time_s and self.max_teleop_time_s > 0
                            and time.perf_counter() - teleop_start_t >= self.max_teleop_time_s):
                        logger.info(f"Teleop time limit ({self.max_teleop_time_s}s) reached -- pausing")
                        say_tts("teleop time limit reached")
                        pause_requested = True
                        break

                # clip_teleop: first teleop segment just ended; record end index
                if (self.clip_teleop and self._clip_teleop_start is not None
                        and self._clip_teleop_end is None):
                    self._clip_teleop_end = self.episode_step

                if not pause_requested:
                    break   # exit_early or max_steps -- episode over

                # --- Phase 4: Paused -- wait for up arrow to resume rollout ---
                self.hold_leaders()
                logger.info("Paused -- release the leaders, then press up arrow to resume rollout")
                say_tts("paused. release leaders, then press up arrow to resume rollout")
                events["switch_to_rollout"] = False  # require a fresh up press

                while not events["switch_to_rollout"] and not events["exit_early"]:
                    if events["switch_to_teleop"]:
                        events["switch_to_teleop"] = False
                    time.sleep(0.1)
                if events["exit_early"]:
                    events["exit_early"] = False
                    break
                events["switch_to_rollout"] = False

                # Re-arm the policy: restore safety clamp and force a fresh
                # replan from the CURRENT robot state (old chunk / RTC / EMA
                # state is stale after the human moved the arms)
                self.restore_max_relative_target()
                self.current_action_chunk = None
                self.action_chunk_idx = 0
                self.prev_normalized_chunk = None
                self._last_action = None
                logger.info(f"Resume check: max_relative_target={self.robot.config.max_relative_target}, "
                            f"chunk cleared={self.current_action_chunk is None}, "
                            f"EMA reset={self._last_action is None}")

                logger.info("Resuming rollout")
                say_tts("rollout resumed")
                # falls through to the top of `while True` -> Phase 1 rollout

        finally:
            self.restore_max_relative_target()

        self.is_running = False
        logger.info(f"Intervention episode completed after {self.episode_step} steps "
                    f"with {n_interventions} intervention(s)")

    def _clip_begin(self, dataset):
        """clip_teleop: start recording into a throwaway temp dataset. Returns
        the temp dataset (the episode runner writes to it)."""
        import os
        import uuid
        import tempfile
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        self._clip_temp_root = os.path.join(
            tempfile.gettempdir(), f"clip_rec_{uuid.uuid4().hex}")        
        self._clip_temp_ds = LeRobotDataset.create(
            f"clip_temp_{int(time.time())}",
            self.control_frequency,
            root=self._clip_temp_root,
            robot_type=self.robot.robot_type,
            features=self.dataset_features,
            use_videos=True,
            image_writer_processes=1,
            image_writer_threads=4 * len(self.robot.cameras),
        )
        return self._clip_temp_ds

    def _clip_finish(self, dataset, task_prompt):
        """clip_teleop: finalize the temp episode, trim it, and add the trimmed
        frames into the real `dataset` (caller then calls dataset.save_episode()).
        Returns the real dataset. Discards the temp."""
        from dataset_trim_utils import trim_and_add_episode
        self._clip_temp_ds.save_episode()
        try:
            dataset = trim_and_add_episode(
                temp_dataset_root=self._clip_temp_root,
                target_dataset=dataset,
                teleop_start=self._clip_teleop_start,
                teleop_end=self._clip_teleop_end,
                tpad_s=self.tpad_s,
                task_prompt=task_prompt,
            )
        finally:
            self._clip_discard()
        return dataset

    def _clip_discard(self):
        """clip_teleop: stop the temp image writer and delete temp files."""
        import shutil
        ds = getattr(self, "_clip_temp_ds", None)
        root = getattr(self, "_clip_temp_root", None)
        try:
            if ds is not None:
                ds.stop_image_writer()
        except Exception:
            pass
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
        self._clip_temp_ds = None
        self._clip_temp_root = None

    def autonomous_mode(self, task_prompt: str = "transfer the cube",
                       dataset=None, num_episodes: int = 1):
        """Run one or more episodes in the configured record_mode."""
        logger.info(f"Starting autonomous mode ({self.record_mode}), {num_episodes} episode(s)")
        listener, events = init_keyboard_listener()

        reset_time_s = 10
        say_tts("reset environment")
        time.sleep(reset_time_s)

        recorded_episodes = 0
        while recorded_episodes < num_episodes:
            if events["stop_recording"]:
                say_tts("stopped recording")
                break

            ep_index = dataset.num_episodes if dataset else recorded_episodes
            logger.info(f"Recording episode {ep_index} ({recorded_episodes + 1}/{num_episodes})")
            say_tts(f"starting episode {ep_index}")
            time.sleep(2)

            # Dispatch to the right episode runner
            rec_dataset = self._clip_begin(dataset) if self.clip_teleop else dataset
            if self.record_mode == "teleoperate":
                self.release_leaders()
                self.run_episode_teleoperate(task_prompt=task_prompt,
                                             dataset=rec_dataset, events=events)
            elif self.record_mode == "intervention":
                self.run_episode_intervention(task_prompt=task_prompt,
                                              dataset=rec_dataset, events=events)
            else:  # "rollout"
                self.run_episode_rollout(task_prompt=task_prompt,
                                         dataset=rec_dataset, events=events)

            # After each episode, freeze followers and hold leaders back in position
            self.robot.teleop_safety_stop()
            self.hold_leaders()

            # Reset period between episodes (unless we're stopping or re-recording last)
            if not events["stop_recording"] and (
                recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
            ):
                say_tts("reset environment")
                time.sleep(reset_time_s)
                if events["exit_early"]:
                    events["exit_early"] = False

            # Handle re-record
            if events["rerecord_episode"]:
                logger.info("Re-recording episode")
                say_tts("re-record episode")
                time.sleep(3)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                if self.clip_teleop:
                    self._clip_discard()
                if dataset is not None:
                    dataset.clear_episode_buffer()
                continue

            # clip_teleop: no teleop captured -> discard temp, don't count episode
            if self.clip_teleop and self._clip_teleop_start is None:
                logger.info("clip_teleop: no teleop captured — discarding")
                say_tts("no teleop, discarding episode")
                self._clip_discard()
                if events["stop_recording"]:
                    break
                continue

            # Save the just-completed episode
            if dataset is not None:
                logger.info(f"Saving episode {ep_index}")
                say_tts(f"saving episode {ep_index}")
                time.sleep(2)
                if self.clip_teleop:
                    dataset = self._clip_finish(dataset, task_prompt)  # trims+adds into real dataset
                dataset.save_episode()
                logger.info(f"Finished saving episode {ep_index}")
                say_tts(f"Finished saving episode {ep_index}")
                time.sleep(3)

            recorded_episodes += 1

            if events["stop_recording"]:
                say_tts("stopped recording")
                time.sleep(2)
                break

        if listener is not None:
            listener.stop()

    def hold_leaders(self):
        """Set leaders to position hold mode (they resist manual movement)."""
        for name in self.robot.leader_arms:
            self.robot.leader_arms[name].driver.set_all_modes(trossen.Mode.position)

    def _set_max_relative_target(self, value):
        """Set the runtime clamp on per-step joint motion. Pass None to disable."""
        self.robot.config.max_relative_target = value

    def restore_max_relative_target(self):
        """Restore max_relative_target to the CLI-configured value (rollout safety default)."""
        self._set_max_relative_target(self._cli_max_relative_target)

    def release_leaders(self):
        """Disable leader torque so user can move them freely for teleop."""
        for name in self.robot.leader_arms:
            self.robot.leader_arms[name].write("Torque_Enable", 0)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.robot.disconnect()

def parse_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")

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
    # --- Recording arguments ---
    parser.add_argument("--max_teleop_time_s", type=float, default=0.0,
                        help="Intervention mode: auto-pause teleop after this many "
                             "seconds, as if the up arrow were pressed (0 = disabled)")
    parser.add_argument("--record_mode", default="rollout",
                        choices=["rollout", "teleoperate", "intervention"],
                        help="Recording mode: rollout, teleoperate, or intervention (DAgger)")
    parser.add_argument("--tpad", type=float, default=2.0,
                        help="clip_teleop: seconds of rollout kept before the "
                             "first teleop segment (default 2.0)")
    parser.add_argument("--clip_teleop", type=parse_bool, default=False,
                        help="Intervention mode only: trim each episode to the "
                             "first teleop segment + tpad lead-in (offline).")
    parser.add_argument("--repo_id", default=None,
                        help="Dataset repo ID (e.g. ANRedlich/my_new_dataset)")
    parser.add_argument("--dataset_root", default=None,
                        help="Local dataset root path")
    parser.add_argument("--resume", type=parse_bool, default=False,
                        help="Resume appending to an existing dataset")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to record")
    parser.add_argument("--use_videos", type=parse_bool, default=True,
                        help="Save videos (vs image sequences)")
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
        record_mode=args.record_mode,
        max_teleop_time_s=args.max_teleop_time_s,
        tpad_s=args.tpad,
        clip_teleop=args.clip_teleop,
    )

    # Create or resume dataset if repo_id and dataset_root provided
    dataset = None
    if args.repo_id and args.dataset_root:
        if args.resume:
            dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
            dataset.start_image_writer(
                num_processes=1,
                num_threads=4 * len(bridge.robot.cameras),
            )
        else:
            dataset = LeRobotDataset.create(
                args.repo_id,
                args.control_freq,
                root=args.dataset_root,
                robot_type=bridge.robot.robot_type,
                features=bridge.dataset_features,
                use_videos=args.use_videos,
                image_writer_processes=1,
                image_writer_threads=4 * len(bridge.robot.cameras),
            )
    else:
        logger.info("No repo_id/dataset_root — running without recording")

    bridge.autonomous_mode(
        task_prompt=args.task_prompt,
        dataset=dataset,
        num_episodes=args.num_episodes,
    )

    if dataset is not None:
        dataset.stop_image_writer()

    bridge.cleanup()