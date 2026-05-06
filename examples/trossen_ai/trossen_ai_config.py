# SPDX-FileCopyrightText: Copyright (c) 2026 ANRedlich. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modality config for the Trossen Stationary AI bimanual robot.
#
# Robot details:
#   - 2 arms x 7 joints each (joint_6 is the gripper on each arm)
#   - 14-dim concatenated state and action vectors
#   - 4 cameras: cam_high, cam_low, cam_left_wrist, cam_right_wrist
#
# Layout of the 14-dim state/action vectors (per meta/info.json column names):
#   indices 0-5   : left_joint_0 .. left_joint_5   (left_arm)
#   index   6     : left_joint_6                   (left_gripper)
#   indices 7-12  : right_joint_0 .. right_joint_5 (right_arm)
#   index   13    : right_joint_6                  (right_gripper)

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


trossen_ai_config = {
    # 4 cameras: high (third-person), low (third-person), left wrist, right wrist.
    # delta_indices=[0] = current frame only (no temporal history).
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
    ),

    # State: split the 14-dim qpos into 4 named blocks for per-block normalization.
    # Splitting separates the gripper (0-0.044 m linear stroke) from the arm joints
    # (radians) so each gets its own normalization range.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "left_gripper", "right_arm", "right_gripper"],
    ),

    # Action: 16-step prediction horizon. Same key structure as state.
    # ABSOLUTE matches the openpi training convention used previously for this robot.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["left_arm", "left_gripper", "right_arm", "right_gripper"],
        action_configs=[
            # left_arm — 6 joint angles
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_gripper — finger displacement (meters)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    # Language: task description from tasks.jsonl, indexed by task_index in parquet.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(trossen_ai_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
