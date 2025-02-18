import torch
import numpy as np
import pandas as pd
from pytorch3d.transforms import euler_angles_to_matrix
from collections import deque
from pathlib import Path
import sys

sys.path.append("../")
from smpl2motorica.utils.bvh import BVHParser
from smpl2motorica.utils.pymo.preprocessing import MocapParameterizer
from smpl2motorica.smpl2keypoint import (
    get_motorica_skeleton_names,
    expand_skeleton,
    skeleton_scaler,
)


class ForwardKinematics:
    def __init__(self):
        self.skeleton = self.get_skeleton()
        self.joints = self.skeleton.keys()
        (
            self.joint_names,
            self.parents,
            self.offsets,
            self.rotation_orders,
            self.has_position,
        ) = self._parse_skeleton(self.skeleton)
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}

    # This is the slowest way to get the skeleton
    def _get_skeleton(self):
        motorica_data_root = Path(
            "/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset"
        )

        motorica_motion_path = (
            motorica_data_root
            / "bvh"
            / "kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.bvh"
        )
        if not motorica_motion_path.exists():
            raise FileNotFoundError(
                f"Motion file {motorica_motion_path} does not exist. "
            )

        # load the motion
        bvh_parser = BVHParser()
        motorica_dummy_data = bvh_parser.parse(motorica_motion_path)
        skeleton = motorica_dummy_data.skeleton
        ratio = 0.01
        my_skeleton = {k: v for k, v in skeleton.items() if k in joints_to_keep}
        # scale the skeleton
        my_skeleton = skeleton_scaler(my_skeleton, ratio)
        return my_skeleton

    def get_skeleton(self):
        return {
            "Hips": {
                "parent": None,
                "channels": [
                    "Xposition",
                    "Yposition",
                    "Zposition",
                    "Zrotation",
                    "Xrotation",
                    "Yrotation",
                ],
                "offsets": np.array([-0.204624, 0.864926, -0.962418]),
                "order": "ZXY",
                "children": ["Spine", "LeftUpLeg", "RightUpLeg"],
            },
            "Spine": {
                "parent": "Hips",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([0.0, 0.0777975, 0.0]),
                "order": "ZXY",
                "children": ["Spine1"],
            },
            "Spine1": {
                "parent": "Spine",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([-1.57670e-07, 2.26566e-01, 7.36298e-07]),
                "order": "ZXY",
                "children": ["Neck", "LeftShoulder", "RightShoulder"],
            },
            "Neck": {
                "parent": "Spine1",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([0.0, 0.249469, 0.0]),
                "order": "ZXY",
                "children": ["Head"],
            },
            "Head": {
                "parent": "Neck",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([0.0, 0.147056, 0.018975]),
                "order": "ZXY",
                "children": ["Head_Nub"],
            },
            "LeftShoulder": {
                "parent": "Spine1",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([0.037925, 0.208193, -0.0005065]),
                "order": "ZXY",
                "children": ["LeftArm"],
            },
            "LeftArm": {
                "parent": "LeftShoulder",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([1.24818e-01, -1.24636e-07, 0.00000e00]),
                "order": "ZXY",
                "children": ["LeftForeArm"],
            },
            "LeftForeArm": {
                "parent": "LeftArm",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([2.87140e-01, 1.34650e-07, 6.52025e-06]),
                "order": "ZXY",
                "children": ["LeftHand"],
            },
            "LeftHand": {
                "parent": "LeftForeArm",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([0.234148, 0.00116565, 0.00321146]),
                "order": "ZXY",
                "children": [
                    "LeftHandThumb1",
                    "LeftHandIndex1",
                    "LeftHandMiddle1",
                    "LeftHandRing1",
                    "LeftHandPinky1",
                ],
            },
            "RightShoulder": {
                "parent": "Spine1",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([-0.0379391, 0.208193, -0.00050652]),
                "order": "ZXY",
                "children": ["RightArm"],
            },
            "RightArm": {
                "parent": "RightShoulder",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([-0.124818, 0.0, 0.0]),
                "order": "ZXY",
                "children": ["RightForeArm"],
            },
            "RightForeArm": {
                "parent": "RightArm",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([-2.87140e-01, -3.94596e-07, 1.22370e-06]),
                "order": "ZXY",
                "children": ["RightHand"],
            },
            "RightHand": {
                "parent": "RightForeArm",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([-0.237607, 0.00081803, 0.00144663]),
                "order": "ZXY",
                "children": [
                    "RightHandThumb1",
                    "RightHandIndex1",
                    "RightHandMiddle1",
                    "RightHandRing1",
                    "RightHandPinky1",
                ],
            },
            "LeftUpLeg": {
                "parent": "Hips",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([0.0948751, 0.0, 0.0]),
                "order": "ZXY",
                "children": ["LeftLeg"],
            },
            "LeftLeg": {
                "parent": "LeftUpLeg",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([2.47622e-07, -3.57160e-01, -1.88071e-06]),
                "order": "ZXY",
                "children": ["LeftFoot"],
            },
            "LeftFoot": {
                "parent": "LeftLeg",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([0.00057702, -0.408583, 0.00046285]),
                "order": "ZXY",
                "children": ["LeftToeBase"],
            },
            "RightUpLeg": {
                "parent": "Hips",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([-0.0948751, 0.0, 0.0]),
                "order": "ZXY",
                "children": ["RightLeg"],
            },
            "RightLeg": {
                "parent": "RightUpLeg",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([-2.56302e-07, -3.57160e-01, -2.17293e-06]),
                "order": "ZXY",
                "children": ["RightFoot"],
            },
            "RightFoot": {
                "parent": "RightLeg",
                "channels": ["Zrotation", "Xrotation", "Yrotation"],
                "offsets": np.array([0.00278006, -0.403849, 0.00049768]),
                "order": "ZXY",
                "children": ["RightToeBase"],
            },
        }

    def _parse_skeleton(self, skeleton):
        # Find root joint
        root = "Hips"

        # Traversal order (BFS)
        joint_names = []
        parents = []
        offsets = []
        rotation_orders = []
        has_position = []
        parent_map = {}

        queue = deque([root])
        while queue:
            joint = queue.popleft()
            if joint not in self.joints:
                continue
            joint_names.append(joint)
            info = skeleton[joint]

            # Parent index
            if info["parent"] is None:
                parents.append(-1)
            else:
                parents.append(parent_map[info["parent"]])

            # Store parent index for children
            parent_map[joint] = len(joint_names) - 1

            # Offset
            offsets.append(torch.tensor(info["offsets"], dtype=torch.float32))

            # Rotation order
            rotation_orders.append(info["order"].upper())

            # Position channels
            has_position.append("Xposition" in info["channels"])

            # Add children to queue
            queue.extend(info["children"])

        return (
            joint_names,
            parents,
            torch.stack(offsets),
            rotation_orders,
            has_position,
        )

    def _df_to_tensors(self, df):
        num_frames = len(df)
        num_joints = len(self.joint_names)

        pos_tensor = torch.zeros((num_frames, num_joints, 3), dtype=torch.float32)
        rot_tensor = torch.zeros((num_frames, num_joints, 3), dtype=torch.float32)

        for j, joint in enumerate(self.joint_names):
            # Handle positions
            if self.has_position[j]:
                pos_cols = [
                    f"{joint}_Xposition",
                    f"{joint}_Yposition",
                    f"{joint}_Zposition",
                ]
                pos_tensor[:, j] = torch.tensor(
                    df[pos_cols].values, dtype=torch.float32
                )

            # Handle rotations
            order = self.rotation_orders[j]
            rot_cols = [f"{joint}_{axis}rotation" for axis in order]
            rotations_deg = torch.tensor(df[rot_cols].values, dtype=torch.float32)
            rot_tensor[:, j] = torch.deg2rad(rotations_deg)  # Convert to radians

        return pos_tensor, rot_tensor

    def forward(self, df):
        pos_values, rot_values = self._df_to_tensors(df)
        device = pos_values.device
        dtype = pos_values.dtype
        num_frames = pos_values.shape[0]

        # Initialize transformations
        global_rot = torch.zeros(
            (num_frames, len(self.joint_names), 3, 3), dtype=dtype, device=device
        )
        global_pos = torch.zeros_like(pos_values)

        for j in range(len(self.joint_names)):
            parent = self.parents[j]

            # Convert to rotation matrices
            local_rot = euler_angles_to_matrix(
                rot_values[:, j], convention=self.rotation_orders[j]
            )

            if parent == -1:  # Root joint
                global_rot[:, j] = local_rot
                global_pos[:, j] = pos_values[:, j]
            else:
                # Combine rotations
                global_rot[:, j] = torch.bmm(global_rot[:, parent], local_rot)

                # Compute position
                offset = self.offsets[j].to(device)
                local_pos = pos_values[:, j] + offset
                rotated_offset = torch.bmm(
                    global_rot[:, parent], local_pos.unsqueeze(-1)
                ).squeeze(-1)
                global_pos[:, j] = global_pos[:, parent] + rotated_offset

        return global_pos

    def convert_to_dataframe(self, positions):
        """Convert output tensor back to DataFrame format"""
        columns = []
        data = {}

        for j, joint in enumerate(self.joint_names):
            pos = positions[:, j].cpu().numpy()
            data[f"{joint}_Xposition"] = pos[:, 0]
            data[f"{joint}_Yposition"] = pos[:, 1]
            data[f"{joint}_Zposition"] = pos[:, 2]

        return pd.DataFrame(data)


if __name__ == "__main__":
    motorica_data_root = Path(
        "/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset"
    )

    motorica_motion_path = (
        motorica_data_root
        / "bvh"
        / "kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.bvh"
    )
    if not motorica_motion_path.exists():
        raise FileNotFoundError(f"Motion file {motorica_motion_path} does not exist. ")

    # load the motion
    bvh_parser = BVHParser()
    motorica_dummy_data = bvh_parser.parse(motorica_motion_path)
    motorica_dummy_df = motorica_dummy_data.values

    # Filter out unnecessary columns
    joints_to_keep = get_motorica_skeleton_names()
    expand_joints_to_keep = expand_skeleton(joints_to_keep)
    # append location to the joint names
    expand_joint_with_location = [
        "Hips_Xposition",
        "Hips_Yposition",
        "Hips_Zposition",
    ] + expand_joints_to_keep

    # make the starting position of the skeleton at the origin
    motorica_dummy_df["Hips_Xposition"] -= motorica_dummy_df["Hips_Xposition"].values[0]
    motorica_dummy_df["Hips_Yposition"] -= motorica_dummy_df["Hips_Yposition"].values[0]
    motorica_dummy_df["Hips_Zposition"] -= motorica_dummy_df["Hips_Zposition"].values[0]

    # filter out unnecessary joints
    motorica_dummy_df = motorica_dummy_df[expand_joint_with_location]
    fk = ForwardKinematics()
    position = fk.forward(motorica_dummy_df)
    position_df = fk.convert_to_dataframe(position)
    print(position_df)

    # # for comparison
    # motorica_dummy_data.values = pd.DataFrame(
    #         (motorica_dummy_df), columns=motorica_dummy_df.columns
    #     )
    # pose_df = MocapParameterizer("position").fit_transform([motorica_dummy_data])[0]
    # pose_df = pose_df.values.reindex(sorted(pose_df.values.columns), axis=1)
    # print(pose_df)
