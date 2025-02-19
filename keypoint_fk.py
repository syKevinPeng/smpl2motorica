from turtle import position
from arrow import get
from sympy import Union
import torch
import numpy as np
import pandas as pd
from pytorch3d.transforms import euler_angles_to_matrix
from collections import deque
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
from typing import Union
from smpl2motorica.utils.bvh import BVHParser
from smpl2motorica.utils.pymo.preprocessing import MocapParameterizer
from smpl2motorica.smpl2keypoint import (
    get_motorica_skeleton_names,
    expand_skeleton,
    skeleton_scaler,
    load_dummy_motorica_data,
    motorica_draw_stickfigure3d,
    motorica2smpl
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
        self.motorica_joint_in_smpl_order = list(motorica2smpl())

    def get_joint_order(self):
        return self.joint_names
    
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
                "children": [],
                # "children": ["Head_Nub"],
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
                "children": [],
                # "children": [
                #     "LeftHandThumb1",
                #     "LeftHandIndex1",
                #     "LeftHandMiddle1",
                #     "LeftHandRing1",
                #     "LeftHandPinky1",
                # ],
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
                "children": [],
                # "children": [
                #     "RightHandThumb1",
                #     "RightHandIndex1",
                #     "RightHandMiddle1",
                #     "RightHandRing1",
                #     "RightHandPinky1",
                # ],
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
                "children": [],
                # "children": ["LeftToeBase"],
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
                "children": [],
                # "children": ["RightToeBase"],
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

    def forward(self, data: Union[pd.DataFrame, torch.Tensor]):
        if isinstance(data, pd.DataFrame):
            pos_values, rot_values = self._df_to_tensors(data) # pos shape (num_frames, num_joints, 3), rot shape (num_frames, num_joints, 3)
        elif isinstance(data, torch.Tensor):
            pos = data[:, :3] # pos shape (num_frames, joint root pos (3 values))
            rot = data[:, 3:]
            assert rot.shape[1] == 57, f"Expected 57 rotation values, got {rot.shape[1]}"
            num_frames = pos.shape[0]
            num_joints = len(self.joint_names)
            pos_values = torch.zeros((num_frames, num_joints, 3), dtype=torch.float32, device=pos.device)
            # position are only for the root joint
            # find the index of the root joint
            root_index = self.motorica_joint_in_smpl_order.index("Hips")
            pos_values[:, root_index,:] = pos
            # process rot
            rot = torch.deg2rad(rot)  # Convert to radians
            rot_values  = rot.reshape(num_frames, -1, 3)
            # reorder the rot_values from motorica order to self.joint_names order
            indices = [self.motorica_joint_in_smpl_order.index(joint) for joint in self.joint_names]
            rot_values = rot_values[:, indices, :]
            # shuffle from xyz to yzx
            rot_values = rot_values[:, :, [1, 2, 0]]
        else:
            raise ValueError("Input data must be a DataFrame or a Tensor")
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
    # forward kinematics
    fk = ForwardKinematics()

    mocap_track = load_dummy_motorica_data()
    mocap_track.skeleton = fk.get_skeleton()
    mocap_df = mocap_track.values
    # force the starting position to be zero all the time
    mocap_df[["Hips_Xposition", "Hips_Yposition", "Hips_Zposition"]] = 0
    mocap_track.values = mocap_df
    position_mocap = MocapParameterizer("position").fit_transform([mocap_track])[0]
    print(position_mocap.values.columns)
    frame = 250
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot(121, projection='3d')
    motorica_ax = motorica_draw_stickfigure3d(
                ax,
                mocap_track=position_mocap,
                frame=frame, draw_names=True
            )
    motorica_ax.set_xlabel('X axis')
    motorica_ax.set_ylabel('Y axis')
    motorica_ax.set_zlabel('Z axis')
    motorica_ax.set_box_aspect([1, 1, 1])
    motorica_ax.set_zlim([-1, 1])
    motorica_ax.set_xlim([-1, 1])
    motorica_ax.set_ylim([-1, 1])
    motorica_ax.set_title('original Figure')

    selected_df = mocap_df[["Hips_Xposition", "Hips_Yposition", "Hips_Zposition"] + expand_skeleton(list(motorica2smpl()))]
    selected_tensor = torch.tensor(selected_df.values, dtype=torch.float32)
    position_df = fk.forward(selected_tensor)
    # position_df = fk.forward(mocap_df)
    position_df = fk.convert_to_dataframe(position_df)
    ax = fig.add_subplot(122, projection='3d')
    keypoint_fk = motorica_draw_stickfigure3d(
                ax,
                mocap_track=mocap_track,
                frame=frame, draw_names=True,
                data= position_df
            )
    keypoint_fk.set_xlabel('X axis')
    keypoint_fk.set_ylabel('Y axis')
    keypoint_fk.set_zlabel('Z axis')
    keypoint_fk.set_box_aspect([1, 1, 1])
    keypoint_fk.set_zlim([-1, 1])
    keypoint_fk.set_xlim([-1, 1])
    keypoint_fk.set_ylim([-1, 1])
    keypoint_fk.set_title('Forward Kinematics')

    
    

    plt.savefig("comparison.png")

        # correct_order = np.array([
    # [ 1.1760e-02,  1.1017e-02,  3.4382e-02], #[0]
    # [ 1.8951e-03, -1.6058e-01,  1.1852e-02], #[1]
    # [-1.8159e-02, -2.2951e-02, -7.1017e-02], #[2]
    # [-2.1993e-02,  2.8106e-02, -3.3211e-02], #[3]
    # [ 5.1560e-03,  1.8165e-01,  1.0596e-02], #[4]
    # [ 1.9840e-05,  1.5830e-01,  1.8508e-04], #[5]
    # [-7.7480e-06,  1.2033e-01, -1.6133e-05], #[6]
    # [ 2.4934e-02, -8.4895e-02, -1.9793e-02], #[7]
    # [ 2.6763e-01, -1.4381e-01,  7.9965e-02], 
    # [-2.0263e-01, -7.9458e-02,  5.1101e-02],
    # [ 3.2295e-02, -1.2899e-01,  8.2248e-02],
    # [ 3.4007e-03, -1.5813e-01, -1.2065e-01],
    # [-2.9071e-02,  3.8841e-02, -2.0848e-02],
    # [-2.8559e-01,  1.2442e-01, -4.0456e-02],
    # [ 2.0005e-01, -9.6870e-03, -1.3214e-01],
    # [ 1.1338e-03,  8.2315e-03, -1.4221e-01],
    # [ 3.6974e-03, -2.4979e-02,  1.4705e-01],
    # [-1.1061e-02, -2.0766e-01,  3.7786e-01],
    # [ 4.0123e-02, -2.4916e-01, -2.8498e-01]])
    # curr_order = np.array(rot_values[0,:])
    # np.set_printoptions(precision=4)
    """
    [[ 3.8841e-02 -2.0848e-02 -2.9071e-02]
    [ 1.1017e-02  3.4382e-02  1.1760e-02]
    [ 1.2442e-01 -4.0456e-02 -2.8559e-01]
    [-1.2899e-01  8.2248e-02  3.2295e-02]
    [ 8.2315e-03 -1.4221e-01  1.1338e-03]
    [-2.0766e-01  3.7786e-01 -1.1061e-02]
    [ 1.5830e-01  1.8508e-04  1.9840e-05]
    [-1.4381e-01  7.9965e-02  2.6763e-01]
    [-2.2951e-02 -7.1017e-02 -1.8159e-02]
    [-8.4895e-02 -1.9793e-02  2.4934e-02]
    [-9.6870e-03 -1.3214e-01  2.0005e-01]
    [-1.5813e-01 -1.2065e-01  3.4007e-03]
    [-2.4979e-02  1.4705e-01  3.6974e-03]
    [-2.4916e-01 -2.8498e-01  4.0123e-02]
    [ 1.2033e-01 -1.6133e-05 -7.7480e-06]
    [-7.9458e-02  5.1101e-02 -2.0263e-01]
    [ 2.8106e-02 -3.3211e-02 -2.1993e-02]
    [-1.6058e-01  1.1852e-02  1.8951e-03]
    [ 1.8165e-01  1.0596e-02  5.1560e-03]]"""