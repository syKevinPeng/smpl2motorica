import torch
import numpy as np
import pandas as pd
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
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
            rot_tensor[:, j] = rotations_deg

        return pos_tensor, rot_tensor

    def forward_df(self, df):
        pos_values, rot_values = self._df_to_tensors(df)
        device = pos_values.device
        dtype = pos_values.dtype
        num_frames = pos_values.shape[0]

        # Initialize transformations
        global_rot = torch.zeros(
            (num_frames, len(self.joint_names), 3, 3), dtype=dtype, device=device
        )
        global_pos = torch.zeros_like(pos_values)
        for j, joint in enumerate(self.joint_names):
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

    # A differentiable forward kinematics function
    def forward(self, data: torch.Tensor):
        # input shape: (batch_size x num_frames, num_joints x 3)
        if data.dim() !=2 or data.shape[1] != 60:
            raise ValueError("Expect input data to have shape (batch_size x num_frames, num_joints x 3)")
        pos = data[:, :3] # pos shape (num_frames, joint root pos (3 values))
        rot = data[:, 3:]
        num_frames = pos.shape[0]
        num_joints = len(self.joint_names)
        device = pos.device
        dtype = pos.dtype
        # rot = torch.deg2rad(rot)
        rot_values = rot.reshape(num_frames, num_joints, 3)
        # convert rot to rotation matrix
        rot_values = euler_angles_to_matrix(rot_values, self.rotation_orders[0])
        
        global_pos_list = []
        global_rot_list = []
        self.offsets = self.offsets.to(device=device, dtype=dtype)
        for j, joint in enumerate(self.joint_names): # we have parsed joint name in hierarchy order
            if joint == "Hips":
                assert j == 0, f"Expected root joint to be at index 0, got {j}"
                # processing root joint
                global_pos_list.append(pos)
                global_rot_list.append(rot_values[:, j,:])
            else:
                # processing other joints
                parent = self.parents[j]
                # compute rotations
                parent_rot = global_rot_list[parent]
                global_rot = torch.bmm(parent_rot, rot_values[:, j,:,:]) # HERE!
                global_rot_list.append(global_rot)

                # compuate positions
                batch_size = pos.shape[0]
                # pos shape:(num_frame, 3), offset shape:(3,)
                local_pose = pos + self.offsets[j].expand(batch_size, -1)
                #parent_rot: torch.Size([num_frame, 3, 3]); local_pose: torch.Size([num_frame, 3])
                rotated_offset = torch.bmm(global_rot_list[parent],local_pose.unsqueeze(-1)).squeeze(-1)
                global_pos = global_pos_list[parent] + rotated_offset
                global_pos_list.append(global_pos)

        global_pos = torch.stack(global_pos_list, dim=1) # joint order in self.joint_names
        return global_pos
        
    
    def convert_to_dataframe(self, positions):
        """Convert output tensor back to DataFrame format"""
        """position shape: (num_frames, num_joints (19), 3)"""
        columns = []
        data = {}

        positions = positions.detach().cpu().numpy()
        # check position shape
        assert positions.shape[1] == len(self.joint_names), f"Expected position shape to be (num_frames, num_joints (19)), got {positions.shape}"

        for j, joint in enumerate(self.joint_names):
            pos = positions[:, j]
            data[f"{joint}_Xposition"] = pos[:, 0]
            data[f"{joint}_Yposition"] = pos[:, 1]
            data[f"{joint}_Zposition"] = pos[:, 2]
        return pd.DataFrame(data)
    
    def grad_check(self,):
        from torch.autograd import gradcheck
        dummy_input = torch.randn(5, 60, requires_grad=True)

        def forward_wrapper(data):
            return self.forward(data).sum()
        test = gradcheck(forward_wrapper, (dummy_input,), eps=1e-6, atol=1e-4)
        
    

if __name__ == "__main__":


    # forward kinematics
    fk = ForwardKinematics()
    torch.autograd.set_detect_anomaly(True)
    # fk.grad_check()

    mocap_track = load_dummy_motorica_data()
    mocap_track.skeleton = fk.get_skeleton()
    mocap_df = mocap_track.values

    # force the starting position to be zero all the time
    mocap_df[["Hips_Xposition", "Hips_Yposition", "Hips_Zposition"]] = 0
    mocap_track.values = mocap_df
    position_mocap = MocapParameterizer("position").fit_transform([mocap_track])[0]
    frame = 150
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot(121, projection='3d')
    motorica_ax = motorica_draw_stickfigure3d(
                ax,
                mocap_track=position_mocap,
                frame=frame, draw_names=False
            )
    motorica_ax.set_xlabel('X axis')
    motorica_ax.set_ylabel('Y axis')
    motorica_ax.set_zlabel('Z axis')
    motorica_ax.set_box_aspect([1, 1, 1])
    motorica_ax.set_zlim([-1, 1])
    motorica_ax.set_xlim([-1, 1])
    motorica_ax.set_ylim([-1, 1])
    motorica_ax.set_title('original Figure')
    input_data_order = ["Hips_Xposition", "Hips_Yposition", "Hips_Zposition"] + expand_skeleton(list(motorica2smpl()))
    selected_df = mocap_df[input_data_order]
    # hiarchy order:
    hiarchy_order =["Hips_Xposition", "Hips_Yposition", "Hips_Zposition"] +  expand_skeleton(fk.get_joint_order(), "ZXY")
    # reorder from input_data_order to hiarchy_order
    selected_df = selected_df[hiarchy_order]

    selected_tensor = torch.tensor(selected_df.values, dtype=torch.float32, requires_grad=True)
    position_tensor, tensor_name = fk.forward(selected_tensor)
    position_df = fk.convert_to_dataframe(position_tensor)
    # check if the output is the same
    ax = fig.add_subplot(122, projection='3d')
    keypoint_fk = motorica_draw_stickfigure3d(
                ax,
                mocap_track=mocap_track,
                frame=frame, draw_names=False,
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

    motorica_ax.view_init(azim=-90, elev=10)
    keypoint_fk.view_init(azim=-90, elev=10)
    

    plt.savefig("comparison.png")

     