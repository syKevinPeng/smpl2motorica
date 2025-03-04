from matplotlib.pylab import f
import torch
import numpy as np
import pandas as pd
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
from collections import deque
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
from smpl2motorica.utils.keypoint_skeleton import get_keypoint_skeleton
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
        self.skeleton = get_keypoint_skeleton()
        self.joints = self.skeleton.keys()
        (
            self.joint_names,
            self.parents,
            self.offsets,
            self.rotation_orders,
            self.has_position,
        ) = self._parse_skeleton(self.skeleton)

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
        print(f'fk: data shape: {data.shape}')
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
                global_rot = torch.bmm(parent_rot, rot_values[:, j,:,:]) 
                global_rot_list.append(global_rot)

                # compuate positions
                batch_size = pos.shape[0]
                #pos shape:(num_frame, 3), offset shape:(3,)
                local_pose = self.offsets[j].expand(batch_size, -1)
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
        
    
def visualize_keypoint_data(ax, frame: int, df: pd.DataFrame, skeleton = None):
    if skeleton is None:
        skeleton = get_keypoint_skeleton()
    joint_names = get_motorica_skeleton_names()
    for idx, joint in enumerate(joint_names):
        # ^ In mocaps, Y is the up-right axis
        parent_x = df[f"{joint}_Xposition"].iloc[frame]
        parent_y = df[f"{joint}_Zposition"].iloc[frame]
        parent_z = df[f"{joint}_Yposition"].iloc[frame]
        # print(f'joint: {joint}: parent_x: {parent_x}, parent_y: {parent_y}, parent_z: {parent_z}')
        ax.scatter(xs=parent_x, ys=parent_y, zs=parent_z, alpha=0.6, c="b", marker="o")

        children_to_draw = [
            c for c in skeleton[joint]["children"] if c in joint_names
        ]

        for c in children_to_draw:
            # ^ In mocaps, Y is the up-right axis
            child_x = df[f"{c}_Xposition"].iloc[frame]
            child_y = df[f"{c}_Zposition"].iloc[frame]
            child_z = df[f"{c}_Yposition"].iloc[frame]
            
            ax.plot(
                [parent_x, child_x],
                [parent_y, child_y],
                [parent_z, child_z],
                # "k-",
                lw=4,
                c="black",
            )

        ax.text(
            x=parent_x - 0.01,
            y=parent_y - 0.01,
            z=parent_z - 0.01,
            s=f"{idx}:{joint}",
            fontsize=5,
        )
    

    return ax
if __name__ == "__main__":
    from smpl2motorica.dataloader import AlignmentDataset
    data_dir = Path("/fs/nexus-projects/PhysicsFall/smpl2motorica/data/alignment_dataset")
    dataset = AlignmentDataset(data_dir, segment_length=50, force_reprocess=False)
    for (keypoint_data, smpl_dict) in dataset:
        break

    # forward kinematics
    fk = ForwardKinematics()
    # torch.autograd.set_detect_anomaly(True)
    # fk.grad_check()
    reshaped_keypoint_data = keypoint_data.reshape(-1, 60)
    keypoint_order = expand_skeleton(get_motorica_skeleton_names(), "ZXY")
    selected_col = [
        "Hips_Xposition",
        "Hips_Yposition",
        "Hips_Zposition",
    ] + keypoint_order
    keypoint_data_df = pd.DataFrame(reshaped_keypoint_data, columns=selected_col)
    keypoint_data_df[keypoint_order] = keypoint_data_df[keypoint_order].apply(np.rad2deg)
    mocap_track = load_dummy_motorica_data()
    mocap_track.skeleton = get_keypoint_skeleton()
    mocap_track.values = keypoint_data_df
    position_mocap = MocapParameterizer("position").fit_transform([mocap_track])[0]
    frame = 30
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

    # selected_df = mocap_df[expand_skeleton(fk.get_joint_order(), "ZXY")].apply(np.deg2rad)
    # selected_df = pd.concat([mocap_df[["Hips_Xposition", "Hips_Yposition", "Hips_Zposition"]], selected_df], axis=1)
    # motion_data = torch.tensor(selected_df.values, dtype=torch.float32)
    # ZXY to XYZ
    # swaped_keypoint_data = keypoint_data[:,:, [1,2, 0]]
    input_data = torch.tensor(keypoint_data, dtype=torch.float32).reshape(-1, 60)
    position_tensor = fk.forward(input_data)
    position_df = fk.convert_to_dataframe(position_tensor)
    # check if the output is the same
    ax = fig.add_subplot(122, projection='3d')
    keypoint_fk = visualize_keypoint_data(ax, frame, position_df)
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

     