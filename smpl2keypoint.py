# This script is used to convert SMPL keypoint to 3D Motorica keypoint format
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import cv2
import smplx
import torch
from tqdm import tqdm


def get_SMPL_skeleton_names():
    return [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ]


def get_motorica_skeleton_names():
    return [
        "Head",
        "Hips",
        "LeftArm",
        "LeftFoot",
        "LeftForeArm",
        "LeftHand",
        "LeftLeg",
        "LeftShoulder",
        # "LeftToeBase",
        "LeftUpLeg",
        "Neck",
        "RightArm",
        "RightFoot",
        "RightForeArm",
        "RightHand",
        "RightLeg",
        "RightShoulder",
        # "RightToeBase",
        "RightUpLeg",
        "Spine",
        "Spine1",
    ]


def smpl2motorica():
    return [
        "head",
        "spine1",
        "left_shoulder",
        "left_ankle",
        "left_elbow",
        "left_wrist",
        "left_knee",
        "left_collar",
        "left_hip",
        "neck",
        "right_shoulder",
        "right_ankle",
        "right_elbow",
        "right_wrist",
        "right_knee",
        "right_collar",
        "right_hip",
        "spine2",
        "spine3",
    ]


def smpl2motorica():
    return [
        "head",
        "spine1",
        "left_shoulder",
        "left_ankle",
        "left_elbow",
        "left_wrist",
        "left_knee",
        "left_collar",
        "left_hip",
        "neck",
        "right_shoulder",
        "right_ankle",
        "right_elbow",
        "right_wrist",
        "right_knee",
        "right_collar",
        "right_hip",
        "spine2",
        "spine3",
    ]


# append x, y, z rotation to each joint
def expand_skeleton(skeleton: list):
    expanded_skeleton = [
        f"{joint}_{axis}rotation" for joint in skeleton for axis in ["X", "Y", "Z"]
    ]
    return expanded_skeleton


def motorica_draw_stickfigure3d(
    fig,
    mocap_track,
    frame,
    data=None,
    joints=None,
    draw_names=True,
):
    from mpl_toolkits.mplot3d import Axes3D

    ax = fig.add_subplot(111, projection="3d")
    # ax.view_init(elev=0, azim=120)

    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints

    if data is None:
        df = mocap_track.values
    else:
        df = data

    for idx, joint in enumerate(joints_to_draw):
        # ^ In mocaps, Y is the up-right axis
        parent_x = df["%s_Xposition" % joint][frame]
        parent_y = df["%s_Zposition" % joint][frame]
        parent_z = df["%s_Yposition" % joint][frame]

        # parent_x = df["%s_Xposition" % joint][frame]
        # parent_y = df["%s_Yposition" % joint][frame]
        # parent_z = df["%s_Zposition" % joint][frame]

        ax.scatter(xs=parent_x, ys=parent_y, zs=parent_z, alpha=0.6, c="b", marker="o")

        children_to_draw = [
            c for c in mocap_track.skeleton[joint]["children"] if c in joints_to_draw
        ]

        for c in children_to_draw:
            # ^ In mocaps, Y is the up-right axis
            child_x = df["%s_Xposition" % c][frame]
            child_y = df["%s_Zposition" % c][frame]
            child_z = df["%s_Yposition" % c][frame]

            ax.plot(
                [parent_x, child_x],
                [parent_y, child_y],
                [parent_z, child_z],
                # "k-",
                lw=2,
                c="black",
            )

        if draw_names:
            ax.text(
                x=parent_x - 0.01,
                y=parent_y - 0.01,
                z=parent_z - 0.01,
                s=f"{idx}:{joint}",
                fontsize=5,
            )

    return ax


if __name__ == "__main__":
    dataset_fps = 60
    target_fps = 6
    aist_data_root = Path("./data/AIST++/")
    smpl_model_path = Path("./smpl/models")

    if not aist_data_root.exists():
        print(f"Please download AIST++ dataset to {aist_data_root}")
        sys.exit(1)
    if not smpl_model_path.exists():
        print(f"Please download SMPL model to {smpl_model_path}")
        sys.exit(1)

    # load all aist data *.pkl files
    aist_data = list(aist_data_root.glob("*.pkl"))
    print(f"Found {len(aist_data)} AIST++ data files")
    for data_file in aist_data:
        print(f"Processing {data_file}")
        data = np.load(data_file, allow_pickle=True)

        # sample to target fps
        sample_indices = np.arange(0, len(data), dataset_fps // target_fps)

        # extract pose, root translation ans scale
        poses = data["smpl_poses"][sample_indices]
        root_trans = data["smpl_trans"][sample_indices]
        scales = data["smpl_scaling"][sample_indices]

        smpl_model = smplx.create(
            model_path=smpl_model_path,
            model_type="smpl",
            return_verts=True,
            batch_size=len(poses),
        )

        # get the joints
        smpl_body_pose = poses[:, 3:]
        smpl_root_rot = poses[:, :3]
        smpl_output = smpl_model(
            body_pose=torch.tensor(smpl_body_pose, dtype=torch.float32),
            transl=torch.tensor(root_trans, dtype=torch.float32),
            # global_orient=torch.tensor(smpl_root_rot, dtype=torch.float32), # TODO: do we need this?
        )

        # get the joints loc
        smpl_joints_loc = smpl_output.joints.detach().cpu().numpy().squeeze()
        smpl_vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()

        # get the SMPL joints (first 23 joints)
        smpl_joints = smpl_joints_loc[:, :23, :]
        print(smpl_joints.shape)

        # Convert SMPL to Motorica Keypoint
        expanded_smpl_joint_names = expand_skeleton(get_SMPL_skeleton_names())
        smpl_joints_df = pd.DataFrame(smpl_body_pose, columns=expanded_smpl_joint_names)
        # get in motorica joint order
        motorica_joint_names = expand_skeleton(smpl2motorica())

        # reorder the columns
        keypoint_smpl_df = smpl_joints_df[motorica_joint_names]
        # convert from radian to degree and keep the same order
        keypoint_smpl_df = keypoint_smpl_df.apply(np.rad2deg)
        # rename the columns to motorica joint names
        keypoint_smpl_df.columns = expand_skeleton(get_motorica_skeleton_names())
        root_pos_df = pd.DataFrame(
            root_trans, columns=["Hips_Xposition", "Hips_Yposition", "Hips_Zposition"]
        )