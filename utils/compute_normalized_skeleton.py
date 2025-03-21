import sys
from pathlib import Path
sys.path.append("../../")
from EDGE.vis import SMPLSkeleton
from smpl2motorica.utils.data import MocapData
from smpl2motorica.utils.bvh import BVHParser
from KeypointFK.keypoint_fk import ForwardKinematics
from smpl2motorica.smpl2keypoint import get_motorica_skeleton_names
from smpl2motorica.utils.keypoint_visualization import visualize_keypoint_data
import torch
from matplotlib import pyplot as plt
import pickle
import numpy as np

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
        raise FileNotFoundError(f"Motion file {motorica_motion_path} does not exist.")
    bvh_parser = BVHParser()
    motorica_dummy_data = bvh_parser.parse(motorica_motion_path)
    skeleton = motorica_dummy_data.skeleton
    selected_joints = get_motorica_skeleton_names()
    # remove joints that are not in the selected joints
    skeleton = {k: v for k, v in skeleton.items() if k in selected_joints}

    # forward kinematics
    fk = ForwardKinematics(skeleton)
    # T-Pose for the skeleton
    pose = torch.zeros(1, 60)
    t_pose_joint_locs = fk.forward(pose)
    t_pose_df = fk.convert_to_dataframe(t_pose_joint_locs)
    

    # visualize the t-pose
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax = visualize_keypoint_data(ax, 0, t_pose_df, skeleton)
    ax.set_title("T-Pose")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    # plt.savefig("t_pose.png")

    # reference skeleton
    # get the head pos
    print(t_pose_df.columns)
    print(f'head pos:')
    print(t_pose_df[["Head_Xposition", "Head_Yposition", "Head_Zposition"]])
    # get the left food
    print(f'left foot pos:')
    print(t_pose_df[["LeftFoot_Xposition", "LeftFoot_Yposition", "LeftFoot_Zposition"]])
    # get the right foot
    print(f'right foot pos:')
    print(t_pose_df[["RightFoot_Xposition", "RightFoot_Yposition", "RightFoot_Zposition"]])

    # height of the skeleton
    # scaling_factor = t_pose_df["Head_Yposition"].iloc[0] - (t_pose_df["LeftFoot_Yposition"].iloc[0] + t_pose_df["RightFoot_Yposition"].iloc[0]) / 2
    scaling_factor = 100
    # normalize the skeleton
    for key, value in skeleton.items():
        value["offsets"] = np.array(value["offsets"]) / scaling_factor
    
    # traverse the skeleton. If joint's children not in the selected joints, remove the children
    for key, value in skeleton.items():
        value["children"] = [c for c in value["children"] if c in selected_joints]
    
    print(f'skeleton after normalization:')
    print(skeleton)
    # save the normalized skeleton
    skeleton_file_name = "normalized_skeleton.pkl"
    output = {
            "skeleton": skeleton,
            "scale": scaling_factor
        }
    with open(skeleton_file_name, "wb") as f:
        pickle.dump(output, f)