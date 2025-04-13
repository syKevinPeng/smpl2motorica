# preprocessing bvh data into npy
from pathlib import Path
import sys

from matplotlib import pyplot as plt
sys.path.append("/fs/nexus-projects/PhysicsFall/")
from smpl2motorica.utils.bvh import BVHParser
from smpl2motorica.smpl2keypoint import get_motorica_skeleton_names, expand_skeleton
from KeypointFK.keypoint_fk import ForwardKinematics
import torch
import numpy as np
from keypoint_visualization import visualize_keypoint_data
import pandas as pd

def parse_bvh_file(bvh_file_path: Path):
    """
    Parse a single BVH file and return the motion data.
    """
    parser = BVHParser()
    motion_data = parser.parse(bvh_file_path)
    skeleton = motion_data.skeleton
    motion_df = motion_data.values
    frame_rate = motion_data.framerate
    file_name = bvh_file_path.stem
    scaling_factor = 100
    # scale the skeleton
    for k, v in skeleton.items():
        v["offsets"] = np.array(v["offsets"]) / 100

    skeleton_names = ["Hips_Xposition","Hips_Yposition", "Hips_Zposition"] + get_motorica_skeleton_names()
    # reorder the dataframe
    pos_df = motion_df[[
        "Hips_Xposition",
        "Hips_Yposition",
        "Hips_Zposition",
    ]]
    # reset starting position to 0,0,0
    pos_df = pos_df - pos_df.iloc[0]
    rot_df = motion_df[expand_skeleton(get_motorica_skeleton_names())]
    rot_df = rot_df.apply(np.deg2rad)
    motion_df = pd.concat([pos_df, rot_df], axis=1)
    motion_data = motion_df.to_numpy().reshape(-1, 20, 3)
    motion_positions = forward_kinematics(motion_data, skeleton)

    data = {
        "file_name": file_name,
        "skeleton": skeleton,
        "scale": scaling_factor,
        "motion_data": motion_data,
        "motion_data_order": skeleton_names,
        "fps": int(1/frame_rate),
        "motion_positions": motion_positions,
        "motion_positions_order": get_motorica_skeleton_names(),
    }

    fk = ForwardKinematics(skeleton, selected_joints=get_motorica_skeleton_names())
    motion_df = fk.convert_to_dataframe(motion_positions)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax = visualize_keypoint_data(ax, 10, motion_df)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.savefig(f"{file_name}.png")
    exit()
    # root = "Hips"
    return data

def forward_kinematics(motion_data, skeleton):
    """
    Apply forward kinematics to the motion data using the provided skeleton.
    """
    seq_leng, num_joints, _ = motion_data.shape
    if type(motion_data) is np.ndarray:
        motion_data = torch.from_numpy(motion_data).float()
    motion_data = motion_data.reshape(seq_leng, num_joints* 3)
    fk = ForwardKinematics(skeleton, selected_joints=get_motorica_skeleton_names())
    motion_positions = fk(motion_data)
    return motion_positions

if __name__ == "__main__":
    motorica_dataset_path = Path("/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset/bvh")
    output_path = Path("/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset/npy")

    if not motorica_dataset_path.exists():
        raise ValueError(f"Motorica dataset path {motorica_dataset_path} does not exist.")
    if not output_path.exists():
        print(f"Output path {output_path} does not exist. Creating it.")
        output_path.mkdir(parents=True, exist_ok=True)
    
    all_bvh_files = list(motorica_dataset_path.glob("*.bvh"))
    print(f'Found {len(all_bvh_files)} bvh files in the dataset.')
    for i, bvh_file in enumerate(all_bvh_files):
        print(f"Processing {bvh_file} ({i+1}/{len(all_bvh_files)})")
        motion_data = parse_bvh_file(bvh_file)
        output_file = output_path / f"{bvh_file.stem}.npy"
        np.save(output_file, motion_data)
        print(f"Saved motion data to {output_file}")


    
    
    