# preprocessing bvh data into npy
from pathlib import Path
import sys
sys.path.append("/fs/nexus-projects/PhysicsFall/")
from smpl2motorica.utils.bvh import BVHParser
from smpl2motorica.smpl2keypoint import get_motorica_skeleton_names, expand_skeleton
from smpl2motorica.keypoint_fk import ForwardKinematics
import torch
import numpy as np

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

    # reorder the dataframe
    skeleton_names = [
        "Hips_Xposition",
        "Hips_Yposition",
        "Hips_Zposition",
    ] + expand_skeleton(get_motorica_skeleton_names())
    motion_data = motion_df[skeleton_names].to_numpy().reshape(-1, len(get_motorica_skeleton_names())+1, 3)
    motion_positions = forward_kinematics(motion_data, skeleton)
    data = {
        "file_name": file_name,
        "skeleton": skeleton,
        "motion_data": motion_data,
        "motion_data_order": skeleton_names,
        "frame_rate": frame_rate,
        "motion_positions": motion_positions,
        "motion_positions_order": get_motorica_skeleton_names(),
    }
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


    
    
    