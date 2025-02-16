import sys
sys.path.append("/fs/nexus-projects/PhysicsFall/")
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import lightning as pl
import smplx
import torch
import pandas as pd
import matplotlib.pyplot as plt
from smpl2motorica.utils import conti_angle_rep
from smpl2keypoint import motorica_draw_stickfigure3d, SMPL_visulize_a_frame, motorica_forward_kinematics

# class SMPLDataset(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = Path(data_dir)
#         self.files = sorted(list(self.data_dir.glob("*_smpl.pkl")))

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         smpl_file = self.files[idx]
#         with open(smpl_file, "rb") as f:
#             smpl_data = pickle.load(f)
#         data_dict = {
#             "smpl_body_pose": smpl_data["smpl_body_pose"],
#             "smpl_transl": smpl_data["smpl_transl"],
#             "smpl_global_orient": smpl_data["smpl_global_orient"],
#             "smpl_joint_loc": smpl_data["smpl_joint_loc"],
#         }
#         return data_dict

# class KeypointDataset(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = Path(data_dir)
#         self.files = sorted(list(self.data_dir.glob("*_motorica.pkl")))

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         keypoint_file = self.files[idx]
#         with open(keypoint_file, "rb") as f:
#             keypoint_data = pickle.load(f)
#         data = torch.tensor(keypoint_data.values, dtype=torch.float32) # data shape (num_frames, num_keypoints x 3)
#         col_name = keypoint_data.columns
#         # convert degree to radian
#         data = torch.deg2rad(data)

#         # data shape (num_frames, num_keypoints, 3)
#         reshaped_euler: torch.Tensor = data.reshape(data.shape[0], -1, 3)
#         # convert to matrix representation
#         data_mat = conti_angle_rep.euler_angles_to_matrix(reshaped_euler, convention="XYZ")

#         return data_mat, col_name

class AlignmentDataset(Dataset):
    def __init__(self, data_dir, segment_length=10):
        self.data_dir = Path(data_dir)
        self.smpl_files = sorted(list(self.data_dir.glob("*_smpl.pkl")))
        self.keypoint_files = sorted(list(self.data_dir.glob("*_motorica.pkl")))

    def __len__(self):
        return len(self.smpl_files)

    def __getitem__(self, idx):
        smpl_file = self.smpl_files[idx]
        # find corresponding keypoint file
        file_name = smpl_file.stem
        # replace _smpl.pkl with _motorica.pkl
        keypoint_file = self.data_dir / (file_name.replace("_smpl", "_motorica") + ".pkl")
        if not keypoint_file.exists():
            raise FileNotFoundError(f"Keypoint file {keypoint_file} not found")
        
        with open(smpl_file, "rb") as f:
            smpl_data = pickle.load(f)
        with open(keypoint_file, "rb") as f:
            keypoint_data = pickle.load(f)
        smpl_dict = {
            "smpl_body_pose": smpl_data["smpl_body_pose"],
            "smpl_transl": smpl_data["smpl_transl"],
            "smpl_global_orient": smpl_data["smpl_global_orient"],
            "smpl_joint_loc": smpl_data["smpl_joint_loc"],
        }
        col_name = keypoint_data.columns
        keypoint_data = torch.tensor(keypoint_data.values, dtype=torch.float32) # data shape (num_frames, num_keypoints x 3)
        # convert degree to radian
        keypoint_data = torch.deg2rad(keypoint_data)

        # data shape (num_frames, num_keypoints, 3)
        reshaped_euler: torch.Tensor = keypoint_data.reshape(keypoint_data.shape[0], -1, 3)
        # convert to matrix representation
        keypoint_data_mat = conti_angle_rep.euler_angles_to_matrix(reshaped_euler, convention="XYZ")

        return keypoint_data_mat, col_name, smpl_dict

class AlignmentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super(AlignmentDataModule, self).__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = AlignmentDataset(self.data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    

def main():
    data_dir = Path("/fs/nexus-projects/PhysicsFall/smpl2motorica/data/alignment_dataset")
    smpl_model_path = Path("/fs/nexus-projects/PhysicsFall/data/smpl/models")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    if not smpl_model_path.exists():
        raise FileNotFoundError(f"SMPL model directory {smpl_model_path} not found")
    batch_size = 1
    num_workers = 1

    alignment_data = AlignmentDataset(data_dir)


    keypoint_batch, keypoint_col_name, smpl_batch = alignment_data[0] #keypoint_batch shape (num_frames, num_keypoints, 3, 3)
    # convert from matrix to euler angles
    keypoint_batch = conti_angle_rep.matrix_to_euler_angles(keypoint_batch, convention="XYZ")
    # reshape back to (num_frames, num_keypoints x 3)
    keypoint_batch = keypoint_batch.reshape(keypoint_batch.shape[0], -1)
    # rad to degree
    keypoint_batch = torch.rad2deg(keypoint_batch)

    # processing SMPL data
    pose = smpl_batch["smpl_body_pose"]
    transl = smpl_batch["smpl_transl"]
    global_orient = smpl_batch["smpl_global_orient"]
    smpl_joint_loc = smpl_batch["smpl_joint_loc"]

    smpl_model = smplx.create(
        model_path=smpl_model_path,
        model_type="smpl",
        return_verts=True,
        batch_size=len(pose),
    )
    smpl_output = smpl_model(
        global_orient=torch.tensor(global_orient, dtype=torch.float32),
        body_pose=torch.tensor(pose, dtype=torch.float32),
        transl=torch.tensor(transl, dtype=torch.float32),
    )
    smpl_joints_loc = smpl_output.joints.detach().cpu().numpy().squeeze()
    smpl_vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()
    smpl_joints_loc = smpl_joints_loc[:, :24, :]
    
    # processing motorica data
    keypoint_df = pd.DataFrame(keypoint_batch, columns=keypoint_col_name)
    position_df, motorica_dummy_data = motorica_forward_kinematics(keypoint_df)

    # frame to visualize
    frame = 0
    fig = plt.figure(figsize=(15, 10))
    smpl_ax = fig.add_subplot(121, projection="3d")
    smpl_ax = SMPL_visulize_a_frame(smpl_ax, smpl_joints_loc[frame], smpl_vertices[frame], smpl_model)
    smpl_ax.set_title("SMPL joints")

    ax_motorica = fig.add_subplot(122, projection="3d")
    ax_motorica = motorica_draw_stickfigure3d(
        ax_motorica, motorica_dummy_data, frame, data=position_df
    )
    ax_motorica.set_title("Motorica")


    plt.savefig("dataloader_testing_fig.png")


if __name__ == "__main__":
    main()