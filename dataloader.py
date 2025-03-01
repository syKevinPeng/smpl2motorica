import sys

from lightning.pytorch.utilities.types import EVAL_DATALOADERS

from utils.pymo.preprocessing import MocapParameterizer

sys.path.append("/fs/nexus-projects/PhysicsFall/")
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import lightning as pl
import smplx
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smpl2motorica.utils.pymo.preprocessing import (
    MocapParameterizer
)
from tqdm import tqdm
from smpl2keypoint import (
    motorica_draw_stickfigure3d,
    SMPL_visulize_a_frame,
    motorica_forward_kinematics,
    get_SMPL_skeleton_names,
    expand_skeleton,
    motorica2smpl,
    load_dummy_motorica_data,
    get_motorica_skeleton_names,
    motorica_to_smpl_mapping
)
from keypoint_fk import ForwardKinematics


class AlignmentDataset(Dataset):
    def __init__(self, data_dir, segment_length=50, force_reprocess=False, mode = "train"):
        self.mode = mode
        if mode not in ["train", "validate", "predict"]:
            raise ValueError("mode should be one of ['train', 'validate', 'predict']")
        self.data_dir = Path(data_dir)
        self.smpl_files = sorted(list(self.data_dir.glob("*_smpl.pkl")))
        self.keypoint_files = sorted(list(self.data_dir.glob("*_motorica.pkl")))
        self.segment_length = segment_length
        self.processed_file_save_path = (
            self.data_dir / f"processed_data_{segment_length}_length.pkl"
        )
        if self.processed_file_save_path.exists() and not force_reprocess:
            print("Loading preprocessed data from pickle")
            self.all_data = pd.read_pickle(self.processed_file_save_path)
        else:
            print("No preprocessed data found or force reprocess is set to True")
            print("Preprocessing data")
            self.all_data = self.preprocess_data()
            self.save_all_data(self.processed_file_save_path)
        if self.mode == "validate":
            # self.all_data = self.all_data.iloc[:100]
            self.all_data = self.all_data
        if self.mode == "predict":
            self.all_data = self.keypoint_files
            self.get_longest_sequence_length()

    def __len__(self):
        return len(self.all_data) // self.segment_length

    def __getitem__(self, idx):
        if self.mode == "train" or self.mode == "validate":
            if isinstance(idx, slice):
                raise NotImplementedError("Slicing is not supported")
            else:
                # consider the segment length
                start_idx = (idx) * self.segment_length
                end_idx = (idx + 1) * self.segment_length
            data = self.all_data.iloc[start_idx:end_idx]

            # smpl col
            smpl_col = [col for col in data.columns if "smpl" in col]
            # keypoint col
            keypoint_col = [col for col in data.columns if "keypoint" in col]

            smpl_df = data[smpl_col]
            keypoint_df = data[keypoint_col]
            sequence_length = len(keypoint_df)

            # processing smpl data
            smpl_dict = {
                "smpl_body_pose": smpl_df[
                    [
                        "smpl_" + name
                        for name in expand_skeleton(get_SMPL_skeleton_names()[1:])
                    ]
                ].values,
                "smpl_transl": smpl_df[
                    ["smpl_transl_x", "smpl_transl_y", "smpl_transl_z"]
                ].values,
                "smpl_global_orient": smpl_df[
                    [
                        "smpl_global_orient_x",
                        "smpl_global_orient_y",
                        "smpl_global_orient_z",
                    ]
                ].values,
            }

            # processing keypoint_df: transfer from dataframe to tensor
            keypoint_data = keypoint_df.values
            keypoint_data = keypoint_data.reshape(sequence_length, -1, 3)
            keypoint_col = keypoint_df.columns
            return (keypoint_data, smpl_dict)
        
        elif self.mode == "predict":
            # prcessing keypoint data
            keypoint_path = self.all_data[idx]
            with open(keypoint_path, "rb") as f:
                keypoint_data_df = pickle.load(f)
            keypoint_data = keypoint_data_df.values
            # padding zero to the longest sequence
            curr_sequence_length = len(keypoint_data)
            keypoint_data_padded = np.concatenate(
                [keypoint_data, np.zeros((self.longest_length - curr_sequence_length, keypoint_data.shape[1]))], axis=0
            )
            keypoint_data_padded = keypoint_data_padded.reshape(self.longest_length, -1, 3)
            padding_mask = np.zeros((self.longest_length))
            padding_mask[:curr_sequence_length] = 1

            # processing smpl data
            smpl_path = keypoint_path.parent / (keypoint_path.stem.replace("_motorica", "_smpl") + ".pkl")
            with open(smpl_path, "rb") as f:
                smpl_data = pickle.load(f)
            # padding to smpl data
            smpl_body_pose = np.concatenate(
                [smpl_data["smpl_body_pose"], np.zeros((self.longest_length - curr_sequence_length, smpl_data["smpl_body_pose"].shape[1]))], axis=0
            )
            smpl_transl = np.concatenate(
                [smpl_data["smpl_transl"], np.zeros((self.longest_length - curr_sequence_length, smpl_data["smpl_transl"].shape[1]))], axis=0)
            smpl_global_orient = np.concatenate(
                [smpl_data["smpl_global_orient"], np.zeros((self.longest_length - curr_sequence_length, smpl_data["smpl_global_orient"].shape[1]))], axis=0
            )
            smpl_dict = {
                "smpl_body_pose": smpl_body_pose,
                "smpl_transl": smpl_transl,
                "smpl_global_orient": smpl_global_orient,
                "padding_mask": padding_mask,
                "name": keypoint_path.stem
            }
            return (keypoint_data_padded, smpl_dict)




    def preprocess_one_pair(self, file_path):
        """
        Preprocess a single pair of SMPL and Motorica data files.

        This function takes the file path of an SMPL data file, finds the corresponding
        Motorica keypoint file, loads both files, processes the data, and combines them
        into a single DataFrame.

        Parameters:
        file_path (Path): The file path of the SMPL data file.

        Returns:
        DataFrame: A pandas DataFrame containing the combined SMPL and Motorica keypoint data.

        Raises:
        FileNotFoundError: If the corresponding Motorica keypoint file is not found.

        Notes:
        - The SMPL data file is expected to contain the following keys:
            - "smpl_body_pose"
            - "smpl_transl"
            - "smpl_global_orient"
            - "smpl_joint_loc"
        - The Motorica keypoint file is expected to be in the same directory as the SMPL data file,
          with the same name but with "_smpl" replaced by "_motorica" in the filename.
        - The resulting DataFrame will have columns prefixed with "smpl_" for SMPL data and "keypoint_"
          for Motorica keypoint data.
        """
        # find corresponding keypoint file
        file_name = file_path.stem
        # replace _smpl.pkl with _motorica.pkl
        keypoint_file = self.data_dir / (file_name.replace("_smpl", "_motorica") + ".pkl")
        if not keypoint_file.exists():
            raise FileNotFoundError(f"Keypoint file {keypoint_file} not found")

        with open(file_path, "rb") as f:
            smpl_data = pickle.load(f)
        with open(keypoint_file, "rb") as f:
            keypoint_data = pickle.load(f)
        smpl_dict = {
            "smpl_body_pose": smpl_data["smpl_body_pose"],
            "smpl_transl": smpl_data["smpl_transl"],
            "smpl_global_orient": smpl_data["smpl_global_orient"],
            "smpl_joint_loc": smpl_data["smpl_joint_loc"],
        }
        # convert dict into a dataframe
        smpl_body_pose_df = pd.DataFrame(
            smpl_dict["smpl_body_pose"],
            columns=[
                "smpl_" + name
                for name in expand_skeleton(get_SMPL_skeleton_names()[1:])
            ],
        )
        smpl_transl_df = pd.DataFrame(
            smpl_dict["smpl_transl"],
            columns=["smpl_transl_x", "smpl_transl_y", "smpl_transl_z"],
        )
        smpl_global_orient_df = pd.DataFrame(
            smpl_dict["smpl_global_orient"],
            columns=[
                "smpl_global_orient_x",
                "smpl_global_orient_y",
                "smpl_global_orient_z",
            ],
        )
        smpl_df = pd.concat(
            [smpl_body_pose_df, smpl_transl_df, smpl_global_orient_df], axis=1
        )

        # reorder the sequence of the columns so that they are in the same order as the smpl data
        keypoint_order = expand_skeleton(get_motorica_skeleton_names(), "ZXY")
        selected_col = [
            "Hips_Xposition",
            "Hips_Yposition",
            "Hips_Zposition",
        ] + keypoint_order
        keypoint_data = keypoint_data[selected_col]
        # convert rotation from degree to radian
        keypoint_data[keypoint_order] = keypoint_data[
            keypoint_order
        ].apply(lambda x: np.deg2rad(x))
        keypoint_data.columns = ["keypoint_" + name for name in keypoint_data.columns]
        combined_df = pd.concat([keypoint_data, smpl_df], axis=1)


        return combined_df

    def preprocess_data(
        self,
    ):
        """
        Preprocesses data by iterating over SMPL files and combining the processed data.

        This method processes each SMPL file in the `self.smpl_files` list by calling the
        `preprocess_one_pair` method on each file. The processed data from each file is
        then appended to a list, which is concatenated into a single DataFrame and returned.

        Returns:
            pd.DataFrame: A concatenated DataFrame containing the processed data from all SMPL files.
        """
        all_data = []
        for i in tqdm(range(len(self.smpl_files)), desc="Processing files"):
            combined_df = self.preprocess_one_pair(self.smpl_files[i])
            all_data.append(combined_df)
        return pd.concat(all_data, ignore_index=True)

    def save_all_data(self, save_path):
        self.all_data.to_pickle(save_path)

    def get_longest_sequence_length(self):
        self.longest_length = 0
        for file in self.keypoint_files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            if len(data) > self.longest_length:
                self.longest_length = len(data)



class AlignmentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, mode):
        super(AlignmentDataModule, self).__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        if self.mode not in ["train", "validate", "predict"]:
            raise ValueError("mode should be one of ['train', 'validate', 'predict']")
        self.setup()

    def setup(self, stage=None):
        if self.mode == "train":
            self.train_dataset = AlignmentDataset(self.data_dir)
        elif self.mode == "validate":
            self.val_dataset = AlignmentDataset(self.data_dir, mode="validate")
        elif self.mode == "predict":
            self.predict_dataset = AlignmentDataset(self.data_dir, mode="predict")
        

    def collate_fn(self, batch):
        # keypoint is a dataframe and smpl is a dict
        keypoint_batch, smpl_dict_batch = zip(*batch)
        keypoint_stacked_batch  = torch.stack(
            [torch.tensor(item, dtype=torch.float32) for item in keypoint_batch]
        )
        # for each key in the dict, stack the values
        smpl_body_pose = torch.stack(
            [
                torch.tensor(item["smpl_body_pose"], dtype=torch.float32)
                for item in smpl_dict_batch
            ]
        )
        smpl_transl = torch.stack(
            [
                torch.tensor(item["smpl_transl"], dtype=torch.float32)
                for item in smpl_dict_batch
            ]
        )
        smpl_global_orient = torch.stack(
            [
                torch.tensor(item["smpl_global_orient"], dtype=torch.float32)
                for item in smpl_dict_batch
            ]
        )
        if self.mode == "predict":
            padding_mask = torch.stack(
                [
                    torch.tensor(item["padding_mask"], dtype=torch.int32)
                    for item in smpl_dict_batch
                ]
            )
            smpl_dict_batch = {
                "smpl_body_pose": smpl_body_pose,
                "smpl_transl": smpl_transl,
                "smpl_global_orient": smpl_global_orient,
                "padding_mask": padding_mask,
                "name": [item["name"] for item in smpl_dict_batch]
            }
        else:   
            smpl_dict_batch = {
                "smpl_body_pose": smpl_body_pose,
                "smpl_transl": smpl_transl,
                "smpl_global_orient": smpl_global_orient,
            }

        return keypoint_stacked_batch, smpl_dict_batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    def val_dataloader(self):
        return  DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=self.collate_fn,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=self.collate_fn,
        )

def main():
    data_dir = Path("/fs/nexus-projects/PhysicsFall/smpl2motorica/data/alignment_dataset")
    smpl_model_path = Path("/fs/nexus-projects/PhysicsFall/data/smpl/models")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    if not smpl_model_path.exists():
        raise FileNotFoundError(f"SMPL model directory {smpl_model_path} not found")

    data_module = AlignmentDataModule(data_dir, batch_size=1, num_workers=1, mode="predict")
    alighment_dataset  = data_module.predict_dataloader()

    # save all the data in a pickle file
    # for i in range(len(alignment_data)):
    #     keypoint_batch, keypoint_col_name, smpl_batch = alignment_data[i]

    # # For visualization and testing
    for keypoint_data, smpl_batch in alighment_dataset:
        break


    keypoint_fk = ForwardKinematics()

    batch_size, len_of_sequence, _ = smpl_batch["smpl_body_pose"].shape
    # # convert from matrix to euler angles
    # # keypoint_batch = conti_angle_rep.matrix_to_euler_angles(keypoint_batch, convention="XYZ")
    # # reshape back to (num_frames, num_keypoints x 3)
    # keypoint_batch = keypoint_batch.reshape(keypoint_batch.shape[0], -1)

    # # processing SMPL data
    pose = smpl_batch["smpl_body_pose"].reshape(-1, 69)
    transl = smpl_batch["smpl_transl"].reshape(-1, 3)
    global_orient = smpl_batch["smpl_global_orient"].reshape(-1, 3)

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
    smpl_joint_names = get_SMPL_skeleton_names()
    smpl_joints_loc = smpl_joints_loc[:, [smpl_joint_names.index(joint) for joint in motorica_to_smpl_mapping().values()],:]
        # swap from XYZ to ZXY
    smpl_joints_loc = smpl_joints_loc[:, :, [2, 0, 1]]

    batch_size, len_of_sequence, num_joints = keypoint_data.shape[:3]
    keypoint_data = keypoint_data.view(batch_size*len_of_sequence, num_joints, 3)
    keypoint_position = keypoint_fk.forward(keypoint_data.reshape(-1, 60))
    keypoint_position = keypoint_position.reshape(batch_size, len_of_sequence, -1, 3)
    keypoint_rot_loc = keypoint_position.view(batch_size, -1, 19, 3)

    def visualize_keypoint_data(ax, frame: int, df: pd.DataFrame, skeleton = None):
        if skeleton is None:
            skeleton = keypoint_fk.get_skeleton()
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
    adjusted_keypoint_loc_df = keypoint_fk.convert_to_dataframe(keypoint_rot_loc.reshape(-1, 19, 3))
    fig = plt.figure(figsize=(20, 10))
    input_loc_ax = fig.add_subplot(121, projection="3d")
    input_loc_ax = visualize_keypoint_data(input_loc_ax, 0, adjusted_keypoint_loc_df)
    input_loc_ax.set_title("Adjusted Keypoint")
    smpl_loc_df = keypoint_fk.convert_to_dataframe(positions=torch.tensor(smpl_joints_loc.reshape(-1, 19, 3)))
    smpl_loc_ax = fig.add_subplot(122, projection="3d")
    smpl_loc_ax = visualize_keypoint_data(smpl_loc_ax, 0, smpl_loc_df)
    smpl_loc_ax.set_title("SMPL Model")
    # set xyz axis
    smpl_loc_ax.set_xlabel("X")
    smpl_loc_ax.set_ylabel("Y")
    smpl_loc_ax.set_zlabel("Z")
    
    plt.savefig("dataloader_testing_fig.png")




    
    # # processing motorica data
    # keypoint_col_name = selected_col = [
    #         "Hips_Xposition",
    #         "Hips_Yposition",
    #         "Hips_Zposition",
    #     ]  + expand_skeleton(get_motorica_skeleton_names(), "ZXY")
    # # remove "smpl_" prefix
    # keypoint_col_name = [col.replace("keypoint_", "") for col in keypoint_col_name]
    # keypoint_fk = ForwardKinematics()
    # keypoint_pos = keypoint_fk.forward(keypoint_batch)
    # position_df = keypoint_fk.convert_to_dataframe(keypoint_pos)

    

    # motorica_dummy_data = load_dummy_motorica_data()
    # # motorica_dummy_data.values = keypoint_batch_df
    # motorica_dummy_data.skeleton = get_skeleton()
    # # frame to visualize
    # frame = 0
    # fig = plt.figure(figsize=(15, 10))
    # smpl_ax = fig.add_subplot(121, projection="3d")
    # smpl_ax = SMPL_visulize_a_frame(
    #     smpl_ax, smpl_joints_loc[frame], smpl_vertices[frame], smpl_model
    # )
    # smpl_ax.set_title("SMPL joints")

    # ax_motorica = fig.add_subplot(122, projection="3d")
    # ax_motorica = motorica_draw_stickfigure3d(
    #     ax_motorica, motorica_dummy_data, frame, data=position_df
    # )
    # ax_motorica.set_title("Motorica")

    # plt.savefig("dataloader_testing_fig.png")


if __name__ == "__main__":
    main()