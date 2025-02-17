import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import pandas as pd
import smplx
from pathlib import Path

sys.path.append("/fs/nexus-projects/PhysicsFall/")
from smpl2motorica.dataloader import AlignmentDataModule
from smpl2motorica.utils import conti_angle_rep
from smpl2motorica.smpl2keypoint import (
    load_dummy_motorica_data,
    get_SMPL_skeleton_names,
    get_motorica_skeleton_names,
    smpl2motorica,
)
from smpl2motorica.utils.pymo.preprocessing import MocapParameterizer


class RotationTranslationNet(nn.Module):
    def __init__(self):
        super(RotationTranslationNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(
                512, 9
            ),  # output size is (batch_size, (1+19)x9), 1 for translation and 19 for rotation
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class KeypointModel(pl.LightningModule):
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.model = RotationTranslationNet().to(self.device)
        self.mse_loss = nn.MSELoss()
        self.motorica_dummy_data = load_dummy_motorica_data()
        self.smpl_model_path = Path("/fs/nexus-projects/PhysicsFall/data/smpl/models")
        if not self.smpl_model_path.exists():
            raise FileNotFoundError(
                f"SMPL model path {self.smpl_model_path} does not exist."
            )

    def forward(self, x):
        return self.model(x)

    def smpl_forward_kinematics(self, smpl_data_dict):
        """
        Computes the forward kinematics for the SMPL model given the SMPL data.

        Args:
            smpl_data_dict (dict): A dictionary containing the SMPL data with the following keys:
                - "smpl_body_pose": The body pose parameters of the SMPL model.
                - "smpl_transl": The translation parameters of the SMPL model.
                - "smpl_global_orient": The global orientation parameters of the SMPL model.

        Returns:
            np.ndarray: A numpy array of shape (batch_size, 24, 3) containing the 3D locations of the 24 SMPL joints.
        """
        # processing SMPL data
        pose = smpl_data_dict["smpl_body_pose"]
        transl = smpl_data_dict["smpl_transl"]
        global_orient = smpl_data_dict["smpl_global_orient"]

        # if there is batch, merge the batch dimension with num pose dimmention
        if len(pose.shape) == 3:
            pose = pose.reshape(-1, pose.shape[-1])
            transl = transl.reshape(-1, transl.shape[-1])
            global_orient = global_orient.reshape(-1, global_orient.shape[-1])

        smpl_model = smplx.create(
            model_path=self.smpl_model_path,
            model_type="smpl",
            return_verts=False,
            batch_size=len(pose),
        ).to(self.device)
        smpl_output = smpl_model(
            global_orient=torch.tensor(global_orient, dtype=torch.float32),
            body_pose=torch.tensor(pose, dtype=torch.float32),
            transl=torch.tensor(transl, dtype=torch.float32),
        )
        smpl_joints_loc = smpl_output.joints.detach().cpu().numpy().squeeze()
        smpl_joints_loc = smpl_joints_loc[:, :24, :].reshape(len(smpl_joints_loc), -1)
        return smpl_joints_loc

    def keypoint_to_rot_mat(self, keypoint_data: pd.DataFrame):
        pos_name = [
            "keypoint_Hips_Xposition",
            "keypoint_Hips_Yposition",
            "keypoint_Hips_Zposition",
        ]
        positions_df = keypoint_data[pos_name]  # (batch_size, 3)
        rot_df = keypoint_data[keypoint_data.columns.difference(pos_name)]
        rot_euler = torch.tensor(rot_df.values, dtype=torch.float32).reshape(
            len(rot_df), -1, 3
        )
        rot_euler_rad = torch.deg2rad(rot_euler)
        # convert to rotation matrix
        keypoint_rot_mat = conti_angle_rep.euler_angles_to_matrix(
            rot_euler_rad, convention="XYZ"
        )  # (batch_size, 19, 3, 3)

        # combine rotation and translation
        keypoint_rot_mat_flat = keypoint_rot_mat.view(
            keypoint_rot_mat.size(0), keypoint_rot_mat.size(1), -1
        )  # (batch_size, 19, 9)
        # pad the translation to match the rotation matrix, from (batch_size, 3) to size(batch_size, 1, 9).
        keypoint_pos = torch.tensor(positions_df.values, dtype=torch.float32).unsqueeze(
            1
        )
        keypoint_pos = torch.nn.functional.pad(keypoint_pos, (0, 6))
        keypoint_combined = torch.cat(
            [
                keypoint_pos,
                keypoint_rot_mat_flat,
            ],
            dim=1,
        )

        return keypoint_combined

    def keypoint_forward_kinematics(self, transl, rotation, keypoint_col_names):
        keypoint_df = pd.DataFrame(
            torch.cat([transl, rotation], dim=1).detach().cpu().numpy(),
            columns=[col.replace("keypoint_", "") for col in keypoint_col_names],
        )
        self.motorica_dummy_data.values = keypoint_df
        position_mocap = MocapParameterizer("position").fit_transform(
            [self.motorica_dummy_data]
        )[0]
        position_df = position_mocap.values
        # select only the keypoint joints
        keypoint_joints = get_motorica_skeleton_names()
        expanded_keypoint_joints = [
            f"{joint}_{axis}position"
            for joint in keypoint_joints
            for axis in ["X", "Y", "Z"]
        ]
        position_df = position_df[expanded_keypoint_joints]
        return position_df

    def smpl_dict_to_cuda(self, smpl_dict):
        for key in smpl_dict:
            smpl_dict[key] = smpl_dict[key].to(self.device)
        return smpl_dict

    def training_step(self, batch, batch_idx):
        keypoint_batch, smpl_batch = batch
        keypoint_col_names = keypoint_batch.columns
        keypoint_combined = self.keypoint_to_rot_mat(keypoint_batch).to(
            self.device
        )  # size (batch_size, 1+19, 9)
        smpl_batch = self.smpl_dict_to_cuda(smpl_batch)
        predicted_rot_transl = self.model(
            keypoint_combined
        )  # size (batch_size, 1+19, 9)
        loss = self.loss(
            predicted_rot_transl, keypoint_combined, smpl_batch, keypoint_col_names
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss(
        self, predicted_rot_transl, keypoint_batch, smpl_batch, keypoint_col_names
    ):
        predicted_transl = predicted_rot_transl[:, 0, :3]
        predicted_rot = predicted_rot_transl[:, 1:, :].view(
            len(predicted_rot_transl), -1, 3, 3
        )

        # update the keypoint data with the predicted rotation and translation
        keypoint_transl = keypoint_batch[:, 0, :3]
        keypoint_rot = keypoint_batch[:, 1:, :].view(
            len(predicted_rot_transl), -1, 3, 3
        )

        # sanity check the shape
        assert keypoint_rot.shape == predicted_rot.shape
        assert keypoint_transl.shape == predicted_transl.shape

        adjusted_transl = keypoint_transl + predicted_transl
        # apply the predicted rotation to the keypoint rotation
        adjusted_rot = torch.matmul(keypoint_rot, predicted_rot)
        # convert the rotation matrix to euler angles
        keypoint_batch = conti_angle_rep.matrix_to_euler_angles(
            adjusted_rot, convention="XYZ"
        )
        # reshape back to (num_frames, num_keypoints x 3)
        keypoint_batch = keypoint_batch.reshape(keypoint_batch.shape[0], -1)
        # rad to degree
        keypoint_rot = torch.rad2deg(keypoint_batch)
        # apply forward kinematics to the adjusted rotation and translation
        keypoint_loc = self.keypoint_forward_kinematics(
            adjusted_transl, keypoint_rot, keypoint_col_names
        )

        # apply forward kinematics to the smpl data
        smpl_loc = self.smpl_forward_kinematics(smpl_batch)
        print(f"shape of keypoint_loc: {keypoint_loc.shape}")
        print(f"shape of smpl_loc: {smpl_loc.shape}")
        exit()


def main():
    data_dir = "/fs/nexus-projects/PhysicsFall/smpl2motorica/data/alignment_dataset"
    batch_size = 2
    num_workers = 4

    data_module = AlignmentDataModule(data_dir, batch_size, num_workers)

    model = KeypointModel()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, data_module.train_dataloader())


if __name__ == "__main__":
    main()
