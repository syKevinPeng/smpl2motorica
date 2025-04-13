from math import isnan
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pytorch3d.transforms
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import sys, os
from tqdm import tqdm
import pandas as pd
import smplx
from pathlib import Path
import pytorch3d
import numpy as np
from matplotlib import pyplot as plt
from lightning.pytorch.utilities import grad_norm
import cv2
from concurrent.futures import ThreadPoolExecutor

import wandb

sys.path.append("/fs/nexus-projects/PhysicsFall/")
from smpl2motorica.dataloader import AlignmentDataModule
from smpl2motorica.utils.keypoint_skeleton import (
    get_keypoint_skeleton,
    get_keypoint_skeleton_scale,
)
from smpl2motorica.smpl2keypoint import (
    load_dummy_motorica_data,
    get_SMPL_skeleton_names,
    motorica_to_smpl_mapping,
    expand_skeleton,
    get_motorica_skeleton_names,
    create_video_from_keypoints,
)
from smpl2motorica.utils.conti_angle_rep import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)
from smpl2motorica.utils.pymo.preprocessing import MocapParameterizer
from smpl2motorica.utils.pymo.Quaternions import Quaternions
from datetime import datetime
from editable_dance_project.src.skeleton.forward_kinematics import ForwardKinematics
from loss import foot_sliding_loss

def grad_hook(module, grad_input, grad_output):
    # Check grad_input (gradients flowing INTO the module)
    for i, g in enumerate(grad_input):
        if g is not None:
            if torch.isnan(g).any():
                print(
                    f"!!!grad_hook: NaN in grad_input[{i}] of {module.__class__.__name__} !!!"
                )

    # Check grad_output (gradients flowing OUT of the module)
    for i, g in enumerate(grad_output):
        if g is not None:
            if torch.isnan(g).any():
                print(
                    f"!!!grad_hook: NaN in grad_output[{i}] of {module.__class__.__name__} !!!"
                )


def _grad_hook(grad, name):
    if torch.isnan(grad).any():
        print(f"!!! _grad_hook: NaN in {name} gradient !!!")
        exit()
    return grad


class EulerAnglesToMatrix(nn.Module):
    def __init__(self):
        super(EulerAnglesToMatrix, self).__init__()
        self.convention = pytorch3d.transforms.euler_angles_to_matrix

    def forward(self, x):
        # x is in the shape of (batch_size, seq_len, num_joints-1, 3)
        # convert to rotation matrix
        x = self.convention(x, convention="ZXY")
        return x


class RotationTranslationNet(nn.Module):
    def __init__(self):
        super(RotationTranslationNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(
                32, 6
            ),  # output size is (batch_size, (1+19),6), 1 for translation and 19 for rotation in 6D representation
        )

        # Initialize the weights of the last layer to be close to zero
        nn.init.normal_(self.layers[-1].weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.layers[-1].bias, 1e-4)
        for layer in self.layers:
            layer.register_full_backward_hook(grad_hook)

    def forward(self, x):
        batch_size, seq_len, num_joints, _ = x.shape
        x = self.layers(x)
        x_trans = x[:, :, 0, :]
        x_rot = x[:, :, 1:, :]
        if torch.isnan(x_rot).any():
            print("!!! x_rot 6d is nan !!!")
        x_rot = rotation_6d_to_matrix(x_rot)
        if torch.isnan(x_rot).any():
            print("!!! x_rot matrix is nan !!!")
        x_rot = x_rot.view(batch_size, seq_len, num_joints - 1, 9)
        trans_zero_padding = torch.zeros(batch_size, seq_len, 1, 3).to(x.device)
        x_trans = torch.cat([x_trans.unsqueeze(2), trans_zero_padding], dim=-1)
        x = torch.cat([x_trans, x_rot], dim=2)
        return x


class LocDiffNet(nn.Module):
    def __init__(self, num_joints=19):
        super(LocDiffNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_joints * 3, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, (num_joints + 1) * 6),
        )
        # Initialize the weights of the last layer to be close to zero
        nn.init.normal_(self.fc[-1].weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.fc[-1].bias, 1e-4)

    def forward(self, x):
        batch_size, seq_len, num_joints, _ = x.shape
        x = x.reshape(batch_size, seq_len, -1)
        x = self.fc(x)
        x = x.reshape(batch_size, seq_len, num_joints + 1, 6)
        # convert from 6D representation to rotation matrix
        x_trans = x[:, :, 0, :]  # shape: (batch_size, seq_len, joints, 6)
        x_rot = x[:, :, 1:, :]  # shape: (batch_size, seq_len, 6)
        x_rot = rotation_6d_to_matrix(
            x_rot
        )  # shape: (batch_size, num_joints, joints, 3,3)
        x_rot = x_rot.view(batch_size, seq_len, num_joints, 9)
        # append zero to the translation
        trans_zero_padding = torch.zeros(batch_size, seq_len, 1, 3).to(x.device)
        x_trans = torch.cat(
            [x_trans.unsqueeze(2), trans_zero_padding], dim=-1
        )  # shape: (batch_size, seq_len, 1, 9)
        x = torch.cat(
            [x_trans, x_rot], dim=2
        )  # output shape: (batch_size, seq_len, num_joints+1, 9)
        return x


class KeypointModel(pl.LightningModule):
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.model = RotationTranslationNet().to(self.device)
        self.mse_loss = self.mpjpe
        self.motorica_dummy_data = load_dummy_motorica_data()
        self.smpl_model_path = Path("/fs/nexus-projects/PhysicsFall/data/smpl/models")
        if not self.smpl_model_path.exists():
            raise FileNotFoundError(
                f"SMPL model path {self.smpl_model_path} does not exist."
            )
        self.keypoint_fk = ForwardKinematics(
            normalized_skeleton_path="/fs/nexus-projects/PhysicsFall/editable_dance_project/data/normalized_skeleton.pkl"
        )

    def mpjpe(self, pred, ref):
        return torch.mean(torch.norm(pred - ref, p=2, dim=-1))

    def forward(self, x):
        return self.model(x)

    def smpl_forward_kinematics(self, smpl_data_dict, return_verts=False):
        """
        Perform forward kinematics using the SMPL model.

        Args:
            smpl_data_dict (dict): A dictionary containing SMPL data with the following keys:
                - "smpl_body_pose": Tensor of shape (batch_size, 72) or (batch_size, num_frames, 72)
                - "smpl_transl": Tensor of shape (batch_size, 3) or (batch_size, num_frames, 3)
                - "smpl_global_orient": Tensor of shape (batch_size, 3) or (batch_size, num_frames, 3)
            return_verts (bool, optional): If True, return the vertices of the SMPL model. Defaults to False.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_selected_joints * 3) containing the locations of the selected joints.
            If return_verts is True, also returns a numpy array of shape (batch_size, num_vertices, 3) containing the vertices of the SMPL model.
        """

        # processing SMPL data
        pose = smpl_data_dict["smpl_body_pose"]
        transl = smpl_data_dict["smpl_transl"]
        global_orient = smpl_data_dict["smpl_global_orient"]
        batch_size = pose.shape[0]
        # if there is batch, merge the batch dimension with num pose dimmention
        if len(pose.shape) == 3:
            pose = pose.reshape(-1, pose.shape[-1])
            transl = transl.reshape(-1, transl.shape[-1])
            global_orient = global_orient.reshape(-1, global_orient.shape[-1])

        smpl_model = smplx.create(
            model_path=self.smpl_model_path,
            model_type="smpl",
            return_verts=return_verts,
            batch_size=len(pose),
        ).to(self.device)
        smpl_output = smpl_model(
            global_orient=global_orient,
            body_pose=pose,
            transl=transl,
        )
        smpl_joints_loc = smpl_output.joints
        smpl_joints_loc = smpl_joints_loc[:, :24, :]  # (batch_size, 24,3)

        if return_verts:
            # for visualization
            smpl_verts = smpl_output.vertices.detach().cpu().numpy()
            smpl_joints_loc = smpl_joints_loc.detach().cpu().numpy()
            return smpl_joints_loc, smpl_verts
        smpl_joint_names = get_SMPL_skeleton_names()
        smpl_joints_loc = smpl_joints_loc[
            :,
            [
                smpl_joint_names.index(joint)
                for joint in motorica_to_smpl_mapping().values()
            ],
            :,
        ]
        # swap from XYZ to ZXY
        smpl_joints_loc[:, :, :] = smpl_joints_loc[:, :, [2, 0, 1]]
        smpl_joints_loc = smpl_joints_loc.reshape(batch_size, -1, 21, 3)
        return smpl_joints_loc

    def keypoint_to_rot_mat(self, keypoint_data: torch.tensor):
        """
        Converts keypoint data to a combined tensor of position and rotation matrices.
        Args:
            keypoint_data (torch.tensor): A tensor of shape (batch_size, len_of_sequence, 22, 3)
                                          containing keypoint data. The last dimension should
                                          represent 3D coordinates.
        Returns:
            torch.tensor: A tensor of shape (batch_size, len_of_sequence, 22, 9) where (batch_size, len_of_sequence, 0, 9) is the position (only first three values are valid)
            and (batch_size, len_of_sequence, 1:22, 9) is the rotation matrix.
            ValueError: If the input tensor does not have the shape (batch_size, len_of_sequence, 22, 3).
        """

        # input shape (batch_size, len_of_sequence, 22, 3)
        batch_size = keypoint_data.shape[0]
        len_of_sequence = keypoint_data.shape[1]
        if keypoint_data.shape[-1] != 3 or keypoint_data.shape[2] != 22:
            raise ValueError(
                "Keypoint data should be of shape (batch_size, len_of_sequence, 22, 3)"
            )
        position_data = keypoint_data[:, :, 0, :]  # (batch_size, len_of_sequence, 3)
        rot_data = keypoint_data[:, :, 1:, :]  # (batch_size, len_of_sequence, 21, 3)
        # convert to rotation matrix
        keypoint_rot_mat = pytorch3d.transforms.euler_angles_to_matrix(
            rot_data, convention="ZXY"
        )  # (batch_size,len_of_sequence, 21, 3, 3)
        keypoint_rot_mat = keypoint_rot_mat.view(batch_size, len_of_sequence, 21, 9)
        # pad the translation to match the rotation matrix, from (batch_size, 3) to size(batch_sizexlen_of_sequence, 1, 9).
        keypoint_pos = position_data.unsqueeze(2)  # (batch_size, len_of_sequence, 1, 3)
        keypoint_pos = torch.nn.functional.pad(keypoint_pos, (0, 6), "constant", 0)
        keypoint_combined = torch.cat(
            [
                keypoint_pos,
                keypoint_rot_mat,
            ],
            dim=2,
        )
        if keypoint_combined.shape != (batch_size, len_of_sequence, 22, 9):
            raise ValueError(
                "Keypoint data should be of shape (batch_size, len_of_sequence, 22, 9)"
            )
        return keypoint_combined

    def rot_mat_to_keypoint(self, keypoint_rot_mat):
        """
        Converts a rotation matrix to keypoint rotations in Euler angles.
        Args:
            keypoint_rot_mat (torch.Tensor): A tensor of shape (batch_size, len_of_sequence, 1+19, 9)
                                             containing rotation matrices and translations.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, len_of_sequence, 20, 3) containing the keypoint
                          rotations in Euler angles.
        Notes:
            - The input tensor `keypoint_rot_mat` contains both translations and rotation matrices.
            - The rotation matrices are converted to Euler angles using the specified convention.
            - The output tensor contains the keypoint rotations in Euler angles and in radian.
        """

        # input shape (batch_size, len_of_sequence, 1+19, 9)
        # output shape is (batch_size, len_of_sequence, 20, 3)
        batch_size = keypoint_rot_mat.shape[0]
        len_of_sequence = keypoint_rot_mat.shape[1]
        # get the translation
        keypoint_transl = keypoint_rot_mat[:, :, 0, :3]
        # get the rotation
        keypoint_rot = keypoint_rot_mat[:, :, 1:, :].view(
            batch_size, len_of_sequence, -1, 3, 3
        )
        # convert the rotation matrix to euler angles
        keypoint_rot = pytorch3d.transforms.matrix_to_euler_angles(
            keypoint_rot, convention="ZXY"
        )
        keypoint_euler = torch.cat([keypoint_transl.unsqueeze(2), keypoint_rot], dim=2)
        return keypoint_euler

    def keypoint_forward_kinematics(self, keypoint_data):
        # keypoint data contains the translation and rotation of the keypoints
        # first 3 columns are the translation, and the rest are the rotation that have the same order as motorica2smpl
        # merge the batch dimension with the len_of_sequence dimension
        # input size: (batch_size, len_of_sequence, 22, 9)
        # output size: (batch_size, len_of_sequence, 22, 3)
        batch_size, len_of_sequence, num_joints, joint_rep = keypoint_data.shape
        keypoint_data = keypoint_data.view(batch_size * len_of_sequence, num_joints, -1)
        keypoint_position = self.keypoint_fk(
            keypoint_data.reshape(batch_size * len_of_sequence, -1)
        )
        keypoint_position = keypoint_position.reshape(
            batch_size, len_of_sequence, -1, 3
        )
        return keypoint_position

    def smpl_dict_to_cuda(self, smpl_dict):
        for key in smpl_dict:
            smpl_dict[key] = smpl_dict[key].to(self.device)
        return smpl_dict

    def training_step(self, batch, batch_idx):
        keypoint_batch, smpl_batch = batch  # keypoint_batch size: (batch_size, 1+19, 3)
        keypoint_combined = self.keypoint_to_rot_mat(keypoint_batch).to(
            self.device
        )  # size (batch_size, 1+19, 9)
        smpl_batch = self.smpl_dict_to_cuda(smpl_batch)

        # for debugging
        self.model = self.model.double()
        keypoint_combined = keypoint_combined.double()

        predicted_rot_transl = self.model(
            keypoint_combined
        )  # size (batch_size, 1+19, 9)
        loss = self.loss(
            predicted_rot_transl, keypoint_combined, smpl_batch, reg_weight=0.1
        )
        # register hook to the loss
        loss.register_hook(lambda g: _grad_hook(g, "loss"))
        # check nan
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model.layers, norm_type=2)
        for key, value in norms.items():
            if torch.isnan(value).any():
                print(f"!!! grad norm {key} is nan !!!")
        self.log_dict(norms)

    def on_after_backward(self):
        # Check for NaN gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient in {name}")

    def validation_step(self, batch, batch_idx):
        print("Performing Validation Step")
        print(f"WARNING: FOR DEBUGGING ONLY; MODEL ONLY RETURN IDENTITY MATRIX")
        keypoint_rot_euler, smpl_batch = batch
        batch_size, len_of_sequence, _, _ = keypoint_rot_euler.shape
        keypoint_rot_mat = self.keypoint_to_rot_mat(keypoint_rot_euler).to(
            self.device
        )  # (batch_size, seq_length, 1+21, 9)
        smpl_batch = self.smpl_dict_to_cuda(smpl_batch)
        smpl_joint_loc = self.smpl_forward_kinematics(smpl_batch, return_verts=False)

        # all ways predict zeros and identity matrix
        predicted_rot_transl = torch.zeros_like(keypoint_rot_mat)
        # for (batch_size, seq_length, 1:22, 9), alwasy return identity matrix
        predicted_rot_transl[:, :, 1:, :] = (
            torch.eye(3)
            .view(1, 1, 1, 9)
            .repeat(predicted_rot_transl.shape[0], predicted_rot_transl.shape[1], 21, 1)
        )
        adjusted_keypoint = self.apply_prediction_to_keypoint(
            predicted_rot_transl, keypoint_rot_mat
        )  # (batch_size, len_of_sequence, 20, 3)
        keypoint_rot_loc = self.keypoint_forward_kinematics(keypoint_rot_euler)
        adjusted_keypoint_loc = self.keypoint_forward_kinematics(adjusted_keypoint)

        loss = self.loss(
            predicted_rot_transl, keypoint_rot_mat, smpl_batch, reg_weight=0
        )

        self.log("val_loss", loss, on_epoch=True)
        # get the first batch
        adjusted_keypoint_loc = adjusted_keypoint_loc[0]
        smpl_joint_loc = smpl_joint_loc[0]
        video_output_dir = Path("videos") / f"valid_test.mp4"
        self.prediction_visualization(
            input_loc=None,
            smpl_loc=smpl_joint_loc,
            predicted_loc=adjusted_keypoint_loc,
            output_path=video_output_dir,
        )
        exit()

    def predict_step(self, batch, batch_idx):
        prediction_output_dir = Path(
            "/fs/nexus-projects/PhysicsFall/smpl2motorica/prediction_result"
        )
        if not prediction_output_dir.exists():
            prediction_output_dir.mkdir(parents=True)

        keypoint_rot_euler, smpl_batch = batch
        file_name = smpl_batch["name"]
        padding_mask = smpl_batch["padding_mask"]
        # remove padding mask from batch
        del smpl_batch["name"]
        batch_size, seq_length, num_joints, _ = keypoint_rot_euler.shape
        keypoint_rot_mat = self.keypoint_to_rot_mat(keypoint_rot_euler).to(
            self.device
        )  # (batch_size, seq_length, 1+19, 9)

        smpl_batch = self.smpl_dict_to_cuda(smpl_batch)
        smpl_joint_loc = self.smpl_forward_kinematics(smpl_batch, return_verts=False)

        predicted_rot_transl = self.model(keypoint_rot_mat)

        # predicted_rot_transl = torch.zeros_like(keypoint_rot_mat)
        # predicted_rot_transl[:, :, 1:, :] = torch.eye(3).view(1, 1, 1, 9).repeat(
        #     predicted_rot_transl.shape[0],predicted_rot_transl.shape[1], 19, 1
        # )
        # setting transl to zero
        predicted_rot_transl[:, :, 0, :] = torch.zeros_like(
            predicted_rot_transl[:, :, 0, :]
        )
        adjusted_keypoint = self.apply_prediction_to_keypoint(
            predicted_rot_transl, keypoint_rot_mat
        )  # (batch_size, len_of_sequence, 20, 3)

        default_skeleton = get_keypoint_skeleton()
        default_scale = get_keypoint_skeleton_scale()
        keypoint_rot_loc = self.keypoint_forward_kinematics(keypoint_rot_euler)
        adjusted_keypoint_loc = self.keypoint_forward_kinematics(adjusted_keypoint)

        joint_order = [
            "Hips_Xposition",
            "Hips_Yposition",
            "Hips_Zposition",
        ] + expand_skeleton(get_motorica_skeleton_names())
        # save the prediction as npy file
        keypoint_dict = {
            "file_name": file_name,
            "skeleton": default_skeleton,
            "scale": default_scale,
            "motion_data": adjusted_keypoint,
            "motion_data_order": joint_order,
            "fps": 30,
            "motion_positions": adjusted_keypoint_loc,
            "motion_positions_order": get_motorica_skeleton_names(),
        }
        # with open(prediction_output_dir/f"{file_name[0]}.npy", "wb") as f:
        #     np.save(f, keypoint_dict)

        # prediction visualization
        motion_position = adjusted_keypoint_loc[0]
        video_output_dir = Path("videos") / f"{file_name[0]}.mp4"
        self.prediction_visualization(
            input_loc=motion_position,
            smpl_loc=smpl_joint_loc[0],
            predicted_loc=motion_position,
            output_path=video_output_dir,
        )

    # visualize the prediction
    def prediction_visualization(
        self, input_loc, smpl_loc, predicted_loc, output_path="predicted_keypoint.png"
    ):
        predicted_position = self.keypoint_fk.convert_to_dataframe(predicted_loc)
        smpl_position = self.keypoint_fk.convert_to_dataframe(smpl_loc)

        create_video_from_keypoints(
            predicted_position,
            smpl=smpl_position,
            title="Predicted Keypoint",
            max_frames=-1,
            output_path=output_path,
            fps=30,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer

    def apply_prediction_to_keypoint(self, predicted_rot_transl, keypoint_batch):
        """
        Apply predicted rotation and translation to keypoint data.

        Args:
            predicted_rot_transl (torch.Tensor): Tensor of shape (batch_size, len_of_sequence, 20, 3) containing the predicted rotation and translation.
            keypoint_batch (torch.Tensor): Tensor of shape (batch_size, len_of_sequence, 20, 3) containing the original keypoint data.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, len_of_sequence, 20, 3) containing the adjusted keypoint data with the applied predicted rotation and translation.

        Raises:
            ValueError: If the output tensor shape does not match (batch_size, len_of_sequence, 20, 3).
        """

        batch_size, len_of_sequence, num_joints, _ = keypoint_batch.shape
        predicted_transl = predicted_rot_transl[:, :, 0, :]
        predicted_rot = predicted_rot_transl[:, :, 1:, :].view(
            batch_size, len_of_sequence, 21, 3, 3
        )
        # update the keypoint data with the predicted rotation and translation
        keypoint_transl = keypoint_batch[:, :, 0, :]
        keypoint_rot = keypoint_batch[:, :, 1:, :].view(
            batch_size, len_of_sequence, 21, 3, 3
        )
        adjusted_transl = (
            keypoint_transl + predicted_transl
        )  # shape: (batch_size, len_of_sequence, 9)
        # apply the predicted rotation to the keypoint rotation

        adjusted_rot = torch.matmul(
            predicted_rot, keypoint_rot
        )  # shape: (batch_size, len_of_sequence, 21, 3, 3)
        keypoint_rot = adjusted_rot.view(batch_size, len_of_sequence, 21, 9)
        predicted_keypoint = torch.cat(
            [adjusted_transl.unsqueeze(2), keypoint_rot], dim=2
        )  # (batch_size, len_of_sequence, 22, 3)
        if predicted_keypoint.shape != (batch_size, len_of_sequence, 22, 9):
            raise ValueError(
                "output should be of shape (batch_size, len_of_sequence, 22, 9)"
            )
        return predicted_keypoint

    def loss(self, predicted_rot_transl, keypoint_batch, smpl_batch, reg_weight=0.1, foot_sliding_weight = 0.1):
        batch_size = keypoint_batch.shape[0]
        len_of_sequence = keypoint_batch.shape[1]
        adjusted_keypoint = self.apply_prediction_to_keypoint(
            predicted_rot_transl, keypoint_batch
        )  # (batch_size, len_of_sequence, 20, 9)
        adjusted_keypoint_loc = self.keypoint_forward_kinematics(adjusted_keypoint)
        # apply forward kinematics to the smpl data
        smpl_loc = self.smpl_forward_kinematics(smpl_batch)

        # calculate the loss
        mpjpe_loss = self.mse_loss(adjusted_keypoint_loc, smpl_loc)
        # mpjpe_loss.register_hook(lambda g: _grad_hook(g, "mpjpe_loss"))
        assert mpjpe_loss > 0, f"mpjpe_loss is negative: {mpjpe_loss}"

        # regularization term
        reg_weight = reg_weight
        rot_mat = predicted_rot_transl[:, :, 1:, :].view(
            batch_size, len_of_sequence, 21, 3, 3
        )
        identity_mat = (
            torch.eye(3)
            .view(1, 1, 1, 3, 3)
            .repeat(batch_size, len_of_sequence, 21, 1, 1)
            .to(self.device)
        )
        reg_loss = reg_weight * torch.mean(
            torch.norm(rot_mat - identity_mat, p=2, dim=(-2, -1))
        )
        assert not torch.isnan(rot_mat - identity_mat).any(), (
            "rot_mat - identity_mat is nan"
        )
        assert reg_loss >= 0, f"reg_loss is negative: {reg_loss}"

        # foot sliding loss
        sliding_loss = foot_sliding_weight * foot_sliding_loss(adjusted_keypoint_loc)
        self.log("sliding_loss", sliding_loss, on_epoch=True)
        self.log("reg_loss", reg_loss, on_epoch=True)
        self.log("mpjpe_loss", mpjpe_loss, on_epoch=True)
        return mpjpe_loss + reg_loss + sliding_loss


class StopOnNaNLoss(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None and (loss.isnan().any() or loss.isinf().any()):
            print("NaN loss detected. Stopping training...")
            trainer.should_stop = True


def main():
    ckpt = Path(
        "/fs/nexus-projects/PhysicsFall/smpl2motorica/checkpoints/20250329_113355_dy5dq0z8/smpl2keypoint-epoch=298-train_loss=0.12.ckpt"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt} does not exist.")
    # mode = "validate"
    # mode = "predict"
    mode = "train"
    num_epoch_to_train = 300
    data_dir = "/fs/nexus-projects/PhysicsFall/smpl2motorica/data/alignment_dataset"
    batch_size = 16
    num_workers = 4

    data_module = AlignmentDataModule(data_dir, batch_size, num_workers, mode=mode)

    model = KeypointModel()
    wandb_logger = WandbLogger(
        project="SMPL2Keypoint",
        name="train_6",
        mode="disabled",
        notes="Adding sliding foot loss",
    )
    saving_dir = Path("/fs/nexus-projects/PhysicsFall/smpl2motorica/checkpoints")
    exp_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{wandb_logger.experiment.id}"
    ckpt_saving_dir = saving_dir / exp_id
    if not ckpt_saving_dir.exists():
        print(f"Creating directory {ckpt_saving_dir} for saving checkpoints.")
        ckpt_saving_dir.mkdir(parents=True)
    if not (saving_dir / exp_id).exists():
        (saving_dir / exp_id).mkdir(parents=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        dirpath=str(ckpt_saving_dir),
        filename="smpl2keypoint-{epoch:02d}-{train_loss:.2f}",
        mode="min",
        save_top_k=5,
        save_last=True,
    )
    nan_callbacks = StopOnNaNLoss()
    trainer = Trainer(
        max_epochs=num_epoch_to_train,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, nan_callbacks],
        gradient_clip_val=1.0,
    )
    if mode == "train":
        trainer.fit(model, data_module.train_dataloader())
    elif mode == "validate":
        trainer.validate(model, data_module.val_dataloader())
    elif mode == "predict":
        print(f"Loading model from {ckpt}")
        model = KeypointModel.load_from_checkpoint(checkpoint_path=str(ckpt))
        trainer.predict(model, data_module.predict_dataloader())
    else:
        raise ValueError("Invalid mode. Choose from 'train', 'validate', or 'predict'.")


if __name__ == "__main__":
    main()

    # # test euler to rotation matrix and then back to euler
    # euler = torch.rand(2, 1, 20, 3)* np.pi
    # # euler = (euler + np.pi) % (2 * np.pi) - np.pi
    # # model = KeypointModel()
    # # rot_mat = model.keypoint_to_rot_mat(euler)
    # # euler_back = model.rot_mat_to_keypoint(rot_mat)
    # rot_mat = pytorch3d.transforms.euler_angles_to_matrix(euler, convention="ZXY")
    # euler_back = pytorch3d.transforms.matrix_to_euler_angles(rot_mat, convention="ZXY")

    # # euler_back = (euler_back + np.pi) % (2 * np.pi) - np.pi

    # print(euler - euler_back)
    # assert torch.allclose(euler, euler_back, atol=1e-4), "euler and euler_back are not close enough"
