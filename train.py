import matplotlib
import pytorch3d.transforms
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import pandas as pd
import smplx
from pathlib import Path
import pytorch3d
import numpy as np
from matplotlib import pyplot as plt
import wandb
sys.path.append("/fs/nexus-projects/PhysicsFall/")
from smpl2motorica.dataloader import AlignmentDataModule
from smpl2motorica.utils import conti_angle_rep
from smpl2motorica.smpl2keypoint import (
    load_dummy_motorica_data,
    get_SMPL_skeleton_names,
    expand_skeleton,
    get_motorica_skeleton_names,
    smpl_motorica_mapping,
    SMPL_visulize_a_frame,
    motorica_draw_stickfigure3d
)
from smpl2motorica.utils.pymo.preprocessing import MocapParameterizer
from smpl2motorica.utils.pymo.Quaternions import Quaternions
from smpl2motorica.keypoint_fk import ForwardKinematics

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
        self.keypoint_fk = ForwardKinematics()

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
            global_orient=torch.tensor(global_orient, dtype=torch.float32),
            body_pose=torch.tensor(pose, dtype=torch.float32),
            transl=torch.tensor(transl, dtype=torch.float32),
        )
        smpl_joints_loc = smpl_output.joints
        smpl_joints_loc = smpl_joints_loc[:, :24, :] # (batch_size, 24,3)
        
        if return_verts:
            # for visualization
            smpl_verts = smpl_output.vertices.detach().cpu().numpy()
            smpl_joints_loc = smpl_joints_loc.detach().cpu().numpy()
            return smpl_joints_loc, smpl_verts
        # only select the joint we needed
        smpl_joint_names = get_SMPL_skeleton_names()
        selected_joints = list(smpl_motorica_mapping().keys())
        # if smpl_joint_name is in selected_joints, then keep the joint
        smpl_joints_loc = smpl_joints_loc[:, [smpl_joint_names.index(joint) for joint in selected_joints],:].reshape(-1, len(selected_joints)*3)
        return smpl_joints_loc

    def keypoint_to_rot_mat(self, keypoint_data: torch.tensor):
        """
        Converts keypoint data to a combined tensor of position and rotation matrices.
        Args:
            keypoint_data (torch.tensor): A tensor of shape (batch_size, len_of_sequence, 20, 3) 
                                          containing keypoint data. The last dimension should 
                                          represent 3D coordinates.
        Returns:
            torch.tensor: A tensor of shape (batch_size, len_of_sequence, 20, 9) where (batch_size, len_of_sequence, 0, 9) is the position (only first three values are valid)
            and (batch_size, len_of_sequence, 1:20, 9) is the rotation matrix. 
            ValueError: If the input tensor does not have the shape (batch_size, len_of_sequence, 20, 3).
        """

        # input shape (batch_size, len_of_sequence, 20, 3)
        batch_size = keypoint_data.shape[0]
        len_of_sequence = keypoint_data.shape[1]
        if keypoint_data.shape[-1] != 3 or keypoint_data.shape[2] != 20:
            raise ValueError("Keypoint data should be of shape (batch_size, len_of_sequence, 20, 3)")
        position_data = keypoint_data[:, :, 0,:] # (batch_size, len_of_sequence, 3)
        rot_data = keypoint_data[:, :, 1:,:] # (batch_size, len_of_sequence, 19, 3)
        # convert to rotation matrix
        keypoint_rot_mat = pytorch3d.transforms.euler_angles_to_matrix(
            rot_data, convention="ZXY"
        )  # (batch_size,len_of_sequence, 19, 3, 3)
        keypoint_rot_mat = keypoint_rot_mat.view(batch_size, len_of_sequence, 19, 9)
        # pad the translation to match the rotation matrix, from (batch_size, 3) to size(batch_sizexlen_of_sequence, 1, 9).
        keypoint_pos = position_data.unsqueeze(2) # (batch_size, len_of_sequence, 1, 3)
        keypoint_pos = torch.nn.functional.pad(keypoint_pos, (0, 6), "constant", 0)
        keypoint_combined = torch.cat(
            [
                keypoint_pos,
                keypoint_rot_mat,
            ],
            dim=2,
        )
        if keypoint_combined.shape != (batch_size, len_of_sequence, 20, 9):
            raise ValueError(
                "Keypoint data should be of shape (batch_size, len_of_sequence, 20, 9)"
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
        keypoint_rot = keypoint_rot_mat[:, :, 1:, :].view(batch_size,len_of_sequence, -1, 3, 3)
        # convert the rotation matrix to euler angles
        keypoint_rot = pytorch3d.transforms.matrix_to_euler_angles(
            keypoint_rot, convention="ZXY" # TODO: check the convention
        )
        keypoint_euler = torch.cat([keypoint_transl.unsqueeze(2), keypoint_rot], dim=2)
        return keypoint_euler

    def keypoint_forward_kinematics(self, keypoint_data):
        # keypoint data contains the translation and rotation of the keypoints
        # first 3 columns are the translation, and the rest are the rotation that have the same order as motorica2smpl
        # merge the batch dimension with the len_of_sequence dimension
        # input size: (batch_size, len_of_sequence, 20, 3)
        batch_size, len_of_sequence, num_joints = keypoint_data.shape[:3]
        keypoint_data = keypoint_data.view(batch_size*len_of_sequence, num_joints, 3)
        keypoint_position = self.keypoint_fk.forward(keypoint_data.reshape(-1, 60))
        return keypoint_position

    def smpl_dict_to_cuda(self, smpl_dict):
        for key in smpl_dict:
            smpl_dict[key] = smpl_dict[key].to(self.device)
        return smpl_dict

    def training_step(self, batch, batch_idx):
        keypoint_batch, smpl_batch = batch # keypoint_batch size: (batch_size, 1+19, 3)
        keypoint_combined = self.keypoint_to_rot_mat(keypoint_batch).to(
            self.device
        )  # size (batch_size, 1+19, 9)
        smpl_batch = self.smpl_dict_to_cuda(smpl_batch)
        predicted_rot_transl = self.model(
            keypoint_combined
        )  # size (batch_size, 1+19, 9)
        loss = self.loss(
            predicted_rot_transl, keypoint_combined, smpl_batch
        )
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        print("Performing Validation Step")
        print(f'WARNING: FOR DEBUGGING ONLY; MODEL ONLY RETURN IDENTITY MATRIX')
        #for testing only
        keypoint_rot_euler, smpl_batch = batch
        # check if euler is in radian
        # assert torch.max(keypoint_rot_euler) <= 3.14, "keypoint rotation is not in radian"
        # keypoint_rot_euler = torch.rand_like(keypoint_rot_euler)
        keypoint_rot_mat = self.keypoint_to_rot_mat(keypoint_rot_euler).to(
            self.device
        ) #(batch_size, seq_length, 1+19, 9)

        # converted_back = self.rot_mat_to_keypoint(keypoint_rot_mat)
        # assert torch.allclose(keypoint_rot_euler, converted_back, atol=1e-4), "keypoint rotation and converted back keypoint rotation are not close enough"

        smpl_batch = self.smpl_dict_to_cuda(smpl_batch)
        smpl_joint_loc, smpl_vertices = self.smpl_forward_kinematics(smpl_batch, return_verts=True)

        # all ways predict zeros and identity matrix
        predicted_rot_transl = torch.zeros_like(keypoint_rot_mat)
        # for (batch_size, seq_length, 1:20, 9), alwasy return identity matrix
        predicted_rot_transl[:, :, 1:, :] = torch.eye(3).view(1, 1, 1, 9).repeat(
            predicted_rot_transl.shape[0],predicted_rot_transl.shape[1], 19, 1
        )
        adjusted_keypoint = self.apply_prediction_to_keypoint(
            predicted_rot_transl, keypoint_rot_mat
        ) # (batch_size, len_of_sequence, 20, 3)
        keypoint_rot_loc = self.keypoint_forward_kinematics(keypoint_rot_euler)
        adjusted_keypoint_loc = self.keypoint_forward_kinematics(adjusted_keypoint)

        # keypoint_rot_euler should have the same value adjusted_keypoint_loc. Check their values
        # print("keypoint_rot_euler", keypoint_rot_euler[0][0])
        # print("adjusted_keypoint", adjusted_keypoint[0][0])
        # assert torch.allclose(keypoint_rot_euler, adjusted_keypoint, atol=1e-6), "SMPL joint location and adjusted keypoint location are not close enough"

        self.prediction_visualization(smpl_loc = smpl_joint_loc, smpl_vertices = smpl_vertices, keypoint_loc=keypoint_rot_loc, predicted_loc=adjusted_keypoint_loc)
        exit()
        return adjusted_keypoint_loc
    
    
    
    def visualize_keypoint_data(self,ax, frame: int, df: pd.DataFrame, skeleton = None):
        if skeleton is None:
            skeleton = self.keypoint_fk.get_skeleton()
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
    
    # visualize the prediction
    def prediction_visualization(self, smpl_loc,smpl_vertices, keypoint_loc, predicted_loc, frame_to_visualize = 0):
        fig = plt.figure(figsize=(30, 10))
        input_data_pos_df = self.keypoint_fk.convert_to_dataframe(keypoint_loc)
        # show all df columns
        input_loc_ax = fig.add_subplot(132, projection="3d")
        input_loc_ax = self.visualize_keypoint_data(input_loc_ax, frame_to_visualize, input_data_pos_df)
        motorica_dummy_data = load_dummy_motorica_data()
        # motorica_dummy_data.values = keypoint_batch_df
        motorica_dummy_data.skeleton = self.keypoint_fk.get_skeleton()
        
        input_loc_ax.set_title("Input Keypoint")
        input_loc_ax.set_xlabel('X axis')
        input_loc_ax.set_ylabel('Y axis')
        input_loc_ax.set_zlabel('Z axis')
        input_loc_ax.set_box_aspect([1, 1, 1])
        input_loc_ax.set_xlim([-1, 1])
        input_loc_ax.set_ylim([-1, 1])
        input_loc_ax.set_zlim([-1, 1])


        predicted_position = self.keypoint_fk.convert_to_dataframe(predicted_loc)
        predicted_loc_ax = fig.add_subplot(133, projection="3d")
        predicted_loc_ax = self.visualize_keypoint_data(predicted_loc_ax, frame_to_visualize, predicted_position)
        predicted_loc_ax.set_title("Predicted Keypoint")
        predicted_loc_ax.set_xlabel('X axis')
        predicted_loc_ax.set_ylabel('Y axis')
        predicted_loc_ax.set_zlabel('Z axis')
        predicted_loc_ax.set_xlim([-1, 1])
        predicted_loc_ax.set_ylim([-1, 1])
        predicted_loc_ax.set_zlim([-1, 1])

        smpl_ax = fig.add_subplot(131, projection="3d")
        smpl_model = smpl_model = smplx.create(
        model_path=self.smpl_model_path,
        model_type="smpl",
        return_verts=True,
        batch_size=len(keypoint_loc),
        )
        smpl_ax = SMPL_visulize_a_frame(smpl_ax, smpl_loc[frame_to_visualize], smpl_vertices[frame_to_visualize], smpl_model)
        smpl_ax.set_title("SMPL Model")
        smpl_ax.set_xlabel('X axis')
        smpl_ax.set_ylabel('Y axis')
        smpl_ax.set_zlabel('Z axis')
        smpl_ax.set_xlim([-1, 1])
        smpl_ax.set_ylim([-1, 1])
        smpl_ax.set_zlim([-1, 1])


        plt.savefig("predicted_keypoint.png")

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
        predicted_transl = predicted_rot_transl[:,:, 0, :3]
        predicted_rot = predicted_rot_transl[:, :,1:, :].view(
           batch_size, len_of_sequence, 19, 3,3
        )
        # update the keypoint data with the predicted rotation and translation
        keypoint_transl = keypoint_batch[:, :, 0, :3]
        keypoint_rot = keypoint_batch[:, :, 1:, :].view(
            batch_size, len_of_sequence, 19, 3, 3
        )
        adjusted_transl = keypoint_transl + predicted_transl
        # apply the predicted rotation to the keypoint rotation

        adjusted_rot = torch.matmul(predicted_rot, keypoint_rot) # shape: (batch_size, len_of_sequence, 19, 3, 3)
        # assert torch.allclose(adjusted_rot, keypoint_rot, atol=1e-6), "adjusted rotation and original rotation are not close enough"
        # convert the rotation matrix to euler angles
        keypoint_rot = pytorch3d.transforms.matrix_to_euler_angles(
            adjusted_rot, convention="ZXY" # TODO: check the convention
        ) #(batch_size, len_of_sequence, 19, 3)
        # form into the same shape as the original keypoint data
        predicted_keypoint = torch.cat([adjusted_transl.unsqueeze(-2), keypoint_rot], dim=2) # (batch_size, len_of_sequence, 20, 3)
        if predicted_keypoint.shape != (batch_size, len_of_sequence, 20, 3):
            raise ValueError(
                "output should be of shape (batch_size, len_of_sequence, 20, 3)"
            )
        return predicted_keypoint

    def loss(
        self, predicted_rot_transl, keypoint_batch, smpl_batch
    ):
        batch_size = keypoint_batch.shape[0]
        len_of_sequence = keypoint_batch.shape[1]
        adjusted_keypoint = self.apply_prediction_to_keypoint(
            predicted_rot_transl, keypoint_batch
        ) # (batch_size, len_of_sequence, 20, 3)
        adjusted_keypoint_loc = self.keypoint_forward_kinematics(adjusted_keypoint)
        # apply forward kinematics to the smpl data
        smpl_loc = self.smpl_forward_kinematics(smpl_batch)
        # calculate the loss
        loss = self.mse_loss(adjusted_keypoint_loc.reshape(batch_size*len_of_sequence,-1), smpl_loc)
        return loss


def main():
    data_dir = "/fs/nexus-projects/PhysicsFall/smpl2motorica/data/alignment_dataset"
    batch_size = 2
    num_workers = 4
    # torch.autograd.set_detect_anomaly(True)
    data_module = AlignmentDataModule(data_dir, batch_size, num_workers)

    model = KeypointModel()
    wandb_logger = WandbLogger(project="SMPL2Keypoint",
                                          name="test1",
                                          )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_loss',
        dirpath='/fs/nexus-projects/PhysicsFall/smpl2motorica/checkpoints',
        filename='smpl2keypoint-{epoch:02d}-{train_loss:.2f}',
        mode='min',
    )
    trainer = Trainer(max_epochs=500, logger=wandb_logger)
    trainer.fit(model, data_module.train_dataloader())
    # trainer.validate(model, data_module.val_dataloader())


if __name__ == "__main__":
    main()

    # # test euler to rotation matrix and then back to euler
    # euler = torch.rand(2, 1, 20, 3)* np.pi
    # euler = (euler + np.pi) % (2 * np.pi) - np.pi
    # # model = KeypointModel()
    # # rot_mat = model.keypoint_to_rot_mat(euler)
    # # euler_back = model.rot_mat_to_keypoint(rot_mat)
    # rot_mat = pytorch3d.transforms.euler_angles_to_matrix(euler, convention="ZXY")
    # euler_back = pytorch3d.transforms.matrix_to_euler_angles(rot_mat, convention="ZXY")

    # euler_back = (euler_back + np.pi) % (2 * np.pi) - np.pi

    # print(euler - euler_back)
    # assert torch.allclose(euler, euler_back, atol=1e-4), "euler and euler_back are not close enough"
