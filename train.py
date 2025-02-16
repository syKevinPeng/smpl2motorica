import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
from smpl2motorica.dataloader import KeypointDataset, SMPLDataset, KeypointDataModule, SMPLDataModule
from smpl2motorica.utils import conti_angle_rep

class RotationTranslationNet(nn.Module):
    def __init__(self):
        super(RotationTranslationNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 128),  # Assuming input is a 3x3 matrix flattened to a vector of size 9
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)  # 6 for continuous rotation representation + 3 for translation
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input matrix to a vector
        return self.fc(x)

class KeypointModel(pl.LightningModule):
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.model = RotationTranslationNet()
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        keypoint_data, smpl_data = batch
        params = self(keypoint_data)
        rotation_matrix = conti_angle_rep.rotation_6d_to_matrix(params[:, :6])
        translation = params[:, 6:]
        transformed_keypoints = self.forward_kinematics(keypoint_data, rotation_matrix, translation)
        loss = self.loss_fn(transformed_keypoints, self.forward_kinematics(smpl_data))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward_kinematics(self, keypoints, rotation_matrix=None, translation=None):
        if rotation_matrix is not None and translation is not None:
            keypoints = torch.matmul(keypoints, rotation_matrix) + translation
        return keypoints

def main():
    data_dir = "/fs/nexus-projects/PhysicsFall/smpl2motorica/data/alignment_dataset"
    batch_size = 32
    num_workers = 4

    keypoint_data_module = KeypointDataModule(data_dir, batch_size, num_workers)
    smpl_data_module = SMPLDataModule(data_dir, batch_size, num_workers)

    model = KeypointModel()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, keypoint_data_module.train_dataloader())

if __name__ == "__main__":
    main()