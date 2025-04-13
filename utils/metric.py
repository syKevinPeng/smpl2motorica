import torch
from torchmetrics import Metric

# https://github.com/CarstenEpic/humos/blob/a176d31f63de9e872bf7e141c5d5a7fbbc2f9924/humos/src/model/metrics.py#L46
# https://github.com/korrawe/guided-motion-diffusion/blob/2f6264a9b793333556ef911981983082a1113049/data_loaders/humanml/utils/metrics.py#L204


class FootSkateRatio(Metric):
    # The foot skate ratio represents the percentage of frames in which either foot skids more than a specified distance (2.5 cm)
    #  while maintaining contact with the fround (foot height < 5 cm).
    def __init__(self, slide_thresh=0.025, contact_thresh=0.05, device='cpu'):
        super().__init__()
        self.slide_thresh = slide_thresh
        self.contact_thresh = contact_thresh
        self.device_ = device

        self.add_state("skate_frame_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_frame_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, foot_joints: torch.Tensor):
        # check input shape
        if foot_joints.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {foot_joints.ndim}D tensor")
        if foot_joints.shape[2] != 2:
            raise ValueError(f"Expected 2 foot joints, got {foot_joints.shape[2]} joints")
        # foot_joints: [B, T, 2, 3]  (x, y, z)
        B, T, F, _ = foot_joints.shape

        # [B, T-1, 2]: horizontal displacement between adjacent frames
        disp_xy = torch.norm(foot_joints[:, 1:, :, :2] - foot_joints[:, :-1, :, :2], dim=-1)

        # [B, T-1, 2]: both frames must have contact
        contact_prev = foot_joints[:, :-1, :, 2] < self.contact_thresh
        contact_next = foot_joints[:, 1:, :, 2] < self.contact_thresh
        in_contact = torch.logical_and(contact_prev, contact_next)

        # [B, T-1, 2]: detect frames where foot is sliding while grounded
        sliding = torch.logical_and(in_contact, disp_xy > self.slide_thresh)

        # [B, T-1]: any foot sliding in that frame
        skate_frames = sliding.any(dim=-1)

        self.skate_frame_count += skate_frames.sum()
        self.total_frame_count += skate_frames.numel()

    def compute(self):
        if self.total_frame_count == 0:
            return torch.tensor(0.0, device=self.device_)
        return self.skate_frame_count.float() / self.total_frame_count
    


if __name__ == "__main__":
    metric = FootSkateRatio(slide_thresh=0.025, contact_thresh=0.05)

    # Fake test input
    # Shape: [B=1, T=5, Feet=2, Coords=3]
    # Foot 0 skates in frames 1->2 and 3->4, Foot 1 is stationary
    foot_joints = torch.tensor([[
        [[0.00, 0.00, 0.04], [0.00, 0.00, 0.04]],  # t=0
        [[0.03, 0.00, 0.04], [0.00, 0.00, 0.04]],  # t=1 (Foot 0 slides)
        [[0.06, 0.00, 0.04], [0.00, 0.00, 0.04]],  # t=2 (Foot 0 slides again)
        [[0.06, 0.00, 0.04], [0.00, 0.00, 0.04]],  # t=3 (Foot 0 stationary)
        [[0.09, 0.00, 0.04], [0.00, 0.00, 0.04]],  # t=4 (Foot 0 slides)
    ]])

    # Apply update
    metric.update(foot_joints)

    # Compute result
    skate_ratio = metric.compute()
    print(f"Skate Ratio: {skate_ratio.item():.4f}")
