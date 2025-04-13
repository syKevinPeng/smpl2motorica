import torch
import sys
sys.path.append("/fs/nexus-projects/PhysicsFall/")
from smpl2keypoint import get_motorica_skeleton_names

def contact_foot_displacement(foot_positions, contact_flags):
    """
    Penalizes foot sliding based on the distance of foot positions from the ground.
    
    Parameters:
    - foot_positions: Tensor of shape (B, T, F, 3), foot positions over T frames for F feet.
    - contact_flags: Tensor of shape (B, T, F), binary flags indicating foot contact with the ground.

    Returns:
    - loss: Scalar loss value.
    """
    displacements = torch.norm(foot_positions[:,1:] - foot_positions[:,:-1], dim=3)
    # Identify frames where the foot is in contact (excluding the last frame)
    contact = contact_flags[:, :-1]
    # Sum displacements during contact frames
    loss = torch.sum(displacements[contact])

    return loss

def foot_sliding_loss(motions_positions, contact_thresh=0.1):
    """
    Computes foot sliding loss for multiple foot positions.
    
    Parameters:
    - motions: List of tensors, each of shape (T, num_joints, 3), representing foot positions over T frames.

    Returns:
    - total_loss: Scalar loss value.
    """
    joint_names = get_motorica_skeleton_names()
    assert motions_positions.ndim == 4, "Input motions_positions must be a 4-dimensional tensor."
    assert motions_positions.shape[-1] == 3, "Joint positions should be 3D coordinates."
    assert motions_positions.shape[2] == len(joint_names), f"Expected {len(joint_names)} joints, but got {motions_positions.shape[2]}."

    foot_names = ['LeftToeBase', 'RightToeBase']
    foot_indices = [joint_names.index(name) for name in foot_names]
    foot_positions = motions_positions[:, :, foot_indices, :]  # Shape: (B, T, F, 3)

    # Assuming Z-axis is vertical
    contact_flags = foot_positions[:, :, :, 2] < contact_thresh  # Shape: (B, T, F)

    total_loss = contact_foot_displacement(foot_positions, contact_flags)


    return total_loss


if __name__ == "__main__":
    # test the foot_sliding_loss function
     # Settings
    B, T = 2, 5  # Batch size, Time steps
    joint_names = get_motorica_skeleton_names()
    num_joints = len(joint_names)

    # Initialize zero motion (no sliding)
    motions = torch.zeros((B, T, num_joints, 3))

    # Set fixed foot positions in contact with ground (Z < 0.05)
    left_toe_idx = joint_names.index('LeftToeBase')
    right_toe_idx = joint_names.index('RightToeBase')

    # Both feet contact ground, but one slides in batch 1
    for b in range(B):
        for t in range(T):
            motions[b, t, left_toe_idx] = torch.tensor([0.0, 0.0, 0.03])
            motions[b, t, right_toe_idx] = torch.tensor([0.0, 0.0, 0.03])

    # Introduce sliding on RightToeBase in batch 1
    for t in range(1, T):
        motions[1, t, right_toe_idx, 0] += t * 0.01  # simulate X sliding

    # Compute and print loss
    loss = foot_sliding_loss(motions)
    print(f"Foot sliding loss: {loss.item():.6f}")




