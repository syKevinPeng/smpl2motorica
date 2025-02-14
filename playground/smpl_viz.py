from pathlib import Path
import numpy as np
import sys, os
import smplx
from tqdm import tqdm
import torch
import vispy
from vispy.scene.visuals import Mesh, Markers, Text
from vispy import scene, app
from vispy.app import use_app


def get_SMPL_skeleton_names():
    return [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ]


def SMPL_output_video(joints, vertices, model):
    for i in tqdm(range(joints.shape[0]), desc="Generating SMPL video"):
        fig = plt.figure(figsize=(10, 10))
        ax = SMPL_visulize_a_frame(fig, joints[i], vertices[i], model)
        ax.set_title(f"SMPL frame {i}")
        plt.savefig(f"smpl_frame_{i:04d}.png")
        plt.close(fig)

    # compile the video
    image_folder = "."
    video_name = "smpl_video.mp4"
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fps = 6  # Set frames per second
    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    # remove the images
    for image in images:
        os.remove(os.path.join(image_folder, image))

    return video_name


def SMPL_visulize_a_frame(fig, joints, vertices, model, output_name="test.png"):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax = fig.add_subplot(111, projection="3d")
    # ax.view_init(elev=0)
    mesh = Poly3DCollection(vertices[model.faces], alpha=0.01)
    face_color = (1.0, 1.0, 0.9)
    edge_color = (0, 0, 0)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color="r")
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    joint_names = get_SMPL_skeleton_names()
    for i, joint in enumerate(joints):
        ax.text(
            joint[0],
            joint[1],
            joint[2],
            f"{i}:{joint_names[i]}",
            color="blue",
            fontsize=5,
        )
    ax.view_init(elev=0, azim=180)
    return ax


if __name__ == "__main__":
    # Load a SMPL model
    amass_data_root = Path("../data/AIST++/gBR_sBM_cAll_d04_mBR0_ch01.pkl")
    smpl_model_path = Path("../smpl/models")
    if not amass_data_root.exists():
        raise FileNotFoundError(f"AMASS data root {amass_data_root} does not exist. ")
    if not smpl_model_path.is_dir():
        raise FileNotFoundError(f"SMPL model path {smpl_model_path} does not exist. ")

    # load the data
    data = np.load(amass_data_root, allow_pickle=True)
    print(data.keys())
    # sample every 10 frames => 6fps
    sample_indices = np.arange(0, data["smpl_poses"].shape[0], 10)
    poses = data["smpl_poses"][sample_indices]
    root_trans = data["smpl_trans"][sample_indices]
    scale = data["smpl_scaling"].reshape(1, 1)
    # create human model
    smpl_model = smplx.create(
        model_path=smpl_model_path,
        model_type="smpl",
        return_verts=True,
        batch_size=len(poses),
    )

    smpl_body_pose = poses[:, 3:]
    smpl_root_rot = poses[:, :3]
    # force the root rotation to be zero
    root_trans = root_trans - root_trans[0]
    # root_trans = np.zeros_like(root_trans)

    smpl_output = smpl_model(
        global_orient=torch.tensor(smpl_root_rot, dtype=torch.float32),
        body_pose=torch.tensor(smpl_body_pose, dtype=torch.float32),
        transl=torch.tensor(root_trans, dtype=torch.float32),
        # global_orient = torch.tensor(smpl_root_rot, dtype=torch.float32),
        scale=torch.tensor(scale, dtype=torch.float32),
    )
    smpl_joints_loc = smpl_output.joints.detach().cpu().numpy().squeeze()
    smpl_vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()

    smpl_joints_loc = smpl_joints_loc[:, : len(get_SMPL_skeleton_names()), :]

    # visualize a frame
    frame_idx = 0
    fig = plt.figure(figsize=(10, 10))
    ax = SMPL_visulize_a_frame(
        fig, smpl_joints_loc[frame_idx], smpl_vertices[frame_idx], smpl_model
    )
