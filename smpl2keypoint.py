# This script is used to convert SMPL keypoint to 3D Motorica keypoint format
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import cv2
import smplx
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

from smpl2motorica.utils.data import MocapData

sys.path.append("../")
from smpl2motorica.utils.bvh import BVHParser
from smpl2motorica.utils.pymo.preprocessing import MocapParameterizer

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


def get_motorica_skeleton_names():
    return [
        "Head",
        "Hips",
        "LeftArm",
        "LeftFoot",
        "LeftForeArm",
        "LeftHand",
        "LeftLeg",
        "LeftShoulder",
        # "LeftToeBase",
        "LeftUpLeg",
        "Neck",
        "RightArm",
        "RightFoot",
        "RightForeArm",
        "RightHand",
        "RightLeg",
        "RightShoulder",
        # "RightToeBase",
        "RightUpLeg",
        "Spine",
        "Spine1",
    ]


def smpl2motorica():
    return [
        "head",
        "pelvis",
        "left_shoulder",
        "left_ankle",
        "left_elbow",
        "left_wrist",
        "left_knee",
        "left_collar",
        "left_hip",
        "neck",
        "right_shoulder",
        "right_ankle",
        "right_elbow",
        "right_wrist",
        "right_knee",
        "right_collar",
        "right_hip",
        "spine2",
        "spine3",
    ]


def expand_skeleton(skeleton: list):
    """
    Expands a list of skeleton joints into a list of joint-axis combinations.

    Each joint in the input list is expanded into three elements, one for each
    axis of rotation (X, Y, Z).

    Args:
        skeleton (list): A list of joint names.

    Returns:
        list: A list of joint-axis combinations in the format "{joint}_{axis}rotation".
    """
    expanded_skeleton = [
        f"{joint}_{axis}rotation" for joint in skeleton for axis in ["X", "Y", "Z"]
    ]
    return expanded_skeleton


def motorica_draw_stickfigure3d(
    ax,
    mocap_track,
    frame,
    data=None,
    joints=None,
    draw_names=True,
):
    """
    Draws a 3D stick figure on the given matplotlib 3D axis based on motion capture data.

    Parameters:
    ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to draw the stick figure on.
    mocap_track (object): The motion capture track containing skeleton and values.
    frame (int): The frame number to draw.
    data (pandas.DataFrame, optional): Custom data to use for drawing. Defaults to None, which uses mocap_track values.
    joints (list, optional): List of joints to draw. Defaults to None, which draws all joints in the skeleton.
    draw_names (bool, optional): Whether to draw joint names. Defaults to True.

    Returns:
    matplotlib.axes._subplots.Axes3DSubplot: The axis with the drawn stick figure.
    """

    # ax.view_init(elev=0, azim=120)

    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints

    if data is None:
        df = mocap_track.values
    else:
        df = data

    for idx, joint in enumerate(joints_to_draw):
        # ^ In mocaps, Y is the up-right axis
        parent_x = df["%s_Xposition" % joint][frame]
        parent_y = df["%s_Zposition" % joint][frame]
        parent_z = df["%s_Yposition" % joint][frame]

        # parent_x = df["%s_Xposition" % joint][frame]
        # parent_y = df["%s_Yposition" % joint][frame]
        # parent_z = df["%s_Zposition" % joint][frame]

        ax.scatter(xs=parent_x, ys=parent_y, zs=parent_z, alpha=0.6, c="b", marker="o")

        children_to_draw = [
            c for c in mocap_track.skeleton[joint]["children"] if c in joints_to_draw
        ]

        for c in children_to_draw:
            # ^ In mocaps, Y is the up-right axis
            child_x = df["%s_Xposition" % c][frame]
            child_y = df["%s_Zposition" % c][frame]
            child_z = df["%s_Yposition" % c][frame]

            ax.plot(
                [parent_x, child_x],
                [parent_y, child_y],
                [parent_z, child_z],
                # "k-",
                lw=2,
                c="black",
            )

        if draw_names:
            ax.text(
                x=parent_x - 0.01,
                y=parent_y - 0.01,
                z=parent_z - 0.01,
                s=f"{idx}:{joint}",
                fontsize=5,
            )

    return ax


def SMPL_output_video(joints, vertices, model):
    """
    Generates a video from SMPL model output.

    This function visualizes each frame of the SMPL model output, saves the frames as images,
    compiles them into a video, and then removes the images.

    Args:
        joints (numpy.ndarray): Array of joint positions for each frame.
        vertices (numpy.ndarray): Array of vertex positions for each frame.
        model (object): SMPL model object used for visualization.

    Returns:
        str: The filename of the generated video.
    """
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


def SMPL_visulize_a_frame(ax, joints, vertices, model, output_name="test.png"):
    """
    Visualizes a single frame of SMPL (Skinned Multi-Person Linear) model.

    Parameters:
    ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to plot on.
    joints (numpy.ndarray): Array of joint coordinates with shape (N, 3).
    vertices (numpy.ndarray): Array of vertex coordinates with shape (M, 3).
    model (object): The SMPL model object containing the faces information.
    output_name (str, optional): The name of the output file. Defaults to "test.png".

    Returns:
    matplotlib.axes._subplots.Axes3DSubplot: The axis with the plotted frame.
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    return ax


def skeleton_scaler(skeleton, ratio):
    """
    Scales the offsets of each joint in the skeleton by a given ratio.

    Args:
        skeleton (dict): A dictionary representing the skeleton, where each key is a joint name and the value is another dictionary containing joint properties, including "offsets".
        ratio (float): The scaling factor to apply to the offsets.

    Returns:
        dict: The scaled skeleton with updated offsets.
    """
    for joint in skeleton:
        skeleton[joint]["offsets"] = np.array(skeleton[joint]["offsets"]) * ratio
    return skeleton


def load_dummy_motorica_data() -> MocapData:
    """
    Loads dummy motorica data from a specified BVH file, scales the skeleton, and returns the parsed data.

    The function performs the following steps:
    1. Defines the root directory for the motorica data.
    2. Constructs the path to the specific BVH file.
    3. Checks if the BVH file exists; raises a FileNotFoundError if it does not.
    4. Parses the BVH file using a BVHParser.
    5. Scales the skeleton data by a specified ratio.
    6. Returns the parsed and scaled motorica data.

    Returns:
        motorica_dummy_data: Parsed and scaled motorica data.

    Raises:
        FileNotFoundError: If the specified BVH file does not exist.
    """
    motorica_data_root = Path(
        "/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset"
    )
    motorica_motion_path = (
        motorica_data_root
        / "bvh"
        / "kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.bvh"
    )
    if not motorica_motion_path.exists():
        raise FileNotFoundError(f"Motion file {motorica_motion_path} does not exist.")
    bvh_parser = BVHParser()
    motorica_dummy_data = bvh_parser.parse(motorica_motion_path)
    # scale the skeleton
    ratio = 0.01
    motorica_dummy_data.skeleton = skeleton_scaler(motorica_dummy_data.skeleton, ratio)
    return motorica_dummy_data


def motorica_forward_kinematics(data_df):
    """
    Perform forward kinematics on the given data using the Motorica model.

    This function takes a DataFrame containing kinematic data, applies the Motorica
    forward kinematics model, and returns the transformed position data along with
    the dummy Motorica data.

    Args:
        data_df (pd.DataFrame): A DataFrame containing the input kinematic data.

    Returns:
        tuple: A tuple containing:
            - position_df (np.ndarray): The transformed position data.
            - motorica_dummy_data (pd.DataFrame): The dummy Motorica data with updated values.
    """
    motorica_dummy_data = load_dummy_motorica_data()
    motorica_dummy_data.values = data_df
    position_mocap = MocapParameterizer("position").fit_transform(
        [motorica_dummy_data]
    )[0]
    position_df = position_mocap.values
    return position_df, motorica_dummy_data


if __name__ == "__main__":
    dataset_fps = 60
    target_fps = 30
    debug = False  # debug flag
    aist_data_root = Path("/fs/nexus-projects/PhysicsFall/data/AIST++/motions-SMPL")
    smpl_model_path = Path("/fs/nexus-projects/PhysicsFall/data/smpl/models")
    output_dir = Path("./data/alignment_dataset")

    if not aist_data_root.exists():
        print(f"Please download AIST++ dataset to {aist_data_root}")
        sys.exit(1)
    if not smpl_model_path.exists():
        print(f"Please download SMPL model to {smpl_model_path}")
        sys.exit(1)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # load all aist data *.pkl files
    aist_data = list(aist_data_root.glob("*.pkl"))
    print(f"Found {len(aist_data)} AIST++ data files")
    for data_file in tqdm(aist_data, desc="Processing AIST++ data files"):
        print(f"Processing {data_file}")
        data = np.load(data_file, allow_pickle=True)

        # sample to target fps
        sample_indices = np.arange(
            0, len(data["smpl_poses"]), dataset_fps // target_fps
        )

        # extract pose, root translation ans scale
        poses = data["smpl_poses"][sample_indices]
        root_trans = data["smpl_trans"][sample_indices]
        scales = data["smpl_scaling"]
        # scale the translation
        root_trans = root_trans / scales
        # reset the starting root position to origin
        root_trans -= root_trans[0]

        # get the joints
        smpl_body_pose = poses[:, 3:]
        smpl_root_rot = poses[:, :3]

        # rotate the SMPL pose to Motorica Pose
        smpl_root_rot_quat = R.from_euler("xyz", smpl_root_rot, degrees=False).as_quat()
        rot_offset = R.from_euler("xyz", [np.pi / 2, 0, 0], degrees=False)
        smpl_root_rot_quat = rot_offset * R.from_quat(smpl_root_rot_quat)
        rotated_smpl_root_rot = smpl_root_rot_quat.as_euler("xyz", degrees=False)
        # Rotate the body translation as well
        rotated_root_trans = rot_offset.apply(root_trans)

        smpl_model = smplx.create(
            model_path=smpl_model_path,
            model_type="smpl",
            return_verts=True if debug else False,
            batch_size=len(poses),
        )

        smpl_output = smpl_model(
            body_pose=torch.tensor(smpl_body_pose, dtype=torch.float32),
            transl=torch.tensor(rotated_root_trans, dtype=torch.float32),
            global_orient=torch.tensor(rotated_smpl_root_rot, dtype=torch.float32),
        )

        # get the joints loc
        smpl_joints_loc = smpl_output.joints.detach().cpu().numpy().squeeze()
        if debug:
            smpl_vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()
        # get the SMPL joints (first 24 joints))
        smpl_joints = smpl_joints_loc[:, :24, :]

        # convert SMPL to Motorica Keypoint
        # create a df for smpl joints
        expanded_smpl_joint_names = expand_skeleton(get_SMPL_skeleton_names())
        len(expanded_smpl_joint_names)
        smpl_joints_df = pd.DataFrame(poses, columns=expanded_smpl_joint_names)
        # get in motorica joint order
        motorica_joint_names = expand_skeleton(smpl2motorica())

        # reorder the columns
        keypoint_smpl_df = smpl_joints_df[motorica_joint_names]
        # convert from radian to degree and keep the same order
        keypoint_smpl_df = keypoint_smpl_df.apply(np.rad2deg)
        # rename the columns to motorica joint names
        keypoint_smpl_df.columns = expand_skeleton(get_motorica_skeleton_names())

        # rotate the pelvis
        pelvis = keypoint_smpl_df[
            ["Hips_Xrotation", "Hips_Yrotation", "Hips_Zrotation"]
        ].values
        pelvis_rot = R.from_euler("xyz", pelvis, degrees=True)
        pelvis_rot_offset = R.from_euler("xyz", [0, 180, 0], degrees=True)
        pelvis_rot = pelvis_rot_offset * pelvis_rot
        pelvis_rot_euler = pelvis_rot.as_euler("xyz", degrees=True)
        keypoint_smpl_df[["Hips_Xrotation", "Hips_Yrotation", "Hips_Zrotation"]] = (
            pelvis_rot_euler
        )

        body_rot_name = list(keypoint_smpl_df.columns)
        body_rot_name.remove("Hips_Xrotation")
        body_rot_name.remove("Hips_Yrotation")
        body_rot_name.remove("Hips_Zrotation")

        # Flip left and right
        joint_rot = keypoint_smpl_df[body_rot_name].values
        joint_rot = np.deg2rad(joint_rot)
        num_frame, num_joint, _ = joint_rot.reshape(len(joint_rot), -1, 3).shape
        joint_rot = joint_rot.reshape(-1, 3)

        joint_matrices = R.from_rotvec(joint_rot).as_matrix()

        reflection_matrix = np.array(
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )  # mirror aross Y-Z plane
        # reflection_matrix = np.identity(3)
        reflected_matrices = reflection_matrix @ joint_matrices @ reflection_matrix.T
        reflected_poses = R.from_matrix(reflected_matrices).as_rotvec()
        reflected_poses = reflected_poses.reshape(num_frame, -1)
        # convert back to degree
        reflected_poses = np.rad2deg(reflected_poses)
        keypoint_smpl_df[body_rot_name] = reflected_poses.astype(np.float32)

        # left side joints
        left_side_joints = [
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "LeftLeg",
            "LeftFoot",
            "LeftShoulder",
            "LeftUpLeg",
        ]
        right_side_joints = [
            "RightArm",
            "RightForeArm",
            "RightHand",
            "RightLeg",
            "RightFoot",
            "RightShoulder",
            "RightUpLeg",
        ]
        left_side_joints = expand_skeleton(left_side_joints)
        right_side_joints = expand_skeleton(right_side_joints)
        # swap left and right joints
        keypoint_smpl_df[left_side_joints], keypoint_smpl_df[right_side_joints] = (
            keypoint_smpl_df[right_side_joints],
            keypoint_smpl_df[left_side_joints].values,
        )

        # add root position
        root_pos_df = pd.DataFrame(
            root_trans, columns=["Hips_Xposition", "Hips_Yposition", "Hips_Zposition"]
        )
        new_df = pd.concat([root_pos_df, keypoint_smpl_df], axis=1)

        # save human pose
        motorica_output_file = output_dir / f"{data_file.stem}_motorica.pkl"
        smpl_output_file = output_dir / f"{data_file.stem}_smpl.pkl"
        smpl_data_dict = {
            "smpl_body_pose": smpl_body_pose,
            "smpl_transl": rotated_root_trans,
            "smpl_global_orient": rotated_smpl_root_rot,
            "smpl_joint_loc": smpl_joints,
        }

        with open(motorica_output_file, "wb") as f:
            pickle.dump(new_df, f)

        with open(smpl_output_file, "wb") as f:
            pickle.dump(smpl_data_dict, f)

        # TODO visualize a frame
        if debug:
            # motorica_dummy_data.values = new_df
            # position_mocap = MocapParameterizer("position").fit_transform(
            #     [motorica_dummy_data]
            # )[0]
            # position_df = position_mocap.values
            position_df, motorica_dummy_data = motorica_forward_kinematics(new_df)

            vis_frame = 0
            fig = plt.figure(figsize=(20, 10))
            ax_motorica = fig.add_subplot(121, projection="3d")
            ax_motorica = motorica_draw_stickfigure3d(
                ax_motorica, motorica_dummy_data, vis_frame, data=position_df
            )
            ax_motorica.set_title("Motorica")

            ax_smpl = fig.add_subplot(122, projection="3d")
            ax_smpl = SMPL_visulize_a_frame(
                ax_smpl,
                smpl_joints[vis_frame],
                smpl_vertices[vis_frame],
                smpl_model,
            )
            ax_smpl.set_title("SMPL")

            ax_motorica.view_init(elev=0, azim=120)
            ax_smpl.view_init(elev=0, azim=120)
            plt.show()
