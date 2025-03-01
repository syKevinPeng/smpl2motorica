# This script is used to convert SMPL keypoint to 3D Motorica keypoint format
from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import cv2
import pytorch3d
import pytorch3d.transforms
import smplx
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
import pytorch3d

sys.path.append("../")
from smpl2motorica.utils.data import MocapData
from smpl2motorica.utils.bvh import BVHParser
from smpl2motorica.utils.pymo.preprocessing import MocapParameterizer
from smpl2motorica.utils.keypoint_skeleton import get_keypoint_skeleton
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


# def get_motorica_skeleton_names():
#     return [
#         "Head",
#         "Hips",
#         "LeftArm",
#         "LeftFoot",
#         "LeftForeArm",
#         "LeftHand",
#         "LeftLeg",
#         "LeftShoulder",
#         # "LeftToeBase",
#         "LeftUpLeg",
#         "Neck",
#         "RightArm",
#         "RightFoot",
#         "RightForeArm",
#         "RightHand",
#         "RightLeg",
#         "RightShoulder",
#         # "RightToeBase",
#         "RightUpLeg",
#         "Spine",
#         "Spine1",
#     ]
def get_motorica_skeleton_names():
    return [
        'Hips', 
        'Spine', 
        'LeftUpLeg', 
        'RightUpLeg', 
        'Spine1', 
        'LeftLeg',
        'RightLeg', 
        'Neck', 
        'LeftShoulder', 
        'RightShoulder', 
        'LeftFoot', 
        'RightFoot', 
        'Head', 
        'LeftArm', 
        'RightArm', 
        'LeftForeArm', 
        'RightForeArm', 
        'LeftHand', 
        'RightHand'
    ]
def smpl_to_motorica_mapping():
    """
    Returns:
        OrderedDict: A mapping of SMPL joints to Motorica joints.
    """
     # ^ In mocaps, Y is the up-right axis
    return OrderedDict(
        [
            ("head", "Head"),
            ("pelvis", "Hips"),
            ("left_shoulder", "LeftArm"),
            ("left_ankle", "LeftFoot"),
            ("left_elbow", "LeftForeArm"),
            ("left_wrist", "LeftHand"),
            ("left_knee", "LeftLeg"),
            ("left_collar", "LeftShoulder"),
            ("left_hip", "LeftUpLeg"),
            ("neck", "Neck"),
            ("right_shoulder", "RightArm"),
            ("right_ankle", "RightFoot"),
            ("right_elbow", "RightForeArm"),
            ("right_wrist", "RightHand"),
            ("right_knee", "RightLeg"),
            ("right_collar", "RightShoulder"),
            ("right_hip", "RightUpLeg"),
            ("spine2", "Spine"),
            ("spine3", "Spine1"),
        ]
    )

def motorica_to_smpl_mapping():
    return OrderedDict(
        [
            ("Hips", "pelvis"),
            ("Spine", "spine2"),
            ("LeftUpLeg", "left_hip"),
            ("RightUpLeg", "right_hip"),
            ("Spine1", "spine3"),
            ("LeftLeg", "left_knee"),
            ("RightLeg", "right_knee"),
            ("Neck", "neck"),
            ("LeftShoulder", "left_collar"),
            ("RightShoulder", "right_collar"),
            ("LeftFoot", "left_ankle"),
            ("RightFoot", "right_ankle"),
            ("Head", "head"),
            ("LeftArm", "left_shoulder"),
            ("RightArm", "right_shoulder"),
            ("LeftForeArm", "left_elbow"),
            ("RightForeArm", "right_elbow"),
            ("LeftHand", "left_wrist"),
            ("RightHand", "right_wrist"),
        ]
    )


def smpl2motorica():
    """
    Reorders the SMPL joints to match the Motorica joints' order.
    Returns:
        list: A list of strings representing the SMPL keypoints corresponding to the Motorica joints' order.
    """
    return smpl_to_motorica_mapping().keys()


def motorica2smpl():
    """
    Reorders the Motorica joints to match the SMPL joints' order.
    Returns:
        list: A list of strings representing the Motorica keypoints corresponding to the SMPL joints' order.
    """
    return smpl_to_motorica_mapping().values()


def expand_skeleton(skeleton: list, order = "XYZ"):
    """
    Expands a list of skeleton joints into a list of joint-axis combinations.

    Each joint in the input list is expanded into three elements, one for each
    axis of rotation (X, Y, Z).

    Args:
        skeleton (list): A list of joint names.

    Returns:
        list: A list of joint-axis combinations in the format "{joint}_{axis}rotation".
    """
    # check if the order is valid
    if len(order) != 3:
        raise ValueError("The order must be a string of length 3")
    if not all([axis in "XYZ" for axis in order]):
        raise ValueError("The order must contain only 'X', 'Y', and 'Z'")
    expanded_skeleton = [
        f"{joint}_{axis}rotation" for joint in skeleton for axis in order
    ]
    return expanded_skeleton

def get_smpl_pelvis_offset(global_trans, global_rot, joint_rot, smpl_model_path = "/fs/nexus-projects/PhysicsFall/data/smpl/models"):
    smpl_model = smplx.create(
        model_path=smpl_model_path,
        model_type="smpl",
        return_verts=False,
        batch_size=len(global_trans),
    )
    smpl_output = smpl_model(
        body_pose=torch.tensor(joint_rot, dtype=torch.float32),
        global_orient=torch.tensor(global_rot, dtype=torch.float32),
        transl=torch.tensor(global_trans, dtype=torch.float32),
    )
    smpl_joints_loc = smpl_output.joints.detach().cpu().numpy().squeeze()
    smpl_joints_loc = smpl_joints_loc[:, : len(get_SMPL_skeleton_names()), :]
    pelvis_loc = smpl_joints_loc[:, 0, :]
    return pelvis_loc
    

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
        parent_x = df[f"{joint}_Xposition"].iloc[frame]
        parent_y = df[f"{joint}_Zposition"].iloc[frame]
        parent_z = df[f"{joint}_Yposition"].iloc[frame]
        ax.scatter(xs=parent_x, ys=parent_y, zs=parent_z, alpha=0.6, c="b", marker="o")
        children_to_draw = [
            c for c in mocap_track.skeleton[joint]["children"] if c in joints_to_draw
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
    # filter out the joint that we don't need
    motorica_dummy_data.skeleton = {
        k: v for k, v in motorica_dummy_data.skeleton.items() if k in get_motorica_skeleton_names()
    }
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
    motorica_dummy_data.skeleton = get_keypoint_skeleton()
    position_mocap = MocapParameterizer("position").fit_transform(
        [motorica_dummy_data]
    )[0]
    position_df = position_mocap.values
    return position_df, motorica_dummy_data

def visualize_keypoint_data(ax, frame: int, df: pd.DataFrame, skeleton = None):
    if skeleton is None:
        skeleton = get_keypoint_skeleton()
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

def save_video(motorica_df, smpl_in_motorica_df, output_file_name = 'sample_motorica_vid.mp4'):
    image_folder = 'tmp'
    os.makedirs(image_folder, exist_ok=True)
    
    for frame in tqdm(range(len(motorica_df)), desc="Generating Videos"):
        fig = plt.figure(figsize=(10, 20))
        motorica_ax = fig.add_subplot(121, projection='3d')
        motorica_ax = visualize_keypoint_data(motorica_ax, frame, motorica_df )
        motorica_ax.set_xlabel('X axis')
        motorica_ax.set_ylabel('Y axis')
        motorica_ax.set_zlabel('Z axis')
        motorica_ax.set_zlim([-1, 1])
        motorica_ax.set_xlim([-1, 1])
        motorica_ax.set_ylim([-1, 1])
        motorica_ax.set_title(f"motorica: frame {frame}")

        smpl_in_motorica_ax = fig.add_subplot(122, projection='3d')
        smpl_in_motorica_ax = visualize_keypoint_data(smpl_in_motorica_ax, frame, smpl_in_motorica_df)
        smpl_in_motorica_ax.set_xlabel('X axis')
        smpl_in_motorica_ax.set_ylabel('Y axis')
        smpl_in_motorica_ax.set_zlabel('Z axis')
        smpl_in_motorica_ax.set_zlim([-1, 1])
        smpl_in_motorica_ax.set_xlim([-1, 1])
        smpl_in_motorica_ax.set_ylim([-1, 1])
        smpl_in_motorica_ax.set_title(f"smpl_in_motorica frame {frame}")
        motorica_ax.view_init(elev=0, azim=90)
        smpl_in_motorica_ax.view_init(elev=0, azim=90)
        plt.savefig(f"{image_folder}/frame_{frame:04d}.png")
        plt.close(fig)
    
    # compile the video
    video_name = output_file_name
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fps = target_fps  # Set frames per second
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
    # remove the images and directory
    for image in images:
        os.remove(os.path.join(image_folder, image))
    os.rmdir(image_folder)
    
    return video_name



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
        # apply pelvis offset
        # pelvis_offset = get_smpl_pelvis_offset(global_trans=root_trans, global_rot=smpl_root_rot, joint_rot=smpl_body_pose)
        # root_trans = root_trans - pelvis_offset

        smpl_root_rot = torch.tensor(smpl_root_rot, dtype=torch.float32)
        smpl_body_pose = torch.tensor(smpl_body_pose, dtype=torch.float32)
        smpl_root_trans = torch.tensor(root_trans, dtype=torch.float32)

        # rotate the SMPL pose to Motorica Pose
        smpl_root_rot = R.from_euler("xyz", smpl_root_rot, degrees=False)
        rot_offset = R.from_euler("z",  -np.pi / 2, degrees=False)
        additiona_rot_offset = R.from_euler("x", -np.pi / 2 , degrees=False)
        smpl_root_rot = additiona_rot_offset * rot_offset * smpl_root_rot
        rotated_smpl_root_rot = smpl_root_rot.as_rotvec()
        rotated_smpl_root_rot = torch.tensor(rotated_smpl_root_rot, dtype=torch.float32)
        # # Rotate the body translation as well
        rotated_root_trans = smpl_root_rot.apply(root_trans)
        # rotated_root_trans += pelvis_offset
        rotated_root_trans = torch.tensor(rotated_root_trans, dtype=torch.float32)


        smpl_model = smplx.create(
            model_path=smpl_model_path,
            model_type="smpl",
            return_verts=True if debug else False,
            batch_size=len(poses),
        )

        smpl_output = smpl_model(
            body_pose=smpl_body_pose,
            transl=rotated_root_trans,
            global_orient=rotated_smpl_root_rot,
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
        smpl_joints_df = pd.DataFrame(poses, columns=expanded_smpl_joint_names)
        # get in motorica joint order
        motorica_joint_names = expand_skeleton(list(motorica_to_smpl_mapping().values()))

        # reorder the columns
        keypoint_smpl_df = smpl_joints_df[motorica_joint_names]
        # convert from radian to degree and keep the same order
        keypoint_smpl_df = keypoint_smpl_df.apply(np.rad2deg)
        # rename the columns to motorica joint names
        keypoint_smpl_df.columns = expand_skeleton(get_motorica_skeleton_names())

        # add root position
        root_pose_df = pd.DataFrame(root_trans, columns=["Hips_Xposition", "Hips_Yposition", "Hips_Zposition"])
        keypoint_smpl_df = pd.concat([root_pose_df, keypoint_smpl_df], axis=1)


        # save human pose
        motorica_output_file = output_dir / f"{data_file.stem}_motorica.pkl"
        smpl_output_file = output_dir / f"{data_file.stem}_smpl.pkl"
        smpl_data_dict = {
            "smpl_body_pose": smpl_body_pose,
            "smpl_transl": rotated_root_trans,
            "smpl_global_orient": rotated_smpl_root_rot,
            "smpl_joint_loc": smpl_joints,
        }
        if not debug:
            with open(motorica_output_file, "wb") as f:
                pickle.dump(keypoint_smpl_df, f)

            with open(smpl_output_file, "wb") as f:
                pickle.dump(smpl_data_dict, f)

        # TODO visualize a frame
        if debug:

            position_df, motorica_dummy_data = motorica_forward_kinematics(keypoint_smpl_df)

            vis_frame = 0
            fig = plt.figure(figsize=(30, 10))
            ax_motorica = fig.add_subplot(131, projection="3d")
            ax_motorica = motorica_draw_stickfigure3d(
                ax_motorica, motorica_dummy_data, vis_frame, data=position_df
            )
            ax_motorica.set_title("Motorica")
            ax_motorica.set_xlim(-1, 1)
            ax_motorica.set_ylim(-1, 1)
            ax_motorica.set_zlim(-1, 1)
            ax_motorica.set_xlabel("X")
            ax_motorica.set_ylabel("Y")
            ax_motorica.set_zlabel("Z")
            ax_motorica.scatter(0, 0, 0, c="red", s=10, marker="o")

            ax_smpl = fig.add_subplot(132, projection="3d")
            ax_smpl = SMPL_visulize_a_frame(
                ax_smpl,
                smpl_joints[vis_frame],
                smpl_vertices[vis_frame],
                smpl_model,
            )
            ax_smpl.set_title("SMPL")
            ax_smpl.set_xlabel("X")
            ax_smpl.set_ylabel("Y")
            ax_smpl.set_zlabel("Z")
            ax_smpl.set_xlim(-1, 1)
            ax_smpl.set_ylim(-1, 1)
            ax_smpl.set_zlim(-1, 1)



            # for debugging:
            
            def convert_to_dataframe(positions):
                """Convert output tensor back to DataFrame format"""
                """position shape: (num_frames, num_joints (19), 3)"""
                columns = []
                data = {}

                positions = positions.detach().cpu().numpy()
                # check position shape
                assert positions.shape[1] == 19, f"Expected position shape to be (num_frames, num_joints (19)), got {positions.shape}"

                for j, joint in enumerate(get_motorica_skeleton_names()):
                    pos = positions[:, j]
                    data[f"{joint}_Xposition"] = pos[:, 0]
                    data[f"{joint}_Yposition"] = pos[:, 1]
                    data[f"{joint}_Zposition"] = pos[:, 2]
                return pd.DataFrame(data)
        
            smpl_joint_names = get_SMPL_skeleton_names()
            smpl_joints_loc = smpl_joints_loc[:, [smpl_joint_names.index(joint) for joint in motorica_to_smpl_mapping().values()],:]
                # swap from XYZ to ZXY
            smpl_joints_loc = smpl_joints_loc[:, :, [2, 0, 1]]
            smpl_loc_df = convert_to_dataframe(positions=torch.tensor(smpl_joints_loc.reshape(-1, 19, 3)))
            smpl_loc_ax = fig.add_subplot(133, projection="3d")
            smpl_loc_ax = visualize_keypoint_data(smpl_loc_ax, 0, smpl_loc_df)
            smpl_loc_ax.set_title("SMPL data in keypoint format")
            smpl_loc_ax.set_xlabel("X")
            smpl_loc_ax.set_ylabel("Y")
            smpl_loc_ax.set_zlabel("Z")

            smpl_loc_ax.set_xlim(-1,1)
            smpl_loc_ax.set_ylim(-1, 1)
            smpl_loc_ax.set_zlim(-1, 1)
            

            ax_motorica.view_init(elev=0, azim=90)
            ax_smpl.view_init(elev=0, azim=90)
            smpl_loc_ax.view_init(elev=0, azim=90)
            # fig.savefig(f"smpl_frame_{vis_frame:04d}.png")
            save_video(position_df, smpl_loc_df, output_file_name=f"{data_file.stem}_motorica.mp4")
            exit()
