import sys
sys.path.append("../../")
from smpl2motorica.smpl2keypoint import get_motorica_skeleton_names, get_keypoint_skeleton
import pandas as pd


def visualize_keypoint_data(ax, frame: int, df: pd.DataFrame, skeleton = None, skeleton_color = "black"):
    if skeleton is None:
        skeleton = get_keypoint_skeleton()
    joint_names = get_motorica_skeleton_names()
    for idx, joint in enumerate(joint_names):
        # ^ In mocaps, Y is the up-right axis
        parent_x = df[f"{joint}_Xposition"].iloc[frame]
        parent_y = df[f"{joint}_Zposition"].iloc[frame]
        parent_z = df[f"{joint}_Yposition"].iloc[frame]
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
                c=skeleton_color,
            )

        ax.text(
            x=parent_x - 0.01,
            y=parent_y - 0.01,
            z=parent_z - 0.01,
            s=f"{idx}:{joint}",
            fontsize=5,
        )
    

    return ax