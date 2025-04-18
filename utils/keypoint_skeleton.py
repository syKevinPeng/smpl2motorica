import pickle
import os
from pathlib import Path

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
if not os.path.exists(script_dir / "normalized_skeleton.pkl"):
    raise FileNotFoundError("normalized_skeleton.pkl not found. Please run the script 'normalize_skeleton.py' to generate it.")

def get_keypoint_skeleton():
    with open(script_dir/'normalized_skeleton.pkl', 'rb') as f:
        skeleton = pickle.load(f)
    return skeleton["skeleton"]

def get_keypoint_skeleton_scale():
    with open(script_dir/'normalized_skeleton.pkl', 'rb') as f:
        skeleton = pickle.load(f)
    return skeleton["scale"]
# import numpy as np
# def get_keypoint_skeleton():
#     return {
#         "Hips": {
#             "parent": None,
#             "channels": [
#                 "Xposition",
#                 "Yposition",
#                 "Zposition",
#                 "Zrotation",
#                 "Xrotation",
#                 "Yrotation",
#             ],
#             "offsets": np.array([-0.204624, 0.864926, -0.962418]),
#             "order": "ZXY",
#             "children": ["Spine", "LeftUpLeg", "RightUpLeg"],
#         },
#         "Spine": {
#             "parent": "Hips",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([0.0, 0.0777975, 0.0]),
#             "order": "ZXY",
#             "children": ["Spine1"],
#         },
#         "Spine1": {
#             "parent": "Spine",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([-1.57670e-07, 2.26566e-01, 7.36298e-07]),
#             "order": "ZXY",
#             "children": ["Neck", "LeftShoulder", "RightShoulder"],
#         },
#         "Neck": {
#             "parent": "Spine1",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([0.0, 0.249469, 0.0]),
#             "order": "ZXY",
#             "children": ["Head"],
#         },
#         "Head": {
#             "parent": "Neck",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([0.0, 0.147056, 0.018975]),
#             "order": "ZXY",
#             "children": [],
#             # "children": ["Head_Nub"],
#         },
#         "LeftShoulder": {
#             "parent": "Spine1",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([0.037925, 0.208193, -0.0005065]),
#             "order": "ZXY",
#             "children": ["LeftArm"],
#         },
#         "LeftArm": {
#             "parent": "LeftShoulder",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([1.24818e-01, -1.24636e-07, 0.00000e00]),
#             "order": "ZXY",
#             "children": ["LeftForeArm"],
#         },
#         "LeftForeArm": {
#             "parent": "LeftArm",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([2.87140e-01, 1.34650e-07, 6.52025e-06]),
#             "order": "ZXY",
#             "children": ["LeftHand"],
#         },
#         "LeftHand": {
#             "parent": "LeftForeArm",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([0.234148, 0.00116565, 0.00321146]),
#             "order": "ZXY",
#             "children": [],
#             # "children": [
#             #     "LeftHandThumb1",
#             #     "LeftHandIndex1",
#             #     "LeftHandMiddle1",
#             #     "LeftHandRing1",
#             #     "LeftHandPinky1",
#             # ],
#         },
#         "RightShoulder": {
#             "parent": "Spine1",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([-0.0379391, 0.208193, -0.00050652]),
#             "order": "ZXY",
#             "children": ["RightArm"],
#         },
#         "RightArm": {
#             "parent": "RightShoulder",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([-0.124818, 0.0, 0.0]),
#             "order": "ZXY",
#             "children": ["RightForeArm"],
#         },
#         "RightForeArm": {
#             "parent": "RightArm",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([-2.87140e-01, -3.94596e-07, 1.22370e-06]),
#             "order": "ZXY",
#             "children": ["RightHand"],
#         },
#         "RightHand": {
#             "parent": "RightForeArm",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([-0.237607, 0.00081803, 0.00144663]),
#             "order": "ZXY",
#             "children": [],
#             # "children": [
#             #     "RightHandThumb1",
#             #     "RightHandIndex1",
#             #     "RightHandMiddle1",
#             #     "RightHandRing1",
#             #     "RightHandPinky1",
#             # ],
#         },
#         "LeftUpLeg": {
#             "parent": "Hips",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([0.0948751, 0.0, 0.0]),
#             "order": "ZXY",
#             "children": ["LeftLeg"],
#         },
#         "LeftLeg": {
#             "parent": "LeftUpLeg",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([2.47622e-07, -3.57160e-01, -1.88071e-06]),
#             "order": "ZXY",
#             "children": ["LeftFoot"],
#         },
#         "LeftFoot": {
#             "parent": "LeftLeg",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([0.00057702, -0.408583, 0.00046285]),
#             "order": "ZXY",
#             "children": [],
#             # "children": ["LeftToeBase"],
#         },
#         "RightUpLeg": {
#             "parent": "Hips",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([-0.0948751, 0.0, 0.0]),
#             "order": "ZXY",
#             "children": ["RightLeg"],
#         },
#         "RightLeg": {
#             "parent": "RightUpLeg",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([-2.56302e-07, -3.57160e-01, -2.17293e-06]),
#             "order": "ZXY",
#             "children": ["RightFoot"],
#         },
#         "RightFoot": {
#             "parent": "RightLeg",
#             "channels": ["Zrotation", "Xrotation", "Yrotation"],
#             "offsets": np.array([0.00278006, -0.403849, 0.00049768]),
#             "order": "ZXY",
#             "children": [],
#             # "children": ["RightToeBase"],
#         },
#     }

