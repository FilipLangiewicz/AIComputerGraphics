import bvhio
import numpy as np

CMU_JOINT_MAP = {
    "HEAD": "Head",
    "NECK": "Neck1",
    "PELVIS": "Hips",
    "RIGHT_SHOULDER": "RightArm",
    "RIGHT_ELBOW": "RightForeArm",
    "RIGHT_WRIST": "RightHand",
    "LEFT_SHOULDER": "LeftArm",
    "LEFT_ELBOW": "LeftForeArm",
    "LEFT_WRIST": "LeftHand",
    "RIGHT_HIP": "RightUpLeg",
    "RIGHT_KNEE": "RightLeg",
    "RIGHT_ANKLE": "RightFoot",
    "LEFT_HIP": "LeftUpLeg",
    "LEFT_KNEE": "LeftLeg",
    "LEFT_ANKLE": "LeftFoot",
}

JOINT_ORDER = [
    "HEAD", "NECK", "PELVIS",
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE",
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
]

def _collect_all_joints(root):
    joints = [root]
    for child in root.Children:
        joints.extend(_collect_all_joints(child))
    return joints

def load_bvh_as_tensor(bvh_path: str) -> np.ndarray:
    """Returns [T, 15, 3] world-space positions."""
    root = bvhio.readAsHierarchy(bvh_path)
    layout = list(root.layout())
    joint_index = {joint.Name: joint for joint, _, _ in layout}

    T = len(root.Keyframes)
    result = np.zeros((T, 15, 3), dtype=np.float32)

    for frame_idx in range(T):
        root.loadPose(frame_idx)
        for j_idx, j_name in enumerate(JOINT_ORDER):
            pos = joint_index[CMU_JOINT_MAP[j_name]].PositionWorld
            result[frame_idx, j_idx] = [pos.x, pos.z, pos.y]  # swap Y and Z
    return result