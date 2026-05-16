import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from enum import IntEnum


class Joint(IntEnum):
    HEAD = 0
    NECK = 1
    PELVIS = 2
    RIGHT_SHOULDER = 3
    RIGHT_ELBOW = 4
    RIGHT_WRIST = 5
    LEFT_SHOULDER = 6
    LEFT_ELBOW = 7
    LEFT_WRIST = 8
    RIGHT_HIP = 9
    RIGHT_KNEE = 10
    RIGHT_ANKLE = 11
    LEFT_HIP = 12
    LEFT_KNEE = 13
    LEFT_ANKLE = 14
    
    
JOINT_CONNECTIONS = [
    (Joint.PELVIS, Joint.NECK),
    (Joint.NECK, Joint.HEAD),
    (Joint.NECK, Joint.RIGHT_SHOULDER),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    (Joint.NECK, Joint.LEFT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.PELVIS, Joint.RIGHT_HIP),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    (Joint.PELVIS, Joint.LEFT_HIP),
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE)
]


OUTPUT_PATH = os.getcwd()


def animate_skeleton_3d(tensor_data, output_filename=None, fps=24, margin=5):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    mins = tensor_data.min(axis=(0, 1))
    maxs = tensor_data.max(axis=(0, 1))
    ax.set_xlim(mins[0] - margin, maxs[0] + margin)
    ax.set_ylim(mins[1] - margin, maxs[1] + margin)
    ax.set_zlim(mins[2] - margin, maxs[2] + margin)

    ax.set_box_aspect([1, 1, 1])
    ax.set_title("3D Stickman Motion Visualization")
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
      
    points_scatter = ax.scatter([], [], [], c='red', s=40, zorder=3)
    lines = [
        ax.plot([], [], [], c='blue', lw=2, zorder=2)[0]
        for _ in range(len(JOINT_CONNECTIONS))
    ]
    
    def init():
        points_scatter._offsets3d = ([], [], [])
        for line in lines:
            line.set_data(np.array([]), np.array([]))
            line.set_3d_properties(np.array([]))
        return [points_scatter] + lines

    def update(frame_idx):
        frame_data = tensor_data[frame_idx]
        
        xs = frame_data[:, 0]
        ys = frame_data[:, 1]
        zs = frame_data[:, 2]
        points_scatter._offsets3d = (xs, ys, zs)
        
        for i, (start_joint, end_joint) in enumerate(JOINT_CONNECTIONS):
            x_coords = np.array(
                [frame_data[start_joint, 0], frame_data[end_joint, 0]]
            )
            y_coords = np.array(
                [frame_data[start_joint, 1], frame_data[end_joint, 1]]
            )
            z_coords = np.array(
                [frame_data[start_joint, 2], frame_data[end_joint, 2]]
            )
            
            lines[i].set_data(x_coords, y_coords)
            lines[i].set_3d_properties(z_coords)
            
        return [points_scatter] + lines

    T = tensor_data.shape[0]
    anim = animation.FuncAnimation(
        fig, update, frames=T, init_func=init, blit=False, interval=1000 / fps
    )
    if output_filename:
        anim.save(
            os.path.join(OUTPUT_PATH, output_filename),
            writer='pillow',
            fps=fps
        )
    plt.show()
    return anim
