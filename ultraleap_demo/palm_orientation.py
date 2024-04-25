import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
from dhg import DHG
import numpy as np
import os

def show_image(img_path, frame=None, title=None):
    if frame is not None:
        img_path = img_path[frame]
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()


def small_difference(normal, thresh=0.1):
    # check if difference between normal components is small enough
    # if so, return True, else return False
    if abs(normal[0] - normal[1]) < thresh or abs(normal[0] - normal[2]) < thresh or abs(normal[1] - normal[2]) < thresh:
        return True
    else:
        return False


def make_orientation_images_v2(gesture_num, finger_num, subject_num, essai_num, indices):
    path = os.path.join(os.getcwd(), "DHGDATA")
    dhg = DHG(path)
    _,_,hand_landmarks = dhg.get_gesture_example(gesture_num, finger_num, subject_num, essai_num)
    img_paths = dhg.get_image_example(gesture_num, finger_num, subject_num, essai_num)

    orientations = []
    for i in range(len(hand_landmarks)):
        hand_landmarks_frame = hand_landmarks[i]

        p1 = hand_landmarks_frame[indices[0]]
        p2 = hand_landmarks_frame[indices[1]]
        p3 = hand_landmarks_frame[indices[2]]

        normal = calculate_normal_vector(p1,p2,p3)
        small_diff = small_difference(normal)
        if small_diff:
            print("small difference")
            orientation = orientations[-1]
        else:
            orientation = determine_palm_orientation(normal)

        orientations.append(orientation)

    orientation_images_path = f"orientation_images-{gesture_num}-{finger_num}-{subject_num}-{essai_num}-{indices}"

    if not os.path.exists(orientation_images_path):
        os.makedirs(orientation_images_path)

    for i in range(len(img_paths)):
        img_path_frame = img_paths[i]
        orientation = orientations[i]
        img = Image.open(img_path_frame)
        plt.imshow(img)
        plt.title(orientation)
        plt.savefig(f"orientation_images-{gesture_num}-{finger_num}-{subject_num}-{essai_num}-{indices}/{i}.png")
        plt.close()


def make_orientation_images(gesture_num, finger_num, subject_num, essai_num, indices):
    path = os.path.join(os.getcwd(), "DHGDATA")
    dhg = DHG(path)
    _,_,hand_landmarks = dhg.get_gesture_example(gesture_num, finger_num, subject_num, essai_num)
    img_paths = dhg.get_image_example(gesture_num, finger_num, subject_num, essai_num)

    orientations = []
    for i in range(len(hand_landmarks)):
        hand_landmarks_frame = hand_landmarks[i]

        p1 = hand_landmarks_frame[indices[0]]
        p2 = hand_landmarks_frame[indices[1]]
        p3 = hand_landmarks_frame[indices[2]]

        normal = normal_vector(p1,p2,p3)
        orientation = classify_normal(normal)

        orientations.append(orientation)

    orientation_images_path = f"orientation_images-{gesture_num}-{finger_num}-{subject_num}-{essai_num}-{indices}"

    if not os.path.exists(orientation_images_path):
        os.makedirs(orientation_images_path)

    for i in range(len(img_paths)):
        img_path_frame = img_paths[i]
        orientation = orientations[i]
        img = Image.open(img_path_frame)
        plt.imshow(img)
        plt.title(orientation)
        plt.savefig(f"orientation_images-{gesture_num}-{finger_num}-{subject_num}-{essai_num}-{indices}/{i}.png")
        plt.close()

def plot_landmarks_with_norm(hand_landmarks, normal_vector):
    # Create the figure
    fig = go.Figure()

    # Add the hand landmarks
    fig.add_trace(go.Scatter3d(
        x=hand_landmarks[:, 0],
        y=hand_landmarks[:, 1],
        z=hand_landmarks[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=hand_landmarks[:, 2],
            colorscale='Viridis',
            opacity=0.8
        )
    ))

    # Add the normal vector
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[1, 0], hand_landmarks[1, 0] + normal_vector[0]],
        y=[hand_landmarks[1, 1], hand_landmarks[1, 1] + normal_vector[1]],
        z=[hand_landmarks[1, 2], hand_landmarks[1, 2] + normal_vector[2]],
        mode='lines',
        line=dict(
            color='red',
            width=10
        ),
        name='Normal vector'
    ))

    # Add the palm position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[1, 0]],
        y=[hand_landmarks[1, 1]],
        z=[hand_landmarks[1, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Palm position'
    ))

    # Add the wrist position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[0, 0]],
        y=[hand_landmarks[0, 1]],
        z=[hand_landmarks[0, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Wrist position'
    ))

    # Add the thumb base position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[2, 0]],
        y=[hand_landmarks[2, 1]],
        z=[hand_landmarks[2, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Thumb base position'
    ))

    # Add the index base position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[6, 0]],
        y=[hand_landmarks[6, 1]],
        z=[hand_landmarks[6, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Index base position'
    ))

    # Add the pink base position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[18, 0]],
        y=[hand_landmarks[18, 1]],
        z=[hand_landmarks[18, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Pinky base position'
    ))

    # Set the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(
                    text='X'
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Y'
                )
            ),
            zaxis=dict(
                title=dict(
                    text='Z'
                )
            )
        )
    )

    # Set the initial camera view
    camera = dict(
        up=dict(
            x=0,
            y=1,
            z=0
        ),
        center=dict(
            x=0,
            y=0,
            z=0
        ),
        eye=dict(
            x=0,
            y=0,
            z=-2
        )
    )

    # Set the camera view
    fig.update_layout(scene_camera=camera)

    # Show the figure
    fig.show()


def plot_landmarks(hand_landmarks):
    # Create the figure
    fig = go.Figure()

    # Add the hand landmarks
    fig.add_trace(go.Scatter3d(
        x=hand_landmarks[:, 0],
        y=hand_landmarks[:, 1],
        z=hand_landmarks[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=hand_landmarks[:, 2],
            colorscale='Viridis',
            opacity=0.8
        )
    ))

    # Add the palm position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[1, 0]],
        y=[hand_landmarks[1, 1]],
        z=[hand_landmarks[1, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Palm'

    ))

    # Add the wrist position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[0, 0]],
        y=[hand_landmarks[0, 1]],
        z=[hand_landmarks[0, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Wrist'
    ))

    # Add the thumb base position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[2, 0]],
        y=[hand_landmarks[2, 1]],
        z=[hand_landmarks[2, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Thumb base'
    ))

    # Add the index base position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[6, 0]],
        y=[hand_landmarks[6, 1]],
        z=[hand_landmarks[6, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Index base'

    ))

    # Add the pink base position
    fig.add_trace(go.Scatter3d(
        x=[hand_landmarks[18, 0]],
        y=[hand_landmarks[18, 1]],
        z=[hand_landmarks[18, 2]],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Pinky base'
    ))

    # Set the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(
                    text='X'
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Y'
                )
            ),
            zaxis=dict(
                title=dict(
                    text='Z'
                )
            )
        )
    )

    # Set the initial camera view
    camera = dict(
        up=dict(
            x=0,
            y=1,
            z=0
        ),
        center=dict(
            x=0,
            y=0,
            z=0
        ),
        eye=dict(
            x=0,
            y=0,
            z=-2
        )
    )

    # Set the camera view
    fig.update_layout(scene_camera=camera)

    # Show the figure
    fig.show()


def normal_vector(p1, p2, p3):
    v1 = p3-p1
    v2 = p2-p3
    n = np.cross(v1,v2)
    return n/np.linalg.norm(n)

def classify_normal(normal):
    max_idx = np.argmax(np.abs(normal))
    if max_idx == 0:
        if normal[0] > 0:
            return "left"
        else:
            return "right"
    elif max_idx == 1:
        if normal[1] > 0:
            return "down"
        else:
            return "up"
    else:
        if normal[2] > 0:
            return "towards"
        else:
            return "away"
        
def calculate_normal_vector(p1,p2,p3):
    # Calculate the vectors that correspond to each joint
    joint_1 = p1
    joint_2 = p2
    joint_3 = p3

    # Calculate two vectors that lie on the palm
    vector_1 = joint_2 - joint_1
    vector_2 = joint_3 - joint_1

    # Calculate the normal vector of the palm
    normal_vector = np.cross(vector_1, vector_2)

    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector
        
def determine_palm_orientation(normal_vector, threshold=0.5):
    # Define the vectors that correspond to each orientation
    orientations_vectors = {
        'Up': np.array([0, 1, 0]),
        'Down': np.array([0, -1, 0]),
        'Left': np.array([-1, 0, 0]),
        'Right': np.array([1, 0, 0]),
        'Towards': np.array([0, 0, -1]),
        'Away': np.array([0, 0, 1])
    }

    # Calculate the dot product of the normal vector with each orientation vector
    dot_products = {orientation: np.dot(normal_vector, vector) for orientation, vector in orientations_vectors.items()}

    # The orientation that has the highest dot product with the normal vector is the orientation of the palm
    max_orientation = max(dot_products, key=dot_products.get)

    # Only assign an orientation if the dot product is above the threshold
    if dot_products[max_orientation] > threshold:
        return max_orientation
    else:
        return 'Uncertain'
