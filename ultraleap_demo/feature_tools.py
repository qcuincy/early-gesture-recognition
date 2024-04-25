from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import numpy as np


# Initialize the scaler
scaler = StandardScaler()

# Palm Orientation
# ----------------------------
def rotate_vector(vector, axis, theta):
    """
    Rotate a 3D vector around the x, y, or z axis by a given angle (theta).
    
    Parameters:
        vector (ndarray): The 3D vector to rotate.
        axis (str): The axis to rotate around ('x', 'y', or 'z').
        theta (float): The angle (in radians) to rotate by.
    
    Returns:
        ndarray: The rotated 3D vector.
    """
    rotation_matrix = np.eye(3)
    
    if axis == 'x':
        rotation_matrix[1, 1] = np.cos(theta)
        rotation_matrix[1, 2] = -np.sin(theta)
        rotation_matrix[2, 1] = np.sin(theta)
        rotation_matrix[2, 2] = np.cos(theta)
    elif axis == 'y':
        rotation_matrix[0, 0] = np.cos(theta)
        rotation_matrix[0, 2] = np.sin(theta)
        rotation_matrix[2, 0] = -np.sin(theta)
        rotation_matrix[2, 2] = np.cos(theta)
    elif axis == 'z':
        rotation_matrix[0, 0] = np.cos(theta)
        rotation_matrix[0, 1] = -np.sin(theta)
        rotation_matrix[1, 0] = np.sin(theta)
        rotation_matrix[1, 1] = np.cos(theta)
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")
    
    rotated_vector = np.dot(rotation_matrix, vector)
    
    return rotated_vector

def get_normal_vector(landmarks):
    # Extract the relevant landmarks (assuming 0-based indexing)
    index_base = np.array(landmarks[6])  # Base of the index finger
    pinky_base = np.array(landmarks[14])  # Base of the pinky finger
    wrist = np.array(landmarks[1])  # Wrist

    # Calculate the vectors between the points
    v1 = index_base - wrist
    v2 = pinky_base - wrist

    # Compute the normal vector of the palm plane
    normal_vector = np.cross(v1, v2)

    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector

def classify_palm_rotation(palm_normal, camera_vector = np.array([0,-1,0]), threshold=0.5):
    # Calculate the dot product between the palm normal and camera vector
    dot_product = np.dot(palm_normal, camera_vector)

    second_dot_product = np.dot(palm_normal, rotate_vector(camera_vector, 'y', np.pi/2))

    third_dot_product = np.dot(palm_normal, rotate_vector(camera_vector, 'y', -np.pi/2))

    condition_1 = second_dot_product > -threshold and second_dot_product < threshold
    condition_2 = third_dot_product > -threshold and third_dot_product < threshold

    product = second_dot_product * third_dot_product

    # Classify the palm rotation based on the dot product value
    if dot_product > threshold:
        if condition_1 and condition_2:
            if product > 0:
                return "down"
            else:
                if (np.abs(product) < 0.1):
                    return "down"
                else:
                    return "up"
        else:
            return "down"
    elif dot_product < -threshold:
        return "up"
    elif second_dot_product > -threshold and second_dot_product < threshold:
        if third_dot_product < threshold:
            return "opposite"
        else:

            return "down"
    else:
        return "opposite"
    


# Moving Direction
# ----------------------------
def moving_direction(vec, avg_distance, stationary_threshold_ratio=0.8, dimensions=3):
    if dimensions == 3:
        return moving_direction_3d(vec, avg_distance, stationary_threshold_ratio)
    else:
        return moving_direction_2d(vec, avg_distance, stationary_threshold_ratio)

def moving_direction_3d(vec, avg_distance, stationary_threshold_ratio=0.8):
    categories = {'up': 0, 'down': 0, 'towards_camera': 0, 'away_from_camera': 0, 'left': 0, 'right': 0, 'stationary': 0}
    

    stationary_threshold = avg_distance * stationary_threshold_ratio
    x, y, z = vec
    if np.linalg.norm(vec) < stationary_threshold:
        categories['stationary'] += 1
    elif abs(x) > abs(y) and abs(x) > abs(z):
        if x > 0:
            categories['right'] += 1
        else:
            categories['left'] += 1
    elif abs(y) > abs(x) and abs(y) > abs(z):
        if y > 0:
            categories['up'] += 1
        else:
            categories['down'] += 1
    elif abs(z) > abs(x) and abs(z) > abs(y):
        if z > 0:
            categories['towards_camera'] += 1
        else:
            categories['away_from_camera'] += 1
    
    return categories, stationary_threshold

def moving_direction_2d(vec, avg_distance, stationary_threshold_ratio=0.8):
    categories = {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'stationary': 0}
    

    stationary_threshold = avg_distance * stationary_threshold_ratio
    x, y, _ = vec
    if np.linalg.norm(vec) < stationary_threshold:
        categories['stationary'] += 1
    elif abs(x) > abs(y):
        if x > 0:
            categories['right'] += 1
        else:
            categories['left'] += 1
    else:
        if y > 0:
            categories['up'] += 1
        else:
            categories['down'] += 1
    
    return categories, stationary_threshold


def get_centroid(frames):
    return np.mean(frames, axis=0)

def moving_directions_sorted(vec, avg_distance, stationary_threshold_ratio=0.8, dimensions=2):

    if dimensions == 3:
        categories = {'up': 0, 'down': 0, 'towards_camera': 0, 'away_from_camera': 0, 'left': 0, 'right': 0, 'stationary': 0}
        

        stationary_threshold = avg_distance * stationary_threshold_ratio
        x, y, z = vec
        
        if x > 0:
            categories['right'] = x
        else:
            categories['left'] = x
        if y > 0:
            categories['up'] = y
        else:
            categories['down'] = y
        if z > 0:
            categories['towards_camera'] = z
        else:
            categories['away_from_camera'] = z
        if np.linalg.norm(vec) < stationary_threshold:
            categories['stationary'] = np.linalg.norm(vec)

        # Get the key, value with the highest magnitude
        
    else:
        categories = {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'stationary': 0}
        stationary_threshold = avg_distance * stationary_threshold_ratio
        x, y, z = vec
        if x > 0:
            categories['right'] = x
        else:
            categories['left'] = x
        if y > 0:
            categories['up'] = y
        else:
            categories['down'] = y
        if np.linalg.norm(vec) < stationary_threshold:
            categories['stationary'] = np.linalg.norm(vec)
    return sorted(categories.items(), key=lambda x: abs(x[1]), reverse=True)


def get_direction(centroid1, centroid2):
    return centroid2 - centroid1

def get_centroids(frames):
    return np.array([get_centroid(frame) for frame in frames])

def get_directions(centroids, L=1):
    directions = [None] * (len(centroids) - 1)
    for i in range(0, len(centroids) - L, L):
        direction = centroids[i + L] - centroids[i]
        for j in range(i, i + L):
            directions[j] = direction
    # Fill the remaining slots with the last calculated direction
    for i in range(len(centroids) - L, len(centroids) - 1):
        directions[i] = direction
    return np.array(directions)

def get_distances(directions):
    return np.array([np.linalg.norm(direction) for direction in directions])

def get_avg_distance(distances):
    return np.mean(distances)

def get_most_common(top_n_frames):
    return Counter(top_n_frames).most_common(1)[0][0]

def get_sorted_direction(sorted_moving_directions, n=2):

    first_direction = sorted_moving_directions[0][0]
    moving_direction_name = first_direction[0]
    
    directions = [moving_direction_name]
    top_n_directions = []
    for i in range(len(sorted_moving_directions)):
        top_n_direction = sorted_moving_directions[i][:n]
        top_n_directions.append(top_n_direction)
        top_n_direction_names = [direction[0] for direction in top_n_direction]
        if moving_direction_name not in top_n_direction_names:
            moving_direction_name = top_n_direction[0][0]
        directions.append(moving_direction_name)
    return np.array(directions), top_n_directions

def get_nonzero_top_n_directions(frames, idxs=None, L = 1, n=5, stationary_threshold_ratio=1.6, dimensions=2, reverse=False, avg_distance=None):
    if idxs is None:
        idxs = [i for i in range(1, 22)]
    frames = frames[::-1] if reverse else frames
    frames = [frame[idxs] for frame in frames]
    centroids = get_centroids(frames)
    direction_vecs = get_directions(centroids)
    distances = get_distances(direction_vecs)
    avg_distance = get_avg_distance(distances) if avg_distance is None else avg_distance
    sorted_moving_directions = [moving_directions_sorted(direction_vec, avg_distance, stationary_threshold_ratio=stationary_threshold_ratio, dimensions=dimensions) for direction_vec in direction_vecs]
    directions, top_n_directions = get_sorted_direction(sorted_moving_directions, n)
    nonzero_top_n_directions = [[d for d in direction if abs(d[1]) > 0] for direction in top_n_directions]
    # add stationary direction to the second element of the list
    return [[('stationary', 0)]] + nonzero_top_n_directions

def scale_distances(distances):
    return scaler.fit_transform(distances)

def get_direction_distances(frame_data):
    distances = {"left":[], "right":[], "up":[], "down":[], "stationary":[]}
    for i in range(len(frame_data)):
        for direction, distance in frame_data[i]:
            distances[direction].append(distance)
    return distances

def scale_frame_data(frame_data):
    distances = get_direction_distances(frame_data)
    left_distances = distances["left"]
    right_distances = distances["right"]
    up_distances = distances["up"]
    down_distances = distances["down"]
    
    scaled_frame_data = []
    for frame in frame_data:
        scaled_frame = []
        for direction, distance in frame:
            if direction == 'stationary':
                scaled_frame.append((direction, 0))
            elif direction == 'left':
                scaled_value = (distance - min(left_distances)) / (max(left_distances) - min(left_distances))
                scaled_frame.append((direction, 1 - scaled_value))
            elif direction == 'right':
                scaled_frame.append((direction, (distance - min(right_distances)) / (max(right_distances) - min(right_distances))))
            elif direction == 'up':
                scaled_frame.append((direction, (distance - min(up_distances)) / (max(up_distances) - min(up_distances))))
            elif direction == 'down':
                scaled_value = (distance - min(down_distances)) / (max(down_distances) - min(down_distances))
                scaled_frame.append((direction, 1 - scaled_value))
        scaled_frame_data.append(scaled_frame)
    return scaled_frame_data

def sort_key(item):
    direction, slope = item
    if direction in ['left', 'down', 'away_from_camera']:
        return -slope  # For 'left' and 'down', a negative slope is considered larger
    else:
        return slope  # For other directions, a positive slope is considered larger
    
def calculate_weights(distances, formula=lambda d: 1 / (d + 1)):
    return [formula(d) for d in distances]

def scale_distances(distances):
    return scaler.fit_transform(distances)


def exponential_decay(d):
    return 2 ** (-d)


def get_motion_directions(frame_data, top_dirs=1, window_size=3, weight_formula=exponential_decay, scaled_frame_data=False, scaled_distances=False):

    if scaled_frame_data:
        frame_data = scale_frame_data(frame_data)

    output = []
    distances = [0 for _ in frame_data]
    distances_history = [dict(left=0, right=0, up=0, down=0, stationary=0) for _ in frame_data]
    weighted_distances_history = [dict(left=0, right=0, up=0, down=0, stationary=0) for _ in frame_data]
    
    for i in range(len(frame_data)):
        frame = frame_data[i]
        # print(i, frame)
        
        if not frame:
            if i == 0:
                output.append('stationary')
            else:
                if len(output) < (window_size//2 + 1):
                    prev_frames = frame_data[max(0, i-window_size//2):i if i > 0 else 1]
                    # print(i, "prev_frames")
                    # print(i, prev_frames)
                else:
                    # use the distances_history to calculate the previous frames
                    prev_distances = distances_history[max(0, i-window_size//2):i].copy()
                    prev_distances = [{k: v for k, v in sorted(prev_distance.items(), key=sort_key, reverse=True)} for prev_distance in prev_distances]
                    prev_frames = [list(prev_distance.items()) for prev_distance in prev_distances]
                
                # prev_frames = frame_data[max(0, i-window_size//2):i]

                prev_frames = [f for f in prev_frames if f]
                        
                next_frames = frame_data[i+1:min(len(frame_data), i+window_size//2+1)]
                next_frames = [f for f in next_frames if f]
                
                prev_distances = range(len(prev_frames), 0, -1)
                next_distances = range(1, len(next_frames) + 1)
                
                prev_weights = calculate_weights(prev_distances, weight_formula)
                next_weights = calculate_weights(next_distances, weight_formula)

                dir_weights = defaultdict(int)
                for prev_frame, weight in zip(prev_frames, prev_weights):
                    for dir, dist in prev_frame[:top_dirs]:
                        dir_weights[dir] += weight * dist
                        distances_history[i][dir] += dist
                        weighted_distances_history[i][dir] += weight * dist
                    
                for next_frame, weight in zip(next_frames, next_weights):
                    for dir, dist in next_frame[:top_dirs]:
                        dir_weights[dir] += weight * dist
                        distances_history[i][dir] += dist
                        weighted_distances_history[i][dir] += weight * dist

                common_dir = max(dir_weights, key=lambda x: abs(dir_weights[x]))
                output.append(common_dir)
                distances.append(dir_weights[common_dir])
                for prev_frame in prev_frames:
                    for dir, dist in prev_frame:
                        if dir == common_dir:
                            distances[i] += dist
                for next_frame in next_frames:
                    for dir, dist in next_frame:
                        if dir == common_dir:
                            distances[i] += dist

        else:
            if frame[0][0] == 'stationary' and len(frame) == 1:
                if len(output) < (window_size//2 + 1):
                    prev_frames = frame_data[max(0, i-window_size//2):i if i > 0 else 1]
                    # print(i, "prev_frames")
                    # print(i, prev_frames)
                else:
                    # use the distances_history to calculate the previous frames
                    prev_distances = distances_history[max(0, i-window_size//2):i].copy()
                    prev_distances = [{k: v for k, v in sorted(prev_distance.items(), key=sort_key, reverse=True)} for prev_distance in prev_distances]
                    prev_frames = [list(prev_distance.items()) for prev_distance in prev_distances]

                # prev_frames = frame_data[max(0, i-window_size//2):i if i > 0 else 1]
                prev_frames = [f for f in prev_frames if f]
                next_frames = frame_data[i+1:min(len(frame_data), i+window_size//2+1)]
                next_frames = [f for f in next_frames if f]
                prev_distances = range(len(prev_frames), 0, -1)
                next_distances = range(1, len(next_frames) + 1)
                
                prev_weights = calculate_weights(prev_distances, weight_formula)
                next_weights = calculate_weights(next_distances, weight_formula)

                dir_weights = defaultdict(int)
                for prev_frame, weight in zip(prev_frames, prev_weights):
                    for dir, dist in prev_frame[:top_dirs]:
                        dir_weights[dir] += weight
                        distances_history[i][dir] += dist
                        weighted_distances_history[i][dir] += weight * dist

                for next_frame, weight in zip(next_frames, next_weights):
                    for dir, dist in next_frame[:top_dirs]:
                        dir_weights[dir] += weight
                        distances_history[i][dir] += dist
                        weighted_distances_history[i][dir] += weight * dist

                common_dir = max(dir_weights, key=lambda x: abs(dir_weights[x]))
                output.append(common_dir)
                for prev_frame in prev_frames:
                    for dir, dist in prev_frame:
                        if dir == common_dir:
                            distances[i] += dist
                for next_frame in next_frames:
                    for dir, dist in next_frame:
                        if dir == common_dir:
                            distances[i] += dist
            else:
                if i == 0 or not output[-1]:
                    output.append(frame[0][0])
                else:
                    if len(output) < (window_size//2 + 1):
                        prev_frames = frame_data[max(0, i-window_size//2):i if i > 0 else 1]
                    else:
                        # use the distances_history to calculate the previous frames
                        prev_distances = distances_history[max(0, i-window_size//2):i].copy()
                        # sort the distances_history by the absolute value of the distance
                        prev_distances = [{k: v for k, v in sorted(prev_distance.items(), key=sort_key, reverse=True)} for prev_distance in prev_distances]
                        prev_frames = [list(prev_distance.items()) for prev_distance in prev_distances]
                    prev_top_dir = output[-1]
                    next_frames = frame_data[i+1:min(len(frame_data), i+window_size//2+1)]
                    # prev_frames = frame_data[max(0, i-window_size//2):i if i > 0 else 1]
                    if prev_top_dir in [dir for dir, _ in frame[:top_dirs]]:
                        output.append(prev_top_dir)
                    elif len(next_frames) > 0 and prev_top_dir in [dir for dir, _ in next_frames[0][:top_dirs]]:
                        output.append(prev_top_dir)
                    else:
                        output.append(frame[0][0])

                    for prev_frame in prev_frames:
                        for dir, dist in prev_frame:
                            distances_history[i][dir] += dist
                            weighted_distances_history[i][dir] += dist
                            if dir == output[-1]:
                                distances[i] += dist
                            
                    for next_frame in next_frames:
                        for dir, dist in next_frame:
                            distances_history[i][dir] += dist
                            weighted_distances_history[i][dir] += dist
                            if dir == output[-1]:
                                distances[i] += dist



    if scaled_distances:
        distances = scale_distances(np.array(distances).reshape(-1, 1)).flatten()

    # Sort each dictionary in each of the history lists by absolute value (descending)
    for i in range(len(distances_history)):
        distances_history[i] = {k: v for k, v in sorted(distances_history[i].items(), key=sort_key, reverse=True)}
        weighted_distances_history[i] = {k: v for k, v in sorted(weighted_distances_history[i].items(), key=sort_key, reverse=True)}
    return output, distances, distances_history, weighted_distances_history


# Siamese Similarity
# ----------------------------
def get_bending_joint_idxs():
    return [
        (0,2,4),(2,4,5), # thumb
        (0,6,8),(6,8,9), # index
        (0,10,12),(10,12,13), # middle
        (0,14,16),(14,16,17), # ring
        (0,18,20),(18,20,21) # little
    ]


def get_bending_angles(jointA, jointB, jointC):
    # Calculate vectors
    AB = np.subtract(jointA, jointB)
    BC = np.subtract(jointC, jointB)

    # Calculate dot product and norms
    dot_product = np.dot(AB, BC)
    norm_ab = np.linalg.norm(AB)
    norm_bc = np.linalg.norm(BC)

    # Calculate angle in radians and make sure no division by zero
    angle_rad = np.arccos(dot_product / (norm_ab * norm_bc + 1e-6))

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# State Encoding
# ----------------------------
import pickle
import os

encoded_states_path = os.path.join(os.getcwd(), "encoded_states")
DHGPATH = os.path.join(os.getcwd(), "DHGDATA")
FILTEREDDHGPATH = os.path.join(os.getcwd(), "FILTEREDDHGDATA")

def make_struct():
    gestures = [f"gesture_{i}" for i in range(1, 15)]
    fingers = [f"finger_{i}" for i in range(1, 3)]
    subjects = [f"subject_{i}" for i in range(1, 21)]
    essais = [f"essai_{i}" for i in range(1, 6)]

    struct = {}
    for gesture in gestures:
        struct[gesture] = {}
        for finger in fingers:
            struct[gesture][finger] = {}
            for subject in subjects:
                struct[gesture][finger][subject] = {}
                for essai in essais:
                    struct[gesture][finger][subject][essai] = []

    return struct


def get_state(moving_directions, palm_orientations, siamese_similarity):
    moving_directions = np.array(moving_directions)[:,np.newaxis]
    palm_orientations = np.array(palm_orientations)[:,np.newaxis]
    siamese_similarity = np.array(siamese_similarity)[:,np.newaxis]

    states = np.concatenate((moving_directions, palm_orientations, siamese_similarity), axis=1)
    states = [list(map(str, state)) for state in states]
    states = ["".join(state) for state in states]
    return states

def get_siamese_similarity_string(multi_digit = True, filtered2 = False):
    siamese_similarity_string = "siamese_similarity"
    if filtered2:
        siamese_similarity_string = "siamese_similarity_48"
    if not multi_digit:
        siamese_similarity_string = siamese_similarity_string + "_1digit"
    return siamese_similarity_string

def make_states_dict(dhg, multi_digit = True, filtered2 = False, use_moving_directions = None):
    siamese_similarity_string = get_siamese_similarity_string(multi_digit, filtered2)
    moving_directions = dhg.clean_features["moving_directions"] if use_moving_directions is None else use_moving_directions
    palm_orientations = dhg.clean_features["palm_orientations"]
    siamese_similarities = dhg.clean_features[siamese_similarity_string]
    states_dict = make_struct()
    
    for gesture_num in range(1, dhg.gesture_num_max+1):
        for finger_num in range(1, dhg.finger_num_max+1):
            for subject_num in range(1, dhg.subject_num_max+1):
                for essai_num in range(1, dhg.essai_num_max+1):
                    moving_direction = moving_directions[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
                    palm_orientation = palm_orientations[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
                    siamese_similarity = siamese_similarities[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
                    
                    states = get_state(moving_direction, palm_orientation, siamese_similarity)

                    states_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}'] = states

    return states_dict

def get_unique_states(dhg, states_dict):
    unique_states = []
    for gesture_num in range(1, dhg.gesture_num_max+1):
        for finger_num in range(1, dhg.finger_num_max+1):
            for subject_num in range(1, dhg.subject_num_max+1):
                for essai_num in range(1, dhg.essai_num_max+1):
                    states = states_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
                    unique_states.extend(states)
    unique_states = list(set(unique_states))
    return unique_states

def get_total_states(dhg, states_dict):
    total_states = 0
    for gesture_num in range(1, dhg.gesture_num_max+1):
        for finger_num in range(1, dhg.finger_num_max+1):
            for subject_num in range(1, dhg.subject_num_max+1):
                for essai_num in range(1, dhg.essai_num_max+1):
                    states = states_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
                    total_states += len(states)
    return total_states

def get_state_counts(dhg, states_dict):
    state_counts = {}
    for gesture_num in range(1, dhg.gesture_num_max+1):
        for finger_num in range(1, dhg.finger_num_max+1):
            for subject_num in range(1, dhg.subject_num_max+1):
                for essai_num in range(1, dhg.essai_num_max+1):
                    states = states_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
                    for state in states:
                        if state in state_counts:
                            state_counts[state] += 1
                        else:
                            state_counts[state] = 1

    # sort by value
    state_counts = {k: v for k, v in sorted(state_counts.items(), key=lambda item: item[1], reverse=True)}
    return state_counts

def get_state_percentage(dhg, states_dict):
    state_counts = get_state_counts(dhg, states_dict)
    total_states = get_total_states(dhg, states_dict)
    state_percentage = {k: v/total_states for k, v in state_counts.items()}
    return state_percentage

def get_top_states(dhg, states_dict, n=10):
    state_counts = get_state_counts(dhg, states_dict)
    top_states = list(state_counts.items())[:n]
    return top_states


def remove_state(dhg, states_dict, state, pct=1, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    # remove pct of state at random (pct = 0 -> remove 0% of state, pct = 1 -> remove 100% of state)
    removed_state_dict = make_struct()
    for gesture_num in range(1, dhg.gesture_num_max+1):
        for finger_num in range(1, dhg.finger_num_max+1):
            for subject_num in range(1, dhg.subject_num_max+1):
                for essai_num in range(1, dhg.essai_num_max+1):
                    states = states_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
                    indexes = np.where(np.array(states) == state)[0]
                    pct_indexes = np.random.choice(indexes, int(len(indexes)*pct), replace=False)
                    state_removed = [state for i, state in enumerate(states) if i not in pct_indexes]
                    removed_state_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}'] = state_removed
    return removed_state_dict


def remove_states(dhg, states_dict, states, pct=1, pcts = None, random_seed=None):
    removed_states_dict = states_dict
    for i, state in enumerate(states):
        if pcts:
            pct = pcts[i]
        removed_states_dict = remove_state(dhg, removed_states_dict, state, pct, random_seed)
    return removed_states_dict

def make_reduced_states_dict(dhg, states, pct=1, pcts = None, multi_digit = True, filtered2 = False, random_seed=None):
    states_dict = make_states_dict(dhg, multi_digit, filtered2)
    removed_states_dict = remove_states(dhg, states_dict, states, pct, pcts, random_seed)
    return states_dict, removed_states_dict


def save_states(states, filename):
    with open(os.path.join(encoded_states_path, filename), 'wb') as f:
        pickle.dump(states, f)

def load_states(filename):
    with open(os.path.join(encoded_states_path, filename), 'rb') as f:
        states = pickle.load(f)
    return states

def get_all_states(dhg, moving_directions, palm_orientations, siamese_similarity):
    states_dict = make_struct()
    for gesture_num in range(1, dhg.gesture_num_max+1):
        for finger_num in range(1, dhg.finger_num_max+1):
            for subject_num in range(1, dhg.subject_num_max+1):
                for essai_num in range(1, dhg.essai_num_max+1):
                    moving_direction_frames = moving_directions[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
                    palm_orientation_frames = palm_orientations[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
                    siamese_similarity_frames = siamese_similarity[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']

                    states = get_state(moving_direction_frames, palm_orientation_frames, siamese_similarity_frames)

                    states_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}'] = states

    return states_dict