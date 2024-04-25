from dhg import DHG
import numpy as np
import os

path = os.path.join(os.getcwd(), "DHGDATA")

# dhg = DHG(path)

def get_test_data(dhg, gesture_name, finger_name, subject_name, essai_name):
    test_data = dhg.gesture_data[gesture_name][finger_name][subject_name][essai_name]
    test_imgs = dhg.img_paths[gesture_name][finger_name][subject_name][essai_name]
    return test_data, test_imgs

def get_frames(dhg, gesture_name, finger_name, subject_name, essai_name):
    test_data, _ = get_test_data(dhg, gesture_name, finger_name, subject_name, essai_name)
    return test_data[2]

def normalize_coordinates(frames):
    x_coordinates = [frame[0] for frame in frames]
    y_coordinates = [frame[1] for frame in frames]
    z_coordinates = [frame[2] for frame in frames]

    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)
    z_min = min(z_coordinates)
    z_max = max(z_coordinates)

    x_norm = [(x - x_min) / (x_max - x_min) for x in x_coordinates]
    y_norm = [(y - y_min) / (y_max - y_min) for y in y_coordinates]
    z_norm = [(z - z_min) / (z_max - z_min) for z in z_coordinates]

    norm_frames = np.array([(x_norm[i], y_norm[i], z_norm[i]) for i in range(len(x_norm))])
    return norm_frames

def compute_direction_vectors(frames, L=1):
    # Compute the direction vector for each point in the sequence except the last L points
    direction_vectors = [frames[i] - frames[i - L] for i in range(L, len(frames))]
    return np.array(direction_vectors)

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



def get_moving_directions(frames, stationary_threshold_ratio=0.8, normalize=True, dimensions=3):
    x_coordinates = [frame[0] for frame in frames]
    y_coordinates = [frame[1] for frame in frames]
    z_coordinates = [frame[2] for frame in frames]

    # min max normalisation
    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)
    z_min = min(z_coordinates)
    z_max = max(z_coordinates)

    x_norm = [(x - x_min) / (x_max - x_min) for x in x_coordinates]
    y_norm = [(y - y_min) / (y_max - y_min) for y in y_coordinates]
    z_norm = [(z - z_min) / (z_max - z_min) for z in z_coordinates]

    norm_frames = np.array([(x_norm[i], y_norm[i], z_norm[i]) for i in range(len(x_norm))])

    frames_to_use = norm_frames if normalize else frames

    # get the avg distance in each frame
    avg_distance = [np.mean([np.linalg.norm(frames_to_use[i] - frames_to_use[i-1]) for i in range(1, len(frames_to_use))])][0]
    direction_vectors = compute_direction_vectors(frames_to_use, L=1)
    moving_directions = [moving_direction(vec, avg_distance=avg_distance, stationary_threshold_ratio=stationary_threshold_ratio, dimensions=dimensions) for vec in direction_vectors]
    thresh = moving_directions[0][1]
    moving_directions = [moving_direction[0] for moving_direction in moving_directions]
    return moving_directions, thresh

def get_directions(moving_directions):
    directions = [list(d.keys())[np.argmax(list(d.values()))] for d in moving_directions]
    # add 'stationary' to the start of the list to account for the first frame
    return ['stationary'] + directions

def get_centroid(frames):
    return np.mean(frames, axis=0)

def make_struct(gestures=(1, 15), fingers=(1, 3), subjects=(1, 21), essais=(1, 6), struct_type=str):

    gestures = [f"gesture_{i}" if struct_type == str else i for i in range(gestures[0], gestures[1])]
    fingers = [f"finger_{i}" if struct_type == str else i for i in range(fingers[0], fingers[1])]
    subjects = [f"subject_{i}" if struct_type == str else i for i in range(subjects[0], subjects[1])]
    essais = [f"essai_{i}" if struct_type == str else i for i in range(essais[0], essais[1])]

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


def get_all_moving_directions(dhg, feature_extractor, best_stationary_threshold = 1.437, indexes = [0, 1, 21, 2], normalize = False, dimensions = 3, filtered=False):
    gestures = [f"gesture_{i}" for i in range(1, 15)]
    fingers = [f"finger_{i}" for i in range(1, 3)]
    subjects = [f"subject_{i}" for i in range(1, 21)]
    essais = [f"essai_{i}" for i in range(1, 6)]

    struct = make_struct()

    for gesture_name in gestures:
        for finger_name in fingers:
            for subject_name in subjects:
                for essai_name in essais:
                    if filtered:
                        frames = get_test_data(dhg, gesture_name, finger_name, subject_name, essai_name)[0]
                    else:
                        frames = get_frames(dhg, gesture_name, finger_name, subject_name, essai_name)
                    frames = frames[:, indexes, :]
                    centroids = [get_centroid(frame) for frame in frames]
                    moving_directions, thresh = get_moving_directions(centroids, stationary_threshold_ratio=best_stationary_threshold, normalize=normalize, dimensions=dimensions)
                    directions = get_directions(moving_directions) 

                    struct[gesture_name][finger_name][subject_name][essai_name] = np.array(feature_extractor.map_directions(directions, dimensions=dimensions))
    
    return struct

