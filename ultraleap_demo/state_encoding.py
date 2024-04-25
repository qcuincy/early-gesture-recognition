# from feature_tools.moving_direction import *
# from feature_tools.palm_orientation import *
# from feature_tools.siamese_similarity import *

from dhg import DHG
import pickle
import os
import numpy as np


# encoded_states_path = os.path.join(os.getcwd(), "encoded_states")
# DHGPATH = os.path.join(os.getcwd(), "DHGDATA")
# FILTEREDDHGPATH = os.path.join(os.getcwd(), "FILTEREDDHGDATA")

# dhg = DHG(DHGPATH)
# filtered_dhg = DHG(FILTEREDDHGPATH)

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
    for gesture_num in states_dict:
        for finger_num in states_dict[gesture_num]:
            for subject_num in states_dict[gesture_num][finger_num]:
                for essai_num in states_dict[gesture_num][finger_num][subject_num]:
                    states = states_dict[gesture_num][finger_num][subject_num][essai_num]
                    for state in states:
                        if state in state_counts:
                            state_counts[state] += 1
                        else:
                            state_counts[state] = 1

    # sort by value
    state_counts = {k: v for k, v in sorted(state_counts.items(), key=lambda item: item[1], reverse=True)}
    return state_counts

def get_state_counts_subset(dhg, states_dict):
    state_counts = {}
    for gesture_num in states_dict:
        for finger_num in states_dict[gesture_num]:
            for subject_num in states_dict[gesture_num][finger_num]:
                for essai_num in states_dict[gesture_num][finger_num][subject_num]:
                    states = states_dict[gesture_num][finger_num][subject_num][essai_num]
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


def save_states(states, filename, encoded_states_path):
    with open(os.path.join(encoded_states_path, filename), 'wb') as f:
        pickle.dump(states, f)

def load_states(filename, encoded_states_path):
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