import numpy as np
import torch
from sklearn.model_selection import train_test_split
from .training_history import *
from .classes import *
from .prep_functions import *
from .feature_tools import *
from .process_tools import *
from .gesturelstm import *
from .transformer import *
from .objects import *

def split_string(s, convert=False):
    if type(s) == int:
        return [s,s,s]
    if convert:
        return [int(s[0]), int(s[1]), int(s[2:])]
    else:
        return [s[0], s[1], s[2:]]

class EncodedSequence:
    def __init__(self, sequence):
        self.sequence = sequence
        self.moving_direction = sequence[:,0]
        self.palm_orientation = sequence[:,1]
        self.hand_pose = sequence[:,2]

class EncodedSequences:
    def __init__(self, encoded_sequences):
        self.sequences = np.array([sequence.sequence for sequence in encoded_sequences])
        self.moving_direction = np.array([sequence.moving_direction for sequence in encoded_sequences])
        self.palm_orientation = np.array([sequence.palm_orientation for sequence in encoded_sequences])
        self.hand_pose = np.array([sequence.hand_pose for sequence in encoded_sequences])
        self.shape = self.sequences.shape

    def __getitem__(self, key):
        return self.sequences[key]
    
    def __len__(self):
        return len(self.sequences)
    
    def __iter__(self):
        return iter(self.sequences)
    
    def __shape__(self):
        return self.shape
    
    def __repr__(self):
        return self.sequences.__repr__()
    
    def __str__(self):
        return self.sequences.__str__()


def get_performer_states_dict(dhg_to_use, smoother=True):
    states_dict = get_new_states_dict(dhg_to_use, smoother=smoother)
    performers = list(states_dict["gesture_1"]["finger_1"].keys())

    performer_states_dict = { performer: { gesture : [] for gesture in states_dict} for performer in performers}
    for gesture in states_dict:
        for finger in states_dict[gesture]:
            for performer in states_dict[gesture][finger]:
                for essai in states_dict[gesture][finger][performer]:
                    performer_states_dict[performer][gesture].append(EncodedSequence(np.array([split_string(s, True) for s in states_dict[gesture][finger][performer][essai]])))

    return performer_states_dict

def normalize_sequence(sequence, target_length):
    normalized_sequence = []
    sequence_length = len(sequence)
    
    for i in range(target_length):
        index = int((i / target_length) * sequence_length)
        normalized_sequence.append(sequence[index])
    
    return np.array(normalized_sequence)

def normalize_performer_sequences(performer_sequences, target_length):
    normalized_performer_sequences = []
    
    for sequence in performer_sequences:
        normalized_sequence = normalize_sequence(sequence, target_length)
        normalized_performer_sequences.append(normalized_sequence)
    
    return np.array(normalized_performer_sequences)

def get_performer_gesture_average_shapes(performer_states_dict):
    performer_gesture_average_shapes = {subject: {gesture: 0 for gesture in performer_states_dict[subject]} for subject in performer_states_dict}
    for subject in performer_states_dict:
        subject_shapes = []
        for gesture in performer_states_dict[subject]:
            subject_gesture_shapes = []
            for i, sequence in enumerate(performer_states_dict[subject][gesture]):
                subject_gesture_shapes.append(performer_states_dict[subject][gesture][i].sequence.shape[0])

            subject_gesture_average_shapes = np.mean(subject_gesture_shapes)
            subject_shapes.append(subject_gesture_average_shapes)
            performer_gesture_average_shapes[subject][gesture] = round(subject_gesture_average_shapes)

    return performer_gesture_average_shapes

def get_performer_average_shapes(performer_states_dict):
    performer_average_shapes = {subject: 0 for subject in performer_states_dict}
    for subject in performer_states_dict:
        subject_shapes = []
        for gesture in performer_states_dict[subject]:
            for i, sequence in enumerate(performer_states_dict[subject][gesture]):
                subject_shapes.append(performer_states_dict[subject][gesture][i].sequence.shape[0])

        subject_average_shape = np.mean(subject_shapes)
        performer_average_shapes[subject] = round(subject_average_shape)

    return performer_average_shapes
    

def normalize_performer_states_dict_by_gesture(performer_states_dict):
    performer_gesture_average_shapes = get_performer_gesture_average_shapes(performer_states_dict)
    normalized_performer_states_dict = {subject: {gesture: [] for gesture in performer_states_dict[subject]} for subject in performer_states_dict}

    for performer in performer_states_dict:
        for gesture in performer_states_dict[performer]:
            sequences = [performer_states_dict[performer][gesture][i].sequence for i in range(len(performer_states_dict[performer][gesture]))]
            norm_sequences = normalize_performer_sequences(sequences, performer_gesture_average_shapes[performer][gesture])
            normalized_performer_states_dict[performer][gesture] = EncodedSequences([EncodedSequence(sequence) for sequence in norm_sequences])

    return normalized_performer_states_dict

def normalize_performer_states_dict_by_subject(performer_states_dict):
    performer_average_shapes = get_performer_average_shapes(performer_states_dict)
    normalized_performer_states_dict = {subject: {gesture: [] for gesture in performer_states_dict[subject]} for subject in performer_states_dict}

    for performer in performer_states_dict:
        for gesture in performer_states_dict[performer]:
            sequences = [performer_states_dict[performer][gesture][i].sequence for i in range(len(performer_states_dict[performer][gesture]))]
            norm_sequences = normalize_performer_sequences(sequences, performer_average_shapes[performer])
            normalized_performer_states_dict[performer][gesture] = EncodedSequences([EncodedSequence(sequence) for sequence in norm_sequences])

    return normalized_performer_states_dict

def normalize_performer_state(performer_states_dict, by_subject=True):
    if by_subject:
        return normalize_performer_states_dict_by_subject(performer_states_dict)
    else:
        return normalize_performer_states_dict_by_gesture(performer_states_dict)
    
def make_train_test_lookup_table(performer_states_dict, normalized = True, test_size = 0.2, random_seed = 7, gestures=None):
    subjects = list(performer_states_dict.keys())
    train_subjects, test_subjects = train_test_split(subjects, test_size=test_size, random_state=random_seed)

    gestures = gestures if gestures else list(performer_states_dict["subject_1"].keys())

    train_lookup_table = {gesture: [] for gesture in gestures}
    test_lookup_table = {gesture: [] for gesture in gestures}

    for subject in performer_states_dict:
        for gesture in gestures:
            for sequence in performer_states_dict[subject][gesture].sequences if normalized else performer_states_dict[subject][gesture]:
                if subject in train_subjects:
                    if normalized:
                        train_lookup_table[gesture].append(sequence)
                    else:
                        train_lookup_table[gesture].append(sequence.sequence)
                else:
                    if normalized:
                        test_lookup_table[gesture].append(sequence)
                    else:
                        test_lookup_table[gesture].append(sequence.sequence)

    return train_lookup_table, test_lookup_table

def lookup_table_tensor(lookup_table):
    prepared_lookup_table = {gesture:[] for gesture in lookup_table}
    for gesture in lookup_table:
        for sequence in lookup_table[gesture]:
            prepared_lookup_table[gesture].append(torch.tensor(sequence, dtype=torch.long))

    return prepared_lookup_table

def normalize_lookup_table(lookup_table, target_length):
    normalized_lookup_table = {gesture: [] for gesture in lookup_table}
    for gesture in lookup_table:
        for sequence in lookup_table[gesture]:
            normalized_lookup_table[gesture].append(normalize_sequence(sequence, target_length))

        normalized_lookup_table[gesture] = np.array(normalized_lookup_table[gesture])

    return normalized_lookup_table