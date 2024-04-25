from .classes import *
from . import *
from .process_tools import *
from .transformer import *
from .objects import *
import numpy as np
import leap
import time
import sys
import pickle
from dtw import dtw
import os


state_mapping_dir = os.path.join('ultraleap_demo','state_mapping')

# load the state mapping for each of the features
moving_direction_state_mapping = pickle.load(open(os.path.join(state_mapping_dir, 'moving_direction_mapping.pkl'), 'rb'))
inverted_moving_direction_state_mapping = {v: k for k, v in moving_direction_state_mapping.items()}
palm_orientation_state_mapping = pickle.load(open(os.path.join(state_mapping_dir, 'palm_orientation_mapping.pkl'), 'rb'))
inverted_palm_orientation_state_mapping = {v: k for k, v in palm_orientation_state_mapping.items()}
hand_pose_state_mapping = pickle.load(open(os.path.join(state_mapping_dir, 'hand_pose_mapping.pkl'), 'rb'))
inverted_hand_pose_state_mapping = {v: k for k, v in hand_pose_state_mapping.items()}

sequence_length = 16
output_window = 4
device = 'cpu'


def separate_sequences(frame_sequences):
    moving_direction_sequences = []
    palm_orientation_sequences = []
    hand_pose_sequences = []
    for sequence in frame_sequences:
        moving_direction_sequence = []
        palm_orientation_sequence = []
        hand_pose_sequence = []
        for state in sequence:
            moving_direction_sequence.append(state[0])
            palm_orientation_sequence.append(state[1])
            hand_pose_sequence.append(state[2])

        moving_direction_sequences.append(moving_direction_sequence)
        palm_orientation_sequences.append(palm_orientation_sequence)
        hand_pose_sequences.append(hand_pose_sequence)

    return moving_direction_sequences, palm_orientation_sequences, hand_pose_sequences

def map_sequence(sequence, mapping):
    return [mapping[state] for state in sequence]

def map_sequences(sequences, mapping):
    return [map_sequence(sequence, mapping) for sequence in sequences]

def combine_mapped_separated_sequences(mapped_moving_direction_states, mapped_palm_orientation_states, mapped_hand_pose_states):
    mapped_frame_sequences = []
    for i, (a,b,c) in enumerate(zip(*[mapped_moving_direction_states, mapped_palm_orientation_states, mapped_hand_pose_states])):
        a = a if isinstance(a, list) else [a]
        b = b if isinstance(b, list) else [b]
        c = c if isinstance(c, list) else [c]
        combined_sequence = []
        for j in range(len(a)):
            combined_sequence.append([a[j], b[j], c[j]])
        mapped_frame_sequences.append(combined_sequence)

    return mapped_frame_sequences
    
def make_predict_frame_sequence(mapped_frame_sequence, sequence_length, output_window):
    if len(mapped_frame_sequence) == sequence_length - output_window:
        return torch.tensor([mapped_frame_sequence + [[0, 0, 0]]*output_window]).to(device)
    else:
        if len(mapped_frame_sequence) > sequence_length - output_window:
            return torch.tensor([mapped_frame_sequence[-(sequence_length - output_window):] + [[0, 0, 0]]*output_window]).to(device)
        else:
            return torch.tensor([[[0, 0, 0]]*(sequence_length - len(mapped_frame_sequence) - output_window) + mapped_frame_sequence + [[0, 0, 0]]*output_window]).to(device)
    
def map_prediction(prediction, inverted_state_mapping):
    mapped_prediction = []
    for i in range(prediction.size(0)):
        mapped_prediction.append([inverted_state_mapping[j.item()] for j in prediction[i]])
    return mapped_prediction


inverted_state_mappings = [inverted_moving_direction_state_mapping, inverted_palm_orientation_state_mapping, inverted_hand_pose_state_mapping]

def make_prediction(model, sequence, output_window):
    model.eval()
    with torch.no_grad():
        outputs = model(sequence)
        # Get the predicted values
        predictions = []
        for i, output in enumerate(outputs):
            prediction = torch.max(output, 2)[1].transpose(0, 1)
            predictions.append(prediction.tolist()[0][-output_window:])
        return predictions
    
def combine_predicted_features(predictions):
    combined_predictions = []
    for i in range(len(predictions[0])):
        combined_predictions.append([predictions[j][i] for j in range(len(predictions))])
    return combined_predictions



def classify_gesture(performed_states, predicted_states, lookup_table):
    combined_states = performed_states + predicted_states
    best_gesture = None
    best_score = float("inf")

    gesture_distances = {gesture: float("inf") for gesture in lookup_table.keys()}

    for gesture, sequences in lookup_table.items():

        for sequence in sequences:
            perform_states = sequence['perform']
            # print(perform_states)
            predict_states = sequence['predict']
            lookup_combined_states = perform_states + predict_states
            distance = dtw(combined_states, lookup_combined_states)
            if distance.distance < gesture_distances[gesture]:
                gesture_distances[gesture] = distance.distance
            if distance.distance < best_score:
                best_score = distance.distance
                best_gesture = gesture

    return best_gesture, gesture_distances