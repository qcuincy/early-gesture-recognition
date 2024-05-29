from .lookup_table import *
from .training_history import *
from .classes import *
from .prep_functions import *
from .feature_tools import *
from .process_tools import *
from .gesturelstm import *
from .transformer import *
from .objects import *
import numpy as np
import leap
import time
import sys
import pickle
import os




RESULTS_DIR = os.path.join(os.getcwd(), 'ultraleap_demo')
TIME_SERIES_MODELS_DIR = os.path.join(RESULTS_DIR, 'time_series_models')
CLASSIFIER_MODELS_DIR = os.path.join(RESULTS_DIR, 'classifier_models')

CLASSIFIER_MAPPING_DIR = os.path.join(CLASSIFIER_MODELS_DIR, 'mapping')
CLASSIFIER_MAPPING_FILES = [os.path.join(CLASSIFIER_MAPPING_DIR, mapping_file) for mapping_file in os.listdir(CLASSIFIER_MAPPING_DIR)]
CLASSIFIER_MODEL_DIR = os.path.join(CLASSIFIER_MODELS_DIR, 'model')
CLASSIFIER_MODEL_FILES = [os.path.join(CLASSIFIER_MODEL_DIR, model_file) for model_file in os.listdir(CLASSIFIER_MODEL_DIR)]

# Load the classifier mappings
classifier_mappings = {}
for mapping_file in CLASSIFIER_MAPPING_FILES:
    keyname = mapping_file.split("\\")[-1].split(".")[0].split("classifier_")[-1]
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    classifier_mappings[keyname] = mapping

gesture_ntokens = len(classifier_mappings["gesture_state_mapping"])

# Load the classifier models
input_size = 3
num_layers = 4
hidden_size = 128
classifier_models = {}
for model_file in CLASSIFIER_MODEL_FILES:
    keyname = model_file.split("\\")[-1].split(".")[0].split("_best")[0]
    model = GestureLSTM(input_size, hidden_size, num_layers, gesture_ntokens)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    classifier_models[keyname] = model


MULTI_STEP_DIR = os.path.join(TIME_SERIES_MODELS_DIR, 'multi_step')
SINGLE_STEP_DIR = os.path.join(TIME_SERIES_MODELS_DIR, 'single_step')

MULTI_RATIO_4_1 = os.path.join(MULTI_STEP_DIR, '4_1')

MUTLI_RATIO_4_1_16_4 = os.path.join(MULTI_RATIO_4_1, '16_4')
HISTORY_16_4 = os.path.join(MUTLI_RATIO_4_1_16_4, 'history')
MAPPING_16_4 = os.path.join(MUTLI_RATIO_4_1_16_4, 'mapping')
MODEL_16_4 = os.path.join(MUTLI_RATIO_4_1_16_4, 'model')

MUTLI_RATIO_4_1_32_8 = os.path.join(MULTI_RATIO_4_1, '32_8')
HISTORY_32_8 = os.path.join(MUTLI_RATIO_4_1_32_8, 'history')
MAPPING_32_8 = os.path.join(MUTLI_RATIO_4_1_32_8, 'mapping')
MODEL_32_8 = os.path.join(MUTLI_RATIO_4_1_32_8, 'model')

MULTI_RATIO_4_2 = os.path.join(MULTI_STEP_DIR, '4_2')

MUTLI_RATIO_4_2_16_8 = os.path.join(MULTI_RATIO_4_2, '16_8')
HISTORY_16_8 = os.path.join(MUTLI_RATIO_4_2_16_8, 'history')
MAPPING_16_8 = os.path.join(MUTLI_RATIO_4_2_16_8, 'mapping')
MODEL_16_8 = os.path.join(MUTLI_RATIO_4_2_16_8, 'model')

MUTLI_RATIO_4_2_32_16 = os.path.join(MULTI_RATIO_4_2, '32_16')
HISTORY_32_16 = os.path.join(MUTLI_RATIO_4_2_32_16, 'history')
MAPPING_32_16 = os.path.join(MUTLI_RATIO_4_2_32_16, 'mapping')
MODEL_32_16 = os.path.join(MUTLI_RATIO_4_2_32_16, 'model')

MULTI_RATIO_4_3 = os.path.join(MULTI_STEP_DIR, '4_3')

MUTLI_RATIO_4_3_16_12 = os.path.join(MULTI_RATIO_4_3, '16_12')
HISTORY_16_12 = os.path.join(MUTLI_RATIO_4_3_16_12, 'history')
MAPPING_16_12 = os.path.join(MUTLI_RATIO_4_3_16_12, 'mapping')
MODEL_16_12 = os.path.join(MUTLI_RATIO_4_3_16_12, 'model')

MUTLI_RATIO_4_3_32_24 = os.path.join(MULTI_RATIO_4_3, '32_24')
HISTORY_32_24 = os.path.join(MUTLI_RATIO_4_3_32_24, 'history')
MAPPING_32_24 = os.path.join(MUTLI_RATIO_4_3_32_24, 'mapping')
MODEL_32_24 = os.path.join(MUTLI_RATIO_4_3_32_24, 'model')

SINGLE_16_1 = os.path.join(SINGLE_STEP_DIR, '16_1')
HISTORY_16_1 = os.path.join(SINGLE_16_1, 'history')
MAPPING_16_1 = os.path.join(SINGLE_16_1, 'mapping')
MODEL_16_1 = os.path.join(SINGLE_16_1, 'model')

SINGLE_32_1 = os.path.join(SINGLE_STEP_DIR, '32_1')
HISTORY_32_1 = os.path.join(SINGLE_32_1, 'history')
MAPPING_32_1 = os.path.join(SINGLE_32_1, 'mapping')
MODEL_32_1 = os.path.join(SINGLE_32_1, 'model')

HISTORY_DIRS = [HISTORY_16_4, HISTORY_32_8, HISTORY_16_8, HISTORY_32_16, HISTORY_16_12, HISTORY_32_24, HISTORY_16_1, HISTORY_32_1]
HISTORY_FILES = [os.path.join(HISTORY_DIR, history_file) for HISTORY_DIR in HISTORY_DIRS for history_file in os.listdir(HISTORY_DIR)]
MAPPING_DIRS = [MAPPING_16_4, MAPPING_32_8, MAPPING_16_8, MAPPING_32_16, MAPPING_16_12, MAPPING_32_24, MAPPING_16_1, MAPPING_32_1]
MAPPING_FILES = [os.path.join(mapping_dir, mapping_file) for mapping_dir in MAPPING_DIRS for mapping_file in os.listdir(mapping_dir)]
MODEL_DIRS = [MODEL_16_4, MODEL_32_8, MODEL_16_8, MODEL_32_16, MODEL_16_12, MODEL_32_24, MODEL_16_1, MODEL_32_1]
MODEL_FILES = [os.path.join(model_dir, model_file) for model_dir in MODEL_DIRS for model_file in os.listdir(model_dir)]

# Load the history
# histories = {}
# for history_file in HISTORY_FILES:
#     keyname = history_file.split("\\")[-1].split("step_")[-1].split("_batch")[0]
#     with open(history_file, 'rb') as f:
#         history = pickle.load(f)
#     histories[keyname] = history

# Load the mappings
mappings = {}
for mapping_file in MAPPING_FILES:
    keyname = mapping_file.split("\\")[-1].split(".")[0].split("step_")[-1]
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    mappings[keyname] = mapping

mapping_16_1 = {mapping:values for mapping, values in mappings.items() if mapping.endswith("16_1")}
ntokens_16_1 = {mapping.split("_")[0]:len(values) + 1 for mapping, values in mapping_16_1.items() if not mapping.split("_")[0] == "inverted"}
mapping_32_1 = {mapping:values for mapping, values in mappings.items() if mapping.endswith("32_1")}
ntokens_32_1 = {mapping.split("_")[0]:len(values) + 1 for mapping, values in mapping_32_1.items() if not mapping.split("_")[0] == "inverted"}
mapping_16_4 = {mapping:values for mapping, values in mappings.items() if mapping.endswith("16_4")}
ntokens_16_4 = {mapping.split("_")[0]:len(values) + 1 for mapping, values in mapping_16_4.items() if not mapping.split("_")[0] == "inverted"}
mapping_32_8 = {mapping:values for mapping, values in mappings.items() if mapping.endswith("32_8")}
ntokens_32_8 = {mapping.split("_")[0]:len(values) + 1 for mapping, values in mapping_32_8.items() if not mapping.split("_")[0] == "inverted"}
mapping_16_8 = {mapping:values for mapping, values in mappings.items() if mapping.endswith("16_8")}
ntokens_16_8 = {mapping.split("_")[0]:len(values) + 1 for mapping, values in mapping_16_8.items() if not mapping.split("_")[0] == "inverted"}
mapping_32_16 = {mapping:values for mapping, values in mappings.items() if mapping.endswith("32_16")}
ntokens_32_16 = {mapping.split("_")[0]:len(values) + 1 for mapping, values in mapping_32_16.items() if not mapping.split("_")[0] == "inverted"}
mapping_16_12 = {mapping:values for mapping, values in mappings.items() if mapping.endswith("16_12")}
ntokens_16_12 = {mapping.split("_")[0]:len(values) + 1 for mapping, values in mapping_16_12.items() if not mapping.split("_")[0] == "inverted"}
mapping_32_24 = {mapping:values for mapping, values in mappings.items() if mapping.endswith("32_24")}
ntokens_32_24 = {mapping.split("_")[0]:len(values) + 1 for mapping, values in mapping_32_24.items() if not mapping.split("_")[0] == "inverted"}


ntokens = {
    16:{
        1:ntokens_16_1,
        4:ntokens_16_4,
        8:ntokens_16_8,
        12:ntokens_16_12
    },
    32:{
        1:ntokens_32_1,
        8:ntokens_32_8,
        16:ntokens_32_16,
        24:ntokens_32_24
    }
}

handpose_model_dir = os.path.join(RESULTS_DIR, 'handpose_models')
handpose_model_paths = {os.path.join(handpose_model_dir, p).split("\\")[-1].split(".")[0].split("_")[0]:os.path.join(handpose_model_dir, p) for p in os.listdir(handpose_model_dir)}

handpose_unfiltered = HandPose(handpose_model_paths["dhg"])
handpose_filtered = HandPose(handpose_model_paths["filtereddhg"])




model_dict = {
    16: {
        1:None,
        4:None,
        8:None,
        12:None,
    },
    32: {
        1:None,
        8:None,
        16:None,
        24:None,
    }
}

models_dir = os.path.join(os.getcwd(), "ultraleap_demo\\time_series_models")
multi_step_dir = os.path.join(models_dir, "multi_step")
single_step_dir = os.path.join(models_dir, "single_step")


device = torch.device('cpu') # inference on CPU
# Load the models
for ratio in os.listdir(multi_step_dir):
    for model_name in os.listdir(os.path.join(multi_step_dir, ratio)):
        sequence_length, output_window = model_name.split("_")

        model_file = os.listdir(os.path.join(multi_step_dir, ratio, model_name, "model"))[-1]
        feature_ntokens=[len(moving_direction_state_mapping)+1, len(palm_orientation_state_mapping)+1, len(hand_pose_state_mapping)+1]
        if model_name == "32_24":
            feature_ntokens[-1] -= 1
        model = TransformerModel(num_features=3, feature_ntokens=feature_ntokens, d_model=512, nhead=8, num_layers=6, max_len=int(sequence_length)).to(device)
        model.load_state_dict(torch.load(os.path.join(multi_step_dir, ratio, model_name, "model", model_file), map_location=device))
        model.eval()
        model_dict[int(sequence_length)][int(output_window)] = model


for model_name in os.listdir(single_step_dir):
    sequence_length, output_window = model_name.split("_")

    model_file = os.listdir(os.path.join(single_step_dir, model_name, "model"))[-1]
    feature_ntokens=[len(moving_direction_state_mapping)+1, len(palm_orientation_state_mapping)+1, len(hand_pose_state_mapping)+1]
    if model_name == "32_24":
        feature_ntokens[-1] -= 1
    model = TransformerModel(num_features=3, feature_ntokens=feature_ntokens, d_model=512, nhead=8, num_layers=6, max_len=int(sequence_length)).to(device)
    model.load_state_dict(torch.load(os.path.join(single_step_dir, model_name, "model", model_file), map_location=device))
    model.eval()
    model_dict[int(sequence_length)][int(output_window)] = model


def get_gesture_result(frames, sequence_length, output_window, lookup_table, target_length, threshold=0.05, device='cpu'):
    gesture_mapped_names = {
        "gesture_7":"Swipe Right",
        "gesture_8":"Swipe Left",
        "gesture_9":"Swipe Up",
        "gesture_10":"Swipe Down",
    }
    mapped_moving_directions = frames.mapped_moving_directions
    mapped_palm_orientations = frames.mapped_palm_orientations
    mapped_similarity_states = frames.similarity_states

    combined = combine_mapped_separated_sequences([mapped_moving_directions], [mapped_palm_orientations], [mapped_similarity_states])[0]
    to_predict_sequence = make_predict_frame_sequence(combined, sequence_length, output_window)
    model_dict[sequence_length][output_window] = model_dict[sequence_length][output_window].to(device)
    model = model_dict[sequence_length][output_window]
    model.eval()
    to_predict_sequence = to_predict_sequence.to(device)
    
    predicted_states = combine_predicted_features(make_prediction(model, to_predict_sequence, output_window))
    performed_states = to_predict_sequence.tolist()[0][:-output_window]
    classify_sequence = performed_states + predicted_states

    gesture, _, scores = classify_gesture(classify_sequence=classify_sequence, lookup_table=lookup_table, target_length=target_length)

    gesture_scores = [l[1] for l in list(scores.values())]
    # Calculate the sum of all scores
    total_score = sum(gesture_scores)

    # Normalize the scores
    normalized_scores = [score / total_score for score in gesture_scores]

    # Sort the normalized scores in descending order
    sorted_scores = sorted(normalized_scores, reverse=True)

    # Calculate the difference between the best score and the second best score
    score_difference = sorted_scores[0] - sorted_scores[1]
    # print(normalized_scores)

    # Check if the score difference is below the threshold
    if score_difference < threshold:
        classified_gesture = "Unknown"
    else:
        classified_gesture = gesture_mapped_names[gesture]

    return classified_gesture, score_difference