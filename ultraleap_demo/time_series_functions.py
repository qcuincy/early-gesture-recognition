from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import itertools
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import time
from IPython.display import clear_output
import plotly.graph_objects as go
from .classes import *
from .process_tools import *
from .gesturelstm import *
from .transformer import *
from .objects import *
from . import *



class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def get_unique_states(dhg, states_dict):
    unique_states = []
    for gesture_num in states_dict:
        for finger_num in states_dict[gesture_num]:
            for subject_num in states_dict[gesture_num][finger_num]:
                for essai_num in states_dict[gesture_num][finger_num][subject_num]:
                    states = states_dict[gesture_num][finger_num][subject_num][essai_num]
                    unique_states.extend(states)
    unique_states = list(set(unique_states))
    return unique_states

def get_num_states(dhg, states_dict):
    unique_states = get_unique_states(dhg, states_dict)
    return len(unique_states)
    
def get_data(encoded_states_dict):
    gestures = [f"gesture_{i}" for i in range(1, 15)]
    fingers = [f"finger_{i}" for i in range(1, 3)]
    subjects = [f"subject_{i}" for i in range(1, 21)]
    essais = [f"essai_{i}" for i in range(1, 6)]

    data = []
    for gesture in gestures:
        for finger in fingers:
            for subject in subjects:
                for essai in essais:
                    data.append(encoded_states_dict[gesture][finger][subject][essai])

    return data


def create_state_mapping(unique_states, pad_zero=True):
    mapping = {state: i for i, state in enumerate(unique_states)}
    if not pad_zero:
        return mapping
    else:
        # Add 1 to each index to avoid 0 as a state
        return {state: i+1 for state, i in mapping.items()}

def invert_state_mapping(state_mapping):
    mapping = {i: state for state, i in state_mapping.items()}
    return mapping

def save_mapping(mapping, filename):
    with open(filename, 'wb') as f:
        pickle.dump(mapping, f)

def load_mapping(filename):
    with open(filename, 'rb') as f:
        mapping = pickle.load(f)
    return mapping

def get_ranges():
    gestures = [f"gesture_{i}" for i in range(1, 15)]
    fingers = [f"finger_{i}" for i in range(1, 3)]
    subjects = [f"subject_{i}" for i in range(1, 21)]
    essais = [f"essai_{i}" for i in range(1, 6)]
    return gestures, fingers, subjects, essais

def get_combinations_dict():
    gestures, fingers, subjects, essais = get_ranges()

    # get the combinations for each gesture in a dictionary
    combinations_dict = {gesture: {} for gesture in gestures}

    # get the combinations for each finger in a dictionary
    for gesture in combinations_dict:
        combinations_dict[gesture] = {finger: list(itertools.product(subjects, essais)) for finger in fingers}

    return combinations_dict

def remove_duplicates(sequences, labels):
    sequences = [tuple(map(int, seq.tolist())) for seq in sequences]
    unique_data = dict(zip(sequences, labels))

    # Convert back to tensors
    sequences, labels = zip(*unique_data.items())
    sequences = torch.tensor(sequences, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences, labels


def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


def create_state_mappings(dhg, encoded_states, pad_zero=True):
    unique_states = get_unique_states(dhg, encoded_states)
    state_mapping = create_state_mapping(unique_states, pad_zero=pad_zero)
    inverted_state_mapping = invert_state_mapping(state_mapping)
    ntokens = len(state_mapping) if not pad_zero else len(state_mapping) + 1
    return state_mapping, inverted_state_mapping, ntokens


def reduce_state(states, pct, side='center', state=None):
    reduced_states = states.copy()
    if state is not None:
        if state in states:
            total_to_remove = int(pct * reduced_states.count(state))
            if side == 'left':
                for _ in range(total_to_remove):
                    reduced_states.remove(state)
            elif side == 'right':
                for _ in range(total_to_remove):
                    reduced_states.reverse()
                    reduced_states.remove(state)
                    reduced_states.reverse()
            elif side == 'center':
                for _ in range(total_to_remove):
                    reduced_states.remove(state)
                    reduced_states.reverse()
                if total_to_remove % 2 != 0:
                    reduced_states.reverse()
            elif side == 'random':
                indexes = np.where(np.array(reduced_states) == state)[0]
                pct_indexes = np.random.choice(indexes, total_to_remove, replace=False)
                reduced_states = [_state for i, _state in enumerate(reduced_states) if i not in pct_indexes]

            else:
                raise ValueError("Invalid side. Choose from 'left', 'right', 'center' or 'random'.")
        else:
            return reduced_states
    else:
        total_to_remove = int(pct * len(reduced_states))
        if side == 'left':
            reduced_states = states[total_to_remove:]
        elif side == 'right':
            reduced_states = states[:-total_to_remove]
        elif side == 'center':
            remove_each_side = total_to_remove // 2
            if total_to_remove % 2 != 0:
                remove_each_side += 1
            reduced_states = states[remove_each_side:-remove_each_side]
        else:
            raise ValueError("Invalid side. Choose from 'left', 'right', or 'center'.")
    return reduced_states

def remove_state(dhg, states_dict, state, pct=1, side='center', random_seed=7):
    np.random.seed(random_seed)
    
    removed_state_dict = make_struct()
    for gesture_num in states_dict:
        for finger_num in states_dict[gesture_num]:
            for subject_num in states_dict[gesture_num][finger_num]:
                for essai_num in states_dict[gesture_num][finger_num][subject_num]:
                    states = states_dict[gesture_num][finger_num][subject_num][essai_num]
                    state_removed = reduce_state(states, pct, side, state)
                    removed_state_dict[gesture_num][finger_num][subject_num][essai_num] = state_removed
    
    return removed_state_dict


def remove_states(dhg, states_dict, states, pct=1, pcts = None, random_seed=7, side='center'):
    removed_states_dict = states_dict
    for i, state in enumerate(states):
        if pcts:
            pct = pcts[i]
        removed_states_dict = remove_state(dhg, removed_states_dict, state, pct = pct, side=side, random_seed=random_seed,)
    return removed_states_dict

def make_reduced_states_dict(dhg, states, pct=1, pcts = None, multi_digit = True, filtered2 = False, side='center', random_seed=None):
    states_dict = make_states_dict(dhg, multi_digit, filtered2)
    removed_states_dict = remove_states(dhg, states_dict, states, pct=pct, pcts = pcts, side=side, random_seed=random_seed)
    return states_dict, removed_states_dict


def get_stationary_states(state_counts, stationary_digit = "4"):
    states = {state:value for state, value in state_counts.items() if state.startswith(stationary_digit)}
    return states

def get_states_percentages(states_counts, all_states_counts):
    states_values = np.array(list(states_counts.values()))
    all_states_values = np.array(list(all_states_counts.values()))
    all_states_sum = np.sum(all_states_values)
    states_percentages = states_values / all_states_sum
    # get back into dictionary
    states_percentages = {state:percentage for state, percentage in zip(states_counts.keys(), states_percentages)}
    return states_percentages

def reduce_stationary_states(dhg, states_dict, stationary_digit = "4", pcts = None, side = 'center', random_seed = None):
    state_counts = get_state_counts(dhg, states_dict)

    stationary_state_counts = get_stationary_states(state_counts, stationary_digit)
    stationary_state_values = np.array(list(stationary_state_counts.values()))
    stationary_state_sum = np.sum(stationary_state_values)
    normalized_stationary_state_counts = [count / stationary_state_sum for count in stationary_state_values]

    pcts = normalized_stationary_state_counts if pcts is None else pcts
    states_to_reduce = list(stationary_state_counts.keys())

    reduced_states_dict = remove_states(dhg, states_dict, states_to_reduce, pcts=pcts, side=side, random_seed=random_seed)
    return reduced_states_dict

def get_new_states_dict(dhg, smoother=False):
    smoother_key = [s for s in list(dhg.clean_features.keys()) if 'smoother' in s][0]
    palm_orientations = dhg.clean_features["palm_orientations"]
    siamese_similarity = dhg.clean_features[smoother_key] if smoother else dhg.clean_features["siamese_similarity"]
    moving_directions = dhg.clean_features["moving_directions"]
    # moving_directions = get_all_moving_directions(dhg, dhg_fe, stationary_threshold, moving_direction_indexes, normalize, dimensions, filtered=filtered)
    states = get_all_states(dhg, moving_directions, palm_orientations, siamese_similarity)
    return states


def generate_distribution(formula = lambda x: x, x_min = 0, x_max = np.pi, num_samples = 20, cutoff_sample = None, seed = None):
    if seed is not None:
        np.random.seed(seed)
    if cutoff_sample is not None:
        x = np.linspace(x_min, x_max, cutoff_sample)
        y = formula(x)
        y = np.concatenate((y, np.full(num_samples - cutoff_sample, y[-1])))
    else:
        x = np.linspace(x_min, x_max, num_samples)
        y = formula(x)
    return x, y

def normalize(values, new_min = 0, new_max = 1):
    old_min = np.min(values)
    old_max = np.max(values)
    return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

np.random.seed(333)
torch.manual_seed(333)

def shuffle_data(X, y=None):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    if y is not None:
        return X[indices], y[indices]
    else:
        return X[indices]


def create_inout_sequences(input_data, sequence_length, state_mapping, output_window=4, remove_duplicates = True):
    if remove_duplicates:
        sequences = create_inout_sequences_unique(input_data, sequence_length, state_mapping, output_window=output_window)
    else:
        sequences = create_inout_sequences_duplicates(input_data, sequence_length, state_mapping, output_window=output_window)

    return sequences


def create_inout_sequences_duplicates(input_data, sequence_length, state_mapping, output_window=4):
    inout_seq = []
    L = len(input_data)
    # map the input data before starting
    input_data = [state_mapping[state] for state in input_data]
    for i in range(L-sequence_length):
        if i + sequence_length + output_window > L:
            break
        train_seq = np.append(input_data[i:i+sequence_length][:-output_window] , output_window * [0])
        train_label = input_data[i:i+sequence_length]
        if any(train_label[-output_window:]) == 0:
            continue
        else:
            inout_seq.append((train_seq, train_label))
    return torch.IntTensor(inout_seq)


def create_inout_sequences_unique(input_data, sequence_length, state_mapping, output_window=4):
    inout_seq = set()
    L = len(input_data)
    # map the input data before starting
    input_data = [state_mapping[state] for state in input_data]
    for i in range(L-sequence_length):
        if i + sequence_length + output_window > L:
            break
        train_seq = np.append(input_data[i:i+sequence_length][:-output_window] , output_window * [0])
        train_label = input_data[i:i+sequence_length]
        # Convert sequences and labels to tuples before adding them to the set
        inout_seq.add((tuple(train_seq), tuple(train_label)))

    # Convert sequences and labels back to lists
    inout_seq = [(np.array(list(seq)), np.array(list(label))) for seq, label in inout_seq]

    return torch.IntTensor(inout_seq)

def get_train_test_sequences(states_dict, sequence_length, state_mapping, output_window = 4, test_size = 0.3, random_state = 333, shuffle = True, remove_duplicates = True, device = 'cuda'):
    np.random.seed(random_state)
    all_sequences = []  # Combine all sequences before splitting
    # print(state_mapping)

    for gesture in states_dict:
        for finger in states_dict[gesture]:
            for subject in states_dict[gesture][finger]:
                for essai in states_dict[gesture][finger][subject]:
                    if len(states_dict[gesture][finger][subject][essai]) > sequence_length + output_window:
                        sequences = create_inout_sequences(states_dict[gesture][finger][subject][essai], sequence_length, state_mapping, output_window=output_window, remove_duplicates=remove_duplicates)
                        if len(sequences) >= 2:  # Only consider sequences with at least 2 samples
                            all_sequences.extend(sequences)


    if shuffle:
        np.random.shuffle(all_sequences)

    # Perform the train-test split
    train_seqs, test_seqs = train_test_split(all_sequences, test_size=test_size, random_state=random_state) if test_size > 0 else (all_sequences, all_sequences)

    return torch.stack(train_seqs).to(device), torch.stack(test_seqs).to(device)

def get_sequences(states_dict, sequence_length, state_mapping, output_window = 4, random_state = 333, shuffle = True, remove_duplicates = True, device = 'cuda'):
    all_sequences = []

    for gesture in states_dict:
        for finger in states_dict[gesture]:
            for subject in states_dict[gesture][finger]:
                for essai in states_dict[gesture][finger][subject]:
                    if len(states_dict[gesture][finger][subject][essai]) > sequence_length + output_window:
                        sequences = create_inout_sequences(states_dict[gesture][finger][subject][essai], sequence_length, state_mapping, output_window=output_window, remove_duplicates=remove_duplicates)
                        if len(sequences) >= 2:  # Only consider sequences with at least 2 samples
                            all_sequences.extend(sequences)

    if shuffle:
        np.random.shuffle(all_sequences)

    return torch.stack(all_sequences).to(device)


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_optimizer(model, lr, decay_rate):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay_rate)

def setup_scheduler(optimizer, warmup_steps):
    return torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
    # return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min((step+1)/warmup_steps, 1))

def setup_criterion():
    return nn.CrossEntropyLoss()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def reduce_states_by_cutoff(dhg, states_dict, cutoff_val=0):
    state_counts = get_state_counts(dhg, states_dict)
    states = {state:value for state, value in state_counts.items() if value > cutoff_val}
    counts = np.array(list(state_counts.values()))
    states = np.array(list(state_counts.keys()))

    cutoff = np.percentile(counts, cutoff_val)

    idxs_to_remove = counts <= cutoff

    states_to_remove = states[idxs_to_remove]
    new_states_dict = remove_states(dhg, states_dict, states=states_to_remove, pct=1)
    return new_states_dict

def preprocess_states(dhg, states_dict, stationary_dropout_pct = 0, cutoff_val = 0, random_seed = 7, side = 'center'):
    state_counts = get_state_counts(dhg, states_dict)
    stationary_state_counts = get_stationary_states(state_counts)

    pcts = [stationary_dropout_pct for _ in stationary_state_counts]
    reduced_stationary_states_dict = reduce_stationary_states(dhg, states_dict, pcts = pcts, random_seed = random_seed, side = side)

    reduced_states_dict = reduce_states_by_cutoff(dhg, reduced_stationary_states_dict, cutoff_val = cutoff_val)

    return reduced_states_dict


import time
from IPython.display import clear_output


def train_test_distribution(train_data, val_data, inverted_state_mapping, lower_bound = 1, upper_bound = 4):
    train_counter = Counter(token for sequence in train_data.tolist() for subseq in sequence for token in subseq)
    valid_counter = Counter(token for sequence in val_data.tolist() for subseq in sequence for token in subseq)

    # Calculate the difference in distributions
    distribution_difference = {token: abs(train_counter.get(token, 0) - valid_counter.get(token, 0)) for token in set(train_counter) | set(valid_counter)}
    bad_tokens = []
    bad_tokens_inverted = []

    # print(inverted_state_mapping)

    print(f"25th Percentile: {np.percentile(list(distribution_difference.values()), 25)}")
    for token, difference in distribution_difference.items():
        ratio = train_counter[token] / valid_counter[token] if valid_counter[token] != 0 else 0
        if ratio > upper_bound or ratio < lower_bound or ratio == 0:
            if not valid_counter[token] > 1000:
                if token == 0:
                    inverted_token = 'padded_zero'
                else:
                    inverted_token = inverted_state_mapping[token]
                print(f'Token: {token} | "{inverted_token}", Difference in Distribution: {difference} | Train Count: {train_counter[token]}, Valid Count: {valid_counter[token]} | Ratio: {ratio}')
                bad_tokens.append(token)
                bad_tokens_inverted.append(inverted_token)

    return bad_tokens, bad_tokens_inverted

def remove_bad_sequence(states_dict, bad_tokens_inverted):
    for gesture in states_dict:
        for finger in states_dict[gesture]:
            for subject in states_dict[gesture][finger]:
                for essai in states_dict[gesture][finger][subject]:
                    states = states_dict[gesture][finger][subject][essai]
                    check = [state for state in states if state in bad_tokens_inverted]
                    if len(check) > 0:
                        states_dict[gesture][finger][subject][essai] = []
    return states_dict


def recursivley_remove_bad_sequences(dhg, sequence_length, stationary_dropout_pct = 0, cutoff_val = 0, test_size = 0.3, random_seed = 7, shuffle = True, remove_duplicates = True, device = 'cuda', output_window = 4, smoother=False, states_dict = None):
    _states_dict = get_new_states_dict(dhg, smoother=smoother) if states_dict is None else states_dict
    new_states_dict = preprocess_states(dhg, _states_dict, stationary_dropout_pct = stationary_dropout_pct, cutoff_val = cutoff_val, random_seed = random_seed)
    state_mapping, inverted_state_mapping, ntokens = create_state_mappings(dhg, new_states_dict, pad_zero=True)
    train_data, val_data = get_train_test_sequences(new_states_dict, sequence_length, state_mapping, output_window=output_window, test_size=test_size, random_state=random_seed, shuffle=shuffle, remove_duplicates=remove_duplicates, device=device)
    bad_tokens, bad_tokens_inverted = train_test_distribution(train_data, val_data, inverted_state_mapping)
    

    start = time.time()
    while len(bad_tokens) > 0:
        new_states_dict = remove_bad_sequence(new_states_dict, bad_tokens_inverted)
        state_mapping, inverted_state_mapping, ntokens = create_state_mappings(dhg, new_states_dict, pad_zero=True)
        train_data, val_data = get_train_test_sequences(new_states_dict, sequence_length, state_mapping, output_window=output_window, test_size=test_size, random_state=random_seed, shuffle=shuffle, remove_duplicates=remove_duplicates, device=device)
        bad_tokens, bad_tokens_inverted = train_test_distribution(train_data, val_data, inverted_state_mapping)

        if time.time() - start > 600:
            break

        clear_output(wait=True)

    return new_states_dict, train_data, val_data, state_mapping, inverted_state_mapping, ntokens

import plotly.graph_objects as go
def plot_distributions(train_data, val_data, save_name = None):
    train_distributions = Counter(token for sequence in train_data.tolist() for subseq in sequence for token in subseq)
    valid_distributions = Counter(token for sequence in val_data.tolist() for subseq in sequence for token in subseq)

    distribution_difference = {token: abs(train_distributions.get(token, 0) - valid_distributions.get(token, 0)) for token in set(train_distributions) | set(valid_distributions)}

    # Get the tokens and their distributions
    tokens = list(distribution_difference.keys())
    train_distributions = [train_distributions.get(token, 0) for token in tokens]
    valid_distributions = [valid_distributions.get(token, 0) for token in tokens]

    # Create a bar chart for training data
    fig = go.Figure(data=[
        go.Bar(x=tokens, y=train_distributions, name='Train',
            marker=dict(color='blue',line=dict(width=0)), # set color to blue
            #    text=train_distributions, # add text
            textposition='auto') # position text
    ])

    # Add validation data to the same chart
    fig.add_trace(go.Bar(x=tokens, y=valid_distributions, name='Valid',
                        marker=dict(color='red', line=dict(width=0)), # set color to red
                        #  text=valid_distributions, # add text
                        textposition='auto')) # position text

    # fig.update_traces(texttemplate='%{text:.2%}') # format text as percentage
    fig.update_layout(
        xaxis_title="Tokens",
        yaxis_title="Count",
        font=dict(
            size=18,
            color="#7f7f7f"
        ),
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=1000,
        height=800,
        barmode='overlay',
    )

    fig.show()

    if save_name is not None:
        fig.write_image(f"{save_name}")

    
def remove_repeated_inputs(tensor_data):
    # Tensor data is a 3D tensor,
    # In each tensor, the first tensor is the input and the second tensor is the output
    # We want to remove any repeated inputs
    inputs = [tensor_data[i][0].tolist() for i in range(len(tensor_data))]

    unique_inputs = []
    unique_indexes = []
    for i, input in enumerate(inputs):
        if tuple(input) not in unique_inputs:
            unique_inputs.append(tuple(input))
            unique_indexes.append(i)

    unique_train_data = tensor_data[unique_indexes]
    return unique_train_data

from sklearn.model_selection import train_test_split
def shrink_data(dhg_to_use, remove_bad_sequences=True,remove_bad_sequences_first = True, cutoff_filtering_first=True, sequence_length = 20, stationary_dropout_pct = 0, cutoff_val = 0, test_size = 0.3, random_seed = 333, shuffle = True, remove_duplicates = True, device = "cuda", output_window = 4, smoother=False, states_to_reduce = {None:0}, states_dict=None):
    if remove_bad_sequences_first and remove_bad_sequences:
       # Remove bad sequences
        new_states_dict, train_data, val_data, state_mapping, inverted_state_mapping, ntokens = recursivley_remove_bad_sequences(dhg_to_use, sequence_length, stationary_dropout_pct = 0, cutoff_val = 0, test_size = test_size, random_seed = random_seed, shuffle = shuffle, remove_duplicates = remove_duplicates, device = device, output_window = output_window, smoother=smoother, states_dict=states_dict)
    
    new_states_dict = get_new_states_dict(dhg_to_use, smoother=smoother)
    state_mapping, inverted_state_mapping, ntokens = create_state_mappings(dhg_to_use, new_states_dict, pad_zero=True)
    if states_to_reduce != {None:0}:
        for state, pct in states_to_reduce.items():
            if (state is not None) and (pct != 0):
                new_states_dict = remove_state(dhg_to_use, new_states_dict, state, pct = pct, side='center', random_seed=random_seed)

    # reduced_states_dict = new_states_dict

    
    if cutoff_filtering_first:
        # # Remove cutoff percentage of states
        cutoff_reduced_states_dict = preprocess_states(dhg_to_use, new_states_dict, stationary_dropout_pct = 0, cutoff_val = cutoff_val, random_seed = random_seed)

        # Remove cutoff percentage of states
        stationary_reduced_states_dict = preprocess_states(dhg_to_use, cutoff_reduced_states_dict, stationary_dropout_pct = stationary_dropout_pct, cutoff_val = 0, random_seed = random_seed)
        reduced_states_dict = stationary_reduced_states_dict
    else:
        # Remove cutoff percentage of states
        stationary_reduced_states_dict = preprocess_states(dhg_to_use, new_states_dict, stationary_dropout_pct = stationary_dropout_pct, cutoff_val = cutoff_val, random_seed = random_seed)

        # # Remove cutoff percentage of states
        cutoff_reduced_states_dict = preprocess_states(dhg_to_use, stationary_reduced_states_dict, stationary_dropout_pct = 0, cutoff_val = cutoff_val, random_seed = random_seed)
        reduced_states_dict = cutoff_reduced_states_dict



    state_mapping, inverted_state_mapping, ntokens = create_state_mappings(dhg_to_use, reduced_states_dict, pad_zero=True)


    train_data, val_data = get_train_test_sequences(reduced_states_dict, sequence_length, state_mapping, output_window=output_window, test_size=test_size, random_state=random_seed, shuffle=shuffle, remove_duplicates=remove_duplicates, device=device)

    train_data = remove_repeated_inputs(train_data)
    val_data = remove_repeated_inputs(val_data)

    if (not remove_bad_sequences_first) and remove_bad_sequences:
        # Remove bad sequences again
        new_states_dict, train_data, val_data, state_mapping, inverted_state_mapping, ntokens = recursivley_remove_bad_sequences(dhg_to_use, sequence_length, stationary_dropout_pct = 0, cutoff_val = cutoff_val, test_size = test_size, random_seed = random_seed, shuffle = shuffle, remove_duplicates = remove_duplicates, device = device, output_window = output_window, smoother=smoother, states_dict=reduced_states_dict)
        reduced_states_dict = new_states_dict
        train_data = remove_repeated_inputs(train_data)
        val_data = remove_repeated_inputs(val_data)
    
    return reduced_states_dict, train_data, val_data, state_mapping, inverted_state_mapping, ntokens


def split_string(s):
    if type(s) == int:
        return [s,s,s]
    return [s[0], s[1], s[2:]]

def get_unique_elements(data_list):
    unique_elements = set()
    for data in data_list:
        for state in data:
            if type(state) == str:
                unique_elements.add(state)
    return unique_elements

def create_unique_mappings(data_list, pad_zero=True):
    unique_states = get_unique_elements(data_list)
    state_mapping = create_state_mapping(unique_states, pad_zero=pad_zero)
    inverted_state_mapping = invert_state_mapping(state_mapping)
    ntokens = len(state_mapping) if not pad_zero else len(state_mapping) + 1
    return state_mapping, inverted_state_mapping, ntokens

def separate_variables(combined_tensor_data, inverted_state_mapping):
    srcs = [combined_tensor_data[i][0].tolist() for i in range(len(combined_tensor_data))]
    srcs_unmapped = [[inverted_state_mapping[i] if i in inverted_state_mapping else i for i in src] for src in srcs]
    trgs = [combined_tensor_data[i][1].tolist() for i in range(len(combined_tensor_data))]
    trgs_unmapped = [[inverted_state_mapping[i] if i in inverted_state_mapping else i for i in trg] for trg in trgs]

    srcs_unmapped_split = [[split_string(state) for state in srcs_unmapped[i]] for i in range(len(srcs_unmapped))]
    trgs_unmapped_split = [[split_string(state) for state in trgs_unmapped[i]] for i in range(len(trgs_unmapped))]

    moving_direction_srcs_unmapped = [[srcs_unmapped_split[j][i][0] for i in range(len(srcs_unmapped_split[j]))] for j in range(len(srcs_unmapped_split))]
    palm_orientation_srcs_unmapped = [[srcs_unmapped_split[j][i][1] for i in range(len(srcs_unmapped_split[j]))] for j in range(len(srcs_unmapped_split))]
    hand_pose_srcs_unmapped = [[srcs_unmapped_split[j][i][2] for i in range(len(srcs_unmapped_split[j]))] for j in range(len(srcs_unmapped_split))]

    moving_direction_trgs_unmapped = [[trgs_unmapped_split[j][i][0] for i in range(len(trgs_unmapped_split[j]))] for j in range(len(trgs_unmapped_split))]
    palm_orientation_trgs_unmapped = [[trgs_unmapped_split[j][i][1] for i in range(len(trgs_unmapped_split[j]))] for j in range(len(trgs_unmapped_split))]
    hand_pose_trgs_unmapped = [[trgs_unmapped_split[j][i][2] for i in range(len(trgs_unmapped_split[j]))] for j in range(len(trgs_unmapped_split))]

    moving_direction_mapping, inverted_moving_direction_mapping, ntokens_moving_direction = create_unique_mappings(moving_direction_trgs_unmapped, pad_zero=True)
    palm_orientation_mapping, inverted_palm_orientation_mapping, ntokens_palm = create_unique_mappings(palm_orientation_trgs_unmapped, pad_zero=True)
    hand_pose_mapping, inverted_hand_pose_mapping, ntokens_hand_pose = create_unique_mappings(hand_pose_trgs_unmapped, pad_zero=True)

    moving_direction_srcs_mapped = [[moving_direction_mapping[s] if s in moving_direction_mapping else s for s in moving_direction_srcs_unmapped[i]] for i in range(len(moving_direction_srcs_unmapped))]
    palm_orientation_srcs_mapped = [[palm_orientation_mapping[s] if s in palm_orientation_mapping else s for s in palm_orientation_srcs_unmapped[i]] for i in range(len(palm_orientation_srcs_unmapped))]
    hand_pose_srcs_mapped = [[hand_pose_mapping[s] if s in hand_pose_mapping else s for s in hand_pose_srcs_unmapped[i]] for i in range(len(hand_pose_srcs_unmapped))]

    moving_direction_trgs_mapped = [[moving_direction_mapping[s] if s in moving_direction_mapping else s for s in moving_direction_trgs_unmapped[i]] for i in range(len(moving_direction_trgs_unmapped))]
    palm_orientation_trgs_mapped = [[palm_orientation_mapping[s] if s in palm_orientation_mapping else s for s in palm_orientation_trgs_unmapped[i]] for i in range(len(palm_orientation_trgs_unmapped))]
    hand_pose_trgs_mapped = [[hand_pose_mapping[s] if s in hand_pose_mapping else s for s in hand_pose_trgs_unmapped[i]] for i in range(len(hand_pose_trgs_unmapped))]

    moving_direction_srcs_mapped_tensor = torch.IntTensor(moving_direction_srcs_mapped).unsqueeze(2)
    palm_orientation_srcs_mapped_tensor = torch.IntTensor(palm_orientation_srcs_mapped).unsqueeze(2)
    hand_pose_srcs_mapped_tensor = torch.IntTensor(hand_pose_srcs_mapped).unsqueeze(2)

    moving_direction_trgs_mapped_tensor = torch.IntTensor(moving_direction_trgs_mapped).unsqueeze(2)
    palm_orientation_trgs_mapped_tensor = torch.IntTensor(palm_orientation_trgs_mapped).unsqueeze(2)
    hand_pose_trgs_mapped_tensor = torch.IntTensor(hand_pose_trgs_mapped).unsqueeze(2)

    moving_direction_mapped_tensor = torch.cat((moving_direction_srcs_mapped_tensor, moving_direction_trgs_mapped_tensor),2)
    palm_orientation_mapped_tensor = torch.cat((palm_orientation_srcs_mapped_tensor, palm_orientation_trgs_mapped_tensor),2)
    hand_pose_mapped_tensor = torch.cat((hand_pose_srcs_mapped_tensor, hand_pose_trgs_mapped_tensor),2)

    return (moving_direction_mapped_tensor, moving_direction_mapping, inverted_moving_direction_mapping, ntokens_moving_direction), (palm_orientation_mapped_tensor, palm_orientation_mapping, inverted_palm_orientation_mapping, ntokens_palm), (hand_pose_mapped_tensor, hand_pose_mapping, inverted_hand_pose_mapping, ntokens_hand_pose)