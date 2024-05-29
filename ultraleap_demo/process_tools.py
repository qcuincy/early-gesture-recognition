from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from .feature_tools import *
import itertools
import pickle
import numpy as np
import torch

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
    for gesture_num in range(1, dhg.gesture_num_max+1):
        for finger_num in range(1, dhg.finger_num_max+1):
            for subject_num in range(1, dhg.subject_num_max+1):
                for essai_num in range(1, dhg.essai_num_max+1):
                    states = states_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
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


def create_sequences(series, sequence_length):
    X, y = [], []

    for i in range(len(series) - sequence_length):
        X.append(series[i:i+sequence_length])
        y.append(series[i+sequence_length])

    return np.array(X), np.array(y)


def create_state_mapping(unique_states, pad_zero=False):
    mapping = {state: i for i, state in enumerate(unique_states)}
    if not pad_zero:
        return mapping
    else:
        # Add 1 to each index to avoid 0 as a state
        return {state: i+1 for state, i in mapping.items()}

def invert_state_mapping(state_mapping, pad_zero=False):
    mapping = {i: state for state, i in state_mapping.items()}
    if not pad_zero:
        return mapping
    else:
        # Subtract 1 from each index as the padding value is 0
        return {i-1: state for i, state in mapping.items()}

def create_dataset(X, sequence_length, state_mapping):
    all_sequences = []
    all_labels = []
    for series in X:
        sequences, labels = create_sequences(series, sequence_length)
        for seq, label in zip(sequences, labels):
            mapped_seq = [state_mapping[state] for state in seq]
            mapped_label = state_mapping[label]

            all_sequences.append(mapped_seq)
            all_labels.append(mapped_label)
            # all_labels.append(to_categorical(mapped_label, num_classes=num_classes))

    return np.array(all_sequences), np.array(all_labels)

def save_mapping(mapping, filename):
    with open(filename, 'wb') as f:
        pickle.dump(mapping, f)

def load_mapping(filename):
    with open(filename, 'rb') as f:
        mapping = pickle.load(f)
    return mapping



def collate_batch(batch):
    # batch is a list of tuples with (sequence, label)
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)

    # Create masks for the sequences where each '1' indicates a real token
    # and '0' indicates a padded position
    masks = sequences_padded == 0

    return sequences_padded, masks, labels

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

def get_train_test_data(states_dict, test_size, random_state):
    gestures, fingers, _, _ = get_ranges()
    combinations_dict = get_combinations_dict()

    # For each gesture, finger combination, split the data into training and testing sets
    for gesture in combinations_dict:
        for finger in combinations_dict[gesture]:
            # Split the data
            train, test = train_test_split(combinations_dict[gesture][finger], test_size=test_size, random_state=random_state)
            # Add the split data to the dictionary
            combinations_dict[gesture][finger] = {"train": train, "test": test}

    train_combinations = []
    test_combinations = []
    for gesture in gestures:
        for finger in fingers:
            train_combs = combinations_dict[gesture][finger]["train"]
            for comb in train_combs:
                train_combinations.append([gesture, finger]+list(comb))

            test_combs = combinations_dict[gesture][finger]["test"]
            for comb in test_combs:
                test_combinations.append([gesture, finger]+list(comb))

    train_states = []
    for comb in train_combinations:
        gesture, finger, subject, essai = comb
        train_states.append(states_dict[gesture][finger][subject][essai])

    test_states = []
    for comb in test_combinations:
        gesture, finger, subject, essai = comb
        test_states.append(states_dict[gesture][finger][subject][essai])

    return train_states, test_states

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

def get_train_test_sequences(states_dict, sequence_length, state_mapping, test_size = 0.3, random_state = 333, duplicates = False, shuffle = True):
    gestures, fingers, subjects, essais = get_ranges()

    train_encoded_sequences = []
    train_encoded_labels = []
    test_encoded_sequences = []
    test_encoded_labels = []

    for gesture in gestures:
        for finger in fingers:
            for subject in subjects:
                for essai in essais:
                    encoded_sequences = []
                    encoded_labels = []
                    sequences, labels = create_sequences(states_dict[gesture][finger][subject][essai], sequence_length)
                    for seq, label in zip(sequences, labels):
                        mapped_seq = [state_mapping[state] for state in seq]
                        mapped_label = state_mapping[label]

                        encoded_sequences.append(mapped_seq)
                        encoded_labels.append(mapped_label)

                    encoded_sequences = np.array(encoded_sequences)
                    encoded_labels = np.array(encoded_labels)

                    if len(encoded_sequences) == 0:
                        continue

                    if not duplicates:
                        encoded_sequences, encoded_labels = remove_duplicates(encoded_sequences, encoded_labels)
                        
                    if encoded_sequences.shape[0] < 2:
                        continue
                    
                    train_seqs, test_seqs, train_labels, test_labels = train_test_split(encoded_sequences, encoded_labels, test_size=test_size, random_state=random_state)
                    train_encoded_sequences.extend(train_seqs)
                    train_encoded_labels.extend(train_labels)
                    test_encoded_sequences.extend(test_seqs)
                    test_encoded_labels.extend(test_labels)

    # Convert to numpy arrays
    train_encoded_sequences = np.array(train_encoded_sequences)
    train_encoded_labels = np.array(train_encoded_labels)
    test_encoded_sequences = np.array(test_encoded_sequences)
    test_encoded_labels = np.array(test_encoded_labels)

    # Shuffle the data
    if shuffle:
        train_encoded_sequences, train_encoded_labels = shuffle_data(train_encoded_sequences, train_encoded_labels)
        test_encoded_sequences, test_encoded_labels = shuffle_data(test_encoded_sequences, test_encoded_labels)

    return np.array(train_encoded_sequences), np.array(train_encoded_labels), np.array(test_encoded_sequences), np.array(test_encoded_labels)
    



def prep_data(dhg, states_dict, state_mapping, sequence_length, batch_size, test_size = 0.2, random_state = 42, duplicates = False, shuffle = True, collate_fn = None, drop_last = False):
    X_train, y_train, X_test, y_test = get_train_test_sequences(states_dict, sequence_length, state_mapping, test_size = test_size, random_state = random_state, duplicates = duplicates, shuffle = shuffle)
    num_states = get_num_states(dhg, states_dict)

    # Convert your data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.long) if not isinstance(X_train, torch.Tensor) else X_train
    y_train_tensor = torch.tensor(y_train, dtype=torch.long) if not isinstance(y_train, torch.Tensor) else y_train

    X_test_tensor = torch.tensor(X_test, dtype=torch.long) if not isinstance(X_test, torch.Tensor) else X_test
    y_test_tensor = torch.tensor(y_test, dtype=torch.long) if not isinstance(y_test, torch.Tensor) else y_test

    # Create datasets for training and test sets
    train_dataset = GestureDataset(X_train_tensor, y_train_tensor)
    test_dataset = GestureDataset(X_test_tensor, y_test_tensor)

    # Create dataloaders for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)

    prep = dict(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_states=num_states
    )

    print("Data prepared")
    print('_'*20)
    print(f"X Train shape: {X_train_tensor.shape}, y Train shape: {y_train_tensor.shape}")
    print(f"X Test shape: {X_test_tensor.shape}, y Test shape: {y_test_tensor.shape}")
    print(f"Test Size: {X_test_tensor.shape[0] / (X_train_tensor.shape[0] + X_test_tensor.shape[0])}")
    print('_'*20)


    return prep

def create_state_mappings(dhg, encoded_states, pad_zero=False):
    unique_states = get_unique_states(dhg, encoded_states)
    state_mapping = create_state_mapping(unique_states, pad_zero=pad_zero)
    inverted_state_mapping = invert_state_mapping(state_mapping, pad_zero=pad_zero)
    return state_mapping, inverted_state_mapping


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
    for gesture_num in range(1, dhg.gesture_num_max+1):
        for finger_num in range(1, dhg.finger_num_max+1):
            for subject_num in range(1, dhg.subject_num_max+1):
                for essai_num in range(1, dhg.essai_num_max+1):
                    states = states_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}'].copy()
                    state_removed = reduce_state(states, pct, side, state)
                    removed_state_dict[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}'] = state_removed
    
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

def get_new_states_dict(dhg_to_use, smoother=False):
    smoother_key = [s for s in list(dhg_to_use.clean_features.keys()) if 'smoother' in s][0]
    palm_orientations = dhg_to_use.clean_features["palm_orientations"]
    siamese_similarity = dhg_to_use.clean_features[smoother_key] if smoother else dhg_to_use.clean_features["siamese_similarity"]
    moving_directions = dhg_to_use.clean_features["moving_directions"]
    # moving_directions = get_all_moving_directions(dhg, dhg_fe, stationary_threshold, moving_direction_indexes, normalize, dimensions, filtered=filtered)
    states = get_all_states(dhg_to_use, moving_directions, palm_orientations, siamese_similarity)
    return states