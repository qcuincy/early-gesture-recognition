from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import h5py


class DHG:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pickle_files = [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p.endswith('.pickle')]
        self.features_path = os.path.join(data_dir, 'features')
        self.clean_features_path = os.path.join(data_dir, 'clean_features')
        self.encoded_states_path = os.path.join(self.features_path, 'encoded_states')
        self.state_to_encoder_path = os.path.join(self.features_path, 'state_to_encoder')

        self.gesture_num_max = 14
        self.finger_num_max = 2
        self.subject_num_max = 20
        self.essai_num_max = 5

        self.gesture_data = {}
        self.bending_angles = {}
        self.moving_directions = {}
        self.palm_orientations = {}
        self.features = {}
        self.clean_features = {}
        self.encoded_states = {}
        self.state_to_encoder_dicts = {}

        self.bending_joint_idxs = [(0,2,4),(2,4,5), # thumb
                                  (0,6,8),(6,8,9), # index
                                  (0,10,12),(10,12,13), # middle
                                  (0,14,16),(14,16,17), # ring
                                  (0,18,20),(18,20,21)] # little


        if len(self.pickle_files) > 0:
            self.load_pickle_files()            
        else:
            print("No pickle files found.")

        self.calculate_metrics()
        self.load_features()
        self.load_clean_features()
        self.load_encoded_states()
        self.load_state_to_encoder()
        

    def load_pickle_files(self):
        print("Loading pickle files...")
        # Get the pickle file with 'gesture_data' in the name
        gesture_data_file = [p for p in self.pickle_files if 'gesture_data' in p][0]
        self.gesture_data = pickle.load(open(gesture_data_file, 'rb'))
        # Get the pickle file with 'img_paths' in the name
        print("Done loading pickle files")

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def calculate_angle(self, v1, v2):
        v1 = self.normalize_vector(v1)
        v2 = self.normalize_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return angle

    def calculate_metrics(self):
        gesture_names = list(self.gesture_data.keys())
        finger_names = list(self.gesture_data[gesture_names[0]].keys())
        subject_names = list(self.gesture_data[gesture_names[0]][finger_names[0]].keys())
        essai_names = list(self.gesture_data[gesture_names[0]][finger_names[0]][subject_names[0]].keys())
        num_frames = []
        for gesture_name in gesture_names:
            for finger_name in finger_names:
                for subject_name in subject_names:
                    for essai_name in essai_names:
                        num_frames.append(
                            self.gesture_data[gesture_name][finger_name][subject_name][essai_name][2].shape[0])
        num_frames = np.array(num_frames)
        self.mean_frames = round(np.mean(num_frames))
        self.std_frames = np.std(num_frames)
        self.most_frames = np.max(num_frames)
        self.least_frames = np.min(num_frames)

    def interp(self, data, num_frames):
        interp_func = interp1d(np.linspace(0, 1, len(data)), data)
        data = interp_func(np.linspace(0, 1, num_frames))
        return data

    def get_gesture_example(self, gesture_num, finger_num, subject_num, essai_num):
        return self.gesture_data[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][
            f'essai_{essai_num}']

    def load_gesture(self, gesture_num, finger_num, subject_num, essai_num):
        gesture = self.get_gesture_example(gesture_num, finger_num, subject_num, essai_num)
        gesture_2d = gesture[0]
        gesture_3d = gesture[2]
        return gesture_2d, gesture_3d

    def recursively_save_dict_contents_to_group(self, h5file, path, dic):
        for key, item in dic.items():
            key_path = f"{path}/{key}"
            if isinstance(item, dict):
                self.recursively_save_dict_contents_to_group(h5file, key_path, item)
            elif isinstance(item, np.ndarray):
                h5file.create_dataset(key_path, data=item)
            else:
                raise ValueError("Cannot save %s type" % type(item))

    def save_dict_to_hdf5(self, dic, filename):
        with h5py.File(filename, 'w') as h5file:
            self.recursively_save_dict_contents_to_group(h5file, '/', dic)

    def recursively_load_dict_contents_from_group(self, h5file, path):
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.group.Group):
                ans[key] = self.recursively_load_dict_contents_from_group(h5file, item.name)
            elif isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
            else:
                raise ValueError("Cannot load %s type" % type(item))
        return ans

    def load_dict_from_hdf5(self, filename):
        with h5py.File(filename, 'r') as h5file:
            return self.recursively_load_dict_contents_from_group(h5file, '/')

    def load_features(self):
        feature_files = [os.path.join(self.features_path, p) for p in os.listdir(self.features_path) if
                            p.endswith('.h5')]
        features = {}
        for feature_file in feature_files:
            key_name = os.path.splitext(os.path.basename(feature_file))[0]
            features[key_name] = self.load_dict_from_hdf5(feature_file)
        self.features = features

    def load_clean_features(self):
        feature_files = [os.path.join(self.clean_features_path, p) for p in os.listdir(self.clean_features_path) if
                            p.endswith(('.h5', '.hdf5'))]
        features = {}
        for feature_file in feature_files:
            key_name = os.path.splitext(os.path.basename(feature_file))[0]
            features[key_name] = self.load_dict_from_hdf5(feature_file)
        self.clean_features = features

    def load_encoded_states(self):
        encoded_state_files = [os.path.join(self.encoded_states_path, p) for p in os.listdir(self.encoded_states_path)
                                if p.endswith(('.h5', '.hdf5'))]
        encoded_states = {}
        for encoded_state_file in encoded_state_files:
            key_name = os.path.splitext(os.path.basename(encoded_state_file))[0]
            encoded_states[key_name] = self.load_dict_from_hdf5(encoded_state_file)
        self.encoded_states = encoded_states

    def load_state_to_encoder(self):
        state_to_encoder_dict_paths = [os.path.join(self.state_to_encoder_path, p) for p in
                                        os.listdir(self.state_to_encoder_path) if p.endswith(".pkl")]

        state_to_encoder_dicts = {}
        for path in state_to_encoder_dict_paths:
            with open(path, "rb") as f:
                keyname = os.path.splitext(os.path.basename(path))[0]
                state_to_encoder_dicts[keyname] = pickle.load(f)
        self.state_to_encoder_dicts = state_to_encoder_dicts

    def load_gesture_angles(self, gesture_num, finger_num, subject_num, essai_num):
        return self.bending_angles[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][
            f'essai_{essai_num}']

    def load_gesture_moving_directions(self, gesture_num, finger_num, subject_num, essai_num):
        return self.moving_directions[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][
            f'essai_{essai_num}']

    def load_gesture_palm_orientations(self, gesture_num, finger_num, subject_num, essai_num):
        return self.palm_orientaitons[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][
            f'essai_{essai_num}']

    def load_gesture_features(self, gesture_num, finger_num, subject_num, essai_num):
        return [
            self.load_gesture_angles(gesture_num, finger_num, subject_num, essai_num),
            self.load_gesture_moving_directions(gesture_num, finger_num, subject_num, essai_num),
            self.load_gesture_palm_orientations(gesture_num, finger_num, subject_num, essai_num)
        ]

    def load_all_angles(self):
        data = []
        for gesture_num in range(1, self.gesture_num_max + 1):
            for finger_num in range(1, self.finger_num_max + 1):
                for subject_num in range(1, self.subject_num_max + 1):
                    for essai_num in range(1, self.essai_num_max + 1):
                        frame_angles = self.load_gesture_angles(gesture_num, finger_num, subject_num, essai_num)
                        data.extend(frame_angles)

        data = np.array(data)
        return data

    def load_angles(self, gesture_num, combined=False):
        data = []
        for finger_num in range(1, self.finger_num_max + 1):
            for subject_num in range(1, self.subject_num_max + 1):
                for essai_num in range(1, self.essai_num_max + 1):
                    frame_angles = self.load_gesture_angles(gesture_num, finger_num, subject_num, essai_num)
                    if combined:
                        data.extend(np.array(frame_angles))
                    else:
                        data.append(np.array(frame_angles))

        if combined:
            data = np.array(data)
        return data

    def load_angles_dict(self, gesture_num):
        data = {}
        for finger_num in range(1, self.finger_num_max + 1):
            data[f'finger_{finger_num}'] = {}
            for subject_num in range(1, self.subject_num_max + 1):
                data[f'finger_{finger_num}'][f'subject_{subject_num}'] = {}
                for essai_num in range(1, self.essai_num_max + 1):
                    data[f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}'] = \
                        self.load_gesture_angles(gesture_num, finger_num, subject_num, essai_num)

        return data