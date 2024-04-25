# import matplotlib.animation as animation
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
import h5py


# DHG dataset class to load data and do some stuff with it easily 
# E.g. load data, split data, visualize a gesture, etc.
class DHG():
    def __init__(self, path, split=0.8, seed=0, load_features=True, load_clean_features=True, load_encoded_states=True, load_state_to_encoder=True):
        self.path = path
        # Check first if any pickle files exist in the path, if so then load them instead of the raw data
        # This will be much faster
        pickle_files = [os.path.join(path, p) for p in os.listdir(self.path) if p.endswith('.pickle')]
        if len(pickle_files) > 0:
            print("Loading pickle files...")
            # Get the pickle file with 'gesture_data' in the name
            gesture_data_file = [p for p in pickle_files if 'gesture_data' in p][0]
            self.gesture_data = pickle.load(open(gesture_data_file, 'rb'))
            # Get the pickle file with 'img_paths' in the name
            img_paths_file = [p for p in pickle_files if 'img_paths' in p][0]
            self.img_paths = pickle.load(open(img_paths_file, 'rb'))
            print("Done loading pickle files")
        else:
            print("No pickle files found, loading raw data...")
            self.gesture_paths = [os.path.join(path, g_path) for g_path in os.listdir(self.path) if not g_path.endswith('.txt')]
            self.gesture_data = None
            self.img_paths = None
            self.make_data_structs()
            self.load_data()
            self.load_img_paths()
        self.split = split
        self.seed = seed
        self.feature_data = None
        self.interp_feature_data = None
        self.labels = None
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.mean_frames = None
        self.std_frames = None
        self.most_frames = None
        self.least_frames = None
        self.calculate_metrics()
        self.bending_joint_idxs = [(0,2,4),(2,4,5), # thumb
                                  (0,6,8),(6,8,9), # index
                                  (0,10,12),(10,12,13), # middle
                                  (0,14,16),(14,16,17), # ring
                                  (0,18,20),(18,20,21)] # little
        self.gesture_num_max = 14
        self.finger_num_max = 2
        self.subject_num_max = 20
        self.essai_num_max = 5

        self.features_path = os.path.join(self.path, 'features')
        if os.path.exists(self.features_path) and load_features:
            self.features = None
            self.load_features()
            self.bending_angles = self.features['bending_angles']
            self.moving_directions = self.features['moving_directions']
            self.palm_orientaitons = self.features['palm_orientations']

        self.clean_features_path = os.path.join(self.features_path, "clean")
        if os.path.exists(self.clean_features_path) and load_clean_features:
            self.clean_features = None
            self.load_clean_features()
            self.clean_siamese_similarity = self.clean_features['siamese_similarity']
            self.clean_moving_directions = self.clean_features['moving_directions']
            self.clean_palm_orientations = self.clean_features['palm_orientations']

        self.encoded_states_path = os.path.join(self.features_path, "encoded_states")
        if os.path.exists(self.encoded_states_path) and load_encoded_states:
            self.encoded_states = None
            self.load_encoded_states()

        self.state_to_encoder_path = os.path.join(self.features_path, "state_to_encoder")
        if os.path.exists(self.state_to_encoder_path) and load_state_to_encoder:
            self.state_to_encoder_dicts = None
            self.load_state_to_encoder()



    def read_last_dir(self, path):
        return path.split('\\')[-1]

    def next_folder(self, path):
        return path + '\\' + os.listdir(path)[0]


    def read_gesture(self, folder):
        last_folder = self.read_last_dir(folder)
        counter = 5
        if last_folder.startswith("essai"):
            return folder
        while not last_folder.startswith("essai"):
            folder = self.next_folder(folder)
            last_folder = self.read_last_dir(folder)
            counter -= 1
            if counter == 0:
                print("No essai folder found")
                return None
        return folder
    
    def read_gesture_num(self, path):
        # Find the 'gesture_n' in a path and extract the n
        path = path.split('\\')
        for p in path:
            if p.startswith('gesture'):
                return int(p.split('_')[-1])

    def read_text_file(self, text_file):
        # read the data from the text file but ignore last line
        with open(text_file, 'r') as f:
            data = f.read().split('\n')[:-1]
        return data


    def convert_text_data(self, text_data):
        data = self.read_text_file(text_data)
        data = [d.split(' ') for d in data]
        data = np.array(data, dtype=np.float32)
        return data
    
    def make_data_structs(self):
        # We need to ensure the self.gesture_data dictionary has the correct keys, should be a recursive dictionary
        # e.g. 'gesture_1':{'finger_1':{'subject_1':{'essai_1':[skeleton_image data, general_info data, skeleton_world data]}}}}
        # We will just make the key names for now and then fill in the data later
        self.gesture_data = {}
        self.img_paths = {}
        self.feature_data = {}
        self.interp_feature_data = {}
        for gesture_path in self.gesture_paths:
            gesture_name = f'gesture_{self.read_gesture_num(gesture_path)}'
            self.gesture_data[gesture_name] = {}
            self.img_paths[gesture_name] = {}
            self.feature_data[gesture_name] = {}
            self.interp_feature_data[gesture_name] = {}
            for finger_path in os.listdir(gesture_path):
                if not finger_path.endswith('.csv'):
                    self.gesture_data[gesture_name][finger_path] = {}
                    self.img_paths[gesture_name][finger_path] = {}
                    self.feature_data[gesture_name][finger_path] = {}
                    self.interp_feature_data[gesture_name][finger_path] = {}
                    for subject_path in os.listdir(os.path.join(gesture_path, finger_path)):
                        self.gesture_data[gesture_name][finger_path][subject_path] = {}
                        self.img_paths[gesture_name][finger_path][subject_path] = {}
                        self.feature_data[gesture_name][finger_path][subject_path] = {}
                        self.interp_feature_data[gesture_name][finger_path][subject_path] = {}
                        for essai_path in os.listdir(os.path.join(gesture_path, finger_path, subject_path)):
                            self.gesture_data[gesture_name][finger_path][subject_path][essai_path] = None
                            self.img_paths[gesture_name][finger_path][subject_path][essai_path] = []
                            self.feature_data[gesture_name][finger_path][subject_path][essai_path] = None
                            self.interp_feature_data[gesture_name][finger_path][subject_path][essai_path] = None


    def read_finger_folder(self, path):
        path = path.split('\\')
        for p in path:
            if p.startswith('finger'):
                return p
            
    def read_subject_folder(self, path):
        path = path.split('\\')
        for p in path:
            if p.startswith('subject'):
                return p
            
    def read_essai_folder(self, path):
        path = path.split('\\')
        for p in path:
            if p.startswith('essai'):
                return p

    
    def load_data(self):
        finger_paths = [os.path.join(gesture_path, s_path) for gesture_path in self.gesture_paths for s_path in os.listdir(gesture_path) if not s_path.endswith('.csv')]
        subject_paths = [os.path.join(finger_path, s_path) for finger_path in finger_paths for s_path in os.listdir(finger_path) if not s_path.endswith('.csv')]

        subject_paths.sort(key=lambda x: int(x.split('\\')[-1].split('_')[-1]))
        subject_paths.sort(key=lambda x: int(x.split('\\')[-2].split('_')[-1]))
        subject_paths.sort(key=lambda x: int(x.split('\\')[-3].split('_')[-1]))

        essai_folder_paths = [os.path.join(subject_path, s_path) for subject_path in subject_paths for s_path in os.listdir(subject_path) if not s_path.endswith('.csv')]

        # In each essai folder there are 3 text files we need to read and convert to numpy arrays and then store in their respective data structure
        # 'gesture_1':{'finger_1':{'subject_1':{'essai_1':[skeleton_image data, general_info data, skeleton_world data]}}}}
        for essai_folder_path in essai_folder_paths:
            skeleton_image = self.convert_text_data(os.path.join(essai_folder_path, 'skeleton_image.txt'))
            # 2d coordinates of 22 joints - (x_1, y_1, x_2, y_2, ..., x_22, y_22)
            skeleton_image = skeleton_image.reshape(-1, 22, 2) 
            general_info = self.convert_text_data(os.path.join(essai_folder_path, 'general_information.txt'))
            # only need the timesteps and convert from 10^-7 seconds to seconds
            general_info = general_info[:, 0] / 10000000
            skeleton_world = self.convert_text_data(os.path.join(essai_folder_path, 'skeleton_world.txt'))
            # 3d coordinates of 22 joints - (x_1, y_1, z_1, x_2, y_2, z_2, ..., x_22, y_22, z_22)
            skeleton_world = skeleton_world.reshape(-1, 22, 3)
            gesture_name = f'gesture_{self.read_gesture_num(essai_folder_path)}'
            finger_name = self.read_finger_folder(essai_folder_path)
            subject_name = self.read_subject_folder(essai_folder_path)
            essai_name = self.read_essai_folder(essai_folder_path)
            self.gesture_data[gesture_name][finger_name][subject_name][essai_name] = [skeleton_image, general_info, skeleton_world]

    def load_img_paths(self):
        # the img_paths dictionary will have the same structure as the gesture_data dictionary
        # 'gesture_1':{'finger_1':{'subject_1':{'essai_1':[depth_1.png, depth_2.png, ..., depth_n.png]}}}}
        gesture_names = list(self.gesture_data.keys())
        finger_names = list(self.gesture_data[gesture_names[0]].keys())
        subject_names = list(self.gesture_data[gesture_names[0]][finger_names[0]].keys())
        essai_names = list(self.gesture_data[gesture_names[0]][finger_names[0]][subject_names[0]].keys())
        for gesture_name in gesture_names:
            for finger_name in finger_names:
                for subject_name in subject_names:
                    for essai_name in essai_names:
                        img_folder_path = os.path.join(self.path, gesture_name, finger_name, subject_name, essai_name)

                        for img_path in os.listdir(img_folder_path):
                            if img_path.endswith('.png'):
                                self.img_paths[gesture_name][finger_name][subject_name][essai_name].append(os.path.join(img_folder_path, img_path))
                                # sort the images in order of the last number in the filename
                                self.img_paths[gesture_name][finger_name][subject_name][essai_name].sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        # img_path = os.path.join(img_path, os.listdir(img_path)[0])
                        # self.img_paths[gesture_name][finger_name][subject_name][essai_name] = img_path


    def calculate_angle(self, p1, p2, p3, dimension=3):
        # calculate the angle between 3 points in 3d space
        # p1, p2, p3 are 3d points
        # p2 is the vertex point
        # dimension is the dimension of the points, e.g. 2d or 3d
        # returns the angle in radians
        v1 = p1 - p2
        v2 = p3 - p2
        if dimension == 2:
            v1 = np.append(v1, 0)
            v2 = np.append(v2, 0)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return angle
    

    def calculate_metrics(self):
        # Calculate some metrics about the data
        # mean frames, std frames, most frames, least frames
        gesture_names = list(self.gesture_data.keys())
        finger_names = list(self.gesture_data[gesture_names[0]].keys())
        subject_names = list(self.gesture_data[gesture_names[0]][finger_names[0]].keys())
        essai_names = list(self.gesture_data[gesture_names[0]][finger_names[0]][subject_names[0]].keys())
        num_frames = []
        for gesture_name in gesture_names:
            for finger_name in finger_names:
                for subject_name in subject_names:
                    for essai_name in essai_names:
                        num_frames.append(self.gesture_data[gesture_name][finger_name][subject_name][essai_name][2].shape[0])
        num_frames = np.array(num_frames)
        self.mean_frames = round(np.mean(num_frames))
        self.std_frames = np.std(num_frames)
        self.most_frames = np.max(num_frames)
        self.least_frames = np.min(num_frames)


    def interp(self, data, num_frames):
        # we want data to have the same length as num_frames (currently has less frames than num_frames)
        # data is a numpy array
        # num_frames is an integer

        # Define interpolation function
        interp_func = interp1d(np.linspace(0, 1, len(data)), data)

        # Interpolate to match the length of num_frames
        data = interp_func(np.linspace(0, 1, num_frames))
        return data



    def make_gif_from_data(self, gesture_name, finger_name, subject_name, essai_name, save_path):
        # Make a gif of the skeleton images on top of the depth images
        fig = plt.figure()
        ims = []
        depth_images = self.img_paths[gesture_name][finger_name][subject_name][essai_name]
        skeleton_images = self.gesture_data[gesture_name][finger_name][subject_name][essai_name][0]
        for i in range(len(depth_images)):
            depth_image = plt.imread(depth_images[i])
            skeleton_image = skeleton_images[i]
            x_coords = skeleton_image[:,0]
            y_coords = skeleton_image[:,1]
            im = plt.imshow(depth_image, cmap='gray')
            im2 = plt.scatter(x_coords, y_coords, c='r', s=10)
            plt.axis('off')
            ims.append([im, im2])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)
        ani.save(save_path)
        plt.axis('off')
        plt.show()


    def get_gesture_example(self, gesture_num, finger_num, subject_num, essai_num):
        return self.gesture_data[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
    
    def get_image_example(self, gesture_num, finger_num, subject_num, essai_num):
        return self.img_paths[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
    
    def load_gesture(self, gesture_num, finger_num, subject_num, essai_num):
        gesture = self.get_gesture_example(gesture_num, finger_num, subject_num, essai_num)
        image = self.get_image_example(gesture_num, finger_num, subject_num, essai_num)

        gesture_2d = gesture[0]
        gesture_3d = gesture[2]

        return gesture_2d, gesture_3d, image


    def recursively_save_dict_contents_to_group(self, h5file, path, dic):
        """
        Take a dictionary with nested dictionaries and arrays, and save it into an HDF5 file.
        """
        for key, item in dic.items():
            key_path = f"{path}/{key}"
            if isinstance(item, dict):
                self.recursively_save_dict_contents_to_group(h5file, key_path, item)
            elif isinstance(item, np.ndarray):
                h5file.create_dataset(key_path, data=item)
            else:
                raise ValueError("Cannot save %s type" % type(item))


    def save_dict_to_hdf5(self, dic, filename):
        """
        Save a dictionary to an HDF5 file.
        """
        with h5py.File(filename, 'w') as h5file:
            self.recursively_save_dict_contents_to_group(h5file, '/', dic)
    

    def recursively_load_dict_contents_from_group(self, h5file, path):
        """
        Load data from an HDF5 group into a nested dictionary structure.
        """
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
        """
        Load a dictionary from an HDF5 file.
        """
        with h5py.File(filename, 'r') as h5file:
            return self.recursively_load_dict_contents_from_group(h5file, '/')
        

    def load_features(self):
        feature_files = [os.path.join(self.features_path, p) for p in os.listdir(self.features_path) if p.endswith('.h5')]
        features = {}
        for feature_file in feature_files:
            key_name = feature_file.split('\\')[-1].replace('.h5', '')
            features[key_name] = self.load_dict_from_hdf5(feature_file)
        self.features = features

    def load_clean_features(self):
        feature_files = [os.path.join(self.clean_features_path, p) for p in os.listdir(self.clean_features_path) if (p.endswith('.h5') or p.endswith('.hdf5'))]
        features = {}
        for feature_file in feature_files:
            key_name = None
            if feature_file.endswith('.hdf5'):
                key_name = feature_file.split('\\')[-1].replace('.hdf5', '')
            else:
                key_name = feature_file.split('\\')[-1].replace('.h5', '')
            features[key_name] = self.load_dict_from_hdf5(feature_file)
        self.clean_features = features

    def load_encoded_states(self):
        encoded_state_files = [os.path.join(self.encoded_states_path, p) for p in os.listdir(self.encoded_states_path) if (p.endswith('.h5') or p.endswith('.hdf5'))]
        encoded_states = {}
        for encoded_state_file in encoded_state_files:
            key_name = None
            if encoded_state_file.endswith('.hdf5'):
                key_name = encoded_state_file.split('\\')[-1].replace('.hdf5', '')
            else:
                key_name = encoded_state_file.split('\\')[-1].replace('.h5', '')
            encoded_states[key_name] = self.load_dict_from_hdf5(encoded_state_file)
        self.encoded_states = encoded_states

    def load_state_to_encoder(self):
        # Load the state to encoder mapping dictionary from the pickle files
        state_to_encoder_dict_paths = os.listdir(self.state_to_encoder_path)
        state_to_encoder_dict_paths = [os.path.join(self.state_to_encoder_path, path) for path in state_to_encoder_dict_paths if path.endswith(".pkl")]

        state_to_encoder_dicts = {}
        for path in state_to_encoder_dict_paths:
            with open(path, "rb") as f:
                keyname = path.split("\\")[-1].split(".")[0]
                state_to_encoder_dicts[keyname] = pickle.load(f)
        self.state_to_encoder_dicts = state_to_encoder_dicts

    def load_gesture_angles(self, gesture_num, finger_num, subject_num, essai_num):
        return self.bending_angles[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
    
    def load_gesture_moving_directions(self, gesture_num, finger_num, subject_num, essai_num):
        return self.moving_directions[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
    
    def load_gesture_palm_orientations(self, gesture_num, finger_num, subject_num, essai_num):
        return self.palm_orientaitons[f'gesture_{gesture_num}'][f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}']
    
    def load_gesture_features(self, gesture_num, finger_num, subject_num, essai_num):
        return [self.load_gesture_angles(gesture_num, finger_num, subject_num, essai_num),
                self.load_gesture_moving_directions(gesture_num, finger_num, subject_num, essai_num),
                self.load_gesture_palm_orientations(gesture_num, finger_num, subject_num, essai_num)]
    
    
    def load_all_angles(self):
        # Load video frame data into a numpy array
        # Each row corresponds to the angles in a frame, the number of rows is the number of frames of all the videos
        # Each column corresponds to a different angle, the number of columns is the number of joint bending angles (10)
        data = []
        for gesture_num in range(1, self.gesture_num_max+1):
            for finger_num in range(1, self.finger_num_max+1):
                for subject_num in range(1, self.subject_num_max+1):
                    for essai_num in range(1, self.essai_num_max+1):
                        # need to extend the data list with the data from each gesture to ensure each element of data is not a list
                        frame_angles = self.load_gesture_angles(gesture_num, finger_num, subject_num, essai_num)
                        data.extend(frame_angles)

        data = np.array(data)
        return data
    
    def load_angles(self, gesture_num, combined=False):
        # Load all angles for a specific gesture
        data = []
        for finger_num in range(1, self.finger_num_max+1):
            for subject_num in range(1, self.subject_num_max+1):
                for essai_num in range(1, self.essai_num_max+1):
                    frame_angles = self.load_gesture_angles(gesture_num, finger_num, subject_num, essai_num)
                    if combined:
                        data.extend(np.array(frame_angles))
                    else:
                        data.append(np.array(frame_angles))
            
        if combined:
            data = np.array(data)
        return data

    def load_angles_dict(self, gesture_num):
        # Load all angles for a specific gesture with the finger, subject, and essai numbers as nested dictionaries
        data = {}
        for finger_num in range(1, self.finger_num_max+1):
            data[f'finger_{finger_num}'] = {}
            for subject_num in range(1, self.subject_num_max+1):
                data[f'finger_{finger_num}'][f'subject_{subject_num}'] = {}
                for essai_num in range(1, self.essai_num_max+1):
                    data[f'finger_{finger_num}'][f'subject_{subject_num}'][f'essai_{essai_num}'] = self.load_gesture_angles(gesture_num, finger_num, subject_num, essai_num)


        return data
