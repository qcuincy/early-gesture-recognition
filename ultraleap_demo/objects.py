import numpy as np
from .feature_tools import *
from .classes import *

class Vector():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.v = np.array([x, y, z])

    def __getitem__(self, index):
        return self.v[index]
    
    def __len__(self):
        return len(self.v)
    
    def __iter__(self):
        return iter(self.v)
    
    def __repr__(self):
        return str(self.v)

class Landmarks():
    def __init__(self, hand):
        self.palm_normal = None
        self.hand = hand
        self.landmarks = self.get_landmarks(hand)
        self.wrist = self.landmarks[0]
        self.palm = self.landmarks[1]
        self.thumb_base = self.landmarks[2]
        self.thumb_art_a = self.landmarks[3]
        self.thumb_art_b = self.landmarks[4]
        self.thumb_tip = self.landmarks[5]
        self.index_base = self.landmarks[6]
        self.index_art_a = self.landmarks[7]
        self.index_art_b = self.landmarks[8]
        self.index_tip = self.landmarks[9]
        self.middle_base = self.landmarks[10]
        self.middle_art_a = self.landmarks[11]
        self.middle_art_b = self.landmarks[12]
        self.middle_tip = self.landmarks[13]
        self.ring_base = self.landmarks[14]
        self.ring_art_a = self.landmarks[15]
        self.ring_art_b = self.landmarks[16]
        self.ring_tip = self.landmarks[17]
        self.pinky_base = self.landmarks[18]
        self.pinky_art_a = self.landmarks[19]
        self.pinky_art_b = self.landmarks[20]
        self.pinky_tip = self.landmarks[21]


    def get_landmarks(self, hand):
        thumb = hand.thumb
        index = hand.index
        middle = hand.middle
        ring = hand.ring
        pinky = hand.pinky
        palm_bone = hand.palm.position
        self.palm_normal = np.array([hand.palm.normal.x, hand.palm.normal.y, hand.palm.normal.z])
        wrist_bone = hand.arm.next_joint

        thumb_base = np.array([thumb.metacarpal.next_joint.x, thumb.metacarpal.next_joint.y, thumb.metacarpal.next_joint.z])
        thumb_art_a = np.array([thumb.proximal.next_joint.x, thumb.proximal.next_joint.y, thumb.proximal.next_joint.z])
        thumb_art_b = np.array([thumb.distal.next_joint.x, thumb.distal.next_joint.y, thumb.distal.next_joint.z])
        thumb_tip = np.array([thumb.distal.prev_joint.x, thumb.distal.prev_joint.y, thumb.distal.prev_joint.z])

        index_base = np.array([index.metacarpal.next_joint.x, index.metacarpal.next_joint.y, index.metacarpal.next_joint.z])
        index_art_a = np.array([index.proximal.next_joint.x, index.proximal.next_joint.y, index.proximal.next_joint.z])
        index_art_b = np.array([index.distal.next_joint.x, index.distal.next_joint.y, index.distal.next_joint.z])
        index_tip = np.array([index.distal.prev_joint.x, index.distal.prev_joint.y, index.distal.prev_joint.z])

        middle_base = np.array([middle.metacarpal.next_joint.x, middle.metacarpal.next_joint.y, middle.metacarpal.next_joint.z])
        middle_art_a = np.array([middle.proximal.next_joint.x, middle.proximal.next_joint.y, middle.proximal.next_joint.z])
        middle_art_b = np.array([middle.distal.next_joint.x, middle.distal.next_joint.y, middle.distal.next_joint.z])
        middle_tip = np.array([middle.distal.prev_joint.x, middle.distal.prev_joint.y, middle.distal.prev_joint.z])

        ring_base = np.array([ring.metacarpal.next_joint.x, ring.metacarpal.next_joint.y, ring.metacarpal.next_joint.z])
        ring_art_a = np.array([ring.proximal.next_joint.x, ring.proximal.next_joint.y, ring.proximal.next_joint.z])
        ring_art_b = np.array([ring.distal.next_joint.x, ring.distal.next_joint.y, ring.distal.next_joint.z])
        ring_tip = np.array([ring.distal.prev_joint.x, ring.distal.prev_joint.y, ring.distal.prev_joint.z])

        pinky_base = np.array([pinky.metacarpal.next_joint.x, pinky.metacarpal.next_joint.y, pinky.metacarpal.next_joint.z])
        pinky_art_a = np.array([pinky.proximal.next_joint.x, pinky.proximal.next_joint.y, pinky.proximal.next_joint.z])
        pinky_art_b = np.array([pinky.distal.next_joint.x, pinky.distal.next_joint.y, pinky.distal.next_joint.z])
        pinky_tip = np.array([pinky.distal.prev_joint.x, pinky.distal.prev_joint.y, pinky.distal.prev_joint.z])

        palm = np.array([palm_bone.x, palm_bone.y, palm_bone.z])
        wrist = np.array([wrist_bone.x, wrist_bone.y, wrist_bone.z])



        landmarks = np.array([
            Vector(*wrist),
            Vector(*palm),
            Vector(*thumb_base),Vector(*thumb_art_a),Vector(*thumb_art_b),Vector(*thumb_tip),
            Vector(*index_base),Vector(*index_art_a),Vector(*index_art_b),Vector(*index_tip),
            Vector(*middle_base),Vector(*middle_art_a),Vector(*middle_art_b),Vector(*middle_tip),
            Vector(*ring_base),Vector(*ring_art_a),Vector(*ring_art_b),Vector(*ring_tip),
            Vector(*pinky_base),Vector(*pinky_art_a),Vector(*pinky_art_b),Vector(*pinky_tip)
        ])
        
        return landmarks

class Frame():
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.palm_normal = landmarks.palm_normal
        self.bending_joint_idxs = get_bending_joint_idxs()
        self.bending_angles = self.joint_bending_angles()
        self.hand_visible = False

    def joint_bending_angles(self):
        joints = self.landmarks.landmarks
        bending_angles = []
        for idxs in self.bending_joint_idxs:
            jointA = np.array(joints[idxs[0]])
            jointB = np.array(joints[idxs[1]])
            jointC = np.array(joints[idxs[2]])

            # Calculate the bending angle
            angle = get_bending_angles(jointA, jointB, jointC)
            bending_angles.append(angle)
        return np.array(bending_angles)
    
    def __getitem__(self, index):
        return self.landmarks[index]
    
    def __len__(self):
        return len(self.landmarks)
    
    def __iter__(self):
        return iter(self.landmarks)


WINDOW_SIZE = 28

SIMILARITY_LOOKBACK = 5

class Frames():
    def __init__(self, 
                 handpose, 
                 moving_direction_mapping, 
                 palm_orientation_mapping, 
                 hand_pose_mapping,
                 sequence_length,
                 max_frames = 200, 
                 window_size = WINDOW_SIZE, 
                 similarity_lookback = SIMILARITY_LOOKBACK, 
                 top_dirs = 2, 
                 weight_formula = exponential_decay, 
                 scaled_frame_data = False, 
                 scaled_distances = False, 
                 idxs = None, L = 1, 
                 stationary_threshold_ratio = 1.6,
                 similarity_threshold = 0.5,
                 moving_direction_indexes = None,
                 avg_distance = None):
        
        self.frames = []
        self.unique_states = []
        self.new_state = False
        self.sequence_length = sequence_length
        self.max_frames = max_frames
        self.top_dirs = top_dirs
        self.weight_formula = weight_formula
        self.scaled_frame_data = scaled_frame_data
        self.scaled_distances = scaled_distances
        self.idxs = idxs
        self.L = L
        self.stationary_threshold_ratio = stationary_threshold_ratio
        self.avg_distance = avg_distance
        
        self.palm_orientations = []
        
        self.window_size = window_size
        self.max_avg_distance = 0
        self.moving_directions = []
        self.moving_direction_indexes = moving_direction_indexes

        self.similarity_lookback = similarity_lookback
        self.similarity_threshold = similarity_threshold
        self.handpose = handpose
        self.similarity_state = 0
        self.frames_bending_angles = []
        self.similarity_states = []

        self.orientation_mapping = {'up': 0, 'down': 1, 'opposite': 2}
        self.direction_mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stationary': 4}

        self.encoded_state = ''
        self.encoded_states = []

        self.moving_direction_mapping = moving_direction_mapping
        self.palm_orientation_mapping = palm_orientation_mapping
        self.hand_pose_mapping = hand_pose_mapping

        self.mapped_moving_directions = []
        self.mapped_palm_orientations = []
        self.mapped_similarity_states = []
        self.unique_mapped_moving_directions = []
        self.unique_mapped_palm_orientations = []
        self.unique_mapped_similarity_states = []

        self.hands_visible = False
        self.frames_hands_visible = []
        self.longer_than_sequence = False



    def add_frame(self, frame, mapped_features = True):
        self.append(frame)

        self.frames_hands_visible.append(frame.hand_visible)
        
        if len(self) >= max(self.window_size, self.similarity_lookback):
            if mapped_features:
                self.add_mapped_features(frame)
            else:
                self.add_features(frame)

            recent_state = self.encoded_states[-1]
            if len(self.unique_states) > 0:
                if recent_state in self.unique_states[-4:]:
                    self.new_state = False
                else:
                    self.unique_states.append(recent_state)
                    self.unique_mapped_moving_directions.append(self.mapped_moving_directions[-1])
                    self.unique_mapped_palm_orientations.append(self.mapped_palm_orientations[-1])
                    self.unique_mapped_similarity_states.append(self.mapped_similarity_states[-1])
                    self.new_state = True
            else:
                self.unique_states.append(recent_state)
                self.unique_mapped_moving_directions.append(self.mapped_moving_directions[-1])
                self.unique_mapped_palm_orientations.append(self.mapped_palm_orientations[-1])
                self.unique_mapped_similarity_states.append(self.mapped_similarity_states[-1])
                self.new_state = True
            

    def add_features(self, frame):
        self.check_features()
        self.palm_orientations.append(self.orientation_mapping[classify_palm_rotation(frame.palm_normal)])
        self.update_similarity_state(frame)
        self.similarity_states.append(self.similarity_state)
        window_size_moving_directions = self.get_previous_frames_motion(n = self.window_size, top_dirs = self.top_dirs, window_size = self.window_size, weight_formula = self.weight_formula, scaled_frame_data = self.scaled_frame_data, scaled_distances = self.scaled_distances, idxs = self.idxs, L = self.L, stationary_threshold_ratio = self.stationary_threshold_ratio, avg_distance = self.avg_distance)
        # Want to only append the same number of items as palm_orientations and similarity_states appended
        moving_direction = self.direction_mapping[window_size_moving_directions[-1]]
        
        self.moving_directions.append(moving_direction)

        self.encoded_state = str(self.moving_directions[-1]) + str(self.palm_orientations[-1]) + str(self.similarity_states[-1])
        self.encoded_states.append(self.encoded_state)
        self.check_features()

    def add_mapped_features(self, frame):
        self.check_features()
        palm_orientation = self.orientation_mapping[classify_palm_rotation(frame.palm_normal)]
        self.palm_orientations.append(palm_orientation)
        self.mapped_palm_orientations.append(self.palm_orientation_mapping[str(palm_orientation)])

        self.update_similarity_state(frame)
        if str(self.similarity_state) not in self.hand_pose_mapping:
            if self.similarity_state < 0:
                self.similarity_state = 0
            else:
                self.similarity_state -= 1
        self.similarity_states.append(self.similarity_state)
        self.mapped_similarity_states.append(self.hand_pose_mapping[str(self.similarity_state)])

        window_size_moving_directions = self.get_previous_frames_motion(n = self.window_size, top_dirs = self.top_dirs, window_size = self.window_size, weight_formula = self.weight_formula, scaled_frame_data = self.scaled_frame_data, scaled_distances = self.scaled_distances, idxs = self.idxs, L = self.L, stationary_threshold_ratio = self.stationary_threshold_ratio, avg_distance = self.avg_distance)
        moving_direction = self.direction_mapping[window_size_moving_directions[-1]]
        self.moving_directions.append(moving_direction)
        self.mapped_moving_directions.append(self.moving_direction_mapping[str(moving_direction)])

        self.encoded_state = str(self.moving_directions[-1]) + str(self.palm_orientations[-1]) + str(self.similarity_states[-1])
        self.encoded_states.append(self.encoded_state)
        self.check_features()


    def update_similarity_state(self, frame):
        lookback_frame = self.get(-self.similarity_lookback)
        similarity_score = self.handpose.get_similarity(lookback_frame.bending_angles, frame.bending_angles)
        evaluated_score = self.evaluate_similarity_score(similarity_score, threshold = self.similarity_threshold)
        self.similarity_state += evaluated_score

    def append(self, frame):
        self.check_frames()
        self.frames.append(frame)
        self.check_frames()

    def pop(self, index):
        self.frames.pop(index)

    def clear(self):
        self.frames.clear()

    def get(self, index):
        return self.frames[index]
    
    def num_frames(self):
        return len(self.frames)
    
    def num_hands_visible(self):
        return sum(self.frames_hands_visible)

    def check_frames(self):
        if self.num_frames() > self.max_frames:
            self.frames.pop(0)

    def check_features(self):
        # Make sure all features are the same length
        if len(self.palm_orientations) > self.max_frames:
            self.palm_orientations.pop(0)
        if len(self.similarity_states) > self.max_frames:
            self.similarity_states.pop(0)
        if len(self.moving_directions) > self.max_frames:
            self.moving_directions.pop(0)
        if len(self.encoded_states) > self.max_frames:
            self.encoded_states.pop(0)
        if len(self.mapped_palm_orientations) > self.max_frames:
            self.mapped_palm_orientations.pop(0)
        if len(self.mapped_similarity_states) > self.max_frames:
            self.mapped_similarity_states.pop(0)
        if len(self.mapped_moving_directions) > self.max_frames:
            self.mapped_moving_directions.pop(0)
        if len(self.frames_hands_visible) > self.max_frames:
            self.frames_hands_visible.pop(0)
        if len(self.frames_hands_visible) > self.sequence_length:
            self.longer_than_sequence = True
        if len(self.unique_states) > self.max_frames:
            self.unique_states.pop(0)
            

        
    def get_recent_frame_data(self):
        return np.array([self.frames[-1].landmarks.landmarks])

    def get_frames_data(self, frames = None):
        frames = self.frames if frames is None else frames
        return np.array([frames[i].landmarks.landmarks for i in range(len(frames))])

    def get_previous_frames_data(self, n):
        return self.get_frames_data(self.frames[-n:])

    def get_frames_directions(self, frames = None, idxs = None, L = 1, n=5, stationary_threshold_ratio=1.6, avg_distance = None):
        return get_nonzero_top_n_directions(self.get_frames_data(frames), idxs = idxs, L = L, n = n, stationary_threshold_ratio = stationary_threshold_ratio, avg_distance = avg_distance)
    
    def get_frames_motion(self, frames_directions = None, top_dirs = 2, window_size = WINDOW_SIZE, weight_formula = exponential_decay, scaled_frame_data = False, scaled_distances = False, idxs = None, L = 1, n=5, stationary_threshold_ratio=1.6, avg_distance = None):
        frames_directions = self.get_frames_directions(idxs = self.moving_direction_indexes, L = L, n = n, stationary_threshold_ratio = stationary_threshold_ratio, avg_distance = avg_distance) if frames_directions is None else frames_directions
        return get_motion_directions(frames_directions, top_dirs = top_dirs, window_size = window_size, weight_formula = weight_formula, scaled_frame_data = scaled_frame_data, scaled_distances = scaled_distances)

    def get_previous_frames_motion(self, n, top_dirs = 2, window_size = WINDOW_SIZE, weight_formula = exponential_decay, scaled_frame_data = False, scaled_distances = False, idxs = None, L = 1, stationary_threshold_ratio=1.6, avg_distance = None):
        return self.get_frames_motion(self.get_frames_directions(self.frames[-n:], idxs = idxs, L = L, stationary_threshold_ratio = stationary_threshold_ratio), top_dirs = top_dirs, window_size = window_size, weight_formula = weight_formula, scaled_frame_data = scaled_frame_data, scaled_distances = scaled_distances, avg_distance=avg_distance)[0]

    def get_frames_bending_angles(self, frames = None):
        frames = self.frames if frames is None else frames
        return np.array([frames[i].bending_angles for i in range(len(frames))])
    
    def evaluate_similarity_score(self, score, threshold = 0.5):
        if score < threshold:
            return 1
        else:
            return 0

    def reset_similarity_state(self):
        self.similarity_state = 0
        self.similarity_states = [0 for _ in range(len(self.palm_orientations))]
        self.mapped_similarity_states = [self.hand_pose_mapping[str(0)] for _ in range(len(self.palm_orientations))]
    
    def update_avg_distance(self):
        frame_data = self.get_frames_data()
        centroids = get_centroids(frame_data)
        avg_distance = get_avg_distance(centroids)
        if avg_distance > self.max_avg_distance:
            self.max_avg_distance = avg_distance


    def current_frame(self):
        return self.frames[-1]
    
    def previous_frame(self):
        return self.frames[-2]
    
    def previous_frames(self, n):
        return self.frames[-n:]
    

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    def __iter__(self):
        return iter(self.frames)
