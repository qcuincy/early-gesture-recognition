from ultraleap_demo.load_demo import *
from data.dhg import DHG
import numpy as np
import argparse
import leap
import cv2
import os

data_dir = os.path.join(os.getcwd(), "data")
dhg_data = DHG(data_dir)

top_dirs = 3
window_size = 16
stationary_threshold_ratio = 1.5
similarity_lookback = 2
similarity_threshold = 0.9
sequence_length = 32
output_window = 1
max_frames = 100
target_length = 32
test_size = 0.4
confidence_threshold = 0.01
device = 'cpu'

def main(
        top_dirs,
        window_size,
        stationary_threshold_ratio,
        similarity_lookback,
        similarity_threshold,
        sequence_length,
        output_window,
        max_frames,
        target_length,
        test_size,
        confidence_threshold,
        device,
):

    moving_direction_indexes = [6, 18, 10, 14, 0, 1,3,7,19]

    orientation_mapping = {'up': 0, 'down': 1, 'opposite': 2}
    inverted_orientation_mapping = {v: k for k, v in orientation_mapping.items()}
    direction_mapping_2d = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stationary': 4}
    inverted_direction_mapping_2d = {v: k for k, v in direction_mapping_2d.items()}

    performer_states_dict = get_performer_states_dict(dhg_data, smoother=True)
    normalized_performer_states_dict = normalize_performer_state(performer_states_dict, by_subject=True)
    gestures = [f"gesture_{i}" for i in range(7, 11)]
    train_lookup_table, test_lookup_table = make_train_test_lookup_table(normalized_performer_states_dict, test_size=test_size, gestures=gestures)

    normalized_train_lookup_table = normalize_lookup_table(train_lookup_table, target_length)
    prepared_train_lookup_table = lookup_table_tensor(normalized_train_lookup_table)

    frames = Frames(
        handpose=handpose_filtered, 
        sequence_length=sequence_length, 
        max_frames=max_frames, 
        window_size=window_size,
        top_dirs=top_dirs,
        similarity_lookback=similarity_lookback,
        stationary_threshold_ratio=stationary_threshold_ratio,
        moving_direction_indexes=moving_direction_indexes,
        similarity_threshold=similarity_threshold,
        moving_direction_mapping=moving_direction_state_mapping, 
        palm_orientation_mapping=palm_orientation_state_mapping, 
        hand_pose_mapping=hand_pose_state_mapping)

    _TRACKING_MODES = {
        leap.TrackingMode.Desktop: "Desktop",
        leap.TrackingMode.HMD: "HMD",
        leap.TrackingMode.ScreenTop: "ScreenTop",
    }


    class Canvas:
        def __init__(self):
            self.name = "Early Hand Gesture Recognition Demo"
            self.screen_size = [500, 700]
            self.hands_colour = (255, 255, 255)
            self.font_colour = (220, 220, 220)
            self.gesture_colour = (0, 255, 40)
            self.hands_format = "Skeleton"
            self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
            self.tracking_mode = None
            self.classification_interval = 2000 # classify gesture every 2 seconds
            self.classification_timer = leap.get_now() / 1000
            self.last_gesture = 'Unknown'
            self.last_gesture_confidence = 1.0

        def set_tracking_mode(self, tracking_mode):
            self.tracking_mode = tracking_mode

        def toggle_hands_format(self):
            self.hands_format = "Dots" if self.hands_format == "Skeleton" else "Skeleton"
            print(f"Set hands format to {self.hands_format}")

        def get_joint_position(self, bone):
            if bone:
                return int(bone.x + (self.screen_size[1] / 2)), int(bone.z + (self.screen_size[0] / 2))
            else:
                return None
            
            

        def render_hands(self, event):
            moving_string = "Moving: None"
            orientation_string = "Orientation: None"
            pose_string = "Pose: None"
            classified_gesture_string = f"Gesture: {self.last_gesture}"
            gesture_confidence_string = f"Confidence: {self.last_gesture_confidence:.2f}"
            if (len(frames.mapped_moving_directions) > 0) and len(event.hands) > 0:
                moving_string = f"moving: {inverted_direction_mapping_2d[int(inverted_moving_direction_state_mapping[frames.mapped_moving_directions[-1]])]}"
                orientation_string = f"orientation: {inverted_orientation_mapping[int(inverted_palm_orientation_state_mapping[frames.mapped_palm_orientations[-1]])]}"
                pose_string = f"hand pose: {frames.similarity_states[-1]}"
                
                current_time = leap.get_now() / 1000
                if current_time - self.classification_timer > self.classification_interval:
                    classified_gesture,confidence = get_gesture_result(frames, sequence_length, output_window, prepared_train_lookup_table, target_length, threshold=confidence_threshold, device=device)
                    if classified_gesture != self.last_gesture:
                        self.last_gesture = classified_gesture
                        self.last_gesture_confidence = confidence
                    classified_gesture_string = f"Gesture: {self.last_gesture}"
                    gesture_confidence_string = f"Confidence: {self.last_gesture_confidence:.2f}"
                    self.classification_timer = current_time

                    if frames.similarity_states[-1] == 5:
                        frames.reset_similarity_state()
                
            # Clear the previous image
            self.output_image[:, :] = 0

            cv2.putText(
                self.output_image,
                f"Tracking Mode: {_TRACKING_MODES[self.tracking_mode]}",
                (10, self.screen_size[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )

            cv2.putText(
                self.output_image,
                moving_string,
                (self.screen_size[1] - 200, self.screen_size[0] - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                orientation_string,
                (self.screen_size[1] - 200, self.screen_size[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                pose_string,
                (self.screen_size[1] - 200, self.screen_size[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                classified_gesture_string,
                (self.screen_size[1] - 200, self.screen_size[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.gesture_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                gesture_confidence_string,
                (self.screen_size[1] - 200, self.screen_size[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.gesture_colour,
                1,
            )

            if len(event.hands) == 0:
                return

            for i in range(0, len(event.hands)):
                hand = event.hands[i]
                landmarks = Landmarks(hand)
                frame = Frame(landmarks)

                frames.add_frame(frame, True)

                for index_digit in range(0, 5):
                    digit = hand.digits[index_digit]
                    for index_bone in range(0, 4):
                        bone = digit.bones[index_bone]
                        if self.hands_format == "Dots":
                            prev_joint = self.get_joint_position(bone.prev_joint)
                            next_joint = self.get_joint_position(bone.next_joint)
                            if prev_joint:
                                cv2.circle(self.output_image, prev_joint, 2, self.hands_colour, -1)

                            if next_joint:
                                cv2.circle(self.output_image, next_joint, 2, self.hands_colour, -1)


                        if self.hands_format == "Skeleton":
                            wrist = self.get_joint_position(hand.arm.next_joint)
                            elbow = self.get_joint_position(hand.arm.prev_joint)
                            if wrist:
                                cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)

                            if elbow:
                                cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)

                            if wrist and elbow:
                                cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

                            bone_start = self.get_joint_position(bone.prev_joint)
                            bone_end = self.get_joint_position(bone.next_joint)

                            if bone_start:
                                cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)

                            if bone_end:
                                cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)

                            if bone_start and bone_end:
                                cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                            if ((index_digit == 0) and (index_bone == 0)) or (
                                (index_digit > 0) and (index_digit < 4) and (index_bone < 2)
                            ):
                                index_digit_next = index_digit + 1
                                digit_next = hand.digits[index_digit_next]
                                bone_next = digit_next.bones[index_bone]
                                bone_next_start = self.get_joint_position(bone_next.prev_joint)
                                if bone_start and bone_next_start:
                                    cv2.line(
                                        self.output_image,
                                        bone_start,
                                        bone_next_start,
                                        self.hands_colour,
                                        2,
                                    )

                            if index_bone == 0 and bone_start and wrist:
                                cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)


    class TrackingListener(leap.Listener):
        def __init__(self, canvas):
            self.canvas = canvas

        def on_connection_event(self, event):
            pass

        def on_tracking_mode_event(self, event):
            self.canvas.set_tracking_mode(event.current_tracking_mode)
            print(f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}")

        def on_device_event(self, event):
            try:
                with event.device.open():
                    info = event.device.get_info()
            except leap.LeapCannotOpenDeviceError:
                info = event.device.get_info()

            print(f"Found device {info.serial}")

        def on_tracking_event(self, event):
            self.canvas.render_hands(event)



    canvas = Canvas()

    print(canvas.name)
    print("")
    print("Press <key> in visualiser window to:")
    print("  q: Exit")
    print("  h: Select HMD tracking mode")
    print("  s: Select ScreenTop tracking mode")
    print("  d: Select Desktop tracking mode")
    print("  f: Toggle hands format between Skeleton/Dots")

    tracking_listener = TrackingListener(canvas)

    connection = leap.Connection()
    connection.add_listener(tracking_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        canvas.set_tracking_mode(leap.TrackingMode.Desktop)

        while running:
            cv2.imshow(canvas.name, canvas.output_image)

            key = cv2.waitKey(1)

            if key == ord("q"):
                running = False
                cv2.destroyAllWindows()

            elif key == ord("h"):
                connection.set_tracking_mode(leap.TrackingMode.HMD)
            elif key == ord("s"):
                connection.set_tracking_mode(leap.TrackingMode.ScreenTop)
            elif key == ord("d"):
                connection.set_tracking_mode(leap.TrackingMode.Desktop)
            elif key == ord("f"):
                canvas.toggle_hands_format()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Customize parameters for the demo.')
    parser.add_argument('-td', '--top_dirs', type=int, help='Top number of directions to consider when determining the moving direction over the sliding window', default=3)
    parser.add_argument('-ws', '--window_size', type=int, help='Size of the sliding window for calculating the moving direction', default=16)
    parser.add_argument('-str', '--stationary_threshold_ratio', type=float, help='Stationary threshold to determine if the hand is moving or stationary', default=1.5)
    parser.add_argument('-sl', '--similarity_lookback', type=int, help='How many frames to look back to when calculating the similarity between the current frame and the chosen frame', default=2)
    parser.add_argument('-st', '--similarity_threshold', type=float, help='Similarity threshold to determine if the current frame is similar to the chosen frame', default=0.9)
    parser.add_argument('-seq', '--sequence_length', type=int, help='The length of the sequence to decide the Transformer model', default=32)
    parser.add_argument('-ow', '--output_window', type=int, help='The output window to decide the Transformer model', default=1)
    parser.add_argument('-tl', '--target_length', type=int, help='Target length for sequence normalisation', default=32)
    parser.add_argument('-mf', '--max_frames', type=int, help='Maximum number of frames to store in Frames object', default=100)
    parser.add_argument('-ts', '--test_size', type=float, help='Test size for lookup table', default=0.4)
    parser.add_argument('-ct', '--confidence_threshold', type=float, help='Gesture classification confidence threshold, if the confidence is below this threshold, the gesture will be classified as Unknown', default=0.01)
    parser.add_argument('-d', '--device', type=str, help='Device to run the model on, requires a CUDA enabled device for GPU', default='cpu')
    parser.add_argument('-T', '--classification_timeout', type=int, help='The time interval to classify gesture', default=2000)

    args = parser.parse_args()
    main(
        args.top_dirs,
        args.window_size,
        args.stationary_threshold_ratio,
        args.similarity_lookback,
        args.similarity_threshold,
        args.sequence_length,
        args.output_window,
        args.max_frames,
        args.target_length,
        args.test_size,
        args.confidence_threshold,
        args.device,
    )
    def get_performer_states_dict(dhg_data, smoother=True):
        performer_states_dict = {}
        for subject in dhg_data.subjects:
            for gesture in dhg_data.gestures:
                for sample in dhg_data.samples:
                    performer_states_dict[(subject, gesture, sample)] = dhg_data.get_performer_state(subject, gesture, sample, smoother=smoother)
        return performer_states_dict

    def normalize_performer_state(performer_states_dict, by_subject=True):
        normalized_performer_states_dict = {}
        for key, performer_state in performer_states_dict.items():
            if by_subject:
                subject = key[0]
                if subject not in normalized_performer_states_dict:
                    normalized_performer_states_dict[subject] = {}
                normalized_performer_states_dict[subject][key[1:]] = performer_state
            else:
                normalized_performer_states_dict[key] = performer_state
        return normalized_performer_states_dict

    def make_train_test_lookup_table(normalized_performer_states_dict, test_size=0.4, gestures=None):
        train_lookup_table = []
        test_lookup_table = []
        for subject, gesture_samples in normalized_performer_states_dict.items():
            for gesture, samples in gesture_samples.items():
                if gestures is None or gesture in gestures:
                    num_samples = len(samples)
                    num_test_samples = int(num_samples * test_size)
                    test_samples = samples[:num_test_samples]
                    train_samples = samples[num_test_samples:]
                    for sample in test_samples:
                        test_lookup_table.append((subject, gesture, sample))
                    for sample in train_samples:
                        train_lookup_table.append((subject, gesture, sample))
        return train_lookup_table, test_lookup_table

    def normalize_lookup_table(lookup_table, target_length):
        normalized_lookup_table = []
        for subject, gesture, sample in lookup_table:
            normalized_lookup_table.append((subject, gesture, sample[:target_length]))
        return normalized_lookup_table

    def lookup_table_tensor(lookup_table):
        tensor = []
        for subject, gesture, sample in lookup_table:
            tensor.append(sample)
        return np.array(tensor)