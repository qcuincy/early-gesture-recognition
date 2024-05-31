from ultraleap_demo.load_demo import *
from data.dhg import DHG
import os
import leap
import numpy as np
import cv2
import argparse


top_dirs = 3
window_size = 16
stationary_threshold_ratio = 1.5
similarity_lookback = 2
similarity_threshold = 0.9
sequence_length = 32
output_window = 1
max_frames = 100
target_length = 32
device = 'cpu'
data_path = os.getcwd()
def main(
        data_path,
        top_dirs,
        window_size,
        stationary_threshold_ratio,
        similarity_lookback,
        similarity_threshold,
        sequence_length,
        max_frames,
):
    moving_direction_indexes = [6, 18, 10, 14, 0, 1,3,7,19]

    orientation_mapping = {'up': 0, 'down': 1, 'opposite': 2}
    inverted_orientation_mapping = {v: k for k, v in orientation_mapping.items()}
    direction_mapping_2d = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stationary': 4}
    inverted_direction_mapping_2d = {v: k for k, v in direction_mapping_2d.items()}

    gesture_mapped_names = {
        "gesture_7":"Swipe Right",
        "gesture_8":"Swipe Left",
        "gesture_9":"Swipe Up",
        "gesture_10":"Swipe Down",
    }


    gestures = [f"gesture_{i}" for i in range(7, 11)]

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

    class Params:
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.gesture_to_record = "Swipe Right"
            self.recording = False
            self.record_timeout = 100 # add a frame every 0.1 seconds
            self.current_time = leap.get_now() / 1000

            self.current_tracking_mode = leap.TrackingMode.Desktop
            self.top_dirs = 3
            self.window_size = 16
            self.stationary_threshold_ratio = 1.5
            self.similarity_lookback = 2
            self.similarity_threshold = 0.9
            self.sequence_length = 32
            self.output_window = 1
            self.max_frames = 100
            self.add_every_n_frame = 10
            self.frame_num = 0
            self.interpolated_frame_num = 0
            self.target_length = 32
            self.test_size = 0.4
            self.confidence_threshold = 0.01
            self.device = 'cpu'
            self.moving_direction_indexes = [6, 18, 10, 14, 0, 1,3,7,19]
            self.orientation_mapping = {'up': 0, 'down': 1, 'opposite': 2}
            self.inverted_orientation_mapping = {v: k for k, v in orientation_mapping.items()}
            self.direction_mapping_2d = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stationary': 4}
            self.inverted_direction_mapping_2d = {v: k for k, v in direction_mapping_2d.items()}
            self.gesture_mapped_names = {
                "gesture_7":"Swipe Right",
                "gesture_8":"Swipe Left",
                "gesture_9":"Swipe Up",
                "gesture_10":"Swipe Down",
            }
            self.frames = Frames(
                handpose=handpose_filtered, 
                sequence_length=sequence_length, 
                max_frames=max_frames, 
                window_size=window_size, 
                similarity_lookback=similarity_lookback,
                stationary_threshold_ratio=stationary_threshold_ratio,
                moving_direction_indexes=moving_direction_indexes,
                similarity_threshold=similarity_threshold,
                moving_direction_mapping=moving_direction_state_mapping, 
                palm_orientation_mapping=palm_orientation_state_mapping, 
                hand_pose_mapping=hand_pose_state_mapping)
            self._TRACKING_MODES = {
                leap.TrackingMode.Desktop: "Desktop",
                leap.TrackingMode.HMD: "HMD",
                leap.TrackingMode.ScreenTop: "ScreenTop",
            }


    class Canvas:
        def __init__(self, params):
            self.params = params
            self.data_dir = params.data_dir
            self.gesture_file_name = None
            self.name = "Hand Gesture Data Collection"
            self.screen_size = [500, 700]
            self.hands_colour = (255, 255, 255)
            self.font_colour = (220, 220, 220)
            self.gesture_colour = (0, 255, 40)
            self.key_colour = (20, 60, 255)
            self.option_1_colour = (0, 255, 40)
            self.option_2_colour = (220, 220, 220)
            self.option_3_colour = (220, 220, 220)
            self.option_4_colour = (220, 220, 220)
            self.record_button_colour = (40, 255, 40)
            self.stop_button_colour = (220, 220, 220)# (40, 40, 255)
            self.hands_format = "Skeleton"
            self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
            self.tracking_mode = None

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
            if (len(self.params.frames.mapped_moving_directions) > 0) and len(event.hands) > 0:
                moving_direction = self.params.frames.mapped_moving_directions[-1]
                palm_orientation = self.params.frames.mapped_palm_orientations[-1]
                hand_pose = self.params.frames.similarity_states[-1]

                frame_data = np.array([moving_direction, palm_orientation, hand_pose])
                current_time = leap.get_now() / 1000
                if self.params.recording:
                    if current_time - self.params.current_time > self.params.record_timeout:
                        self.save_frame_to_gesture_file(frame_data)
                        self.params.current_time = current_time

                moving_string = f"moving: {self.params.inverted_direction_mapping_2d[int(inverted_moving_direction_state_mapping[moving_direction])]}"
                orientation_string = f"orientation: {self.params.inverted_orientation_mapping[int(inverted_palm_orientation_state_mapping[palm_orientation])]}"
                pose_string = f"hand pose: {hand_pose}"
                
            # Clear the previous image
            self.output_image[:, :] = 0

            cv2.putText(
                self.output_image,
                f"Change Gesture To Record:",
                (self.screen_size[1] - 225, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )

            cv2.putText(
                self.output_image,
                f"  1: Swipe Right",
                (self.screen_size[1] - 150, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.option_1_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                f"  2: Swipe Left",
                (self.screen_size[1] - 150, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.option_2_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                f"  3: Swipe Up",
                (self.screen_size[1] - 150, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.option_3_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                f"  4: Swipe Down",
                (self.screen_size[1] - 150, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.option_4_colour,
                1,
            )

            cv2.putText(
                self.output_image,
                "Selected Gesture:",
                (self.screen_size[1] - 225, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                f"{self.params.gesture_to_record}",
                (self.screen_size[1] - 130, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.gesture_colour,
                1,
            )

            cv2.putText(
                self.output_image,
                f"Press:",
                (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )

            cv2.putText(
                self.output_image,
                f"  q: Exit",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                f"  r: Start Recording",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.record_button_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                f"  s: Stop Recording",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.stop_button_colour,
                1,
            )

            cv2.putText(
                self.output_image,
                f"Tracking Mode: {self.params._TRACKING_MODES[self.tracking_mode]}",
                (10, self.screen_size[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )

            cv2.putText(
                self.output_image,
                moving_string,
                (self.screen_size[1] - 170, self.screen_size[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                orientation_string,
                (self.screen_size[1] - 170, self.screen_size[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )
            cv2.putText(
                self.output_image,
                pose_string,
                (self.screen_size[1] - 170, self.screen_size[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )

            if len(event.hands) == 0:
                return

            for i in range(0, len(event.hands)):
                hand = event.hands[i]
                landmarks = Landmarks(hand)
                frame = Frame(landmarks)

                self.params.frames.add_frame(frame, True)

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

        def make_gesture_dir(self, gesture_name):
            gesture_dir_name = gesture_name.lower().replace(" ", "_")
            if not os.path.exists(os.path.join(self.data_dir, gesture_dir_name)):
                os.makedirs(os.path.join(self.data_dir, gesture_dir_name))
            return os.path.join(self.data_dir, gesture_dir_name)
        
        def make_gesture_file(self, gesture_name):
            gesture_dir = self.make_gesture_dir(gesture_name)
            gesture_name = gesture_name.lower().replace(" ", "_")
            # get the number of files in the directory
            file_count = len([name for name in os.listdir(gesture_dir) if os.path.isfile(os.path.join(gesture_dir, name))])
            file_name = os.path.join(gesture_dir, f"{gesture_name}_{file_count}.npy")
            return file_name
        
        def save_frame_to_gesture_file(self, frame_data):
            # Load existing data
            if os.path.exists(self.gesture_file_name):
                existing_data = np.load(self.gesture_file_name)
                data = np.vstack((existing_data, frame_data))
            else:
                data = frame_data

            # Save the data back to the file
            np.save(self.gesture_file_name, data)




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


    def make_data_dir(data_path):
        data_dir = os.path.join(data_path, "hand_gesture_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir



    data_dir = make_data_dir(data_path)

    params = Params(data_dir)
    canvas = Canvas(params)

    print(canvas.name)
    print("")
    print("Press <key> to:")
    print("  q: Exit")
    print("  r: Start Recording")
    print("  s: Stop Recording")
    print("\nGesture To Record:")
    print("  1: Swipe Right")
    print("  2: Swipe Left")
    print("  3: Swipe Up")
    print("  4: Swipe Down")


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
            elif key == ord("1"):
                canvas.params.gesture_to_record = "Swipe Right"
                canvas.option_1_colour = (0, 255, 40)
                canvas.option_2_colour = (220, 220, 220)
                canvas.option_3_colour = (220, 220, 220)
                canvas.option_4_colour = (220, 220, 220)
            elif key == ord("2"):
                canvas.params.gesture_to_record = "Swipe Left"
                canvas.option_2_colour = (0, 255, 40)
                canvas.option_1_colour = (220, 220, 220)
                canvas.option_3_colour = (220, 220, 220)
                canvas.option_4_colour = (220, 220, 220)
            elif key == ord("3"):
                canvas.params.gesture_to_record = "Swipe Up"
                canvas.option_3_colour = (0, 255, 40)
                canvas.option_1_colour = (220, 220, 220)
                canvas.option_2_colour = (220, 220, 220)
                canvas.option_4_colour = (220, 220, 220)
            elif key == ord("4"):
                canvas.params.gesture_to_record = "Swipe Down"
                canvas.option_4_colour = (0, 255, 40)
                canvas.option_1_colour = (220, 220, 220)
                canvas.option_2_colour = (220, 220, 220)
                canvas.option_3_colour = (220, 220, 220)
            elif key == ord("r"):
                gesture_file_name = canvas.make_gesture_file(canvas.params.gesture_to_record)
                canvas.gesture_file_name = gesture_file_name
                canvas.params.recording = True
                print(f"Recording gesture {canvas.params.gesture_to_record} to {gesture_file_name}")
                canvas.stop_button_colour = (40, 40, 255)
                canvas.record_button_colour = (220, 220, 220)
            elif key == ord("s"):
                canvas.params.recording = False
                print(f"Stopped recording gesture {canvas.params.gesture_to_record}")
                canvas.record_button_colour = (40, 255, 40)
                canvas.stop_button_colour = (220, 220, 220)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Customize parameters for data collection.')
    parser.add_argument('-dp', '--data_path', type=str, help='Default path to store gesture sequences.', default=data_path)
    parser.add_argument('-td', '--top_dirs', type=int, help='Top number of directions to consider when determining the moving direction over the sliding window', default=3)
    parser.add_argument('-ws', '--window_size', type=int, help='Size of the sliding window for calculating the moving direction', default=16)
    parser.add_argument('-str', '--stationary_threshold_ratio', type=float, help='Stationary threshold to determine if the hand is moving or stationary', default=1.5)
    parser.add_argument('-sl', '--similarity_lookback', type=int, help='How many frames to look back to when calculating the similarity between the current frame and the chosen frame', default=2)
    parser.add_argument('-st', '--similarity_threshold', type=float, help='Similarity threshold to determine if the current frame is similar to the chosen frame', default=0.9)
    parser.add_argument('-seq', '--sequence_length', type=int, help='The length of the sequence to decide the Transformer model', default=32)
    parser.add_argument('-mf', '--max_frames', type=int, help='Maximum number of frames to store in Frames object', default=100)

    args = parser.parse_args()
    main(
        args.data_path,
        args.top_dirs,
        args.window_size,
        args.stationary_threshold_ratio,
        args.similarity_lookback,
        args.similarity_threshold,
        args.sequence_length,
        args.max_frames,
    )