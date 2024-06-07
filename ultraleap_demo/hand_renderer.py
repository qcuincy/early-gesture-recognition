import cv2

class HandRenderer:
    def __init__(self, output_image, hands_format="Dots", hands_colour=(0, 255, 0), circle_radius=3):
        self.output_image = output_image
        self.hands_format = hands_format
        self.hands_colour = hands_colour
        self.circle_radius = circle_radius
        
    def render_hand_data(self, hand_data):
        # Clear the previous image
        self.output_image[:, :] = 0

        for hand_dict in hand_data:
            digits = hand_dict['digits']

            for index_digit, digit_dict in enumerate(digits):
                bones = digit_dict['bones']

                for index_bone, bone_dict in enumerate(bones):
                    prev_joint = bone_dict['prev_joint']
                    next_joint = bone_dict['next_joint']

                    if self.hands_format == "Dots":
                        if prev_joint:
                            cv2.circle(self.output_image, prev_joint, self.circle_radius, self.hands_colour, -1)

                        if next_joint:
                            cv2.circle(self.output_image, next_joint, self.circle_radius, self.hands_colour, -1)

                    if self.hands_format == "Skeleton":
                        wrist = hand_dict['wrist']
                        elbow = hand_dict['elbow']

                        if wrist:
                            cv2.circle(self.output_image, wrist, self.circle_radius, self.hands_colour, -1)

                        if elbow:
                            cv2.circle(self.output_image, elbow, self.circle_radius, self.hands_colour, -1)

                        if wrist and elbow:
                            cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

                        bone_start = prev_joint
                        bone_end = next_joint

                        if bone_start:
                            cv2.circle(self.output_image, bone_start, self.circle_radius, self.hands_colour, -1)

                        if bone_end:
                            cv2.circle(self.output_image, bone_end, self.circle_radius, self.hands_colour, -1)

                        if bone_start and bone_end:
                            cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                        if ((index_digit == 0) and (index_bone == 0)) or (
                                (index_digit > 0) and (index_digit < 4) and (index_bone < 2)
                            ):
                                index_digit_next = index_digit + 1
                                digit_next = digits[index_digit_next]
                                bone_next = digit_next["bones"][index_bone]
                                bone_next_start = bone_next["prev_joint"]
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