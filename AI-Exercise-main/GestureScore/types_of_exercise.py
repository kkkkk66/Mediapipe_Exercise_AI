import numpy as np
from GestureScore.body_part_angle import BodyPartAngle
from GestureScore.utils import *
import cv2


class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    # 俯卧撑
    def push_up(self, counter, status, avg_score):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_left_arm()
        # 求两个手肘的平均角度
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2

        standard = [45, 170]
        standard_sum = 2 * sum(standard)

        if status:
            # 如果角度小于70度，说明动作标准，个数加一
            if avg_arm_angle < 70:
                counter += 1
                status = False
            avg_score = 0
        else:
            if avg_arm_angle > 160:
                status = True
            left_arm_score = (1 - abs((self.angle_of_the_left_arm() - standard[0]) / standard_sum)) * 100
            right_arm_score = (1 - abs((self.angle_of_the_right_arm() - standard[0]) / standard_sum)) * 100
            left_leg_score = (1 - abs((self.angle_of_the_left_leg() - standard[1]) / standard_sum)) * 100
            right_leg_score = (1 - abs((self.angle_of_the_right_leg() - standard[1]) / standard_sum)) * 100
            avg_score = (left_arm_score + right_arm_score + left_leg_score + right_leg_score) / 4

        return [counter, status, avg_score]

    # 引体向上
    def pull_up(self, counter, status, avg_score):
        nose = detection_body_part(self.landmarks, "NOSE")
        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        avg_shoulder_y = (left_elbow[1] + right_elbow[1]) / 2

        standard = [30, 45]
        standard_sum = 2 * sum(standard)

        if status:
            # 如果鼻子的纵坐标大于肩膀的纵坐标，说明动作标准，个数加一
            if nose[1] > avg_shoulder_y:
                counter += 1
                status = False
            left_arm_score = (1 - abs((self.angle_of_the_left_arm() - standard[0]) / standard_sum)) * 100
            right_arm_score = (1 - abs((self.angle_of_the_right_arm() - standard[0]) / standard_sum)) * 100
            left_shoulder_score = (1 - abs((self.angle_of_the_left_shoulder() - standard[1]) / standard_sum)) * 100
            right_shoulder_score = (1 - abs((self.angle_of_the_right_shoulder() - standard[1]) / standard_sum)) * 100
            avg_score = (left_arm_score + right_arm_score + left_shoulder_score + right_shoulder_score) / 4
        else:
            if nose[1] < avg_shoulder_y:
                status = True
            avg_score = 0

        return [counter, status, avg_score]

    # 下蹲
    def squat(self, counter, status, avg_score):
        left_leg_angle = self.angle_of_the_right_leg()
        right_leg_angle = self.angle_of_the_left_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2

        standard = [45, 50]
        standard_sum = 2 * sum(standard)

        if status:
            if avg_leg_angle < 70:
                counter += 1
                status = False
            avg_score = 0
        else:
            if avg_leg_angle > 160:
                status = True
            left_leg_score = (1 - abs((self.angle_of_the_left_leg() - standard[0]) / standard_sum)) * 100
            right_leg_score = (1 - abs((self.angle_of_the_right_leg() - standard[0]) / standard_sum)) * 100
            abdomen_score = (1 - abs((self.angle_of_the_abdomen() - standard[1]) / standard_sum)) * 100
            avg_score = (left_leg_score + right_leg_score + abdomen_score) / 3

        return [counter, status, avg_score]

    # 走路
    def walk(self, counter, status):
        right_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")

        if status:
            if left_knee[0] > right_knee[0]:
                counter += 1
                status = False

        else:
            if left_knee[0] < right_knee[0]:
                counter += 1
                status = True

        return [counter, status]

    # 仰卧起坐
    def sit_up(self, counter, status, avg_score):
        angle = self.angle_of_the_abdomen()

        standard = [45, 60]
        standard_sum = 2 * sum(standard)

        if status:
            if angle < 55:
                counter += 1
                status = False
            avg_score = 0
        else:
            if angle > 105:
                status = True
            abdomen_score = (1 - abs((self.angle_of_the_abdomen() - standard[0]) / standard_sum)) * 100
            left_leg_score = (1 - abs((self.angle_of_the_left_leg() - standard[1]) / standard_sum)) * 100
            right_leg_score = (1 - abs((self.angle_of_the_right_leg() - standard[1]) / standard_sum)) * 100
            avg_score = (abdomen_score + left_leg_score + right_leg_score) / 3

        return [counter, status, avg_score]

    # 运动的种类
    def calculate_exercise(self, exercise_type, counter, status, avg_score):
        if exercise_type == "push-up":
            counter, status, avg_score = TypeOfExercise(self.landmarks).push_up(
                counter, status, avg_score)
        elif exercise_type == "pull-up":
            counter, status, avg_score = TypeOfExercise(self.landmarks).pull_up(
                counter, status, avg_score)
        elif exercise_type == "squat":
            counter, status, avg_score = TypeOfExercise(self.landmarks).squat(
                counter, status, avg_score)
        elif exercise_type == "walk":
            counter, status = TypeOfExercise(self.landmarks).walk(
                counter, status)
        elif exercise_type == "sit-up":
            counter, status, avg_score = TypeOfExercise(self.landmarks).sit_up(
                counter, status, avg_score)

        return [counter, status, avg_score]

    def score_table(self, exercise, counter, status, avg_score, isPause):
        score_table = cv2.imread("./images/score_table.png")
        # LINE_AA表示使用的算法计算出的属于线段上的像素点，相邻的两点之间是大于八个方向的，比如十六个、三十二个
        cv2.putText(score_table, "Activity : " + exercise.replace("-", " "),
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2,
                    cv2.LINE_AA)
        # cv2.putText(image, text, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # 参数依次为：图片，文字，文字的位置，字体类型，字体大小，字体颜色，字体粗细
        cv2.putText(score_table, "Counter : " + str(counter), (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
        cv2.putText(score_table, "Status : " + str(status), (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
        if exercise == "push-up":
            cv2.putText(score_table, "Score : " + str(avg_score), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left arm : " + str(self.angle_of_the_left_arm()), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right arm : " + str(self.angle_of_the_right_arm()), (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left leg : " + str(self.angle_of_the_left_leg()), (10, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right leg : " + str(self.angle_of_the_right_leg()), (10, 570),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)

        elif exercise == "pull-up":
            cv2.putText(score_table, "Score : " + str(avg_score), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left arm : " + str(self.angle_of_the_left_arm()), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right arm : " + str(self.angle_of_the_right_arm()), (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left shoulder : " + str(self.angle_of_the_left_shoulder()), (10, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right shoulder : " + str(self.angle_of_the_right_shoulder()), (10, 570),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)

        elif exercise == "squat":
            cv2.putText(score_table, "Score : " + str(avg_score), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left leg : " + str(self.angle_of_the_left_leg()), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right leg : " + str(self.angle_of_the_right_leg()), (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of abdomen : " + str(self.angle_of_the_abdomen()), (10, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
        elif exercise == "walk":
            cv2.putText(score_table, "right_knee : " + str(detection_body_part(self.landmarks, "RIGHT_KNEE")), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "left_knee : " + str(detection_body_part(self.landmarks, "LEFT_KNEE")), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
        elif exercise == "sit-up":
            cv2.putText(score_table, "Score : " + str(avg_score), (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of left leg : " + str(self.angle_of_the_left_leg()), (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of right leg : " + str(self.angle_of_the_right_leg()), (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
            cv2.putText(score_table, "Angle of abdomen : " + str(self.angle_of_the_abdomen()), (10, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (182, 158, 128), 2, cv2.LINE_AA)
        # cv2.imShow()函数可以在窗口中显示图像。该窗口和图像的原始大小自适应（自动调整到原始尺寸）,
        # 第一个参数是一个窗口名称，它是一个字符串类型;
        # 第二个参数是图像,可以创建任意数量的窗口，但必须使用不同的窗口名称。
        cv2.imshow("Score Table", score_table)

