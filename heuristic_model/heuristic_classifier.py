import numpy as np
import math
import cv2

import sys
sys.path.insert(1, 'utility')
import utils 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class HeuristicClassifier:
    NOSTATE = 0
    STRAIGHT = 1
    BENT = 2
    CROSSED = 3

    OPEN_PALM = 0
    VICTORY = 1
    CLOSED_FIST = 2
    POINTING_UP = 3
    NOGESTURE = 4

    def __init__(self, frame_size):
        self.frame_width = frame_size[0]
        self.frame_height = frame_size[1]

    def __gesture(self, state):
        if (np.all(state == self.STRAIGHT)):
                return self.OPEN_PALM

        if (np.all(state == self.BENT)):
            return self.CLOSED_FIST
        
        thumb_state = state[0]
        index_state = state[1]
        middle_state = state[2]
        ring_state = state[3]
        pinky_state = state[4]

        if (thumb_state == self.CROSSED and
            index_state == self.STRAIGHT and
            middle_state == self.STRAIGHT and
            ring_state == self.BENT and
            pinky_state == self.BENT):
        
            return self.VICTORY
        
        if ((thumb_state == self.CROSSED and
             index_state == self.STRAIGHT and
             middle_state == self.BENT and
             ring_state == self.BENT and
             pinky_state == self.BENT) or \
            (thumb_state == self.BENT and
             index_state == self.STRAIGHT and
             middle_state == self.BENT and
             ring_state == self.BENT and
             pinky_state == self.BENT)):
        
            return self.POINTING_UP

        return self.NOGESTURE

    def process(self, landmarks):
        hand_img = np.ones((self.frame_width, self.frame_height)) * 255
        hand_img = utils.draw_hand(hand_img, landmarks)
     
        n_fingers = 5
        state = np.ones(n_fingers) * -1
    
        # Compute the distances
        center_kp = (landmarks[5][:2] + landmarks[9][:2] + landmarks[17][:2]) // 3
        norm_landmarks = utils.normalize_wrt_keypoint(landmarks, center_kp)
        center_kp = np.zeros(2)

        lm_index = 4
        for i in range(n_fingers):
            lm = norm_landmarks[lm_index]
            d = math.sqrt((lm[0] - center_kp[0])**2 + (lm[1]- center_kp[1])**2)

            if (d > 0.6):
                state[i] = self.STRAIGHT
            else:
                state[i] = self.BENT

            lm_index = lm_index + 4
        
        # Compute the angle
        h, _ = self.perspective_correction(hand_img, landmarks)  

        to_transform = [landmarks[0], landmarks[1], landmarks[5]]

        transformed = self.transform_points(to_transform, h)

        norm_transformed_landmarks = utils.normalize_wrt_keypoint(transformed, transformed[0])
        
        v1 = norm_transformed_landmarks[1]
        v2 = norm_transformed_landmarks[2]
        angle = utils.angle_between(v1, v2)

        if angle < 0.8:
            state[0] = self.CROSSED

        # Classify the gesture
        hand_gesture = self.__gesture(state)
        return hand_gesture

    def perspective_correction(self, frame, landmarks):
        dict_circles = dict()
        pts2 = []

        ref_img = cv2.imread('heuristic_model/HomographyRef.png', 0)

        # Compute the circles on homography reference image
        circles = cv2.HoughCircles(ref_img, cv2.HOUGH_GRADIENT, 1, 46, param1=50, param2=1, minRadius=4, maxRadius=30)
        if circles is not None:
            circles = np.uint16(np.round(circles[0, :]))

            for (x, y, r) in circles:
                dict_circles[r] = (x, y)
        
        # Sort the circles with respect to their radius
        for elem in sorted(dict_circles.items()):
            pts2.append(elem[1])
        pts2 = np.float32(pts2)

        pts1 = np.float32([[landmarks[0][:2]],
                           [landmarks[5][:2]],
                           [landmarks[9][:2]],
                           [landmarks[17][:2]]])

        h = cv2.getPerspectiveTransform(pts1, pts2)
        im_out = cv2.warpPerspective(frame, h, (ref_img.shape[1], ref_img.shape[0]))

        return h, im_out

    def transform_points(self, landmarks, h):
        points = []
        for lm in landmarks:
            points.append([[lm[0], lm[1]]])
        points = np.float32(points)
        
        transformed = cv2.perspectiveTransform(points, h)
        return transformed.squeeze()

    def evaluate(self, X_test, y_test, matrix=True):
        frame_width = self.frame_height
        frame_height = self.frame_width

        # Pre processing the landmarks
        n_landmarks = X_test.shape[0]
        y_pred = []
        for i in range(n_landmarks):
            one_d_landmarks = X_test[i]
            
            two_d_landmarks = one_d_landmarks.reshape(-1, 2)

            landmarks = []
            for j in range(21):
                lm = two_d_landmarks[j]
                lm_x = min(int(lm[0] * frame_width), frame_width - 1)
                lm_y = min(int(lm[1] * frame_height), frame_height - 1)
                lm = np.array([lm_x, lm_y], dtype=np.float64)

                landmarks.append(lm)
            
            res = self.process(landmarks)
            y_pred.append(res)

        y_pred = np.array(y_pred)

        # Compute the confusion matrix
        def print_confusion_matrix(y_true, y_pred):
            labels = sorted(list(set(y_true)))
            cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
            
            labels = ['OpenPalm', 'Victory', 'ClosedFist', 'PoitingUp', 'NoGesture']
            df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
        
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
            ax.set_ylim(len(set(y_true)), 0)
            plt.show()
        
        print('Classification Report')
        print(classification_report(y_test, y_pred))

        if matrix:
            print_confusion_matrix(y_test, y_pred)

    
if __name__ == '__main__':
    # Run the current file in order to evaluate the heuristic classifier

    RANDOM_SEED = 142
    dataset_path = './nn_model/keypoint.csv'

    X_dataset = np.loadtxt(dataset_path, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset_path, delimiter=',', dtype='int32', usecols=(0))

    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

    model = HeuristicClassifier((1028, 720))
    model.evaluate(X_test, y_test)