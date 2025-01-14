import cv2
import mediapipe as mp 
import pickle
import math
import numpy as np

import os
import sys


def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.normpath(os.path.join(base_path, relative_path))


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Loading model from disk
# Model is trained on Random forest classifier with z score normalisation of data
with open(resource_path(r'.\utils\model_random_forest_z_score.pkl'), 'rb') as input:
    clf = pickle.load(input)

bg = None

frame_skip_counter = 0
hand_keypoints_stored = None
void_points_detection_counter = 0


def z_score_normalisation(dimension):
    """
    Normalise an array using z score method xi = (xi-mean)/standard deviation

    :param dimension: each dimension of the hand key points or a single array of same values
    :return: returns the array after normalising the values 
    """
    # Calculate the Standard Deviation in Python
    
    mean = np.mean(dimension)
    standard_deviation = np.std(dimension)

    z_score = (dimension - mean) / standard_deviation

    return z_score


def min_max_normaliser(dimension):
    """
    Formalise an array using min max method and to a range [0,1]

    :param dimension: each dimension of the hand key points or a single array of same values
    :return: returns the array after normalising the values 
    """
    normalizedData = (dimension-np.min(dimension))/(np.max(dimension)-np.min(dimension))
  
    # normalized data using min max value
    return normalizedData


def hand_point_normaliser(handpoints_1d_numpy_vector, z_score_flag=True):
    """
    From 1d hand key points find the x,y,z and normalise the values and convert back to the 1d array

    :param handpoints_1d: 1d values of handkey points
    :param z_score_flag: Set default value to True will call z score else min max
    :return: 1d value of hand key points after normalising  
    """ 
    # unpacking 1d array of handpoints to 3d
    if not handpoints_1d_numpy_vector:
        return None
    
    x = handpoints_1d_numpy_vector[0::3]
    y = handpoints_1d_numpy_vector[1::3]
    z = handpoints_1d_numpy_vector[2::3]
    
    if z_score_flag:
        #normalisation approaches
        # z score normalisation
        x = z_score_normalisation(x)
        y = z_score_normalisation(y)
        z = z_score_normalisation(z)
    else:   
        # min max normalisation
        x = min_max_normaliser(x)
        y = min_max_normaliser(y)
        z = min_max_normaliser(z)
    
    out = np.ones_like(handpoints_1d_numpy_vector)
    out[0::3] = x
    out[1::3] = y
    out[2::3] = z

    return out


def hand_perimeter(MP_Hand):
    """
    Calculate the perimeter for the hand palm

    :param MP_Hand: keypoint results of input hand
    :return: returns the perimeter of the bounding box for hand
    """

    for id, lm in enumerate(MP_Hand.landmark):
        if id == 5:
            point5 = [MP_Hand.landmark[id].x, MP_Hand.landmark[id].y, MP_Hand.landmark[id].z]
        if id == 9:
            point9 = [MP_Hand.landmark[id].x, MP_Hand.landmark[id].y, MP_Hand.landmark[id].z]
        if id == 13:
            point13 = [MP_Hand.landmark[id].x, MP_Hand.landmark[id].y, MP_Hand.landmark[id].z]
        if id == 17:
            point17 = [MP_Hand.landmark[id].x, MP_Hand.landmark[id].y, MP_Hand.landmark[id].z]
        if id == 0:
            point0 = [MP_Hand.landmark[id].x, MP_Hand.landmark[id].y, MP_Hand.landmark[id].z]

    perimeter = math.dist(point0, point5) + math.dist(point0, point17) + math.dist(point5, point9) + \
                math.dist(point9, point13) + math.dist(point13, point17)

    return perimeter


def hand_landmarks_to_1D_array(hand_landmarks):
    """
    hand_landmarks_to_1D_array take 21 landmark objects and return an array of points

    :param hand_landmarks: array of landmark objects
    :return: returns and array of x,x,z point coordinates
    """
    out = []
    if not hand_landmarks:
        return None
    for i in range(len(hand_landmarks.landmark)):
        out.append(hand_landmarks.landmark[i].x)
        out.append(hand_landmarks.landmark[i].y)
        out.append(hand_landmarks.landmark[i].z)
    return out


def hand_keypoint_extract_from_image(im):
    """
    This function extract the 21 hand key point features from the biggest hand in the input image.

    :param frame: current video frame
    :return: returns the 21 hand key points in mediapipe format
    """
    # Extract all hands from the image
    mediapipe_hand_results = hands.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # Iterate to determine the biggest hand using the perimiter
    max_perimeter = 0
    hand_index = 0
    if mediapipe_hand_results.multi_hand_landmarks is not None:
        # return only the biggest hand perimeter
        for i in range(len(mediapipe_hand_results.multi_hand_landmarks)):
            perimeter = hand_perimeter(mediapipe_hand_results.multi_hand_landmarks[i])
            if perimeter > max_perimeter:
                max_perimeter = perimeter
                hand_index = i

        hand_keypoints = mediapipe_hand_results.multi_hand_landmarks[hand_index]
    else:
        hand_keypoints = None

    # Adding some logic to minimize lost hand detection
    global void_points_detection_counter, hand_keypoints_stored
    persistence_frames_amount = 10
    if hand_keypoints != None:
        hand_keypoints_stored = hand_keypoints
        void_points_detection_counter = 0

    if hand_keypoints == None and void_points_detection_counter <= persistence_frames_amount:
        hand_keypoints = hand_keypoints_stored
        void_points_detection_counter = void_points_detection_counter + 1

    if hand_keypoints == None and void_points_detection_counter > persistence_frames_amount:
        hand_keypoints = None
        void_points_detection_counter = 0

    return hand_keypoints


def draw_hand_keypoint(frame, frame_skip=0):
    """
    This function detects hand from frame and display it to the screen
    :param frame: current video frame
    :return: frame marked the landmark of hands that has highest perimeter
    """
    global frame_skip_counter, hand_keypoints_stored

    if frame_skip == frame_skip_counter:
        hand_keypoints = hand_keypoint_extract_from_image(frame)
        hand_keypoints_stored = hand_keypoints
        frame_skip_counter = 0
    else:
        frame_skip_counter = frame_skip_counter + 1
        hand_keypoints = hand_keypoints_stored

    if hand_keypoints != None:
        mp_drawing.draw_landmarks(frame,
                                  hand_keypoints,
                                  mp_hands.HAND_CONNECTIONS,  # if we want to show the connected fingers within the box
                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                  mp_drawing_styles.get_default_hand_connections_style())

    return frame


def hand_gesture_prediction(frame):
    """
    :param frame: current video frame
    :return: predicted gesture for the hand points 0:Rock, 1:Paper, 2:Scissors, 3:Not a game gesture, 4:No landmarks
    """

    hand_keypoints = hand_keypoint_extract_from_image(frame)
    if not hand_keypoints:
        y_pred = 4  # no landmark detect case
        return y_pred
    
    hand_keypoints_1d = hand_landmarks_to_1D_array(hand_keypoints)
    hand_keypoints_1d_normalised = hand_point_normaliser(hand_keypoints_1d)
    #clf - random forest classifier with z score normaliser
    y_pred = clf.predict([hand_keypoints_1d_normalised])
  
    return y_pred[0]