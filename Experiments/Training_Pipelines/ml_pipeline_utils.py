import math
import copy
from os import path
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from scipy.spatial.transform import Rotation
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8)
def hands_keypoint_extract_from_file(filepath, flip_flag = False):
    """
    hands_keypoint_extract_from_file will extract the 21 hand key point features from an image.

    :param filepath: image file path
    :return: returns the 21 hand key points
    """

    im = cv2.imread(filepath)

    if flip_flag:
        im = cv2.flip(im, 1)
    
    # Error flag will handle situations when the dataset does not have the path file or error in loading dataset
    error_flag = False

    try:
        mediapipe_hand_results = hands.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    except:
        print("An exception occurred")
        error_flag = True

    hands_key_points = []

    max_perimeter = 0
    hand_index = 0

    # Iterate to determine the biggest hand
    if not error_flag:
        if mediapipe_hand_results.multi_hand_landmarks is not None:
            for i in range(len(mediapipe_hand_results.multi_hand_landmarks)):
                perimeter = hand_perimeter(mediapipe_hand_results, i)
                if perimeter > max_perimeter:
                    max_perimeter = perimeter
                    hand_index = i

        # Iterates to select the biggest hand
        if mediapipe_hand_results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(mediapipe_hand_results.multi_hand_landmarks):
                if hand_no == hand_index:
                    hands_key_points.append(hand_landmarks)
    else:
        hands_key_points = None        
    # Adding cases to handle the no hands in the screen
    if len(hands_key_points) == 0:
        hands_key_points = None
    else:
        hands_key_points = hands_key_points[0]

    return hands_key_points


def hand_perimeter(mediapipe_hand_results, handNo):
    """
    hand_perimeter will define bounding boxes for hands

    :param mediapipe_hand_results: keypoint results of all hands
    :param handNo: The hand number

    :return: returns the perimeter of the bounding box for hand #handNo
    """

    if mediapipe_hand_results.multi_hand_landmarks is not None:
        myHand = mediapipe_hand_results.multi_hand_landmarks[handNo]
        point5 = 0
        point9 = 0
        point13 = 0
        point17 = 0
        point0 = 0
        for id, lm in enumerate(myHand.landmark):
            if id == 5:
                point5 = [myHand.landmark[id].x, myHand.landmark[id].y, myHand.landmark[id].z]
            if id == 9:
                point9 = [myHand.landmark[id].x, myHand.landmark[id].y, myHand.landmark[id].z]
            if id == 13:
                point13 = [myHand.landmark[id].x, myHand.landmark[id].y, myHand.landmark[id].z]
            if id == 17:
                point17 = [myHand.landmark[id].x, myHand.landmark[id].y, myHand.landmark[id].z]
            if id == 0:
                point0 = [myHand.landmark[id].x, myHand.landmark[id].y, myHand.landmark[id].z]
        perimeter = math.dist(point0, point5) + math.dist(point0, point17) + math.dist(point5, point9) + math.dist(
            point9, point13) + math.dist(point13, point17)
        return perimeter


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


def hand_point_normaliser(handpoints_1d_numpy_vector, z_score_flag = True):
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


def Data_read_from_csv(csv_file_path):
    """
    Data_read_from_csv will read all the lines in a csv file and extract the paths and labels

    :param csv_file_path: csv file path
    :return: returns a list of image path and a list of labels
    """
    list_img_path = []
    list_labels = []
    out = None
    with open(csv_file_path, 'r') as fd:
        for row in fd:
            img_path = row.rstrip().split(',')
            list_img_path.append(img_path[0])
            list_labels.append(img_path[1])
        out = [list_img_path, list_labels]
    return out


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


def hand_landmarks_3D_Rotation(hand_landmarks, x_rotation_angle, y_rotation_angle, z_rotation_angle):
    """
    hand_landmarks_3D_Rotation take 21 landmark objects and return an array of points

    :param hand_landmarks: array of landmark points
    :param x_rotation_angle: rotation angle in x direction
    :param y_rotation_angle: rotation angle in y direction
    :param z_rotation_angle: rotation angle in z direction
    :return: returns and array of x,x,z rotated point coordinates
    """
    hand_landmarks = copy.deepcopy(hand_landmarks)
    for i in range(len(hand_landmarks.landmark)):
        original_point = (hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z)
        # rotation in each principal direction:
        rot = Rotation.from_euler('xyz', (x_rotation_angle, y_rotation_angle, z_rotation_angle), degrees=True)
        rotated_point = rot.apply(original_point)  # Rotated points
        (hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z) = rotated_point

    return hand_landmarks


def draw_all_lines_between_points(ax, x, y, z):
    """
    draw_all_lines_between_points will draw the connections between the landmarks

    :ax: plot where to add the elements
    :x: x coordinates array
    :y: y coordinates array
    :z: z coordinates array
    """
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
        [9, 10], [10, 11], [11, 12],  # Middle finger
        [13, 14], [14, 15], [15, 16],  # Ring Finger
        [0, 17], [17, 18], [18, 19], [19, 20],  # Pinky
        [5, 9], [9, 13], [13, 17],  # Palm
    ]

    for connection in connections:
        ind_1 = connection[0]
        ind_2 = connection[1]
        x_d = [x[ind_1], x[ind_2]]
        y_d = [y[ind_1], y[ind_2]]
        z_d = [z[ind_1], z[ind_2]]
        ax.plot(x_d, y_d, z_d, color='black')


def hand_landmarks_3D_plotting(hand_landmarks, title):
    """
    hand_landmarks_3D_plotting take 21 hand key points and will create a 3d graph

    :param hand_landmarks: array of landmark points
    """
    # Extracting x y z coordinates
    x = []
    y = []
    z = []
    for i in range(len(hand_landmarks.landmark)):
        x.append(hand_landmarks.landmark[i].x)
        y.append(hand_landmarks.landmark[i].y)
        z.append(hand_landmarks.landmark[i].z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    draw_all_lines_between_points(ax, x, y, z)
    if title is not None:
        plt.title(title)
    plt.show()


class data_generator():

    def __init__(self, images_path, maximum_size, tag, randomize_img_list=True, flip_flag = False):
        self.index = 0

        if randomize_img_list:
            random.seed(4)
            random.shuffle(images_path)
            self.images = images_path
        else:
            self.images = images_path

        self.dataset_size = len(images_path)
        self.maximum_size = maximum_size
        self.tag = tag
        self.flip_flag = flip_flag

    def __iter__(self):
        self.index = 0
        self.ending_index = 0
        return self

    ######For testing purposes######
    def set_index(self, index):
        self.index = index

    ################################


    def __next__(self):
        hand_keypoints = None
        if self.ending_index == self.maximum_size:
            raise StopIteration
        else:
            self.ending_index += 1

        if self.index >= (self.dataset_size - 2):
            self.index = 0

        while hand_keypoints is None or path.exists("." + self.images[self.index]) == False:
        
            if path.exists("." + self.images[self.index]):
                hand_keypoints = hands_keypoint_extract_from_file("." + self.images[self.index],self.flip_flag)
            else:
                hand_keypoints = None
            self.index += 1

        x_angle = random.randint(0, 360)
        y_angle = random.randint(0, 360)
        z_angle = random.randint(0, 360)
        rotated_keypoints = hand_landmarks_3D_Rotation(hand_keypoints, x_angle, y_angle, z_angle)
        return rotated_keypoints


def test_3D_rotation_visualization(file_path):
    """
    test_3D_rotation_visualization will test the rotation of the points in a given direction

    :filepath:image file path
    """
    landmarks_original = hands_keypoint_extract_from_file(file_path)
    landmarks_rotated = hand_landmarks_3D_Rotation(landmarks_original, 30, 30, 30)
    hand_landmarks_3D_plotting(landmarks_original, "Original")
    hand_landmarks_3D_plotting(landmarks_rotated, "Rotated")


def test_iterator():
    all_gestures = Data_read_from_csv("./data_source_list.csv")
    all_imgs = all_gestures[0]
    all_labels = all_gestures[1]
    rock_imgs = []
    paper_imgs = []
    scissors_imgs = []
    ngg_imgs = []
    for i in range(len(all_imgs)):
        if all_labels[i] == "rock":
            rock_imgs.append(all_imgs[i])
        elif all_labels[i] == "paper":
            paper_imgs.append(all_imgs[i])
        elif all_labels[i] == "scissors":
            scissors_imgs.append(all_imgs[i])
        else:
            ngg_imgs.append(all_imgs[i])

    max = len(ngg_imgs)
    rock = data_generator(rock_imgs, max, "rock")
    rock_iterator = iter(rock)

    paper = data_generator(paper_imgs, max, "paper")
    paper_iterator = iter(paper)

    scissors = data_generator(scissors_imgs, max, "scissors")
    scissors_iterator = iter(scissors)

    ngg = data_generator(ngg_imgs, max, "ngg")
    ngg_iterator = iter(ngg)

    # Testing for index <= dataset_size
    for i in range(5):
        r = next(rock_iterator)
        p = next(paper_iterator)
        s = next(scissors_iterator)
        n = next(ngg_iterator)

        hand_landmarks_3D_plotting(r, "rock")
        hand_landmarks_3D_plotting(p, "paper")
        hand_landmarks_3D_plotting(s, "scissors")
        hand_landmarks_3D_plotting(n, "None game gesture")

    # Testing for index > dataset_size
    data_generator.set_index(rock_iterator, 30000)
    data_generator.set_index(paper_iterator, 30000)
    data_generator.set_index(scissors_iterator, 30000)
    r = next(rock_iterator)
    p = next(paper_iterator)
    s = next(scissors_iterator)
    hand_landmarks_3D_plotting(r, "rock")
    hand_landmarks_3D_plotting(p, "paper")
    hand_landmarks_3D_plotting(s, "scissors")


def test_multi_hands():
    landmarks_original = hands_keypoint_extract_from_file("./2_hands.jpg")[0]
    hand_landmarks_3D_plotting(landmarks_original, None)

def metric_values(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print('Confusion matrix:')
    print(cm)
    print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred, normalize=True)}')
    print(f'Balanced Accuracy: {metrics.balanced_accuracy_score(y_test, y_pred)}')
    print(f'F1 Score: {metrics.f1_score(y_test, y_pred, average=None)}')
    print(f'Precision: { metrics.precision_score(y_test, y_pred, average=None)}')
    print(f'Recall: {metrics.recall_score(y_test, y_pred, average=None)}')

def Data_load(data_csv_path="./data_source_list.csv", max_items_per_class=None):
    all_imgs, all_labels = Data_read_from_csv(data_csv_path)
    rock_imgs = []
    paper_imgs = []
    scissors_imgs = []
    ngg_imgs = []

    for i in range(len(all_imgs)):
        if all_labels[i] == "rock":
            rock_imgs.append(all_imgs[i])
        elif all_labels[i] == "paper":
            paper_imgs.append(all_imgs[i])
        elif all_labels[i] == "scissors":
            scissors_imgs.append(all_imgs[i])
        else:
            ngg_imgs.append(all_imgs[i])

    # Seting the maximum amount per class
    if max_items_per_class==None:
        max_items = len(ngg_imgs)
    else:
        max_items = max_items_per_class

    rock = data_generator(rock_imgs, max_items, "rock")
    rock_iterator = iter(rock)

    paper = data_generator(paper_imgs, max_items, "paper")
    paper_iterator = iter(paper)

    scissors = data_generator(scissors_imgs, max_items, "scissors")
    scissors_iterator = iter(scissors)

    ngg = data_generator(ngg_imgs, max_items, "ngg")
    ngg_iterator = iter(ngg)

    data_X = []
    data_y = []

    for i in tqdm(range(max_items)):
        r = next(rock_iterator)
        p = next(paper_iterator)
        s = next(scissors_iterator)
        n = next(ngg_iterator)

        data_X.append(hand_point_normaliser(hand_landmarks_to_1D_array(r)))
        data_X.append(hand_point_normaliser(hand_landmarks_to_1D_array(p)))
        data_X.append(hand_point_normaliser(hand_landmarks_to_1D_array(s)))
        data_X.append(hand_point_normaliser(hand_landmarks_to_1D_array(n)))

        del(r)
        del(p)
        del(s)
        del(n)

        data_y.append(0)  # rock
        data_y.append(1)  # paper
        data_y.append(2)  # scissors
        data_y.append(3)  # ngg not game gesture
    
    # applying flip image logic by creating datagenerator object with flip flag set True
    
    rock_flip = data_generator(rock_imgs, max_items, "rock", flip_flag=True)
    rock_iterator_flip = iter(rock_flip)

    paper_flip = data_generator(paper_imgs, max_items, "paper", flip_flag=True)
    paper_iterator_flip = iter(paper_flip)

    scissors_flip = data_generator(scissors_imgs, max_items, "scissors", flip_flag=True)
    scissors_iterator_flip = iter(scissors_flip)

    ngg_flip = data_generator(ngg_imgs, max_items, "ngg", flip_flag=True)
    ngg_iterator_flip = iter(ngg_flip)

    for i in tqdm(range(max_items)):
        r_flip = next(rock_iterator_flip)
        p_flip = next(paper_iterator_flip)
        s_flip = next(scissors_iterator_flip)
        n_flip = next(ngg_iterator_flip)
        
        data_X.append(hand_point_normaliser(hand_landmarks_to_1D_array(r_flip)))
        data_X.append(hand_point_normaliser(hand_landmarks_to_1D_array(p_flip)))
        data_X.append(hand_point_normaliser(hand_landmarks_to_1D_array(s_flip)))
        data_X.append(hand_point_normaliser(hand_landmarks_to_1D_array(n_flip)))

        del(r_flip)
        del(p_flip)
        del(s_flip)
        del(n_flip)

        data_y.append(0)  # rock
        data_y.append(1)  # paper
        data_y.append(2)  # scissors
        data_y.append(3)  # ngg not game gesture

    return [data_X, data_y]


if __name__ == "__main__":
    
    # test_3D_rotation_visualization(file_path)
    # print(hand_point_normaliser(hand_landmarks_to_1D_array(hands_keypoint_extract_from_file(file_path))))
    #test_iterator()
    #test_multi_hands()
    
    print(len(Data_load(data_csv_path="./data_source_list.csv", max_items_per_class=1)[0]))
