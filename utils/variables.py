# TODO: Add file description
import cv2
from utils.gui import applying_mask

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


class FILES(object):
    """
    This class contains all the hardcoded paths.
    """
    paper_blue = './Images/robot_paper.png'
    paper_red = './Images/paper.png'
    rock_blue = './Images/robot_rock.png'
    rock_red = './Images/rock.png'
    scissors_blue = './Images/robot_scissors.png'
    scissors_red = './Images/scissors.png'
    game_name = './Images/game_name.png'
    language_selection = './Images/language_selection.png'
    play_button_en = './Images/play_eng.png'
    play_button_fr = './Images/play_fr.png'
    one = './Images/1.png'
    two = './Images/2.png'
    three = './Images/3.png'
    language_in_game = './Images/language_instr_in_game.png'


class POP_SCREEN(object):
    #TODO: Check if this need to be deleted

    mask_game_name, mask_inv_game_name, game_name_img = applying_mask(cv2.imread(resource_path(FILES.game_name)))
    mask_language_selection, mask_inv_language_selection, language_selection_img = applying_mask(cv2.imread(resource_path(FILES.language_selection)))
    mask_play_button_en, mask_inv_play_button_en, play_button_en_img = applying_mask(cv2.imread(resource_path(FILES.play_button_en)))
    mask_play_button_fr, mask_inv_play_button_fr, play_button_fr_img = applying_mask(cv2.imread(resource_path(FILES.play_button_fr)))


class SCISSORS:
    #TODO: Check if this need to be deleted
    """
    This class stores all variables related to scissors.

    mask_scissors_blue: mask to be applied to the image
    mask_inv_scissors_blue: inverse mask
    scissors_blue_img: original image
    mask_scissors_red: mask to be applied to the image
    mask_inv_scissors_red: inverse mask
    scissors_red_img:original image
    """
    mask_scissors_blue, mask_inv_scissors_blue, scissors_blue_img = applying_mask(cv2.imread(resource_path(FILES.scissors_blue)))
    mask_scissors_red, mask_inv_scissors_red, scissors_red_img = applying_mask(cv2.imread(resource_path(FILES.scissors_red)))


class ROCK:
    #TODO: Check if this need to be deleted

    """
    This class stores all variables related to rock.

    mask_rock_blue: mask to be applied to the image
    mask_inv_rock_blue: inverse mask
    rock_blue_img: original image
    mask_rock_red: mask to be applied to the image
    mask_inv_rock_red: inverse mask
    rock_red_img:original image
    """
    mask_rock_blue, mask_inv_rock_blue, rock_blue_img = applying_mask(cv2.imread(resource_path(FILES.rock_blue)))
    mask_rock_red, mask_inv_rock_red, rock_red_img = applying_mask(cv2.imread(resource_path(FILES.rock_red)))


class PAPER:
    #TODO: Check if this need to be deleted

    """
    This class stores all variables related to paper.

    mask_paper_blue: mask to be applied to the image
    mask_inv_paper_blue: inverse mask
    paper_blue_img: original image
    mask_paper_red: mask to be applied to the image
    mask_inv_paper_red: inverse mask
    paper_red_img:original image
    """
    mask_paper_blue, mask_inv_paper_blue, paper_blue_img = applying_mask(cv2.imread(resource_path(FILES.paper_blue)))
    mask_paper_red, mask_inv_paper_red, paper_red_img = applying_mask(cv2.imread(resource_path(FILES.paper_red)))


class FLAGS(object):
    """
    This class stores all the flags used.

    COUNTDOWN : Initialized to 3, it will decrement to 1 with this function
    PLAY_FLAG: Determines if the player started playing
    """
    COUNTDOWN = 0
    LANG_SEL_FLAG = True
    INST_FLAG = True
    PLAY_FLAG = False
    RESULT_FLAG = 0
    EXIT = False

class BACKGROUND(object):
    """
    This class stores all the variables for background subtraction

    aweigth : Average weight for robust background subtraction
    num_frames: Number of accumulating frames
    bg: Background
    """
    aweight = 0.5
    bg = None
    num_frames = 0


class SIZE(object):
    """
        This class stores all the dimensions of objects on the screen

        gesture_size : Size of the gesture
        im_size: Size of the image
        """
    gesture_size = 150
    im_size = (100, 100)


class ROI(object):
    """
            This class stores all variables related to ROI

            roi_height_start: where the height of the roi start
            roi_height_end: where the height of the roi end
            roi_width_start: where the width of the roi start
            roi_width_end: where the width of the roi end
            """
    roi_height_start = 50
    roi_height_end = 350
    roi_width_start = 300
    roi_width_end = 600