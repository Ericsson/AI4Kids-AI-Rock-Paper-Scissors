import cv2
from utils.language import UsedLanguage, FrenchLanguage, EnglishLanguage, change_language
import os
import sys

# TODO: Repeated in main CHECK
usedlanguages = UsedLanguage()
englishlanguage = EnglishLanguage()
frenchlanguages = FrenchLanguage()


# TODO: Repeated in main CHECK
# Default Language selection
change_language(usedlanguages, englishlanguage)

# A 'Play' button class
def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.normpath(os.path.join(base_path, relative_path))


class Button(object):
    def __init__(self, text, x, y, width, height, command=None):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.left = x
        self.top = y
        self.right = x + width - 1
        self.bottom = y + height - 1

        self.hover = False
        self.clicked = False
        self.command = command

    def handle_event(self, event, x, y, flags, param):
        self.hover = (self.left <= x <= self.right and
                      self.top <= y <= self.bottom)

        if self.hover and flags == 1:
            self.clicked = True

    def draw(self, frame):
       
        mask_play_button, mask_inv_play_button, play_button_img = applying_mask(cv2.imread(resource_path(usedlanguages.button_img)))
        mask_play_instr_string, mask_inv_play_instr_string, play_instr_string_img = applying_mask(cv2.imread(resource_path(usedlanguages.play_instr_string_img)))
        mask_play_instr_string_2, mask_inv_play_instr_string_2, play_instr_string_img_2 = applying_mask(cv2.imread(resource_path(usedlanguages.language_in_game)))

        
        if not self.hover:
            roi = frame[360:440, 250:385]
            frame[360:440, 250:385] = resizing(play_button_img, 135, 80, mask_play_button,mask_inv_play_button, roi)
            
            roi = frame[320:340, 200:435]
            frame[320:340, 200:435] = resizing(play_instr_string_img, 235, 20, mask_play_instr_string, mask_inv_play_instr_string, roi)
            # cv2.putText(frame, UsedLanguage.play_instr_string_img, (190, 470), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 127), 2)
            # cv2.rectangle(frame, (290, 440), (290 + 100, 440 - 50), (255, 255, 127), 2)
            roi = frame[460:480, 200:435] 
            frame[460:480, 200:435] = resizing(play_instr_string_img_2, 235, 20, mask_play_instr_string_2, mask_inv_play_instr_string_2, roi)
            
        else:
            roi = frame[360:440, 250:385]
            frame[360:440, 250:385] = resizing(play_button_img, 135, 80, mask_play_button,mask_inv_play_button, roi)
       

def choose_language(choose_language_flag, key):
    
    """
    This function change the language of the game based on the user key input

    :param choose_language_flag: Status of whether the user selected language input or not
    :param key: Checking for user key input expecting values 1 or 2

    :return:
    choose_language_flag: Return the flag turned off if user selected language
    """
   
    # Wait for user input
    
    
    if key == ord('1'):
        print('English selected')
        change_language(usedlanguages, englishlanguage)
        choose_language_flag = False
    elif key == ord('2'):
        print('French selected')
        change_language(usedlanguages, frenchlanguages)
        choose_language_flag = False
    
    return choose_language_flag 


def show_text_image(frame, image_name_path, center_position, img_dimension=(0, 0)):
    """
    This function works as putText function from opencv for each text it takes the image object
    and position it in the frame

    :param frame: This is the current frame
    :param image_name_path: This is a objects to all gesture images
    :param center_position: the center position of the image
    :param img_dimension: custom values of the image width and height

    :return:
    frame: Loads all content to the position and return the frame
    """

    img = cv2.imread(image_name_path)
    mask_var, mask_inv_var, var_img = applying_mask(img)

    # Get the height and width of the image
    height_frame, width_frame = frame.shape[:2]
    height_img, width_img = img.shape[:2]


    if img_dimension != (0, 0):
        height_img, width_img = img_dimension

    # Calculate the bottom-left corner of the image
    x = int(center_position[0] - width_img / 2)
    y = int(center_position[1] - height_img / 2)

    # Check if the new frame values exceed the maximum allowed values
    frame_height = height_img
    frame_width = width_img
    if y + height_img > height_frame:
        frame_height = height_frame - y
    if x + width_img > width_frame:
        frame_width = width_frame - x

    # Calculate the new height and width of the image after capping
    new_height, new_width = frame_height, frame_width

    roi = frame[y:y + frame_height, x:x + frame_width]
    frame[y:y + frame_height, x:x + frame_width] = resizing(var_img, new_width, new_height, mask_var, mask_inv_var, roi)
    return frame


   

def game_instr_screen(frame, ROCK, PAPER, SCISSORS):
    """
    This function loads the game screen with game instruction input image

    :param frame: This is the current frame
    :param ROCK, PAPER , SCISSORS: This is a objects to all gesture images

    :return:
    frame: Loads all content to the position and return the frame
    """

    
    # Robotic hand gestures
    roi = frame[180:260, 10:160]
    frame[180:260, 10:160] = resizing(PAPER.paper_blue_img, 150, 80, PAPER.mask_paper_blue,
                                        PAPER.mask_inv_paper_blue, roi)
    roi = frame[280:360, 10:160]
    frame[280:360, 10:160] = resizing(ROCK.rock_blue_img, 150, 80, ROCK.mask_rock_blue,
                                        ROCK.mask_inv_rock_blue, roi)
    roi = frame[380:460, 10:160]
    frame[380:460, 10:160] = resizing(SCISSORS.scissors_blue_img, 150, 80, SCISSORS.mask_scissors_blue,
                                        SCISSORS.mask_inv_scissors_blue, roi)
    # Hang gestures
    roi = frame[180:260, 530:610]
    frame[180:260, 530:610] = resizing(PAPER.paper_red_img, 80, 80, PAPER.mask_paper_red,
                                         PAPER.mask_inv_paper_red, roi)
    roi = frame[280:360, 530:610]
    frame[280:360, 530:610] = resizing(ROCK.rock_red_img, 80, 80, ROCK.mask_rock_red,
                                        ROCK.mask_inv_rock_red, roi)
    roi = frame[380:460, 530:610]
    frame[380:460, 530:610] = resizing(SCISSORS.scissors_red_img, 80, 80, SCISSORS.mask_scissors_red,
                                        SCISSORS.mask_inv_scissors_red, roi)
    # game instruction
    mask_game_instr, mask_inv_game_instr, game_instr_img = applying_mask(cv2.imread(resource_path(usedlanguages.game_instr_img)))

    roi = frame[30:310, 200:435]
    frame[30:310, 200:435] = resizing(game_instr_img, 235, 280, mask_game_instr,
                                        mask_inv_game_instr, roi)
    # Return the modified frame
    return frame

def show_count_down(frame, count, FILES):
    """
    This function loads the pop up screen with language selection input image

    :param frame: This is the current frame
    :param count: This is text values of the count 
    :param FILES: This is a objects to all count images

    :return:
    frame: Loads all content to the position and return the frame
    """
    # show count numbers
    if count == "1":
        mask_one, mask_inv_one, one_img = applying_mask(cv2.imread(resource_path(FILES.one)))

        roi = frame[30:310, 200:435]
        frame[30:310, 200:435] = resizing(one_img, 235, 280, mask_one,
                                            mask_inv_one, roi)
    
    if count == "2":
        mask_two, mask_inv_two, two_img = applying_mask(cv2.imread(resource_path(FILES.two)))

        roi = frame[30:310, 200:435]
        frame[30:310, 200:435] = resizing(two_img, 235, 280, mask_two,
                                            mask_inv_two, roi)
        
    if count == "3":
        mask_three, mask_inv_three, three_img = applying_mask(cv2.imread(resource_path(FILES.three)))

        roi = frame[30:310, 200:435]
        frame[30:310, 200:435] = resizing(three_img, 235, 280, mask_three,
                                            mask_inv_three, roi)

    return frame

def pop_up_screen(frame, POP_SCREEN,  ROCK, PAPER, SCISSORS):
    """
    This function loads the pop up screen with language selection input image

    :param frame: This is the current frame
    :param POP_SCREEN: This is object to pop screen content , heading and langugae selection image
    :param POP_SCREEN: This is a objects to all gesture images

    :return:
    frame: Loads all content to the position and return the frame
    """

    # Game name
    roi = frame[10:160, 30:605]
    frame[10:160, 30:605] = resizing(POP_SCREEN.game_name_img, 575, 150, POP_SCREEN.mask_game_name,
                                         POP_SCREEN.mask_inv_game_name, roi)
    # Robotic hand gestures
    # Robotic hand gestures
    roi = frame[180:260, 10:160]
    frame[180:260, 10:160] = resizing(PAPER.paper_blue_img, 150, 80, PAPER.mask_paper_blue,
                                        PAPER.mask_inv_paper_blue, roi)
    roi = frame[280:360, 10:160]
    frame[280:360, 10:160] = resizing(ROCK.rock_blue_img, 150, 80, ROCK.mask_rock_blue,
                                        ROCK.mask_inv_rock_blue, roi)
    roi = frame[380:460, 10:160]
    frame[380:460, 10:160] = resizing(SCISSORS.scissors_blue_img, 150, 80, SCISSORS.mask_scissors_blue,
                                        SCISSORS.mask_inv_scissors_blue, roi)
    # Hang gestures
    roi = frame[180:260, 530:610]
    frame[180:260, 530:610] = resizing(PAPER.paper_red_img, 80, 80, PAPER.mask_paper_red,
                                         PAPER.mask_inv_paper_red, roi)
    roi = frame[280:360, 530:610]
    frame[280:360, 530:610] = resizing(ROCK.rock_red_img, 80, 80, ROCK.mask_rock_red,
                                        ROCK.mask_inv_rock_red, roi)
    roi = frame[380:460, 530:610]
    frame[380:460, 530:610] = resizing(SCISSORS.scissors_red_img, 80, 80, SCISSORS.mask_scissors_red,
                                        SCISSORS.mask_inv_scissors_red, roi)
    # language selection instruction
    roi = frame[180:460, 200:435]
    frame[180:460, 200:435] = resizing(POP_SCREEN.language_selection_img, 235, 280, POP_SCREEN.mask_language_selection,
                                        POP_SCREEN.mask_inv_language_selection, roi)
    # Return the modified frame
    return frame


# Load and resize hand gesture images to show computer and player results
def load_hand_gesture_imgs(gesture_size):
    paper_blue = cv2.imread('Images/paper_blue.png')
    paper_blue = cv2.resize(paper_blue, (gesture_size, gesture_size))
    rock_blue = cv2.imread('Images/rock_blue.png')
    rock_blue = cv2.resize(rock_blue, (gesture_size, gesture_size))
    scissors_blue = cv2.imread('Images/scissors_blue.png')
    scissors_blue = cv2.resize(scissors_blue, (gesture_size, gesture_size))
    paper_red = cv2.imread('Images/paper_red.png')
    paper_red = cv2.resize(paper_red, (gesture_size, gesture_size))
    rock_red = cv2.imread('Images/rock_red.png')
    rock_red = cv2.resize(rock_red, (gesture_size, gesture_size))
    scissors_red = cv2.imread('Images/scissors_red.png')
    scissors_red = cv2.resize(scissors_red, (gesture_size, gesture_size))
    return paper_blue, paper_red, rock_blue, rock_red, scissors_blue, scissors_red


def applying_mask(img):
    #TODO: we need to fix the issues with the mask

    # Check if the input image has an alpha channel
    if img.shape[2] == 4:
        # Extract the alpha channel
        alpha_channel = img[:, :, 3]

        # Define a threshold value for alpha channel (e.g., 128 for semi-transparent)
        alpha_threshold = 128

        # Create the mask using the alpha channel and the threshold value
        _, mask = cv2.threshold(alpha_channel, alpha_threshold, 255, cv2.THRESH_BINARY)
    else:
        # Create the mask for the object using grayscale conversion as before
        objectGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(objectGray, 10, 255, cv2.THRESH_BINARY)

    # Create the inverted mask for the object
    mask_inv = cv2.bitwise_not(mask)

    # Ensure the image is in 3-channel BGR format
    img = img[:, :, 0:3]

    return mask, mask_inv, img

# def applying_mask(img):
#     """
#     This function allows to create the mask of each image used during the game.
#     The output gives the mask of the image, its inverse mask also which will be inserted in the game.

#     :param img: This is the image to which the mask is applied
#     :return:
#     mask: Creation of a new single-layer image uses for masking
#     mask_inv: The initial mask will define the area for the image, and the inverse mask will be for the region
#               around the image
#     img: The image in 3-channel BGR format
#     """
#     # Create the mask for the object
#     objectGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, mask = cv2.threshold(objectGray, 1, 255, cv2.THRESH_BINARY)

#     # Create the inverted mask for the object
#     mask_inv = cv2.bitwise_not(mask)
#     img = img[:, :, 0:3]

#     return mask, mask_inv, img




# def applying_mask(img, rect=None):
#     """
#     This function allows creating the mask of each image used during the game.
#     The output gives the mask of the image, its inverse mask also which will be inserted in the game.
#     :param img: This is the image to which the mask is applied
#     :param rect: A tuple (x, y, w, h) representing a rectangle around the foreground object
#     :return:
#     mask: Creation of a new single-layer image used for masking
#     mask_inv: The initial mask will define the area for the image, and the inverse mask will be for the region
#               around the image
#     img: The image in 3-channel BGR format
#     """

#     if img.shape[2] == 4:  # Check if the image has an alpha channel
#         # Use the alpha channel as the mask
#         mask = img[:, :, 3]
#         mask_inv = cv2.bitwise_not(mask)
#         img = img[:, :, 0:3]
#     else:
#         if rect is None:
#             # Set a smaller default rectangle inside the image
#             x = img.shape[1] // 8
#             y = img.shape[0] // 8
#             w = img.shape[1] * 3 // 4
#             h = img.shape[0] * 3 // 4
#             rect = (x, y, w, h)

#         # Initialize the mask for the GrabCut algorithm
#         grabcut_mask = np.zeros(img.shape[:2], np.uint8)
#         bgdModel = np.zeros((1, 65), np.float64)
#         fgdModel = np.zeros((1, 65), np.float64)

#         # Apply the GrabCut algorithm
#         cv2.grabCut(img, grabcut_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

#         # Create the final mask where the value of the pixels marked as 'probable foreground' or 'foreground' is set to 255
#         mask = np.where((grabcut_mask == cv2.GC_PR_FGD) | (grabcut_mask == cv2.GC_FGD), 255, 0).astype(np.uint8)

#         # Create the inverted mask for the object
#         mask_inv = cv2.bitwise_not(mask)

#         img = img[:, :, 0:3]

#     return mask, mask_inv, img


def resizing(image, width, height, orig_mask, orig_mask_inv, roi_image):
    """
    This function allows you to resize the images (foreground) so that they fit in the frame (background)
    :param image: Image that will be resized to match the desired size in the game
    :param width: width of the resizing image
    :param height: height of the resizing image
    :param orig_mask: mask that will also be re-imaged
    :param orig_mask_inv: inv-mask that will also be re-imaged
    :param roi_image: region of interest where the image will be inserted
    :return:
    dst: Represents the addition of the background and the foreground. Adding the image with the right dimensions
         in the desired area of interest
    """
    img = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (width, height), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (width, height), interpolation=cv2.INTER_AREA)

    # roi_bg contains the original image only where the image is not in the region that is the size of the image.
    roi_bg = cv2.bitwise_and(roi_image, roi_image, mask=mask_inv)

    # roi_fg contains the image of the image only where the image is
    roi_fg = cv2.bitwise_and(img, img, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    return dst

def display_hand_gesture_computer(frame, computerGesture,hand_pred, ROCK, PAPER, SCISSORS,roi):
    if  hand_pred in [3,4]:
        return frame
    roi = frame[100:180, 30:180]
    if computerGesture == 1:
        frame[100:180, 30:180] = resizing(PAPER.paper_blue_img, 150, 80, PAPER.mask_paper_blue,
                                         PAPER.mask_inv_paper_blue, roi)
        frame = show_text_image(frame, usedlanguages.paper_img, (115, 205))
    elif computerGesture == 0:
        frame[100:180, 30:180] = resizing(ROCK.rock_blue_img, 150, 80, ROCK.mask_rock_blue,
                                         ROCK.mask_inv_rock_blue, roi)
        frame = show_text_image(frame, usedlanguages.rock_img, (115, 205))
    elif computerGesture == 2:
        frame[100:180, 30:180] = resizing(SCISSORS.scissors_blue_img, 150, 80, SCISSORS.mask_scissors_blue,
                                         SCISSORS.mask_inv_scissors_blue, roi)
        frame = show_text_image(frame, usedlanguages.scissors_img, (115, 205))
    return frame


def display_hand_gesture_player(frame, hand_pred, ROCK, PAPER ,SCISSORS, roi2):
    
    roi2 = frame[100:180, 495:575]
    if hand_pred == 1:
        frame[100:180, 495:575] = resizing(PAPER.paper_red_img, 80, 80, PAPER.mask_paper_red,
                                          PAPER.mask_inv_paper_red, roi2)
        frame = show_text_image(frame, usedlanguages.paper_img, (530, 205))
    elif hand_pred == 0:
        frame[100:180, 495:575] = resizing(ROCK.rock_red_img,  80, 80, ROCK.mask_rock_red,
                                          ROCK.mask_inv_rock_red, roi2)
        frame = show_text_image(frame, usedlanguages.rock_img, (530, 205))
    elif hand_pred == 2:
        frame[100:180, 495:575] = resizing(SCISSORS.scissors_red_img,  80, 80, SCISSORS.mask_scissors_red,
                                          SCISSORS.mask_inv_scissors_red, roi2)
        frame = show_text_image(frame, usedlanguages.scissors_img, (530, 205) )    

    return frame

def display_game_results(frame, computerGesture, hand_pred):
    diff = computerGesture - hand_pred

    #handling non game gesture scenarios
    if hand_pred == 3:
        result = usedlanguages.ngg_img
        frame = show_text_image(frame, result, (320, 100) )    

    
    #handling no hand detected scenarios
    elif hand_pred == 4:

        result = usedlanguages.no_landmark_img
        frame = show_text_image(frame, result, (320, 100) )    

        
       
    
    elif diff in [-2, 1]:
        result = usedlanguages.computer_win_img
        frame = show_text_image(frame, usedlanguages.computer_img,(115, 45),(80,140) )    
        frame = show_text_image(frame, usedlanguages.you_img, (530, 45), (60,100) )    
        frame = show_text_image(frame, result, (320, 400) )    


    elif diff in [2, -1]:
        result = usedlanguages.player_win_img
        frame = show_text_image(frame, usedlanguages.computer_img,(115, 45),(80,140) )    
        frame = show_text_image(frame, usedlanguages.you_img, (530, 45), (60,100) )    
        frame = show_text_image(frame, result, (320, 400) ) 

    else:
        result = usedlanguages.tie_img
        frame = show_text_image(frame, usedlanguages.computer_img,(115, 45),(80,140) )    
        frame = show_text_image(frame, usedlanguages.you_img,(530, 45), (60,100) )    
        frame = show_text_image(frame, result, (320, 400) ) 
    return frame