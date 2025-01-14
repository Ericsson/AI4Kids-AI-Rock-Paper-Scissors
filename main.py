import utils.gui as gui
from utils.cv_ai_logic import *
from Experiments.Training_Pipelines.ml_pipeline_utils import *
from utils.language import UsedLanguage, FrenchLanguage, EnglishLanguage, change_language
from utils.variables import FLAGS, SIZE, ROCK, PAPER, SCISSORS, BACKGROUND, POP_SCREEN, FILES

import cv2
import time

usedlanguages = UsedLanguage()
englishlanguage = EnglishLanguage()
frenchlanguages = FrenchLanguage()

# Language selection
change_language(usedlanguages, englishlanguage)

# Play game setting
button = gui.Button("test", 250, 380, 135, 80)  # Create button instance

# Read the video stream
cap = cv2.VideoCapture(0)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

# changing variables to make game window full screen
cv2.namedWindow("Game window", cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Game window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Main game loop
while True:

    # Choosing language section
    while FLAGS.LANG_SEL_FLAG:
        print('Choosing language section')
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        frame = cv2.flip(frame, 1)
        frame = gui.pop_up_screen(frame, POP_SCREEN, ROCK, PAPER, SCISSORS)
        cv2.imshow("Game window", frame)
        key = cv2.waitKey(1)
        # FLAGS.LANG_SEL_FLAG is set to false when the language is selected
        FLAGS.LANG_SEL_FLAG = gui.choose_language(FLAGS.LANG_SEL_FLAG, key)

        # Exit management
        if key == ord('q'):
            FLAGS.EXIT = True
            break


    # Gaming instructions section
    while FLAGS.INST_FLAG:
        print('Gaming instructions section')
        ret, frame = cap.read()

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        frame = cv2.flip(frame, 1)
        frame = draw_hand_keypoint(frame=frame, frame_skip=0)

        # Draw a button on current frame
        button.draw(frame)
        # draw game icons and instructions
        frame = gui.game_instr_screen(frame, ROCK, PAPER, SCISSORS)

        cv2.imshow('Game window', frame)

        cv2.setMouseCallback("Game window", button.handle_event)
        key = cv2.waitKey(1)
        # Adding the possibility to change language
        gui.choose_language(FLAGS.LANG_SEL_FLAG, key)

        # Once 'Play' button is clicked or any key except q/Esc/1/2 pressed, a countdown begins.
        # Then the game starts recognize hand gestures
        if button.clicked or (key > -1 and (key not in [ord('q'), ord('Q'), 27, ord('1'), ord('2')])):
            FLAGS.LANG_SEL_FLAG = False
            FLAGS.INST_FLAG = False
            FLAGS.PLAY_FLAG = False
            FLAGS.RESULT_FLAG = 0
            FLAGS.COUNTDOWN = 3
            button.clicked = False

        # Exit management
        if key == ord('q'):
            FLAGS.EXIT = True
            break


    # Count down section
    time_start = time.time()
    while FLAGS.COUNTDOWN > 0:
        print('Count down section')
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        frame = draw_hand_keypoint(frame=frame, frame_skip=0)  # keeping hand detection through out the countdown
        frame = gui.show_count_down(frame, str(FLAGS.COUNTDOWN), FILES)
        cv2.imshow('Game window', frame)
        key = cv2.waitKey(1)

        # time counting
        time_end = time.time()
        if (time_end - time_start) > 1:
            FLAGS.COUNTDOWN -= 1
            time_start = time.time()

        if FLAGS.COUNTDOWN == 0:
            FLAGS.LANG_SEL_FLAG = False
            FLAGS.INST_FLAG = False
            FLAGS.PLAY_FLAG = True
            FLAGS.RESULT_FLAG = 0
            FLAGS.COUNTDOWN = 0  # Not needed but is here for clarity

        # Exit management
        if key == ord('q'):
            FLAGS.EXIT = True
            break


    # Player gesture capture section
    time_start = time.time()
    while FLAGS.PLAY_FLAG:
        print('Player gesture capture section')
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        # wait some miliseconds before capture
        # time counting
        time_end = time.time()
        if (time_end - time_start) > 0.5:
            # Hand detection
            hand_pred = hand_gesture_prediction(frame)
            # Setting FLAGS
            FLAGS.LANG_SEL_FLAG = False
            FLAGS.INST_FLAG = False
            FLAGS.PLAY_FLAG = False
            FLAGS.RESULT_FLAG = 5
            FLAGS.COUNTDOWN = 0

        frame = draw_hand_keypoint(frame=frame, frame_skip=0)  # keeping hand detection all the time
        cv2.imshow('Game window', frame)
        key = cv2.waitKey(1)


    # Computer gestures are generated by random numbers
    computerGesture = random.randint(0, 2)


    # Display results section
    time_start = time.time()
    while FLAGS.RESULT_FLAG > 0:
        print('Display results section')
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        roi = frame[50:200, 30:180]  # TODO: MAke screen size adjustable
        roi2 = frame[50:200, 455:605]  # TODO: MAke screen size adjustable

        # Display hand gesture images for computer results
        frame = gui.display_hand_gesture_computer(frame, computerGesture, hand_pred, ROCK, PAPER, SCISSORS, roi)

        # Display hand gesture images for player results
        frame = gui.display_hand_gesture_player(frame, hand_pred, ROCK, PAPER, SCISSORS, roi2)

        # The game logic to decide who wins
        frame = gui.display_game_results(frame, computerGesture, hand_pred)

        cv2.imshow('Game window', frame)
        key = cv2.waitKey(1)

        # time counting
        time_end = time.time()
        if (time_end - time_start) > 1:
            FLAGS.RESULT_FLAG -= 1
            time_start = time.time()

        if FLAGS.RESULT_FLAG == 0:
            FLAGS.LANG_SEL_FLAG = False
            FLAGS.INST_FLAG = True
            FLAGS.PLAY_FLAG = False
            FLAGS.RESULT_FLAG = 0
            FLAGS.COUNTDOWN = 0

        # Exit management
        if key == ord('q'):
            FLAGS.EXIT = True
            break

    # Handling Exit
    if FLAGS.EXIT:
        break

cap.release()
cv2.destroyAllWindows()