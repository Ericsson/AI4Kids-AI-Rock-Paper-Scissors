# TODO: Add file description

from PIL import Image, ImageDraw, ImageFont
import os
import sys


def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):  # Running as PyInstaller executable
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        print('Running from PyInstaller binary')
    else:
        base_path = os.path.abspath(".")

    return os.path.normpath(os.path.join(base_path, relative_path))


class UsedLanguage(object):
    def __init__(self):
        # Default FONT, path is updated if PyInstaller binary is used
        self.embeded_font = resource_path('./utils/Roboto-VariableFont_wdth,wght.ttf')

        self.language_in_game = self.create_text_image('./Images/languages/instr_in_game.png',
                                                       "Press 1 for English / Appuyez sur 2 pour le français")
        self.countdown_number_1 = self.create_number_image(path='./Images/1.png', number_str='1')
        self.countdown_number_2 = self.create_number_image(path='./Images/2.png', number_str='2')
        self.countdown_number_3 = self.create_number_image(path='./Images/3.png', number_str='3')

    def create_text_image(self, path, text, font_path=None, font_size=50, font_color=(166, 202, 240), bg_color=None):
        """
        Create an image containg a text
        :param path: path where the image will be saved
        :param text: Text on the image
        :param font_path: font path
        :param font_size: font size
        :param font_color: font color
        :param bg_color: background color
        :return: image path
        """

        if not font_path:
            font_path = self.embeded_font

        # Load the font
        font = ImageFont.truetype(font_path, font_size)

        # Calculate the size of the text
        text_size = font.getsize(text)

        # Create an image with a transparent background
        image = Image.new("RGBA", text_size, (0, 0, 0, 0))

        # Create a drawing context for the image
        draw = ImageDraw.Draw(image)

        # Draw the text on the image
        draw.text((0, 0), text, font=font, fill=font_color)

        # Crop the image to remove any extra transparent space
        bbox = image.getbbox()
        cropped_image = image.crop(bbox)

        # If a background color was specified, create a new image with the specified size and color
        if bg_color is not None:
            bg_image = Image.new("RGBA", cropped_image.size, bg_color)

            # Paste the cropped image onto the background image
            bg_image.paste(cropped_image, (0, 0), cropped_image)

            # Set the image variable to the background image
            image = bg_image

        # Update Path so it works with PyInstaller binary
        path = resource_path(path)

        # Save image to file
        image.save(path)

        return path

    def create_button(self, path, text, button_size=(400, 200), font_path=None, font_size=80,
                      font_color=(0, 0, 0), button_color=(166, 202, 240)):
        """
        Create the button as image
        :param path: path where the image will be saved
        :param text: Text on the button
        :param button_size: buttonsize
        :param font_path: font path
        :param font_size: font size
        :param font_color: font color
        :param button_color: button color
        :return: image path
        """
        # Create an Image object
        img = Image.new("RGB", button_size, button_color)

        # Create a Draw object
        draw = ImageDraw.Draw(img)

        if not font_path:
            font_path = self.embeded_font

        # Load font
        font = ImageFont.truetype(font_path, font_size)

        # Draw the text on the image
        text_size = draw.textsize(text, font=font)
        text_x = (button_size[0] - text_size[0]) // 2
        text_y = (button_size[1] - text_size[1]) // 2
        draw.text((text_x, text_y), text, font=font, fill=font_color)

        # Update Path so it works with PyInstaller binary
        path = resource_path(path)
        img.save(path, format='PNG', transparent=True)

        return path

    def create_number_image(self, path, number_str, font_path=None, font_size=150, image_size=(200, 200)):
        """
        Creates an image of a number as a string with a transparent background.

        Args:
            number_str (str): The number to display.
            font_path (str): The file path to the font file to use.
            font_size (int): The font size to use.
            image_size (tuple): The width and height of the resulting image.
            img_path (str): path where the image will be saved.

        Returns:
            A PIL Image object with the number drawn on it.
        """
        # Create a new transparent image with the given size.
        image = Image.new('RGBA', image_size, (0, 0, 0, 0))

        # Create a new drawing context.
        draw = ImageDraw.Draw(image)

        if not font_path:
            font_path = self.embeded_font

        # Load the font.
        font = ImageFont.truetype(font_path, font_size)

        # Calculate the position to draw the number.
        text_width, text_height = draw.textsize(number_str, font=font)
        x = (image_size[0] - text_width) / 2
        y = (image_size[1] - text_height) / 2

        # Draw the number on the image.
        draw.text((x, y), number_str, font=font, fill=(166, 202, 240, 255))

        # Update Path so it works with PyInstaller binary
        path = resource_path(path)
        image.save(path)

        return path

    def create_instructions(self, path, instructions_text, font_path=None, font_size=45,
                            background_color=(255, 255, 255), font_color=(166, 202, 240)):
        # Create an Image object
        img = Image.new("RGBA", (1000, 1000), background_color)

        # Create a Draw object
        draw = ImageDraw.Draw(img)

        if not font_path:
            font_path = self.embeded_font

        # Load font
        font = ImageFont.truetype(font_path, font_size)

        # Split instructions into lines
        lines = instructions_text.split("\n")

        # Draw each line on the image
        y = 50
        for line in lines:
            text_size = draw.textsize(line, font=font)
            x = (img.width - text_size[0]) // 2
            draw.text((x, y), line, font=font, fill=font_color)
            y += text_size[1] + 20

        # Update Path so it works with PyInstaller binary
        path = resource_path(path)
        img.save(path, format='PNG', transparent=True)

        return path


class FrenchLanguage(UsedLanguage):
    def __init__(self):
        super().__init__()
        self.computer_win_img = self.create_text_image("./Images/languages/french/computer_win.png", "Vous perdez!")
        self.player_win_img = self.create_text_image("./Images/languages/french/player_win.png", "Vous gagnez!")
        self.tie_img = self.create_text_image("./Images/languages/french/tie.png", "Égalité")
        self.computer_img = self.create_text_image("./Images/languages/french/computer.png", "Ordinateur")
        self.you_img = self.create_text_image("./Images/languages/french/you.png", "Vous")
        self.play_string_img = self.create_text_image("./Images/languages/french/play_string.png", "Jouer")
        self.play_instr_string_img = self.create_text_image("./Images/languages/french/play_instr_string.png",
                                                            "Cliquez sur lecture ou appuyez sur n'importe " + \
                                                            "quelle touche pour commencer")
        self.rock_img = self.create_text_image("./Images/languages/french/rock.png", "Roche")
        self.paper_img = self.create_text_image("./Images/languages/french/paper.png", "Papier")
        self.scissors_img = self.create_text_image("./Images/languages/french/scissors.png", "Ciseaux")
        self.ngg_img = self.create_text_image("./Images/languages/french/ngg.png", "Pas un geste valable")
        self.no_landmark_img = self.create_text_image("./Images/languages/french/no_landmark.png",
                                                      "Pas de main detectée")
        self.language_in_game = self.create_text_image('./Images/languages/instr_in_game.png',
                                                       "Press 1 for English / Appuyez sur 2 pour le français")
        self.button_img = self.create_button(path="./Images/languages/french/play_fr.png", text="Jouer")

        # Create instructions image
        instructions_fr = """
        Bienvenue dans ROCK PAPER SCISSORS 
        1. Gardez vos mains plus près de la caméra 
        2. Assurez-vous que le système détecte votre main 
        3. Pendant le compte à rebours, gardez vos mains stables
        """
        self.game_instr_img = self.create_instructions(path="./Images/languages/french/game_instructions.png",
                                                       instructions_text=instructions_fr,
                                                       background_color=(0, 0, 0, 0),
                                                       font_color=(166, 202, 240))


class EnglishLanguage(UsedLanguage):
    def __init__(self):
        super().__init__()
        self.computer_win_img = self.create_text_image("./Images/languages/english/computer_win.png", "Computer wins!")
        self.player_win_img = self.create_text_image("./Images/languages/english/player_win.png", "Player wins!")
        self.tie_img = self.create_text_image("./Images/languages/english/tie.png", "Tie")
        self.computer_img = self.create_text_image("./Images/languages/english/computer.png", "Computer")
        self.you_img = self.create_text_image("./Images/languages/english/you.png", "You")
        self.play_string_img = self.create_text_image("./Images/languages/english/play_string.png", "Play")
        self.play_instr_string_img = self.create_text_image("./Images/languages/english/play_instr_string.png",
                                                            "Click play or press any key to start")
        self.rock_img = self.create_text_image("./Images/languages/english/rock.png", "Rock")
        self.paper_img = self.create_text_image("./Images/languages/english/paper.png", "Paper")
        self.scissors_img = self.create_text_image("./Images/languages/english/scissors.png", "Scissors")
        self.ngg_img = self.create_text_image("./Images/languages/english/ngg.png", "Not a valid gesture")
        self.no_landmark_img = self.create_text_image("./Images/languages/english/no_landmark.png", "No hand detected")
        self.language_in_game = self.create_text_image('./Images/languages/instr_in_game.png',
                                                       "Press 1 for English / Appuyez sur 2 pour le français")
        self.button_img = self.create_button(path="./Images/languages/english/play_eng.png", text="Play")
        # Create instructions image
        instructions_eng = """
                Welcome to Rock PAPER SCISSORS 
                1. Keep your hands closer to the camera 
                2. Make sure system detects your hand 
                3. During count down keep yours hands steady
                """
        self.game_instr_img = self.create_instructions(path="./Images/languages/english/game_instructions.png",
                                                       instructions_text=instructions_eng,
                                                       background_color=(0, 0, 0, 0),
                                                       font_color=(166, 202, 240))


# TODO: Include into Used Language Class
def change_language(UsedLanguage, NewLanguage):
    UsedLanguage.computer_win_img = NewLanguage.computer_win_img
    UsedLanguage.player_win_img = NewLanguage.player_win_img
    UsedLanguage.computer_img = NewLanguage.computer_img
    UsedLanguage.you_img = NewLanguage.you_img
    UsedLanguage.tie_img = NewLanguage.tie_img
    UsedLanguage.play_string_img = NewLanguage.play_string_img
    UsedLanguage.paper_img = NewLanguage.paper_img
    UsedLanguage.scissors_img = NewLanguage.scissors_img
    UsedLanguage.rock_img = NewLanguage.rock_img
    UsedLanguage.ngg_img = NewLanguage.ngg_img
    UsedLanguage.no_landmark_img = NewLanguage.no_landmark_img
    UsedLanguage.play_instr_string_img = NewLanguage.play_instr_string_img
    UsedLanguage.button_img = NewLanguage.button_img
    UsedLanguage.game_instr_img = NewLanguage.game_instr_img
    UsedLanguage.language_in_game = NewLanguage.language_in_game
