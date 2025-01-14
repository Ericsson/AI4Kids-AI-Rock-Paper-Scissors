# Create EXE from the Python file

Once have a environment where the code is running perfectly this is a good start.

If it is in a conda env please refer the following page to install python and the dependencies 

[Install python in windows]()

Once you have everything setup we can start creating the EXE 

1. Install PyInstaller: Before creating an executable file using PyInstaller, you need to ensure that PyInstaller is installed on your system. You can install PyInstaller by running the following command in your terminal or command prompt:
    
    ```bash
    
    pip install pyinstaller
    ```
    
2. Prepare your code: Ensure that your code is running as expected and that all dependencies are installed.
3. Open a terminal or command prompt in the directory where your Python code is located.
4. Run PyInstaller: The command to create an executable file using PyInstaller is:
    
    ```bash
    
    pyinstaller --add-data "utils;utils" --add-data "Images;Images" -w main.py
    
    ```
    
    Let's break down the command and explain what each argument means:
    
    - **`-add-data`**: This argument specifies the additional data files or directories to be included in the executable file. The argument is followed by a path to the data file or directory, and a semicolon-separated list of target directories in the executable file. In the given command, we are adding two directories, **`utils`** and **`Images`**, to the executable file. The target directories in the executable file will be named **`utils`** and **`Images`**, respectively.
    - **`w`**: This argument tells PyInstaller to create a windowed application. If you omit this argument, PyInstaller will create a console application.
    - **`main.py`**: This is the name of your Python script.
5. After running the PyInstaller command, PyInstaller will create a **`dist`** directory in the same location as your Python script. Inside the **`dist`** directory, you will find the executable file with the same name as your Python script.

### Notes
- Media pipe has issues with reading the library values directly from the generated main.spec file. You have to update the code for mediapipe related files. The main.spec file attached to this project has been already updated with the given changes.

- If you don't use -w command in the exe generation it is hard to track where the code is failing.
