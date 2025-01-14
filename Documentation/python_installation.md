# Install python in windows

Open windows powershell in an Administration mode
### Step 1

### **Go to User Directory**

### `cd ~`

### **Change Execution Policy Settings**

### `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### **Check List of Execution Policy**

### `Get-ExecutionPolicy -List`

**Explanation:**
Step 1 involves changing the PowerShell execution policy settings to allow running of unsigned scripts. The first line changes the current directory to the user directory (`~`). The second line changes the execution policy to RemoteSigned, which allows running of scripts that are locally created but blocks running of scripts downloaded from the internet that are not signed. The third line checks the current execution policy settings**.**

### Step 2

### **Connect to WebClient (shares Internet connection settings)**

### `$script = New-Object Net.WebClient`

### **Check if script is working + options (commands)**

### `$script | Get-Member`

### **Download & Check Signatures**

### `$script.DownloadString("**[https://chocolatey.org/install.ps1](https://chocolatey.org/install.ps1)**")`

### **Install Chocolatey**

### `iwr **[https://chocolatey.org/install.ps1](https://chocolatey.org/install.ps1)** -UseBasicParsing | iex`

### **Check if Chocolatey is installed**

### `choco -?`

### **Update Chocolatey**

### `choco upgrade chocolatey`

**Explanation:**
Step 2 installs Chocolatey, a package manager for Windows that makes it easy to install and manage software packages. The first two lines create a new WebClient object and check its properties. The third line downloads the Chocolatey installation script and checks its signature. The fourth line installs Chocolatey by running the downloaded script. The fifth line checks if Chocolatey is installed properly by running **`choco -?`**. The last line upgrades Chocolatey to the latest version.

### Step 3

### **Install Python via Chocolatey**

### `choco install python --version=3.8.0`

### **Refresh Environment Vars**

### `refreshenv`

### **Check if Python is installed**

### `python -V`

### **Make Sure PIP is installed and upgraded**

### `python -m pip install --upgrade pip`

**Explanation:**
Step 3 installs Python 3 using Chocolatey, refreshes the environment variables to make the installation visible to PowerShell, checks the Python version, and ensures that pip (Python package manager) is installed and upgraded to the latest version. The first line installs Python 3 using Chocolatey. The second line refreshes the environment variables so that PowerShell can see the newly installed Python. The third line checks the installed Python version. The last line upgrades pip to the latest version.

### Step 4

### **Install Required Packages using pip**

### `pip install -r requirements.txt`

**Explanation:**
Step 4 involves installing the required packages for your Python project using pip. If your project has external dependencies, you can list them in a **`requirements.txt`** file, which can be used to install all required packages at once. The **`pip install -r requirements.txt`** command installs all the packages listed in the **`requirements.txt`** file.

### Notes
- Mediapipe has dependency issues with python 3.10 and above.