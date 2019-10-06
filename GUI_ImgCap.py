import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QErrorMessage, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, QRect, QTimer, Qt
from pyqtgraph import ImageView
import os
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.WARN)
import paramiko
from scp import SCPClient, SCPException

'''

PURPOSE:
To capture a short video from the client's webcam, from which we will extract frames.
We will transfer these frames to the server, and will use them for to train our facial recognition.

USAGE:
From terminal, run python3 GUI_ImgCap.py. Assuming packages are installed correctly,
popup window will prompt to capture video. Let the video capture for ~10 seconds, it will stop on its own.
Please press the 'Exit' button on the popup window instead of the red 'X' to close the window :).

'''

# Local folders for outgoing videos
VID_FOLDER = 'train_videos/'

# Remote folder where we will be writing videos to
AAC_FOLDER = 'AAC_Accuracy'
SERVER_HOME_DIR = '~/{}/server/'.format(AAC_FOLDER)
REMOTE_VID_FOLDER = '~/{}/server/training/videos/'.format(AAC_FOLDER)
REMOTE_IMG_FOLDER = '~/{}/server/training/images/'.format(AAC_FOLDER)

def ssh_connection():
    # Info for SSH login
    USERNAME = 'vipaacc'
    PASSWORD_FILE = 'password.txt'
    SERVER = 'deepspeed4.ecn.purdue.edu'

    if not os.path.isdir(VID_FOLDER): os.mkdir(VID_FOLDER)
    # Get password
    if os.path.isfile(PASSWORD_FILE):
        with open(PASSWORD_FILE) as fp:
            PASSWORD = fp.read() # To avoid storing password on GitHub
    else:
        PASSWORD = input('Please enter the password to {}@{}: '.format(USERNAME, SERVER))
        with open(PASSWORD_FILE, 'w') as fp:
            fp.write(PASSWORD)

    # Set up SSH connection
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect with deepspeed4 server
    ssh.connect(SERVER, username=USERNAME, password=PASSWORD)

    #Make sure the REMOTE_FOLDER is a valid directory in DeepSpeed4
    _, stdout, _ = ssh.exec_command('ls {}'.format(SERVER_HOME_DIR))
    if 'training' not in stdout.readlines():
        ssh.exec_command('mkdir {}/training'.format(SERVER_HOME_DIR))
        ssh.exec_command('mkdir {}'.format(REMOTE_IMG_FOLDER))
        ssh.exec_command('mkdir {}'.format(REMOTE_VID_FOLDER))

    return ssh

class WebCam:
    def __init__(self, cam_num, ssh):
        self.cam_num = cam_num
        self.last_frame = np.zeros((1,1))
        self.ssh = ssh
        self.scp = SCPClient(self.ssh.get_transport(), socket_timeout=None)

    def initialize(self):
        self.cap = cv2.VideoCapture(self.cam_num)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

    def getSingleCap(self):
        ret, self.last_frame = self.cap.read()
        return self.last_frame

    def getMultiFrames(self, frames, name):
        VID_NAME = '{}.mp4'.format(name)
        vid_out = cv2.VideoWriter(VID_FOLDER + VID_NAME, cv2.VideoWriter_fourcc(*'MP4V'), 10, (self.frame_width, self.frame_height))
        for i in range(frames):
            frame = self.getSingleCap()
            vid_out.write(frame)
        vid_out.release()
        self.videoTransfer(VID_FOLDER + VID_NAME, VID_NAME)

    def videoTransfer(self, filename, VID_NAME):
        try:
            self.scp.put(filename, REMOTE_VID_FOLDER + VID_NAME)
        except SCPException:
            raise IOError("Remote directory not available")

    def close_camera(self):
        self.cap.release()
        return self.ssh

class MovieThread(QThread):
    def __init__(self, webcam, name, btnCap, btnExit, statusUpdate):
        super().__init__()
        self.webcam = webcam
        self.name = name
        self.btnCapture = btnCap
        self.btnExit = btnExit
        self.statusUpdate = statusUpdate
    def run(self):
        self.webcam.getMultiFrames(100, self.name)
        self.btnCapture.setEnabled(True)
        self.btnExit.setEnabled(True)
        self.statusUpdate.setText("Ready to Capture")

class StartWindow(QMainWindow):
    def __init__(self, webcam):
        super().__init__()
        self.webcam = webcam

        self.central_widget = QWidget()
        self.btnCapture = QPushButton("Capture", self.central_widget)
        self.btnExit = QPushButton("Exit", self.central_widget)
        self.nameText = QLabel("Please Enter Your Name: ", self.central_widget)
        self.userName = QLineEdit(self.central_widget)
        self.statusBar = QLabel("Status:", self.central_widget)
        self.statusUpdate = QLabel("Ready to Capture", self.central_widget)
        self.image_view = ImageView()

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.image_view)
        self.layout.addWidget(self.nameText)
        self.layout.addWidget(self.userName)
        self.layout.addWidget(self.statusBar)
        self.layout.addWidget(self.statusUpdate)
        self.layout.addWidget(self.btnCapture)
        self.layout.addWidget(self.btnExit)
        self.setCentralWidget(self.central_widget)

        #Create a timer object. Every timer timeout results in imageviewer being updated
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_movie)

        #Connect the capture button with start_movie()
        self.btnCapture.clicked.connect(self.start_movie)
        self.btnExit.clicked.connect(self.exitApp)
        self.userName.textChanged.connect(self.btnChange)

        self.btnCapture.setEnabled(False)

    def btnChange(self):
        #Only enable the capture button when a name is entered
        if self.userName.text():
            self.btnCapture.setEnabled(True)
        else:
            self.btnCapture.setEnabled(False)

    def update_movie(self):
        #Update the image viewer with the last frame of the self.webcam object
        self.image_view.setImage(self.webcam.last_frame.T)

    def start_movie(self):
        #Start the MovieThread if user clicks "Capture" button
        #Set timer to timeout every 30 ms
        #Disable buttons during capturing
        self.nameEntered = self.userName.text()
        self.btnCapture.setEnabled(False)
        self.btnExit.setEnabled(False)
        self.statusUpdate.setText("Capturing...")
        self.movie_thread = MovieThread(self.webcam, self.nameEntered, self.btnCapture, self.btnExit, self.statusUpdate)
        self.movie_thread.start()
        self.update_timer.start(30)

    def exitApp(self):
        #Upon exiting the application, run vid2faces.py on deepspeed4
        ssh = self.webcam.close_camera()
        self.btnCapture.setEnabled(False)
        self.btnExit.setEnabled(False)
        self.statusUpdate.setText("Converting videos into images on deepspeed4")
        stdin, stdout, stderr = ssh.exec_command('python3 {}vid2faces.py'.format(SERVER_HOME_DIR))
        print("Output from vid2faces.py:")
        for line in stdout.readlines():
            print(line.replace('\n', '').replace('\t',''))
        if not stderr:
            for line in stderr.readlines():
                print(line.replace('\n', '').replace('\t',''))
            raise ValueError("Remote execution throws the exception above")
        self.close()


if __name__ == "__main__":
    ssh = ssh_connection()
    main_cam = WebCam(0, ssh)
    main_cam.initialize()
    currentApp = QApplication(sys.argv)
    currWindow = StartWindow(main_cam)
    currWindow.show()
    currentApp.exit(currentApp.exec_())
