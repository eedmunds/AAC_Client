import socket
import time
import imutils
import imagezmq
import cv2

sender = imagezmq.ImageSender(connect_to='tcp://deepspeed4.ecn.purdue.edu:5556')
rpi_name = socket.gethostname()
time.sleep(2.0)
cam = cv2.VideoCapture(0)
if not cam.isOpened(): raise IOError('Was not able to access webcam')

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=600, height=400)
    recv_msg = sender.send_image(rpi_name, frame)
    print(recv_msg.decode())