from flask import Flask, render_template, Response
from camera import VideoCamera
import path
import numpy as np 
import argparse
import imutils 
import time
import cv2
import os


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
	
    while True:
        frame = camera.get_frame()
        #time.sleep(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    ap= argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,help="path to input video")
	#ap.add_argument("-o", "--output", required=True,help="path to output video")
	#ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
	#ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())
    path.s = args["input"]
    
	
    app.run(host='0.0.0.0', debug=True)	
