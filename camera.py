
import numpy as np 
import argparse
import imutils 
import time
import cv2
import os
import path






class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        
        ####################################################
    
        self.labelsPath ="yolo/yolov3.txt"
        self.LABELS = open(self.labelsPath).read().strip().split("\n")
        np.random.seed(42)        
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),dtype="uint8")
        # derive the paths to the YOLO weights and model configuration
        self.weightsPath="yolo/yolov3.weights"
        self.configPath = "yolo/yolov3.cfg"

        print("WAIT Running the yolo model...")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.vs = cv2.VideoCapture(path.s)
        self.vs.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # # X = desired FPS
        # self.FPS = 500
        # self.FPS_MS = int(self.FPS * 1000)

        #self.vs= cv2.VideoCapture(0)
        (self.W, self.H) = (None, None)
        

   

    def __del__(self):
        self.vs.release()


    def get_frame(self):
        # loop over frames from the video file stream
        (grabbed, frame) = self.vs.read()
        # time.sleep(self.FPS)


        # if not grabbed:
        #     break

        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:

                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5, 0.3)
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],confidences[i])
                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
            

