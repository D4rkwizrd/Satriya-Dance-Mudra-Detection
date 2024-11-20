import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re 
import requests
import shutil
import time
import glob

from ultralytics import YOLO



app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')


opdir='runs/detect'
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            # print("printing predict_img :::::::", predict_img)
            print("printing predict_img :::::::", predict_img.imgpath)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.jpg', cv2.UMat(img)) [1].tobytes()
                print("continue...........................................................................")
                image = Image.open(io.BytesIO(frame))
                print("continue...........................................................................")
                # perfome the detection
                yolo = YOLO('best.pt')
                # detections = yolo.predict(image, save=True)
                results = yolo(filepath, conf=0.4, save=True, project=opdir)

                print("continue...........................................................................")
                print(results)
                return display(f.filename)              
            
            elif file_extension == 'mp4':
                video_path = filepath #replace with your video path
                cap = cv2.VideoCapture(video_path)

                #get video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                #Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                #initialize the YOLOv8 model here
                model = YOLO('best.pt')

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # do YOLOv8 detection on the frame here
                    results = model(frame, save =True) #working
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)

                    #write the frame to the output video
                    out.write(res_plotted)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

                    # for result in results:
                        # #class_id, confidence, bbox = result
                        # boxes = result.boxes  # Boxes object for bbox outputs
                        # probs = result.probs  # Class probabilities for classification outputs
                        # cls = boxes.cls
                        # xyxy = boxes.xyxy
                        # xywh = boxes.xywh  # box with xywh format, (N, 4)
                        # conf = boxes.conf
                        # print("boxes : ", boxes)
                        # print("probs : ", probs)
                        # print("cls - cls, (N, 1) : ", cls)
                        # print("conf - confidence score, (N, 1): ", conf)
                        # print("box with xyxy format, (N, 4) : ", xyxy)
                        # print("box with xywh format, (N, 4) : ", xywh)


                return video_feed()
                
    folder_path = 'runs/detect'
    # subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    subfolders = [f for f in os.listdir(opdir) if os.path.isdir(os.path.join(opdir, f))]
    # latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(opdir, x)))

    # image_path = folder_path+'/'+latest_subfolder+'/'+f.filename
    image_path = opdir+'/'+latest_subfolder+'/'+f.filename
    return render_template('index.html',image_path=image_path )
    #return "done"

## The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path)if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

    directory = folder_path+'/'+latest_subfolder
    print("printing directory:", directory)
    files = os.listdir(directory)
    latest_file = files[0]

    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file, environ) # shows the result in seperate tab
    
    else:
        return "Invalid file format"
    

    
def get_frame():
    folder_path = os.getcwd()
    mp4_file = 'output.mp4'
    video = cv2.VideoCapture(mp4_file)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)  # control the frame rate to display one frame every 100 milliseconds

# Function to stop the video feed
@app.route("/stop_video")
def stop_video_feed():
    global stop_video
    stop_video = True
    return "Video feed stopped."

# funtion to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Flask app exposing yolov8 models")
    parser.add_argument("--port", default = 5000, type=int, help="port number")
    args = parser.parse_args()

    


app.run(debug=True)
