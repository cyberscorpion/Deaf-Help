from django.shortcuts import render
import numpy as np
import urllib
import json
import cv2
import os
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
import keras
#import pandas as pd
#import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import imutils
#import easydict
#import keyboard
from keras.models import load_model
from django.core.files.storage import FileSystemStorage

from django.http import JsonResponse
# define the path to the face detector which would be an xml file that comes installed with haarcascades
# find it here -> https://github.com/opencv/opencv/tree/master/data/haarcascades
# download and save it in your project repository

face_detector = "/home/rajat/Desktop/monor lip reading/face-detection-opencv-api/face_detector/haarcascade_frontalface_default.xml"

@csrf_exempt

def rect_to_bb(rect):
    x=rect.left()
    y=rect.top()
    w=rect.right()-x
    h=rect.bottom()-y
    
    return (x,y,w,h)

def empty_distance(number):
    shape=np.zeros((20,number))
    return shape

def preprocess_dis(dis_array):
    x=dis_array-np.mean(dis_array)
    if np.std(dis_array)!=0:
        x/=np.std(dis_array)
    return x

def convert_points_to_distance(points):

    mid_x=(points[14,0]+points[18,0])/2
    mid_y=(points[14,1]+points[18,1])/2
    mid_point=np.array([mid_x,mid_y])
    points=(points-mid_point)**2
    distances_array=np.sqrt((points[:,0]+points[:,1]))
    
    return preprocess_dis(distances_array)
    
def face_landmarks(image):
    #all_images=[]
    #for image in images:
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor(args["shape_predictor"])
    
    rects=detector(gray,1)
    shape=np.zeros((20,2))
    for rect in rects:
        shape=predictor(gray,rect)
        shape=shape_to_np(shape)
        shape=shape[48:68]
    #all_images.append(shape)

    return shape   

def shape_to_np(shape,dtype='int'):
    coords=np.zeros((68,2),dtype=dtype)
    for i in range(0,68):
        coords[i]=(shape.part(i).x,shape.part(i).y)
        
    return coords

args={
    "shape_predictor":"/home/rajat/Desktop/monor lip reading/face-detection-opencv-api/facedetection/shape_predictor_68_face_landmarks.dat",
}
    
    
@csrf_exempt
def face_detection(request):
    default = {"safely executed": False} 
    if request.method == "POST":
        if request.FILES.get("video", None) is not None:
#            image_to_read = read_image(stream = request.FILES["image"])
            fs = FileSystemStorage()
            filename = fs.save('rajat.mp4', request.FILES["video"])
            uploaded_file_url = fs.url(filename)
            cap=cv2.VideoCapture('/home/rajat/Desktop/monor lip reading/face-detection-opencv-api/'+uploaded_file_url)
    else:
        print("not")
#    cap=cv2.VideoCapture('/home/rajat/Desktop/face detection/face-detection-opencv-api/face_detector/vid.mp4')
    ret, frame = cap.read()
    
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tf.reset_default_graph()
    tf.global_variables_initializer()
    
    
    images_array=[]
    count=0
    while ret:
        print(count)
        count+=1
        points=face_landmarks(frame)
        points=convert_points_to_distance(points)
        #cv2.putText(frame,"Start",(150,150),cv2.FONT_HERSHEY_COMPLEX,1,(150,150,150),2,cv2.LINE_AA)
        images_array.append(points)
        ret,frame=cap.read()
        ret,frame=cap.read()
        ret,frame=cap.read()
        ret,frame=cap.read()
        ret,frame=cap.read()
        ret,frame=cap.read()
        ret,frame=cap.read()
        ret,frame=cap.read()
        ret,frame=cap.read()
        
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
#        ret,frame=cap.read()
        
        #print(ret)
        
    print("eee")
    images_size=len(images_array)
    if images_size<=10:
        images_array=np.append(images_array,empty_distance(10-images_size)).reshape(10,20)
    else:
        move=images_size/10.0
        images_array=np.array(images_array)
        images_array=images_array[[int(i) for i in np.arange(0,images_size,move)]]

    images_array=np.array(images_array)
        
    with keras.backend.get_session().graph.as_default():
        model2=load_model('/home/rajat/Desktop/monor lip reading/face-detection-opencv-api/face_detector/minor2.h5')
        print(model2.predict_classes(images_array[None]))
        xx=model2.predict_classes(images_array[None])
    
    
#    count =0 
#    print(success)
#    while success:
#        print("dd")
#        count+=1
        
#    if request.method == "POST":
            
#        if request.FILES.get("image", None) is not None:
#            vidcap=cv2.VideoCapture(request.FILES["image"])
           
#            image_to_read = read_image(stream = request.FILES["image"])

    keywords=[ 'Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web',
    ]
    contest={
        'predicted_word':keywords[xx[0]],
        'video_url':uploaded_file_url,
    }
    return render(request,'video_result.html',contest)



#def requested_url(request):
#    default = {"safely executed": False} 
#    
#    if request.method == "POST":
#        if request.FILES.get("image", None) is not None:
#            image_to_read = read_image(stream = request.FILES["image"])
#
#        else: # URL is provided
#            url_provided = request.POST.get("url", None)
#
#            if url_provided is None:
#                default["error_value"] = "There is no URL Provided"
#
#                return JsonResponse(default)
#
#            image_to_read = read_image(url = url_provided)
#
#        #save file in local storage
#        x=cv2.imwrite('media/test.png',image_to_read)
#        image_to_read = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2GRAY)
#
#        detector_value = cv2.CascadeClassifier(face_detector)
#            #passing the face detector path make sure to pass the complete path to the .xml file
#
#        values = detector_value.detectMultiScale(image_to_read,
#                                                 scaleFactor=1.1,
#                                                 minNeighbors = 5,
#                                                 minSize=(30,30),
#                                                 flags = cv2.CASCADE_SCALE_IMAGE)
#
#        ###dimensions for boxes that will pop up around the face
#        values=[(int(a), int(b), int(a+c), int(b+d)) for (a,b,c,d) in values]
#
#        default.update({"no_of_faces": len(values),
#                        "faces":values,
#                        "safely_executed": True })
#
#    contest={
#        'default':default
#    }
#    return render(request,'results.html',contest)
#    return JsonResponse(default)

def read_image(path=None, stream=None, url=None):
    ## load the image from your local repository

    if path is not None:
        image = cv2.imread(path)

    else:
        if url is not None:

            response = urllib.request.urlopen(url)
            data_temp = response.read()

        elif stream is not None:
            data_temp = stream.read()

        image = np.asarray(bytearray(data_temp), dtype="uint8")

        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def upload_page(request):
    return render(request,'ss.html')

def video_page(request):
    return render(request,'video.html')

def home(request):
    return render(request,'index.html')