import streamlit as st
import time

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

import dlib
import PIL.Image
from imutils import face_utils
import argparse
from pathlib import Path
import os
import ntpath
import streamlit as st


def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


def encode_face(image):
    face_locations = face_detector(image, 1)
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # DETECT FACES
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list


def easy_face_reco(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    # ENCODING FACE
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Unknown"
        face_names.append(name) 

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        #cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1) # bgr couleur du nom

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)
    return face_names


def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)


def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()


def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()


#------------------------------------------------------------------------------------
if __name__ == "__main__":    
    st.title("Webcam Application")
    run = st.button('Lancer la caméra',key=1)
    stop = st.button('Stopper la caméra')
    print("salut")
    print(stop)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    FRAME_WINDOW = st.image([]) 
    #chargement des modèles pré-entrainés pour les noms
    pose_predictor_68_point = dlib.shape_predictor("../pretrained_model/shape_predictor_68_face_landmarks.dat")
    pose_predictor_5_point = dlib.shape_predictor("../pretrained_model/shape_predictor_5_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("../pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
    face_detector = dlib.get_frontal_face_detector()
    st.write('[INFO] Importing pretrained model..')
    
    if run:
        with st.spinner('Wait for it...'):
            time.sleep(2)
        
    
    while run:
        
        ret, frame = cap.read()
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        st.write('[INFO] Importing faces...')
        #print('[INFO] Importing faces...')
    
        face_to_encode_path = Path("../known_faces")
        #print(face_to_encode_path)
        files = [file_ for file_ in face_to_encode_path.glob('*.jpg')]
        #print(files)
    
        for file_ in face_to_encode_path.glob('*.png'):
            files.append(file_)
        if len(files)==0:
            raise ValueError('No faces detect in the directory: {}'.format(face_to_encode_path))
        known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]
    
        known_face_encodings = []
        for file_ in files:
            image = PIL.Image.open(file_)
            image = np.array(image)
            face_encoded = encode_face(image)[0][0]
            known_face_encodings.append(face_encoded)
    
        st.write('[INFO] Faces well imported')
        st.write('[INFO] Starting Webcam...')
    
        face_classifier=cv2.CascadeClassifier('../haarcascades_models/haarcascade_frontalface_default.xml')
        emotion_model = load_model('../models/emotion_detection_model_100epochs.h5')
        class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
    
    
        # The gender model architecture
        GENDER_MODEL = 'weights/deploy_gender.prototxt'
        # The gender model pre-trained weights
    
        GENDER_PROTO = 'weights/gender_net.caffemodel'
        # Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
        # substraction to eliminate the effect of illunination changes
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        # Represent the gender classes
        GENDER_LIST = ['Male', 'Female']
        FACE_PROTO = "weights/deploy.prototxt.txt"
        FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        # The model architecture
        AGE_MODEL = 'weights/deploy_age.prototxt'
        # The model pre-trained weights
        AGE_PROTO = 'weights/age_net.caffemodel'
        # Represent the 8 age classes of this CNN probability layer
        AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                         '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
        # Initialize frame size
        frame_width = 1280
        frame_height = 720
        # load face Caffe model
        face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        # Load age prediction model
        age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
        # Load gender prediction model
        gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)
        
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) - 14
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        result = cv2.VideoWriter('filename.mp4', fourcc, fps,size)
        #stop = st.button('Stopper la caméra')
        list_sise = []
        
        while True:
            labels = []
            
            _, img = cap.read()
            # Take a copy of the initial image and resize it
            
            #FRAME_WINDOW = st.image([])
            frame = img.copy()
            
            #FRAME_WINDOW.image(frame)
            
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            # resize if higher than frame_width
            if frame.shape[1] > frame_width:
                frame = image_resize(frame, width=frame_width)
            # predict the faces
            faces = get_faces(frame)
            # Loop over the faces detected
            # for idx, face in enumerate(faces):
            for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
                
                roi_gray=gray[start_y:end_y,start_x:end_x]
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
                roi=roi_gray.astype('float')/255.0  #Scale
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)  #Expand dims to get it ready for prediction (1, 48, 48, 1)
                
                preds=emotion_model.predict(roi)[0]  #Yields one hot encoded result for 7 classes
                label=class_labels[preds.argmax()]  #Find the label
                label_position=(start_x,start_y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                
                
                face_img = frame[start_y: end_y, start_x: end_x]
                # predict age
                age_preds = get_age_predictions(face_img)
                # predict gender
                gender_preds = get_gender_predictions(face_img)
                i = gender_preds[0].argmax()
                gender = GENDER_LIST[i]
                gender_confidence_score = gender_preds[0][i]
                i = age_preds[0].argmax()
                age = AGE_INTERVALS[i]
                age_confidence_score = age_preds[0][i]
                # Draw the box
                label2 = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%"
                # label2 = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
                #print(label2)
                yPos = start_y - 15
                
                # detection si la personne appartient à la promo
                names = easy_face_reco(frame, known_face_encodings, known_face_names)
                
                while yPos < 15:
                    yPos += 15
                box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
                # Label processed image
                cv2.putText(frame, label2, (start_x, yPos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, box_color, 2)
                
                list_sise = list_sise + names
                
                # Display processed image
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            #cv2.imshow("Gender Estimator", frame)
            result.write(frame)
            result.release()
            liste_sise = set(list_sise)
            
            print(liste_sise)
            
            if (stop):
                print("hello")
                run=False
            
                with open('filename.txt', 'w') as f:
                    for item in liste_sise:
                        print(item)
                        f.write("%s\n" % item)
                break
                
        #break
            
            # if stop:
            #     print(run)
            #     print("Fermeture de la caméra")
            #     st.write("Fermeture de la caméra")
            #     #result.release()
            #     #st.write("Fermeture de la caméra")
            #     with open('filename.txt', 'w') as f:
            #         for item in liste_sise:
            #             print(item)
            #             f.write("%s\n" % item)
            #     break
        
            # else:
            #     print("ok")
    
    
    
    
    
        
    
    
