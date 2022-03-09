# It helps in identifying the faces
import cv2, sys, numpy, os
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'Photo_Sise_2'
 
# Part 1: Create fisherRecognizer
print('Recognizing Face Please Be in sufficient Lights...')
 
# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
subjectpath = os.path.join(datasets)
for filename in os.listdir(subjectpath):
    #print(filename)
    names[id]=filename.split(".")[0]
    print(names[id])
    path = subjectpath + '/' + filename
    label = id
    images.append(cv2.imread(path, 0))
    labels.append(int(label))
    id += 1
(width, height) = (130, 100)
 
# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
 
# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
print("odel")
# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        #print(prediction[0])
        #print(prediction[1])
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print("test")
        #print(prediction)
        print(names[prediction[0]])
        if prediction[1]<500:
 
           cv2.putText(im, '% s' %(names[prediction[0]]), (x-10, y-10),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
          cv2.putText(im, 'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
 
    cv2.imshow('OpenCV', im)
     
    key = cv2.waitKey(10)
    if key == 27:
        break