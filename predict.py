import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.initializers import glorot_uniform

#Loading the model
model=load_model("iSpeakCNNModel.h5")

#Capturing ROI
def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

cam_capture = cv2.VideoCapture(0)

while True:  
    _, image_frame = cam_capture.read()  
    image_frame = cv2.flip(image_frame, 1)
    #Select ROI
    cv2.rectangle(image_frame, (300,100), (600,400), (255,0,0) ,1)
    
    #Capture image from ROI
    im2 = crop_image(image_frame, 300,100,300,300)
    
    #Processing the captured image
    image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
    im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)
    im4 = np.resize(im3, (28, 28, 1))
    im5 = np.expand_dims(im4, axis=0)
    img=im5.reshape(1,28,28,1)
    
    #Sending the image to the model
    img_class = model.predict_classes(img)
    prediction = img_class[0]
    category = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    pred=category[prediction]
    
    # Displaying the predictions
    cv2.putText(image_frame, pred, (10, 120), cv2.FONT_HERSHEY_PLAIN, 6, (0,0,0), 3)    
    cv2.imshow("Frame", image_frame)
    
    cv2.imshow("test", im3)
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cam_capture.release()
cv2.destroyAllWindows()
