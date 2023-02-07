# -*- coding: utf-8 -*-

import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime
from PIL import Image
import dlib

'''
# UNCOMMENT THE FOLLOWING CODE TO TRAIN THE CNN FROM SCRATCH

# BUILDING MODEL TO CLASSIFY BETWEEN MASK AND NO MASK

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=16 ,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')

model_saved=model.fit_generator(
        training_set,
        epochs=10,
        validation_data=test_set,

        )

model.save('mymodel.h5',model_saved)

#To test for individual images

mymodel=load_model('mymodel.h5')
#test_image=image.load_img('C:/Users/Karan/Desktop/ML Datasets/Face Mask Detection/Dataset/test/without_mask/30.jpg',target_size=(150,150,3))
#test_image=image.load_img(r'C:/Users/karan/Desktop/FaceMaskDetector/test/with_mask/1-with-mask.jpg',
#                          target_size=(150,150,3))
#test_image=keras.utils.load_img(r'C:/Users/karan/Desktop/FaceMaskDetector/test/with_mask/1-with-mask.jpg',
#                          target_size=(150,150,3))
test_image=keras.utils.load_img('./test/with_mask/1-with-mask.jpg', target_size=(150,150,3))
#test_image=image.img_to_array(test_image)
test_image=keras.utils.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
mymodel.predict(test_image)[0][0]

'''
# IMPLEMENTING LIVE DETECTION OF FACE MASK

mymodel=load_model('mymodel.h5')

# cap = cv2.VideoCapture("C:\\Users\\Misaki Sato\\Desktop\\mask\\test.mp4")
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

mouth = "mouth.jpg"
cv2_img = cv2.imread(mouth, cv2.IMREAD_UNCHANGED)

def overlayImage(src, overlay, location, size):
    overlay_height, overlay_width = overlay.shape[:2]

    # webカメラの画像をPIL形式に変換
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_src = Image.fromarray(src)
    pil_src = pil_src.convert('RGBA')

    # 合成したい画像をPIL形式に変換
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    pil_overlay = Image.fromarray(overlay)
    pil_overlay = pil_overlay.convert('RGBA')
    #顏の大きさに合わせてリサイズ
    pil_overlay = pil_overlay.resize(size)

    # 画像を合成
    pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
    pil_tmp.paste(pil_overlay, location, pil_overlay)
    result_image = Image.alpha_composite(pil_src, pil_tmp)

    # OpenCV形式に変換
    return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

while cap.isOpened():
    _,img=cap.read()
    face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)

    for(x,y,w,h) in face:

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        print(x,y,w,h)

        # 認識した顔を切り出す
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))      
        # cv2.imshow('face_img', face_img)

        # 下半分に顎を貼り付ける
        # overlayImage(src, overlay, location, size)
        processing_img = overlayImage(face_img, cv2_img, (0,100), (200, 100))
        # processing_img = cv2.cvtColor(processing_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('processing_img', processing_img)

        # # OpenFaceにかけて視線を抽出する：今はOpenCVの特徴点抽出ができるか試してみてる
        # dlib_shape = landmark_predictor(face_img,face)
        # shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
        # for s in shape_2d:
        #     cv2.circle(face_img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imshow('face_img', face_img)
          
    cv2.imshow('img',img)

    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()