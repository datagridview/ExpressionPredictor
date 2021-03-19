#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
@Time        : 2021-03-17 16:47
@Author      : heyunfan
@Project     : micro-expression-recognition
@File        : models.py
@Description :
"""

import os
from typing import Any

import cv2
import numpy as np
import imageio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution3D, MaxPooling3D
from keras import backend as K
import tensorflow as tf

WEIGHTS1 = "/Users/yunfan/PycharmProjects/Predictor/micro_predictor/weights-improvement-53-0.88.hdf5"
WEIGHTS2 = "/Users/yunfan/PycharmProjects/Predictor/micro_predictor/weights-improvement-26-0.69.hdf5"
CASME_DEPTH = 96
SMIC_DEPTH = 18
CASME_TYPE = ["angry", "happy", "disgust"]
SMIC_TYPE = ["negative", "positive", "surprise"]

CASC_PATH = "/Users/yunfan/PycharmProjects/Predictor/env/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"


def init_model(model_type: str) -> Sequential:
    # input size definition
    image_rows, image_columns = 64, 64
    if model_type == "CASME":
        image_depth = CASME_DEPTH
    else:
        image_depth = SMIC_DEPTH

    # definition of models
    model = Sequential()
    model.add(Convolution3D(32, (3, 3, 15), input_shape=(image_rows, image_columns, image_depth, 1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, kernel_initializer='normal'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    model.summary()

    print("loading weights...")
    if model_type == "CASME":
        model.load_weights(WEIGHTS1)
    else:
        model.load_weights(WEIGHTS2)
    return model


def predict(model: Sequential, images: np.ndarray) -> np.ndarray:
    result = model.predict(images)
    return result


def crop(image: Any) -> Any:
    """

    :param image: a single image
    :return: cropped image
    """
    cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if not len(faces) > 0:
        return None

    # select the biggest face
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    face_coor = max_are_face

    # corp the face field
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    try:
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        print("problem during resize")
        return None
    return gray


def video2facelist(filename, depth) -> np.ndarray:
    """

    :param filename: video need dealing with
    :param depth: number of images in a set
    :return: images set list in form of ndarray
    """
    # check the past result
    if os.path.exists("{}_{}_result.npy".format(filename, str(depth))):
        image_sets = np.load("{}_{}_result.npy".format(filename, str(depth)))
        return image_sets
    videos = []
    frames = []
    loaded_video = imageio.get_reader(filename, 'ffmpeg')
    count_inset = 0
    index = 0
    total = len(list(enumerate(loaded_video)))
    while index < total:
        print("{}:{}".format(index, count_inset))
        image = loaded_video.get_data(index)
        image = crop(image)
        if image is None:
            frames = []
            index += 1
            count_inset = 0
            continue
        frames.append(image)
        tmp = np.asarray(frames)
        print(tmp.shape)
        count_inset += 1
        index += 1
        if count_inset % depth == 0:
            # f = copy.deepcopy(frames)
            # f = np.asarray(f)
            frames = np.asarray(frames)
            print(frames.shape)
            video = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
            videos.append(video)
            frames = []
            count_inset = 0

        print("length frames:", len(frames))
        print("length videos:", len(videos))

    n = len(videos)
    face_set = np.asarray(videos)

    a = np.zeros((n, 1, 64, 64, depth))
    for h in range(n):
        print(h)
        a[h][0][:][:][:] = face_set[h, :, :, :]
    a = a.astype("float32")
    a -= np.mean(a)
    a /= np.max(a)

    image_sets = []
    for i in a:
        a = tf.expand_dims(i[0], -1)
        image_sets.append(a)
    image_sets = np.array(image_sets)

    np.save("{}_{}_result.npy".format(filename, str(depth)), image_sets)
    return image_sets


def MicroExpSTCNN_with_CASME(model: Sequential, filename: str) -> np.ndarray:
    input_data = video2facelist(filename, CASME_DEPTH)
    output_data = predict(model, input_data)
    output = []
    for i in output_data:
        index = np.argmax(i)
        if i[index] > 0.5:
            output.append(CASME_TYPE[index])
        else:
            output.append("nothing")
    return output


def MicroExpSTCNN_with_SMIC(model: Sequential, filename: str) -> np.ndarray:
    input_data = video2facelist(filename, SMIC_DEPTH)
    output_data = predict(model, input_data)
    output = []
    for i in output_data:
        index = np.argmax(i)
        if i[index] > 0.55:
            output.append(SMIC_TYPE[index])
        else:
            output.append("nothing")
    return output
