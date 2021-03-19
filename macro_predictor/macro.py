import cv2
import imageio
import tensorflow.compat.v1 as tf
import numpy as np

from macro_predictor.model import *

tf.disable_v2_behavior()


CASC_PATH = '/Users/yunfan/PycharmProjects/Predictor/env/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
MODEL = "./macro_predictor/models"
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


def crop(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if not len(faces) > 0:
        return None, None
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face

    face_coor = max_are_face
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("problem during resize")
        return None, None

    return image, face_coor


def predict(filename):
    results = []
    face_x = tf.placeholder(tf.float32, [None, 2304])
    y_conv = deepnn(face_x)
    probs = tf.nn.softmax(y_conv)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(MODEL)
    sess = tf.Session()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    loaded_video = imageio.get_reader(filename, 'ffmpeg')
    total = len(list(enumerate(loaded_video)))
    index = 0
    while index < total:
        print(index)
        image = loaded_video.get_data(index)
        detected_face, face_coor = crop(image)
        if detected_face is not None:
            tensor = image_to_tensor(detected_face)
            result = sess.run(probs, feed_dict={face_x: tensor})
        else:
            index += 1
            continue

        if result is not None:
            results.append(EMOTIONS[np.argmax(result[0])])
        else:
            results.append("")
        index += 1
    return results